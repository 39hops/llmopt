"""Tabula rasa round 1 (spec: 2026-07-07-tabula-rasa-design.md): the
from-scratch lineage's first expert-iteration step. The proposer
trained ONLY on round-0 random-search wins (proposer_tr_r1.pt) drives
the search; eval stays count_ops (no HCE/NNUE — knowledge must come
from the lineage's own data, only the verifier is given). Race vs the
round-0 random engine on FRESH problems (r0 roots excluded by srepr),
harvest r1 wins for the next round's curriculum.

  python scripts/tabula_rasa_r1.py --per-cell 20
Output: data/tr_round1.jsonl + roots json + per-cell r0-vs-r1 table.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import sympy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, successors
from llmopt.search.proposer import hf_score_fn, make_proposer
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 180
BUDGET = 200
RESTARTS = 3
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


def count_ops_eval(state: State) -> float:
    return float(sp.count_ops(state.expr))


def load_tr_proposer():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load("checkpoints/proposer_tr_r1.pt", weights_only=True,
                   map_location="cpu"), strict=False)
    model.eval()
    return make_proposer(hf_score_fn(model, tok, device))


def random_proposer(seed: str):
    rng = random.Random(seed)

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


def _root(rng, level, kind):
    if kind == "diff":
        f = _expression(rng, level)
        return sp.Derivative(f, X), sp.diff(f, X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X), g


def _check(kind, expr, truth):
    if kind == "diff":
        return sp.simplify(expr - truth) == 0
    return sp.simplify(sp.diff(expr, X) - truth) == 0


def solve_r0(root, seed):
    per = BUDGET // RESTARTS
    for i in range(RESTARTS):
        signal.alarm(WALL)
        try:
            r = beam_search(root, width=2, max_plies=20, max_nodes=per,
                            eval_fn=count_ops_eval, verify_p=0.1,
                            proposer=random_proposer(f"{seed}-{i}"),
                            propose_k=1)
        except _Timeout:
            continue
        finally:
            signal.alarm(0)
        if r.solved:
            return r
    return None


def solve_r1(root, prop):
    signal.alarm(WALL)
    try:
        r = beam_search(root, width=2, max_plies=20, max_nodes=BUDGET,
                        eval_fn=count_ops_eval, verify_p=0.1,
                        proposer=prop, propose_k=3)
    except _Timeout:
        return None
    finally:
        signal.alarm(0)
    return r if r.solved else None


def path_rows(root, history):
    rows, cur = [], State(root)
    for chosen in history:
        kids = list(successors(cur))
        labels = [name for name, _ in kids]
        if chosen not in labels:
            return rows
        idx = labels.index(chosen)
        if len(labels) > 1:
            rows.append({"state": sp.sstr(cur.expr), "moves": labels,
                         "answer": idx})
        cur = kids[idx][1]
    return rows


def main(per_cell: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    prop = load_tr_proposer()
    exclude = set(json.loads(Path("data/tr_round0_roots.json").read_text()))
    out = Path("data/tr_round1.jsonl")
    roots, rows_n = [], 0
    print(f"{'kind':>4} {'lvl':>3} {'r0-random':>10} {'r1-prop3':>9}")
    tot0 = tot1 = 0
    with out.open("w") as f:
        for kind in ("diff", "int"):
            for level in (1, 2, 3, 4):
                rng = random.Random(f"tr-r1-{kind}-{level}-0")
                ok0 = ok1 = 0
                done = 0
                while done < per_cell:
                    root, truth = _root(rng, level, kind)
                    if sp.srepr(root) in exclude:
                        continue  # never race on round-0 curriculum
                    done += 1
                    r0 = solve_r0(root, f"r1base-{kind}-{level}-{done}")
                    ok0 += (r0 is not None
                            and _check(kind, r0.state.expr, truth))
                    r1 = solve_r1(root, prop)
                    if r1 is not None and _check(kind, r1.state.expr, truth):
                        ok1 += 1
                        roots.append(sp.srepr(root))
                        for row in path_rows(root, r1.state.history):
                            f.write(json.dumps(row) + "\n")
                            rows_n += 1
                tot0 += ok0
                tot1 += ok1
                print(f"{kind:>4} {level:>3} {ok0:>7}/{per_cell:<2} "
                      f"{ok1:>6}/{per_cell:<2}", flush=True)
    Path("data/tr_round1_roots.json").write_text(json.dumps(roots))
    print(f"ROUND 1: r0 {tot0} vs r1 {tot1} "
          f"(delta {tot1 - tot0:+d}), {rows_n} rows harvested")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=20)
    main(ap.parse_args().per_cell)

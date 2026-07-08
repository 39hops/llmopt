"""Tabula rasa round 2: r2 proposer (trained on rounds 0+1) vs r1
proposer, head-to-head on FRESH problems (roots from both prior rounds
excluded). The from-scratch lineage's curve: r0 random 112 -> r1 138
(+26). Does round 2 keep climbing, or plateau early like the mature
lineage's 40v40? Eval stays count_ops (lineage purity); winners
harvested for a possible round 3.

  python scripts/tabula_rasa_r2.py --per-cell 20
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
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


def count_ops_eval(state: State) -> float:
    return float(sp.count_ops(state.expr))


def load_proposer(ckpt: str):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load(ckpt, weights_only=True, map_location="cpu"),
        strict=False)
    model.eval()
    return make_proposer(hf_score_fn(model, tok, device))


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


def solve_with(root, prop):
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
    # one model in memory at a time is kinder to MPS; swap per arm
    # would reload constantly — load both, bf16 0.5B x2 fits in 36GB
    p1 = load_proposer("checkpoints/proposer_tr_r1.pt")
    p2 = load_proposer("checkpoints/proposer_tr_r2.pt")
    exclude = set(json.loads(Path("data/tr_round0_roots.json").read_text()))
    exclude |= set(json.loads(Path("data/tr_round1_roots.json").read_text()))
    out = Path("data/tr_round2.jsonl")
    roots, rows_n = [], 0
    print(f"{'kind':>4} {'lvl':>3} {'r1':>7} {'r2':>7}")
    tot1 = tot2 = 0
    with out.open("w") as f:
        for kind in ("diff", "int"):
            for level in (1, 2, 3, 4):
                rng = random.Random(f"tr-r2-{kind}-{level}-0")
                ok1 = ok2 = done = 0
                while done < per_cell:
                    root, truth = _root(rng, level, kind)
                    if sp.srepr(root) in exclude:
                        continue
                    done += 1
                    r1 = solve_with(root, p1)
                    ok1 += (r1 is not None
                            and _check(kind, r1.state.expr, truth))
                    r2 = solve_with(root, p2)
                    if r2 is not None and _check(kind, r2.state.expr, truth):
                        ok2 += 1
                        roots.append(sp.srepr(root))
                        for row in path_rows(root, r2.state.history):
                            f.write(json.dumps(row) + "\n")
                            rows_n += 1
                tot1 += ok1
                tot2 += ok2
                print(f"{kind:>4} {level:>3} {ok1:>4}/{per_cell:<2} "
                      f"{ok2:>4}/{per_cell:<2}", flush=True)
    Path("data/tr_round2_roots.json").write_text(json.dumps(roots))
    print(f"ROUND 2: r1 {tot1} vs r2 {tot2} (delta {tot2 - tot1:+d}), "
          f"{rows_n} rows harvested")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=20)
    main(ap.parse_args().per_cell)

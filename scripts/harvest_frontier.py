"""Expert-iteration round 2, harvest phase (spec:
2026-07-07-expert-iteration-r2-design.md).

Frontier = baseline engine (full enumeration + HCE, 200 nodes) FAILS.
Harvest = full engine (prop3 + HCE, 400 nodes; 3 k1-random restarts on
failure) SOLVES. Winning paths from harvested problems are the round-2
training rows: derivations that provably exist only because of the
learned components. Runs where the proposer model lives (CUDA box).

  python scripts/harvest_frontier.py --per-cell 40
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
BASE_WALL, FULL_WALL = 120, 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


def load_proposer():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load("checkpoints/proposer_lora.pt", weights_only=True,
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


def _run(root, wall, **kw):
    signal.alarm(wall)
    try:
        return beam_search(root, width=8, max_plies=24, **kw)
    except _Timeout:
        return None
    finally:
        signal.alarm(0)


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
    prop = load_proposer()
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    out = Path("data/frontier_r1.jsonl")
    out.parent.mkdir(exist_ok=True)
    roots: list[str] = []
    n_frontier = n_harvest = n_rows = 0
    with out.open("w") as f:
        for kind in ("diff", "int"):
            rng = random.Random(f"eir2-harvest-{kind}-4-0")
            for i in range(per_cell):
                root, truth = _root(rng, 4, kind)
                base = _run(root, BASE_WALL, max_nodes=200)
                if base is not None and base.solved and _check(
                        kind, base.state.expr, truth):
                    continue  # baseline solves it: not frontier
                n_frontier += 1
                r = _run(root, FULL_WALL, max_nodes=400, proposer=prop,
                         propose_k=3)
                if not (r is not None and r.solved
                        and _check(kind, r.state.expr, truth)):
                    for j in range(3):  # k1 random restarts
                        r = _run(root, BASE_WALL, max_nodes=133,
                                 proposer=random_proposer(f"h-{kind}-{i}-{j}"),
                                 propose_k=1)
                        if (r is not None and r.solved
                                and _check(kind, r.state.expr, truth)):
                            break
                    else:
                        continue  # frontier stays unsolved
                n_harvest += 1
                roots.append(sp.srepr(root))
                for row in path_rows(root, r.state.history):
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
                if (i + 1) % 10 == 0:
                    print(f"{kind}: {i + 1}/{per_cell} problems, "
                          f"frontier {n_frontier}, harvested {n_harvest}, "
                          f"rows {n_rows}", flush=True)
    Path("data/frontier_r1_roots.json").write_text(json.dumps(roots))
    yld = 100 * n_harvest / max(1, n_frontier)
    print(f"FRONTIER {n_frontier}, HARVESTED {n_harvest} ({yld:.0f}%), "
          f"rows {n_rows}")
    if yld < 5:
        print("NULL: yield < 5% — guidance is not the binding constraint; "
              "new rewrite rules are (rung 3). Per spec, STOP here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=40)
    a = ap.parse_args()
    main(a.per_cell)

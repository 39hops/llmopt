"""Tabula rasa round 0 (spec: 2026-07-07-tabula-rasa-design.md): the
AlphaZero-way lineage's first harvest. NO hand-crafted knowledge:
random k=1 dives with restarts, eval = count_ops ONLY (no unsolved-
weighting — that's HCE knowledge), no proposer, no NNUE. Only the
verifier survives (the game rules). Winning paths from whatever random
search solves become the from-scratch lineage's first training data.

  python scripts/tabula_rasa_r0.py --per-cell 20
Output: data/tr_round0.jsonl + roots json + yield stats.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, successors

X = sp.Symbol("x")
WALL = 60
BUDGET = 200
RESTARTS = 3


class _Timeout(BaseException):
    pass


def count_ops_eval(state: State) -> float:
    return float(sp.count_ops(state.expr))  # knowledge-free tie-break


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
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    out = Path("data/tr_round0.jsonl")
    out.parent.mkdir(exist_ok=True)
    roots, n_solved, n_rows, n_total = [], 0, 0, 0
    with out.open("w") as f:
        for kind in ("diff", "int"):
            for level in (1, 2, 3, 4):
                rng = random.Random(f"tr-r0-{kind}-{level}-0")
                for i in range(per_cell):
                    root, truth = _root(rng, level, kind)
                    n_total += 1
                    r = solve_r0(root, f"{kind}-{level}-{i}")
                    if r is None or not _check(kind, r.state.expr, truth):
                        continue
                    n_solved += 1
                    roots.append(sp.srepr(root))
                    for row in path_rows(root, r.state.history):
                        f.write(json.dumps(row) + "\n")
                        n_rows += 1
                print(f"{kind} L{level}: solved {n_solved}/{n_total}, "
                      f"rows {n_rows}", flush=True)
    Path("data/tr_round0_roots.json").write_text(json.dumps(roots))
    print(f"ROUND 0: {n_solved}/{n_total} solved ({100*n_solved/n_total:.0f}%), "
          f"{n_rows} rows — the from-scratch lineage's first curriculum")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=20)
    a = ap.parse_args()
    main(a.per_cell)

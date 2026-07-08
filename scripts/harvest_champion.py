"""Champion harvest: winning paths from the CURRENT best structural
engine (engine.solve: markov3 @ w2, all autopsy rules, smoothing) on
fresh problems. Motivation: the prior-pollution null — a mined prior
inherits the policy quality of its paths, so the proper re-mine needs
paths from an engine at least as strong as the prior's user, and no
such harvest exists post-rules. Output rows feed the prior re-mine
(and future proposer training).

  python scripts/harvest_champion.py --per-cell 40
Output: data/champion_harvest.jsonl + per-cell yields.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, successors
from llmopt.search.engine import solve

X = sp.Symbol("x")
WALL = 180
BUDGET = 300


class _Timeout(BaseException):
    pass


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


def path_rows(root, history):
    rows, cur = [], State(root)
    for chosen in history:
        kids = list(successors(cur, use_macros=True))
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
    out = Path("data/champion_harvest.jsonl")
    rows_n = solved = total = 0
    with out.open("w") as f:
        for kind in ("diff", "int"):
            for level in (1, 2, 3, 4):
                rng = random.Random(f"champ-harvest-{kind}-{level}-0")
                ok = 0
                for _ in range(per_cell):
                    root, truth = _root(rng, level, kind)
                    total += 1
                    signal.alarm(WALL)
                    try:
                        r = solve(root, budget=BUDGET)
                    except _Timeout:
                        continue
                    finally:
                        signal.alarm(0)
                    if not (r.solved and _check(kind, r.state.expr, truth)):
                        continue
                    ok += 1
                    solved += 1
                    for row in path_rows(root, r.state.history):
                        f.write(json.dumps(row) + "\n")
                        rows_n += 1
                print(f"{kind} L{level}: {ok}/{per_cell}, rows {rows_n}",
                      flush=True)
    print(f"CHAMPION HARVEST: {solved}/{total} solved, {rows_n} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=40)
    main(ap.parse_args().per_cell)

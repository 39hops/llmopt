"""Rung-1 solve-rate bench + macro ablation (spec: macros earn a slot
only if they win on solve-rate-per-node).

  python scripts/bench_derivation.py --levels 1 2 3 --n 30
  python scripts/bench_derivation.py --macros            # ablation arm
"""

from __future__ import annotations

import argparse
import random
import statistics

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search

X = sp.Symbol("x")


def run(levels: list[int], n: int, width: int, max_plies: int,
        max_nodes: int | None, use_macros: bool) -> None:
    tag = "macros ON" if use_macros else "core rules only"
    print(f"# rung-1 bench — {tag}, width={width}, max_plies={max_plies}, "
          f"max_nodes={max_nodes}")
    print(f"{'level':>5} {'solved':>10} {'mean nodes':>11} {'mean plies':>11}")
    for level in levels:
        rng = random.Random(f"bench-deriv-{level}-0")  # string seed
        solved, nodes, plies = 0, [], []
        for _ in range(n):
            f = _expression(rng, level)
            r = beam_search(sp.Derivative(f, X), width=width,
                            max_plies=max_plies, max_nodes=max_nodes,
                            use_macros=use_macros)
            ok = r.solved and sp.simplify(r.state.expr - sp.diff(f, X)) == 0
            solved += ok
            nodes.append(r.nodes)
            if ok:
                plies.append(r.state.plies)
        mp = statistics.mean(plies) if plies else float("nan")
        print(f"{level:>5} {solved:>6}/{n:<3} {statistics.mean(nodes):>11.1f} "
              f"{mp:>11.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--macros", action="store_true")
    a = ap.parse_args()
    run(a.levels, a.n, a.width, a.max_plies, a.max_nodes, a.macros)

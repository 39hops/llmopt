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


def _make_problem(rng: random.Random, level: int, kind: str):
    """Returns (root_expr, oracle_check). Integrands are reverse-sampled
    (draw F, present F') so every problem is solvable in principle."""
    if kind == "diff":
        f = _expression(rng, level)
        truth = sp.diff(f, X)
        return sp.Derivative(f, X), lambda e: sp.simplify(e - truth) == 0
    while True:
        integrand = sp.simplify(sp.diff(_expression(rng, level), X))
        if integrand != 0:
            break
    return (sp.Integral(integrand, X),
            lambda e: sp.simplify(sp.diff(e, X) - integrand) == 0)


def run(levels: list[int], n: int, width: int, max_plies: int,
        max_nodes: int | None, use_macros: bool, kind: str) -> None:
    tag = "macros ON" if use_macros else "core rules only"
    print(f"# rung bench — kind={kind}, {tag}, width={width}, "
          f"max_plies={max_plies}, max_nodes={max_nodes}")
    print(f"{'level':>5} {'solved':>10} {'mean nodes':>11} {'mean plies':>11}")
    for level in levels:
        rng = random.Random(f"bench-deriv-{kind}-{level}-0")  # string seed
        solved, nodes, plies = 0, [], []
        for _ in range(n):
            root, check = _make_problem(rng, level, kind)
            r = beam_search(root, width=width, max_plies=max_plies,
                            max_nodes=max_nodes, use_macros=use_macros)
            ok = r.solved and check(r.state.expr)
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
    ap.add_argument("--kind", choices=["diff", "int"], default="diff")
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="loop max_nodes over these budgets (the chart)")
    a = ap.parse_args()
    if a.budgets:
        for b in a.budgets:
            run(a.levels, a.n, a.width, a.max_plies, b, a.macros, a.kind)
    else:
        run(a.levels, a.n, a.width, a.max_plies, a.max_nodes, a.macros, a.kind)

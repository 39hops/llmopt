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
from llmopt.search.parallel import pmap

X = sp.Symbol("x")


def _solve_one(args) -> tuple[bool, int, int]:
    """Module-level worker (fork-pool picklable). Returns
    (ok, nodes, plies)."""
    root_s, truth_s, kind, width, max_plies, max_nodes, use_macros, vp = args
    root, truth = sp.sympify(root_s), sp.sympify(truth_s)
    r = beam_search(root, width=width, max_plies=max_plies,
                    max_nodes=max_nodes, use_macros=use_macros, verify_p=vp)
    if kind == "diff":
        ok = r.solved and sp.simplify(r.state.expr - truth) == 0
    else:
        ok = r.solved and sp.simplify(sp.diff(r.state.expr, X) - truth) == 0
    return bool(ok), r.nodes, r.state.plies


def run(levels: list[int], n: int, width: int, max_plies: int,
        max_nodes: int | None, use_macros: bool, kind: str,
        jobs: int | None = None, verify_p: float = 1.0) -> None:
    tag = "macros ON" if use_macros else "core rules only"
    print(f"# rung bench — kind={kind}, {tag}, width={width}, "
          f"max_plies={max_plies}, max_nodes={max_nodes}, jobs={jobs}")
    print(f"{'level':>5} {'solved':>10} {'mean nodes':>11} {'mean plies':>11}")
    for level in levels:
        rng = random.Random(f"bench-deriv-{kind}-{level}-0")  # string seed
        items = []
        for _ in range(n):
            if kind == "diff":
                f = _expression(rng, level)
                root, truth = sp.Derivative(f, X), sp.diff(f, X)
            else:
                root, _ = None, None
                while True:
                    truth = sp.simplify(sp.diff(_expression(rng, level), X))
                    if truth != 0:
                        break
                root = sp.Integral(truth, X)
            items.append((sp.srepr(root), sp.srepr(truth), kind, width,
                          max_plies, max_nodes, use_macros, verify_p))
        results = pmap(_solve_one, items, jobs=jobs)
        solved = sum(ok for ok, _, _ in results)
        nodes = [nd for _, nd, _ in results]
        plies = [pl for ok, _, pl in results if ok]
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
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers (default: cpu_count-2; 1=serial)")
    ap.add_argument("--verify-p", type=float, default=1.0)
    a = ap.parse_args()
    if a.budgets:
        for b in a.budgets:
            run(a.levels, a.n, a.width, a.max_plies, b, a.macros, a.kind,
                jobs=a.jobs, verify_p=a.verify_p)
    else:
        run(a.levels, a.n, a.width, a.max_plies, a.max_nodes, a.macros,
            a.kind, jobs=a.jobs, verify_p=a.verify_p)

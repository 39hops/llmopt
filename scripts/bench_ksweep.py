"""Depth-vs-breadth sweep (Artin's hypothesis, 2026-07-07: breadth can
be synthesized — cf. LazySMP). Random pruning at k in {1,2,3,5} vs full
enumeration vs k=1 x R randomized restarts at EQUAL total node budget.
No model: the random proposer isolates pure depth/diversity effects
from move-choice quality (which bench_proposer.py measures).

  python scripts/bench_ksweep.py --n 20 --budgets 25 50 100
"""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search
from llmopt.search.parallel import pmap

X = sp.Symbol("x")
WALL = 120
RESTARTS = 3


class _Timeout(BaseException):
    pass


def random_proposer(seed: str):
    rng = random.Random(seed)

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


def restart_search(root, total_budget: int, restarts: int, seed: str):
    """k=1 deep dives with different orderings, budget split evenly;
    first solve wins. Depth + diversity instead of breadth."""
    per = max(1, total_budget // restarts)
    used = 0
    for i in range(restarts):
        r = beam_search(root, width=8, max_plies=20, max_nodes=per,
                        proposer=random_proposer(f"{seed}-restart-{i}"),
                        propose_k=1)
        used += r.nodes
        if r.solved:
            return r, used
    return r, used


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


def _sweep_one(args) -> bool:
    """Module-level worker: one (problem, config) cell entry. SIGALRM
    lives inside the worker process."""
    root_s, truth_s, kind, level, p, k, budget = args
    root, truth = sp.sympify(root_s), sp.sympify(truth_s)
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(WALL)
    try:
        if k == "restarts":
            r, _ = restart_search(root, budget, RESTARTS,
                                  f"{kind}-{level}-{p}")
        else:
            prop = (random_proposer(f"sweep-{kind}-{level}-{p}-{k}")
                    if k else None)
            r = beam_search(root, width=8, max_plies=20, max_nodes=budget,
                            proposer=prop, propose_k=k)
        return bool(r.solved and _check(kind, r.state.expr, truth))
    except _Timeout:
        return False
    finally:
        signal.alarm(0)


def main(n: int, budgets: list[int], jobs: int | None = None) -> None:
    ks = [None, 5, 3, 2, 1]  # None = full enumeration
    names = ["full", "k5", "k3", "k2", "k1", f"k1x{RESTARTS}"]
    print(f"# depth-vs-breadth sweep — random pruning, n={n}/cell, jobs={jobs}")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6}" +
          "".join(f" {nm:>6}" for nm in names))
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                cells = []
                for k in ks + ["restarts"]:
                    rng = random.Random(f"ksweep-{kind}-{level}-0")
                    items = []
                    for p in range(n):
                        root, truth = _root(rng, level, kind)
                        items.append((sp.srepr(root), sp.srepr(truth),
                                      kind, level, p, k, budget))
                    ok = sum(pmap(_sweep_one, items, jobs=jobs))
                    cells.append(f"{ok:>3}/{n:<2}")
                print(f"{kind:>4} {level:>3} {budget:>6} " + " ".join(cells),
                      flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    main(a.n, a.budgets, jobs=a.jobs)

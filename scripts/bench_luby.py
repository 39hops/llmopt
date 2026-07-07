"""Luby restart schedule vs equal-thirds (Artin's budget-reallocation
thread). Luby (1,1,2,1,1,2,4,...) is provably within a log factor of
the optimal restart policy without knowing the difficulty
distribution. Same seeds as every race. Refs: k1x3 got 267/360."""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search

X = sp.Symbol("x")
WALL = 120


class _Timeout(BaseException):
    pass


def luby(i: int) -> int:
    """1-indexed Luby sequence."""
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    if (1 << k) - 1 == i:
        return 1 << (k - 1)
    return luby(i - (1 << (k - 1)) + 1)


def random_proposer(seed: str):
    rng = random.Random(seed)

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


def restarts_equal(root, budget, seed):
    per = max(1, budget // 3)
    for i in range(3):
        r = beam_search(root, width=8, max_plies=20, max_nodes=per,
                        proposer=random_proposer(f"{seed}-e{i}"), propose_k=1)
        if r.solved:
            return r
    return r


def restarts_luby(root, budget, seed, unit):
    used, i = 0, 1
    r = None
    while used < budget:
        slice_ = min(unit * luby(i), budget - used)
        if slice_ < 2:
            break
        r = beam_search(root, width=8, max_plies=20, max_nodes=slice_,
                        proposer=random_proposer(f"{seed}-l{i}"), propose_k=1)
        used += r.nodes
        if r.solved:
            return r
        i += 1
    return r


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


def main(n: int, budgets: list[int]) -> None:
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    print(f"# luby vs equal-thirds restarts — n={n}/cell (ref k1x3: 267/360)")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'equal3':>7} {'luby':>7}")
    totals = {"equal3": 0, "luby": 0}
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                row = {}
                for name in ("equal3", "luby"):
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for p in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            if name == "equal3":
                                r = restarts_equal(root, budget,
                                                   f"{kind}-{level}-{p}")
                            else:
                                r = restarts_luby(root, budget,
                                                  f"{kind}-{level}-{p}",
                                                  unit=max(4, budget // 12))
                            ok += (r is not None and r.solved
                                   and _check(kind, r.state.expr, truth))
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[name] = ok
                    totals[name] += ok
                print(f"{kind:>4} {level:>3} {budget:>6} "
                      f"{row['equal3']:>4}/{n:<2} {row['luby']:>4}/{n:<2}",
                      flush=True)
    print(f"TOTALS: equal3 {totals['equal3']}  luby {totals['luby']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    a = ap.parse_args()
    main(a.n, a.budgets)

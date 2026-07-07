"""Golden-angle restart diversity (Artin's fib thread, the legit
version): restart i rotates a base ordering by the golden-angle stride
(low-discrepancy: successive restarts maximally spread over orderings)
vs iid random shuffles. Expectation calibrated by the Luby null: at 3
restarts, schedule/diversity effects may not bite. n=15, same seeds."""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search

X = sp.Symbol("x")
WALL = 120
PHI = (5 ** 0.5 - 1) / 2  # golden ratio conjugate


class _Timeout(BaseException):
    pass


def iid_proposer(seed: str):
    rng = random.Random(seed)

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


def golden_proposer(seed: str, i: int):
    """One fixed base shuffle per problem; restart i rotates it by the
    golden-angle stride — deterministic, maximally-spread orderings."""
    rng = random.Random(seed)  # same base across restarts of a problem

    def prop(state, children):
        children = list(children)
        rng2 = random.Random(f"{seed}-base-{len(children)}")
        rng2.shuffle(children)
        off = int(round(i * PHI * len(children))) % max(1, len(children))
        return children[off:] + children[:off]

    return prop


def run_restarts(root, budget, seed, kind_):
    per = max(1, budget // 3)
    for i in range(3):
        prop = (iid_proposer(f"{seed}-{i}") if kind_ == "iid"
                else golden_proposer(seed, i))
        r = beam_search(root, width=2, max_plies=20, max_nodes=per,
                        proposer=prop, propose_k=1, verify_p=0.1)
        if r.solved:
            return r
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
    print(f"# golden-angle vs iid restart diversity — n={n}/cell, w=2, k=1x3")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'iid':>6} {'golden':>7}")
    totals = {"iid": 0, "golden": 0}
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in budgets:
                row = {}
                for name in ("iid", "golden"):
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for p in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            r = run_restarts(root, budget,
                                             f"{kind}-{level}-{p}", name)
                            ok += (r is not None and r.solved
                                   and _check(kind, r.state.expr, truth))
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[name] = ok
                    totals[name] += ok
                print(f"{kind:>4} {level:>3} {budget:>6} {row['iid']:>4}/{n:<2}"
                      f" {row['golden']:>4}/{n:<2}", flush=True)
    print(f"TOTALS: iid {totals['iid']}  golden {totals['golden']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100])
    a = ap.parse_args()
    main(a.n, a.budgets)

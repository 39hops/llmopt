"""Do the timeout campaign's winners COMPOSE? Lazy expansion (+2
solves, timeouts 4v10) and the magic detector (+1, 71 certified cuts)
won independently; engine.solve() integration wants the interaction
term. Four arms, paired, one run: classical / lazy / magic / both.
bf-nnue + markov, the hard cells.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import random
import signal
import sys

import sympy as sp

sys.path.insert(0, "scripts")
from bench_lazy import lazy_children, make_rankers  # noqa: E402
from bench_magic import is_dead, load_nnue  # noqa: E402
from llmopt.mathgen.problems import _expression  # noqa: E402
from llmopt.search.derivation import State, is_solved, successors  # noqa: E402

X = sp.Symbol("x")
WALL = 240


class _Timeout(BaseException):
    pass


def best_first(root, budget, expand, h, magic):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        for _, child in expand(s):
            if child.key() in visited:
                continue
            visited.add(child.key())
            if magic and is_dead(child):
                continue
            nodes += 1
            if is_solved(child):
                return child
            heapq.heappush(pq, (h(child), next(tie), child))
            if nodes >= budget:
                break
    return None


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


def main(n: int, budget: int) -> None:
    prop, rule_rank = make_rankers()
    h = load_nnue("checkpoints/nnue_eval.pt")

    def eager(s):
        return prop(s, list(successors(s, use_macros=True,
                                       verify_p=0.1)))[:3]

    def lazy(s):
        return lazy_children(s, rule_rank)

    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    arms = {"classical": (eager, False), "lazy": (lazy, False),
            "magic": (eager, True), "both": (lazy, True)}
    print(f"# winner stacking — n={n}/cell, budget={budget}, wall={WALL}s")
    print(f"{'kind':>4} {'lvl':>3}" + "".join(f" {c:>10}" for c in arms))
    tot = {c: 0 for c in arms}
    for kind, levels in (("diff", (4,)), ("int", (3, 4))):
        for level in levels:
            row = {}
            for cfg, (expand, magic) in arms.items():
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        sol = best_first(root, budget, expand, h, magic)
                        ok += (sol is not None
                               and _check(kind, sol.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                row[cfg] = ok
                tot[cfg] += ok
            print(f"{kind:>4} {level:>3}"
                  + "".join(f" {row[c]:>7}/{n:<2}" for c in arms),
                  flush=True)
    print("TOTALS: " + "  ".join(f"{c}: {t}" for c, t in tot.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budget", type=int, default=400)
    a = ap.parse_args()
    main(a.n, a.budget)

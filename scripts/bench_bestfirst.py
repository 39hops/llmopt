"""Best-first (priority-queue) search vs synchronized beam — the
skeleton where the Dijkstra g+h question is actually askable (the
beam g-sweep tied 92=92=92=92 with a structural proof: equal-depth
comparisons cancel g). Frontier mixes depths; lambda weights g.
Markov top-3 pruning both, width-2 beam as the incumbent."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import signal
from collections import Counter, defaultdict

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, is_solved, successors

X = sp.Symbol("x")
WALL = 120
UNSOLVED = (sp.Derivative, sp.Integral, sp.Limit)


class _Timeout(BaseException):
    pass


def markov():
    d = json.load(open("checkpoints/markov_prior.json"))
    uni, bi = d["unigram"], d["bigram"]

    def prop(state, children):
        prev = state.history[-1].split("@")[0] if state.history else None
        t = bi.get(prev) if prev else None

        def s(n):
            r = n.split("@")[0]
            return (t.get(r, 0) if t else 0) + 0.01 * uni.get(r, 0)

        return sorted(children, key=lambda c: -s(c[0]))

    return prop


def h(state: State) -> float:
    return (100.0 * sum(1 for _ in state.expr.atoms(*UNSOLVED))
            + float(sp.count_ops(state.expr)))


def best_first(root: sp.Expr, budget: int, prop, lam: float):
    """Pop min(lam*g + h); expand markov-top-3; sampled verification."""
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start, 1
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:3]
        for _, child in kids:
            if child.key() in visited:
                continue
            visited.add(child.key())
            nodes += 1
            if is_solved(child):
                return child, nodes
            heapq.heappush(pq, (lam * child.plies + h(child), next(tie), child))
            if nodes >= budget:
                break
    return None, nodes


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


def main(n: int) -> None:
    prop = markov()
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    cfgs = ["beam-w2", "bf-g0", "bf-g1", "bf-g5"]
    print(f"# best-first vs beam — markov top-3 both, n={n}/cell")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6}" +
          "".join(f" {c:>8}" for c in cfgs))
    totals = {c: 0 for c in cfgs}
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in (25, 50):
                row = {}
                for cfg in cfgs:
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            if cfg == "beam-w2":
                                r = beam_search(root, width=2, max_plies=20,
                                                max_nodes=budget, proposer=prop,
                                                propose_k=3, verify_p=0.1,
                                                use_macros=True)
                                sol = r.state if r.solved else None
                            else:
                                lam = float(cfg.split("g")[1])
                                sol, _nd = best_first(root, budget, prop, lam)
                            ok += (sol is not None
                                   and _check(kind, sol.expr, truth))
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[cfg] = ok
                    totals[cfg] += ok
                print(f"{kind:>4} {level:>3} {budget:>6}" +
                      "".join(f" {row[c]:>5}/{n:<2}" for c in cfgs),
                      flush=True)
    print("TOTALS: " + "  ".join(f"{c}: {t}" for c, t in totals.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    main(a.n)

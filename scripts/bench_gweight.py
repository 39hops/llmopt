"""The Dijkstra component of Artin's Google-Maps analogy: our beam
ranks by (almost) pure heuristic h; Dijkstra ranks by path cost g; A*
by g+h. Sweep the g-weight (plies coefficient) in the eval:
lambda in {0, 0.1 (current hce), 1, 5}. markov3 @ w2 engine, held-out
seeds, n=15, budgets 25/50."""

from __future__ import annotations

import argparse
import json
import random
import signal
from collections import Counter, defaultdict

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search

X = sp.Symbol("x")
WALL = 120
UNSOLVED = (sp.Derivative, sp.Integral, sp.Limit)


def markov_proposer():
    d = json.load(open("checkpoints/markov_prior.json"))
    uni, bi = d["unigram"], d["bigram"]

    def prop(state, children):
        prev = state.history[-1].split("@")[0] if state.history else None
        table = bi.get(prev) if prev else None

        def s(name):
            r = name.split("@")[0]
            return (table.get(r, 0) if table else 0) + 0.01 * uni.get(r, 0)

        return sorted(children, key=lambda c: -s(c[0]))

    return prop


def eval_with_g(lam: float):
    def ev(state: State) -> float:
        unsolved = sum(1 for _ in state.expr.atoms(*UNSOLVED))
        return (100.0 * unsolved + float(sp.count_ops(state.expr))
                + lam * state.plies)
    return ev


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
    prop = markov_proposer()
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    lams = [0.0, 0.1, 1.0, 5.0]
    print(f"# g-weight sweep (Dijkstra component) — markov3@w2, n={n}/cell")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6}" +
          "".join(f" {('g=' + str(l)):>7}" for l in lams))
    totals = {l: 0 for l in lams}
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in (25, 50):
                row = {}
                for lam in lams:
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            r = beam_search(root, width=2, max_plies=20,
                                            max_nodes=budget, proposer=prop,
                                            propose_k=3, verify_p=0.1,
                                            eval_fn=eval_with_g(lam),
                                            use_macros=True)
                            ok += r.solved and _check(kind, r.state.expr, truth)
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[lam] = ok
                    totals[lam] += ok
                print(f"{kind:>4} {level:>3} {budget:>6}" +
                      "".join(f" {row[l]:>4}/{n:<2}" for l in lams), flush=True)
    print("TOTALS: " + "  ".join(f"g={l}: {t}" for l, t in totals.items()))


class _Timeout(BaseException):
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    main(a.n)

"""Markov prior IN-SEARCH race: rule-bigram scores (zero inference
cost) driving propose_k=3 pruning, on the same held-out seeds as the
proposer race. Reference totals (n=15): full+hce 265, rand3 277,
prop3(LLM) 288, adapt-T0.1 300. If markov3 lands near 288, the LLM's
pruning value is rule grammar and the wall-clock tax is optional."""

from __future__ import annotations

import argparse
import json
import random
import signal
from collections import Counter, defaultdict

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search

X = sp.Symbol("x")
WALL = 120


class _Timeout(BaseException):
    pass


def build_prior():
    train = [json.loads(l) for l in open("data/proposer_train.jsonl")]
    unigram: Counter = Counter()
    bigram: dict[str, Counter] = defaultdict(Counter)
    prev = None
    for row in train:
        chosen = row["moves"][row["answer"]].split("@")[0]
        unigram[chosen] += 1
        if prev is not None:
            bigram[prev][chosen] += 1
        prev = chosen
    return unigram, bigram


def make_markov_proposer(unigram, bigram):
    # per-search closure: track the previous applied rule via the
    # state's history (last entry), no external state needed
    def proposer(state, children):
        prev = state.history[-1].split("@")[0] if state.history else None
        table = bigram.get(prev) if prev else None
        def s(name):
            r = name.split("@")[0]
            return (table.get(r, 0) if table else 0) + 0.01 * unigram.get(r, 0)
        return sorted(children, key=lambda c: -s(c[0]))
    return proposer


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
    unigram, bigram = build_prior()
    prop = make_markov_proposer(unigram, bigram)
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    print(f"# markov3 race — n={n}/cell (compare: full 265, rand3 277, "
          f"prop3-LLM 288, adapt 300)")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'markov3':>8}")
    total = 0
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        r = beam_search(root, width=8, max_plies=20,
                                        max_nodes=budget, proposer=prop,
                                        propose_k=3)
                        ok += r.solved and _check(kind, r.state.expr, truth)
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                total += ok
                print(f"{kind:>4} {level:>3} {budget:>6} {ok:>5}/{n:<2}",
                      flush=True)
    print(f"TOTAL: {total}/{6 * len(budgets) * n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    a = ap.parse_args()
    main(a.n, a.budgets)

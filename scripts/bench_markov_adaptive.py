"""The zero-GPU champion candidate: adaptive-k with MARKOV confidence.
Entropy over the bigram's count-normalized distribution gates k
(1..6). If this lands near adapt-LLM's 300/360, the entire champion
engine needs no neural network at all. Reference totals (n=15):
full 265, rand3 277, prop3-LLM 288, markov3 293, adapt-LLM 300."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import statistics
from collections import Counter, defaultdict

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search

X = sp.Symbol("x")
WALL = 120
K_MIN, K_MAX = 1, 6


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


def make_markov_adaptive(unigram, bigram, ks_seen):
    """Scoring proposer + entropy policy in one: counts give both the
    ranking and a natural probability distribution (no temperature to
    calibrate — counts are already calibrated by frequency)."""

    def proposer(state, children):
        prev = state.history[-1].split("@")[0] if state.history else None
        table = bigram.get(prev) if prev else None

        def raw(name):
            r = name.split("@")[0]
            return (table.get(r, 0) if table else 0) + 0.01 * unigram.get(r, 0)

        scored = sorted(((raw(n), (n, st)) for n, st in children),
                        key=lambda t: -t[0])
        return [c for _, c in scored], [s for s, _ in scored]

    def policy(state, ranked, scores):
        n = len(scores)
        if n <= 1:
            return K_MIN
        z = sum(scores)
        if z <= 0:
            return K_MAX  # no information: hedge wide
        ps = [s / z for s in scores]
        h = -sum(p * math.log(p) for p in ps if p > 0) / math.log(n)
        k = K_MIN + round(h * (K_MAX - K_MIN))
        ks_seen.append(k)
        return k

    return proposer, policy


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
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    print(f"# markov-adaptive race — n={n}/cell "
          f"(refs: markov3 293, adapt-LLM 300)")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'madapt':>7} {'mean-k':>7}")
    total = 0
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                ks: list[int] = []
                prop, policy = make_markov_adaptive(unigram, bigram, ks)
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        r = beam_search(root, width=8, max_plies=20,
                                        max_nodes=budget, proposer=prop,
                                        propose_k=policy)
                        ok += r.solved and _check(kind, r.state.expr, truth)
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                total += ok
                mk = statistics.mean(ks) if ks else float("nan")
                print(f"{kind:>4} {level:>3} {budget:>6} {ok:>4}/{n:<2} "
                      f"{mk:>7.2f}", flush=True)
    print(f"TOTAL: {total}/{6 * len(budgets) * n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    a = ap.parse_args()
    main(a.n, a.budgets)

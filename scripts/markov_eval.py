"""Absorbing-Markov eval (Artin's Markov thread, part 2): bucket
states by coarse structure, estimate P(solve | bucket) from fast
probes, use -P(solve) as eval_fn. A probability-theoretic eval raced
against HCE's hand-tuned weights — both model-free.

Phase 1 (this run): generate probe-labeled states with the FAST
engine (markov3 pruning, width 2, verify_p=0.1 — the compounded
speedups), bucket, report bucket stats, race P-eval vs hce on
held-out problems (markov3 pruning both, only eval_fn differs).
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from collections import Counter, defaultdict

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, hce, is_solved

X = sp.Symbol("x")
WALL = 60
UNSOLVED = (sp.Derivative, sp.Integral, sp.Limit)


class _Timeout(BaseException):
    pass


def build_markov_proposer():
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

    def prop(state, children):
        prev_r = state.history[-1].split("@")[0] if state.history else None
        table = bigram.get(prev_r) if prev_r else None

        def s(name):
            r = name.split("@")[0]
            return (table.get(r, 0) if table else 0) + 0.01 * unigram.get(r, 0)

        return sorted(children, key=lambda c: -s(c[0]))

    return prop


def bucket(expr: sp.Expr) -> tuple:
    """Coarse structural key: (n_unsolved, ops-quartile, deepest-kind)."""
    unsolved = list(expr.atoms(*UNSOLVED))
    ops = sp.count_ops(expr)
    ops_b = 0 if ops < 10 else 1 if ops < 25 else 2 if ops < 60 else 3
    kinds = tuple(sorted({type(u).__name__ for u in unsolved}))
    return (min(len(unsolved), 4), ops_b, kinds)


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


def main(n_probe: int, n_race: int) -> None:
    prop = build_markov_proposer()
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))

    # phase 1: probe-labeled states -> bucket stats
    solved_c: Counter = Counter()
    total_c: Counter = Counter()
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            rng = random.Random(f"markov-eval-probe-{kind}-{level}-0")
            for _ in range(n_probe):
                root, _t = _root(rng, level, kind)
                trace: list[State] = []
                signal.alarm(WALL)
                try:
                    beam_search(root, width=2, max_plies=20, max_nodes=150,
                                proposer=prop, propose_k=3, verify_p=0.1,
                                trace=trace)
                except _Timeout:
                    pass
                finally:
                    signal.alarm(0)
                rng2 = random.Random(f"sub-{kind}-{level}")
                sample = rng2.sample(trace, min(6, len(trace)))
                for s in sample:
                    if is_solved(s):
                        continue
                    b = bucket(s.expr)
                    total_c[b] += 1
                    signal.alarm(30)
                    try:
                        r = beam_search(s.expr, width=2, max_plies=20,
                                        max_nodes=80, proposer=prop,
                                        propose_k=3, verify_p=0.1)
                        solved_c[b] += bool(r.solved)
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
    p_solve = {b: solved_c[b] / total_c[b] for b in total_c if total_c[b] >= 5}
    print(f"buckets with >=5 samples: {len(p_solve)}")
    for b, p in sorted(p_solve.items(), key=lambda t: t[1])[:6]:
        print(f"  P(solve)={p:.2f} n={total_c[b]:3d} bucket={b}")

    def markov_eval(state: State) -> float:
        return -p_solve.get(bucket(state.expr), 0.5)

    # phase 2: race eval_fns (markov3 pruning both; only eval differs)
    print(f"\n# eval race — markov3 pruning, width 2, n={n_race}/cell")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'hce':>6} {'P-eval':>7}")
    totals = {"hce": 0, "P": 0}
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in (25, 50):
                row = {}
                for name, ev in (("hce", hce), ("P", markov_eval)):
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for _ in range(n_race):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            r = beam_search(root, width=2, max_plies=20,
                                            max_nodes=budget, proposer=prop,
                                            propose_k=3, eval_fn=ev,
                                            verify_p=0.1)
                            ok += r.solved and _check(kind, r.state.expr, truth)
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[name] = ok
                    totals[name] += ok
                print(f"{kind:>4} {level:>3} {budget:>6} {row['hce']:>4}/{n_race:<2}"
                      f" {row['P']:>4}/{n_race:<2}", flush=True)
    print(f"TOTALS: hce {totals['hce']}  P-eval {totals['P']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probe", type=int, default=10)
    ap.add_argument("--n-race", type=int, default=15)
    a = ap.parse_args()
    main(a.n_probe, a.n_race)

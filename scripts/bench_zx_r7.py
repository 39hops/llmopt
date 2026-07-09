"""ZX rung 7: push the phase-teleport win — markov prior on the new
move set, then bigger Toffoli nets.

The rung-3 prior predates M:phase_teleport (prior-pollution lesson:
hygiene >> content — never rank moves with a prior that has never
seen the current move set), so --harvest re-mines it from best-first
descents WITH the macro into zx_markov_prior_v2.json.

--race arms (per circuit, alarm-bounded, extraction-verified):
  greedy    : full_reduce                       (deposed champion)
  pipeline  : teleport + phase_block            (rung-5 winner)
  bf        : bf_extract, macro in move set     (rung-6 winner)
  bf-mk3    : bf + fresh bigram prior, top-3    (rung-7 question)

Bars: bf-mk3 vs bf wins >= 20% (does the prior still pay with the
macro dominating descents?); at q8 verification is tier-2 (documented).
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from collections import Counter, defaultdict
from pathlib import Path

import pyzx as zx

from bench_zx_r3 import bf_extract, rule_of, structured_toffoli
from bench_zx_r5 import _teleported_circuit
from llmopt.search.zx_engine import best_first_zx, tcount, verify_equal

PRIOR_PATH = Path("checkpoints/zx_markov_prior_v2.json")


class _Timeout(BaseException):
    pass


def harvest(n: int, qubits: int, tofs: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    uni: Counter = Counter()
    bi: dict = defaultdict(Counter)
    rng = random.Random(f"zx-harvest2-{qubits}-{tofs}")
    got = 0
    for i in range(n):
        c = structured_toffoli(qubits, tofs, rng)
        g = c.to_graph()
        zx.simplify.to_gh(g)
        signal.alarm(180)
        try:
            best, _ = best_first_zx(g, budget=budget)
        except _Timeout:
            continue
        finally:
            signal.alarm(0)
        if tcount(best.g) >= tcount(g):
            continue
        got += 1
        prev = None
        for lab in best.history:
            r = rule_of(lab)
            uni[r] += 1
            if prev is not None:
                bi[prev][r] += 1
            prev = r
        print(f"harvest {i}: T {tcount(g)} -> {tcount(best.g)}, "
              f"path {len(best.history)}", flush=True)
    PRIOR_PATH.write_text(json.dumps(
        {"unigram": dict(uni), "bigram": {k: dict(v) for k, v in bi.items()}}))
    print(f"harvested {got}/{n} descents -> {PRIOR_PATH}")
    print("unigram:", dict(uni))


def race(n: int, qubits: int, tofs: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    prior = json.loads(PRIOR_PATH.read_text())
    rng = random.Random(f"zx-race7-{qubits}-{tofs}")  # fresh stream
    arms = ("greedy", "pipeline", "bf", "bf-mk3")
    stats = {a: {"T": 0, "to": 0, "bad": 0} for a in arms}
    wg = {a: 0 for a in arms}
    mk_vs_bf = 0
    print(f"# rung-7 race — q={qubits} tofs={tofs} n={n} budget={budget}"
          + (" [tier-2 verify]" if qubits > 8 else ""))
    print(f"{'i':>3} " + " ".join(f"{a:>10}" for a in arms))
    for i in range(n):
        c = structured_toffoli(qubits, tofs, rng)
        row = {}
        for arm in arms:
            signal.alarm(240)
            try:
                if arm == "greedy":
                    g = c.to_graph()
                    zx.full_reduce(g)
                    t = zx.tcount(g)
                elif arm == "pipeline":
                    ct = zx.optimize.phase_block_optimize(
                        _teleported_circuit(c))
                    t = ct.tcount()
                    if not verify_equal(c, ct.to_graph(), qubits):
                        stats[arm]["bad"] += 1
                        t = 10**9
                else:
                    gs = c.to_graph()
                    zx.simplify.to_gh(gs)
                    best, bt = bf_extract(
                        gs, budget,
                        prior=prior if arm == "bf-mk3" else None)
                    t = bt if bt is not None else 10**9
                    if bt is not None and not verify_equal(
                            c, best.g, qubits):
                        stats[arm]["bad"] += 1
                        t = 10**9
            except _Timeout:
                t = 10**9
                stats[arm]["to"] += 1
            except Exception:
                t = 10**9
                stats[arm]["bad"] += 1
            finally:
                signal.alarm(0)
            row[arm] = t
        for arm in arms:
            stats[arm]["T"] += min(row[arm], 10**6)
            wg[arm] += row[arm] < row["greedy"]
        mk_vs_bf += row["bf-mk3"] < row["bf"]
        print(f"{i:>3} " + " ".join(f"{row[a]:>10}" for a in arms),
              flush=True)
    print("TOTALS")
    for arm in arms:
        print(f"  {arm:>10}: wins-vs-greedy {wg[arm]:>2}  "
              f"mean T {stats[arm]['T'] / n:.1f}  "
              f"timeouts {stats[arm]['to']} fails {stats[arm]['bad']}")
    print(f"bf-mk3 wins vs bf: {mk_vs_bf} (bar >= {0.2 * n:.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--race", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--tofs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=500)
    a = ap.parse_args()
    if a.harvest:
        harvest(a.n, a.qubits, a.tofs, a.budget)
    if a.race:
        race(a.n, a.qubits, a.tofs, a.budget)

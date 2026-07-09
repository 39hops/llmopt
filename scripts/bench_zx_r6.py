"""ZX rung 6: composition — does SEARCH around the phase-teleport
macro beat the bare pipeline?

Rung 5 found the first greedy-beater (teleport + phase_block, 7/30
wins charged with every crash). It is now a macro move
(M:phase_teleport in zx_engine.macro_moves). Same seed stream as
rung 5, so rows compare directly.

Arms: greedy full_reduce | bare pipeline | bf_extract (macro inside).
Composition bar: bf-extract wins vs the PIPELINE on >= 20%, else the
macro alone is the whole story (the magic>lazy=both lesson).
"""

from __future__ import annotations

import argparse
import random
import signal

import pyzx as zx

from bench_zx_r3 import bf_extract, structured_toffoli
from bench_zx_r5 import _teleported_circuit
from llmopt.search.zx_engine import verify_equal


class _Timeout(BaseException):
    pass


def main(n: int, qubits: int, tofs: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    rng = random.Random(f"zx-race5-{qubits}-{tofs}")  # rung-5 stream
    arms = ("greedy", "pipeline", "bf-extract")
    stats = {a: {"T": 0, "to": 0, "bad": 0} for a in arms}
    wins_vs_greedy = {a: 0 for a in arms}
    wins_vs_pipeline = 0
    print(f"# rung-6 composition — q={qubits} tofs={tofs} n={n} "
          f"budget={budget}")
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
                    best, bt = bf_extract(gs, budget)
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
            wins_vs_greedy[arm] += row[arm] < row["greedy"]
        wins_vs_pipeline += row["bf-extract"] < row["pipeline"]
        print(f"{i:>3} " + " ".join(f"{row[a]:>10}" for a in arms),
              flush=True)
    print("TOTALS")
    for arm in arms:
        print(f"  {arm:>10}: wins-vs-greedy {wins_vs_greedy[arm]:>2}  "
              f"mean T {stats[arm]['T'] / n:.1f}  "
              f"timeouts {stats[arm]['to']} fails {stats[arm]['bad']}")
    print(f"bf-extract wins vs pipeline: {wins_vs_pipeline} "
          f"(composition bar >= {0.2 * n:.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--tofs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=500)
    a = ap.parse_args()
    main(a.n, a.qubits, a.tofs, a.budget)

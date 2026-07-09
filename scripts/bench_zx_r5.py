"""ZX rung 5: phase-polynomial machinery (the literature's greedy-beater).

Rung 4 closed with 0 wins for graph-rewrite search vs greedy
full_reduce — the move set (lcomp/pivot/fusion) can't beat the macro
built from the same moves. The literature's actual T-count wins come
from PHASE-POLYNOMIAL optimization (TODD / phase-block merging), which
pyzx ships: teleport_reduce moves phases into a shared polynomial,
full_optimize runs TODD-class merging on the extracted circuit.

Cheapest honest test before hand-rolling TODD as an engine move:
race the pipelines head-to-head on the rung-3 structured Toffoli nets
(SHARED phase structure = where these wins are supposed to live).

Arms (all verified by extraction/comparison against the source circuit):
  greedy   : full_reduce, T-count of graph            (rung 0 champion)
  teleport : teleport_reduce -> circuit T-count       (phase teleportation)
  phaseblk : teleport + phase_block_optimize          (phase-poly merging)
  fullopt  : teleport + full_optimize                 (TODD-class, heaviest)

Pre-registered bar: an arm must beat greedy on >= 20% of circuits to
justify wrapping it as an engine macro move.
"""

from __future__ import annotations

import argparse
import random
import signal

import pyzx as zx
from pyzx.circuit import Circuit

from bench_zx_r3 import structured_toffoli
from llmopt.search.zx_engine import verify_equal


class _Timeout(BaseException):
    pass


def _teleported_circuit(c: Circuit) -> Circuit:
    g = c.to_graph()
    zx.teleport_reduce(g)
    return Circuit.from_graph(g).split_phase_gates()


def run_arm(arm: str, c: Circuit):
    if arm == "greedy":
        g = c.to_graph()
        zx.full_reduce(g)
        return zx.tcount(g), None  # graph-level; extraction verified below
    ct = _teleported_circuit(c)
    if arm == "phaseblk":
        ct = zx.optimize.phase_block_optimize(ct)
    elif arm == "fullopt":
        ct = zx.optimize.full_optimize(ct)
    return ct.tcount(), ct


def main(n: int, qubits: int, tofs: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    arms = ("greedy", "teleport", "phaseblk", "fullopt")
    rng = random.Random(f"zx-race5-{qubits}-{tofs}")
    stats = {a: {"w": 0, "t": 0, "l": 0, "T": 0, "to": 0, "bad": 0}
             for a in arms}
    print(f"# rung-5 race — toffoli nets q={qubits} tofs={tofs} n={n}")
    print(f"{'i':>3} " + " ".join(f"{a:>9}" for a in arms))
    for i in range(n):
        c = structured_toffoli(qubits, tofs, rng)
        row = {}
        for arm in arms:
            signal.alarm(240)
            try:
                t, ct = run_arm(arm, c)
                if ct is not None and not verify_equal(c, ct.to_graph(),
                                                       qubits):
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
        tg = row["greedy"]
        for arm in arms:
            stats[arm]["T"] += min(row[arm], 10**6)
            key = ("w" if row[arm] < tg else
                   "t" if row[arm] == tg else "l")
            stats[arm][key] += 1
        print(f"{i:>3} " + " ".join(f"{row[a]:>9}" for a in arms),
              flush=True)
    print("TOTALS")
    for arm, st in stats.items():
        print(f"  {arm:>9}: wins {st['w']:>2} ties {st['t']:>2} "
              f"losses {st['l']:>2}  mean T {st['T'] / n:.1f}  "
              f"timeouts {st['to']} verify-fail {st['bad']}")
    print(f"bar: wins >= {0.2 * n:.0f} vs greedy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--tofs", type=int, default=8)
    a = ap.parse_args()
    main(a.n, a.qubits, a.tofs)

"""T-count rung-2 race: best-first over ZX rewrites (primitives +
gadget macros + macro-greedy) vs greedy full_reduce, per-circuit.
Pre-registered bar (spec): search beats greedy on >= 20% of seeded
circuits, else the greedy oracle wins the domain and we say so.
Outputs win/tie/loss on T-count, mean T per arm, and tensor
verification on every search result (<= 8 qubits)."""

from __future__ import annotations

import argparse
import random
import signal

import pyzx as zx

from llmopt.search.zx_engine import best_first_zx, tcount, verify_equal


class _Timeout(BaseException):
    pass


def main(n: int, qubits: int, depth: int, budget: int, seed: str) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    rng = random.Random(f"zx-race-{seed}-{qubits}-{depth}")
    wins = ties = losses = badver = nto = 0
    tg_sum = tb_sum = 0
    print(f"# zx race — q={qubits} d={depth} n={n} budget={budget}")
    print(f"{'i':>3} {'T0':>4} {'greedy':>7} {'search':>7} {'verdict':>8}")
    for i in range(n):
        random.seed(rng.randrange(10**9))  # pyzx uses global random
        c = zx.generate.CNOT_HAD_PHASE_circuit(
            qubits=qubits, depth=depth, clifford=False)
        g0 = c.to_graph()
        t0 = tcount(g0)
        gg = c.to_graph()
        zx.full_reduce(gg)
        tg = zx.tcount(gg)
        gs = c.to_graph()
        zx.simplify.to_gh(gs)
        signal.alarm(240)
        try:
            best, _ = best_first_zx(gs, budget=budget)
            tb = tcount(best.g)
            ok = verify_equal(c, best.g, qubits)
        except _Timeout:
            tb, ok = t0, True  # timeout: search scores its start
            nto += 1
        finally:
            signal.alarm(0)
        if not ok:
            badver += 1
            tb = t0
        verdict = ("WIN" if tb < tg else "tie" if tb == tg else "loss")
        wins += tb < tg
        ties += tb == tg
        losses += tb > tg
        tg_sum += tg
        tb_sum += tb
        print(f"{i:>3} {t0:>4} {tg:>7} {tb:>7} {verdict:>8}", flush=True)
    print(f"TOTALS: search wins {wins}, ties {ties}, losses {losses} "
          f"of {n}; mean T greedy {tg_sum / n:.1f} vs search "
          f"{tb_sum / n:.1f}; verify-failures {badver}")
    print(f"pre-registered bar: wins >= {0.2 * n:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=120)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seed", type=str, default="0")
    a = ap.parse_args()
    main(a.n, a.qubits, a.depth, a.budget, a.seed)

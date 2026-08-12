"""Deconfounder for the hybrid 349/360: markov3 fixed-k3 (the
engine.solve default) rerun on the same 24-cell matrix WITH today's
new rules. The old markov3 reference (316) predates i_cyclic/i_unprod/
i_ansatz_exp/i_linear_basis/smoothing. If this control lands near 349,
the operators explain the record and hybrid confidence adds ~nothing;
if it lands well below, the LLM-gated k earns real credit."""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp

from llmopt.search.benchkit import _check, _root
from llmopt.search.engine import solve

X = sp.Symbol("x")
WALL = 300


class _Timeout(BaseException):
    pass


def main(n: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# control: markov3-k3 @ w2 + new rules — n={n}/cell; "
          f"hybrid scored 349/360 on these cells")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'control':>8}")
    tot = 0
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in (25, 50, 100, 200):
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        r = solve(root, budget=budget)
                        ok += (r.solved
                               and _check(kind, r.state.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                tot += ok
                print(f"{kind:>4} {level:>3} {budget:>6} {ok:>5}/{n:<2}",
                      flush=True)
    print(f"TOTAL: control {tot}/360")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    main(ap.parse_args().n)

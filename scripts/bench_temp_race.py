"""Solve-level race: temperature ladder vs const 0.7 in real chains.

The diversity probe (2026-07-14) measured the ladder lifting
distinct verified steps/state 0.33 -> 0.42 (+27%); this decides
whether that converts to SOLVES at equal budget — hotter streams
also mint more invalid steps, and every sample bills the budget.

Bar: ladder solves >= const AND validity >= const - 0.1pts on the
race band; ships as sample_batch default only after a second band
agrees (the hints-flip discipline).

    .venv/bin/python scripts/bench_temp_race.py --seed0 9400000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEVELS = (2, 3, 4, 5)
LADDER = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45]


def main(n_per: int, budget: int, seed0: int) -> None:
    import sympy as sp

    import bench_step_tokens as bst
    from bench_step_tokens import _gen_isolated, load, solve_chain

    tok, model = load("checkpoints/step_lora.pt")
    probs = []
    for lv in LEVELS:
        for i in range(n_per):
            p = _gen_isolated(lv, seed0 + 1000 * lv + i)
            if p is not None:
                probs.append((lv, sp.sstr(p._expr)))
    print(f"# temp race — {len(probs)} problems, budget {budget}, "
          f"band {seed0}")
    for arm, temps in (("const", None), ("ladder", LADDER)):
        bst.TEMP_LADDER = temps
        solved = valid = tried = 0
        t0 = time.time()
        for k, (lv, s) in enumerate(probs):
            ok, _p, v, t = solve_chain(tok, model, s, budget,
                                       seed0=seed0 + k)
            solved += ok
            valid += v
            tried += t
        print(f"{arm:>7s}: solves {solved}/{len(probs)} validity "
              f"{100 * valid / max(tried, 1):.2f}% "
              f"wall {time.time() - t0:.0f}s", flush=True)
    bst.TEMP_LADDER = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=12)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--seed0", type=int, default=9_400_000)
    a = ap.parse_args()
    main(a.n_per, a.budget, a.seed0)

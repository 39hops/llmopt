"""Autonomous expert-iteration loop driver (spec:
docs/superpowers/specs/2026-07-12-step-expert-iteration-design.md).
Round = evaluate -> mine -> train -> gate; state on disk; tripwires
halt the loop. All sympy touches forked (pathologies #7/#8/#10)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(tok, model, levels, n_per, seed_base, budget=768):
    """Frontier scan: solve rate per level (stop below 20%), overall
    step validity, and the verified chains from solved traces (the
    on-policy mining source)."""
    import sympy as sp

    from bench_step_tokens import _gen_isolated, solve_chain
    sb: dict = {"solves": {}, "validity": 0.0, "chains": {}}
    valid = tried = 0
    for lv in levels:
        s = 0
        sb["chains"][lv] = []
        for i in range(n_per):
            p = _gen_isolated(lv, seed_base + 1000 * lv + i)
            if p is None:
                continue
            ok, pairs, v, t = solve_chain(
                tok, model, sp.sstr(p._expr), budget,
                seed0=seed_base + 1000 * lv + i)
            s += ok
            valid += v
            tried += t
            if ok:
                sb["chains"][lv].extend(pairs)
        sb["solves"][lv] = s
        if s < 0.2 * n_per:      # frontier scan stops below 20%
            break
    sb["validity"] = 100.0 * valid / max(tried, 1)
    return sb


def frontier(sb: dict, n_per: int) -> int:
    """Highest level in the 20-80% solve band; else highest evaluated."""
    band = [lv for lv, s in sb["solves"].items()
            if 0.2 * n_per <= s <= 0.8 * n_per]
    return max(band) if band else max(sb["solves"])


def gate_verdict(prev: dict, new: dict, frontier: int) -> tuple[bool, str]:
    """PROMOTE iff no level <= frontier regresses by more than 2
    solves AND (frontier solves improve OR validity gains >= 2 pts)."""
    for lv, s in prev["solves"].items():
        if lv <= frontier and new["solves"].get(lv, 0) < s - 2:
            return False, f"L{lv} regressed {s}->{new['solves'].get(lv, 0)}"
    gain_frontier = (new["solves"].get(frontier, 0)
                     > prev["solves"].get(frontier, 0))
    gain_validity = new["validity"] >= prev["validity"] + 2.0
    if gain_frontier or gain_validity:
        return True, "frontier gain" if gain_frontier else "validity gain"
    return False, "no improvement"

"""Autonomous expert-iteration loop driver (spec:
docs/superpowers/specs/2026-07-12-step-expert-iteration-design.md).
Round = evaluate -> mine -> train -> gate; state on disk; tripwires
halt the loop. All sympy touches forked (pathologies #7/#8/#10)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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

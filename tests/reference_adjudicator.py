"""Test-only reference evaluator for the pre-registration predicates.

Written independently of llmopt/lab/metrics.py and llmopt/lab/prereg.py
from the law as documented, so the production adjudicator is checked
against a second implementation rather than against itself. Nothing
here imports from llmopt.

Vocabulary: a predicate evaluates to one of
    "FIRE", "NO-FIRE", "UNRESOLVED", or "REFUSE"
where REFUSE means the operands cannot be compared at all (the
production code raises a typed error there). Arithmetic runs in exact
rationals so float rounding in the production path is observable.
"""
from __future__ import annotations

import math
from fractions import Fraction

FIRE, NOFIRE, UNRESOLVED, REFUSE = "FIRE", "NO-FIRE", "UNRESOLVED", "REFUSE"


def _real(x):
    """Exact Fraction for a finite int/float; None for anything else."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return Fraction(x)


def predicate(observed, bar, direction, fence=None):
    """Three-valued comparison with an optional half-width fence."""
    o, b = _real(observed), _real(bar)
    if o is None or b is None or direction not in ("above", "below"):
        return REFUSE
    if fence is not None:
        w = _real(fence)
        if w is None or w < 0:
            return REFUSE
        if abs(o - b) < w:
            return UNRESOLVED
    if direction == "below":
        return FIRE if o <= b else NOFIRE
    return FIRE if o >= b else NOFIRE


def conjunction(parts):
    """A bar is the conjunction of its main predicate and conjuncts.
    REFUSE anywhere refuses; UNRESOLVED anywhere leaves the bar
    unresolved; otherwise FIRE only when every part fires."""
    parts = list(parts)
    if any(p == REFUSE for p in parts):
        return REFUSE
    if any(p == UNRESOLVED for p in parts):
        return UNRESOLVED
    return FIRE if all(p == FIRE for p in parts) else NOFIRE


def _fence_width(obj):
    f = obj.get("fence")
    return None if f is None else f.get("width")


def _triple(m, spec, bar):
    """Measurement metric triple must equal the registered one (a
    conjunct inherits the bar's triple where it declares none)."""
    for k in ("metric", "population", "aggregation"):
        if m.get(k) != spec.get(k, bar[k]):
            return False
    return True


def bar_outcome(bar, obs):
    """Reference for one bar. Returns (outcome, kind) where kind is
    "outcome" for FIRE/NO-FIRE/UNRESOLVED and "refuse" for a contract
    refusal."""
    if not obs.get("measurement_valid", False):
        return UNRESOLVED, "outcome"
    arms = obs.get("arms", {})
    for a in bar["arms"]:
        if a not in arms or not arms[a].get("admissible", False):
            return UNRESOLVED, "outcome"
    con = obs.get("contrasts", {}).get(str(bar["id"]))
    if con is not None and not con.get("admissible", False):
        return UNRESOLVED, "outcome"
    ms = obs.get("measurements", {})
    m = ms.get(str(bar["id"]))
    if m is None:
        return UNRESOLVED, "outcome"
    if not _triple(m, bar, bar):
        return REFUSE, "refuse"
    parts = [predicate(m.get("value"), bar["value"], bar["direction"],
                       _fence_width(bar))]
    if parts[0] == REFUSE:
        return REFUSE, "refuse"
    for cj in bar.get("conjuncts", []):
        cm = ms.get(cj["measurement"])
        if cm is None:
            parts.append(UNRESOLVED)
            break
        if not _triple(cm, cj, bar):
            return REFUSE, "refuse"
        parts.append(predicate(cm.get("value"), cj["value"],
                               cj["direction"], _fence_width(cj)))
    out = conjunction(parts)
    return out, ("refuse" if out == REFUSE else "outcome")


def refutation(doc, obs, bar_outcomes):
    """Reference for the refutation clause. Returns None when the
    document has no structured predicate or no measurement; otherwise
    one of REFUTED / NOT-REFUTED / UNRESOLVED / UNADJUDICATED / REFUSE
    (the production strings carry a parenthetical the reference does
    not reproduce)."""
    rp = doc.get("refuted_if_predicate")
    if rp is None:
        return None
    prec = doc.get("refutation_precedence") or {}
    for i in prec.get("suppressed_unless_bars_fire", []):
        if bar_outcomes[i] != FIRE:
            return "UNADJUDICATED"
    for i in prec.get("applies_only_if_bars_fire", []):
        if bar_outcomes[i] == UNRESOLVED:
            return UNRESOLVED
        if bar_outcomes[i] != FIRE:
            return "NOT-REFUTED"
    m = obs.get("measurements", {}).get(rp["measurement"])
    if m is None:
        return None
    v = predicate(m.get("value"), rp["value"], rp["direction"],
                  _fence_width(rp))
    if v == REFUSE:
        return REFUSE
    if v == UNRESOLVED:
        return UNRESOLVED
    return "REFUTED" if v == FIRE else "NOT-REFUTED"

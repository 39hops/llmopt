"""Axiom oracle adapter — Phase A of the axiom backend
(docs/superpowers/specs/2026-07-18-axiom-backend.md).

Wraps the axiom_sym pybind module behind the doctrine:
- sympy is the ORACLE OF RECORD until the live shadow gate passes
  (>=1e5 verifications, zero axiom-EQUIVALENT/sympy-unequal).
- UNDECIDED / parse-reject / import-missing always falls back to
  sympy and NEVER counts as valid on axiom's word alone.
- SHADOW mode (default): run axiom alongside sympy, log every
  disagreement to a JSONL audit trail, RETURN SYMPY'S ANSWER.
  PRIMARY mode (post-gate): return axiom's decided verdicts,
  sympy only on UNDECIDED.

Usage:
    from llmopt.search.axiom_oracle import AxiomOracle
    orc = AxiomOracle(mode="shadow", audit="axiom_shadow.jsonl")
    verdict = orc.equivalent(lhs_sstr, rhs_sstr)   # True/False
"""
from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import axiom_sym  # pybind bridge (built per-platform, opt-in)
    HAVE_AXIOM = True
except ImportError:
    axiom_sym = None
    HAVE_AXIOM = False


class AxiomOracle:
    def __init__(self, mode: str = "shadow",
                 audit: str = "axiom_shadow.jsonl",
                 sympy_equiv=None):
        assert mode in ("shadow", "primary")
        self.mode = mode if HAVE_AXIOM else "shadow"
        self.audit = Path(audit)
        self.n = self.n_axiom_decided = self.n_disagree = 0
        # injected sympy-side check: (lhs_str, rhs_str) -> bool.
        # Kept injectable so the farm's existing verify path (with
        # its fork walls) stays the reference implementation.
        self._sympy_equiv = sympy_equiv
        if self._sympy_equiv is None:
            from llmopt.search.derivation import _timeboxed  # noqa
            import sympy as sp

            def _default(l: str, r: str) -> bool:
                dl = sp.simplify(sp.sympify(l) - sp.sympify(r))
                return dl == 0
            self._sympy_equiv = _default

    def _axiom_verdict(self, lhs: str, rhs: str) -> str | None:
        """EQUIVALENT / NOT_EQUIVALENT / UNDECIDED, or None on any
        bridge failure (parse reject, missing module, exception)."""
        if not HAVE_AXIOM:
            return None
        try:
            return axiom_sym.equivalent(lhs, rhs, "x")
        except Exception:
            return None

    def equivalent(self, lhs: str, rhs: str) -> bool:
        self.n += 1
        av = self._axiom_verdict(lhs, rhs)
        if self.mode == "primary" and av in ("EQUIVALENT",
                                             "NOT_EQUIVALENT"):
            self.n_axiom_decided += 1
            return av == "EQUIVALENT"
        # shadow (or undecided-in-primary): sympy answers
        sv = self._sympy_equiv(lhs, rhs)
        if av in ("EQUIVALENT", "NOT_EQUIVALENT"):
            self.n_axiom_decided += 1
            agree = (av == "EQUIVALENT") == sv
            if not agree:
                self.n_disagree += 1
                with self.audit.open("a") as f:
                    f.write(json.dumps({
                        "t": time.time(), "lhs": lhs, "rhs": rhs,
                        "axiom": av, "sympy": bool(sv)}) + "\n")
        return sv

    def stats(self) -> dict:
        return {"n": self.n, "axiom_decided": self.n_axiom_decided,
                "disagreements": self.n_disagree,
                "decided_rate": self.n_axiom_decided / max(self.n, 1),
                "mode": self.mode, "have_axiom": HAVE_AXIOM}

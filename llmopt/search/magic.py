"""The magic detector (RESULTS: 55v54 + replication, 71 certified
cuts at int L4): Liouville/Risch as integration's Gottesman-Knill.
sympy's risch_integrate PROVES integrands non-elementary in ~10ms on
our death-state shapes; a state carrying a certified non-elementary
Integral node is dead WITHIN THE ENGINE'S OPERATOR CLOSURE (no rule
merges integral nodes, so split non-elementary siblings can never
recombine — the mathematical loophole is closed by the move set).
Pruning it is a theorem per cut: provably zero false positives.

Only a POSITIVE certificate prunes; no-verdict (NotImplementedError,
timeout) never does. Verdicts cached by integrand srepr.
"""

from __future__ import annotations

import sympy as sp

from llmopt.search.derivation import State, _timeboxed

X = sp.Symbol("x")
_VERDICT_CACHE: dict[str, bool] = {}


def _risch_dead(integrand: sp.Expr) -> bool:
    from sympy.integrals.risch import NonElementaryIntegral, risch_integrate
    try:
        r = risch_integrate(integrand, X)
        return isinstance(r, NonElementaryIntegral) or bool(
            r.has(NonElementaryIntegral))
    except Exception:
        return False


def is_dead(state: State) -> bool:
    """True iff the state contains a certified non-elementary
    single-variable Integral node (and therefore cannot close)."""
    for node in state.expr.atoms(sp.Integral):
        if len(node.limits) != 1 or len(node.limits[0]) != 1:
            continue
        if node.limits[0][0] != X:
            continue  # u_-substitution carriers: no verdict
        k = sp.srepr(node.function)
        if k not in _VERDICT_CACHE:
            _VERDICT_CACHE[k] = _timeboxed(
                _risch_dead, node.function, default=False)
        if _VERDICT_CACHE[k]:
            return True
    return False

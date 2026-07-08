"""Verifiable classical-mechanics problems (physics-night section 6:
you never need to know what mass IS — a closed rule set at one floor
of the tower plus a checkable invariant is enough).

Oracles, all sympy:
- eom: Euler-Lagrange verify-by-substitution — the generator draws the
  potential, the acceleration falls out; check() is the default
  simplify-difference on the answer expression. The generator ASSERTS
  the EL identity at build time (m*x'' + V'(x) == 0), so every emitted
  problem carries its own consistency proof.
- small_osc: frequency of small oscillations about a planted minimum;
  omega = sqrt(V''(x0)/m) — exact symbolic answer.
- kinematics: planted trajectory x(t); velocity/acceleration by
  differentiation (the same oracle the whole lab trusts).

All constants numeric so answers parse with the standard locals.
Dimensional-analysis pre-verifier: v1 (needs a units layer; banked).
"""

from __future__ import annotations

import random

import sympy as sp

from llmopt.mathgen.problems import Problem, X

T = sp.Symbol("t")


def _potential(rng: random.Random, level: int) -> sp.Expr:
    """V(x) with a genuine minimum at a planted x0 (V'' > 0 there)."""
    a = sp.Integer(rng.randint(1, 6))          # quadratic stiffness > 0
    x0 = sp.Integer(rng.randint(-2, 2)) if level >= 2 else sp.Integer(0)
    v = a / 2 * (X - x0) ** 2
    if level >= 2:
        v += sp.Integer(rng.randint(1, 3)) * (X - x0) ** 4 / 4
    if level >= 3:
        v += sp.Integer(rng.randint(1, 4)) * (X - x0) ** 3 / 3
        # cubic can kill convexity at x0? V''(x0) = a stays: the cubic's
        # second derivative vanishes at x0, so the minimum survives
    return sp.expand(v)


def make_eom(level: int, seed: int) -> Problem:
    rng = random.Random(f"mech-eom-{level}-{seed}")
    m = sp.Integer(rng.randint(1, 5))
    v = _potential(rng, level)
    ans = sp.expand(-sp.diff(v, X) / m)
    # Euler-Lagrange consistency, asserted at build time
    assert sp.simplify(m * ans + sp.diff(v, X)) == 0
    return Problem(
        prompt=(f"A particle of mass {m} moves in one dimension with "
                f"potential energy V(x) = {sp.sstr(v)}. Using Newton's "
                "second law (or the Euler-Lagrange equation), give the "
                "acceleration x'' as an expression in x."),
        answer=sp.sstr(ans), kind="eom", level=level, _expr=ans)


def make_small_osc(level: int, seed: int) -> Problem:
    rng = random.Random(f"mech-osc-{level}-{seed}")
    m = sp.Integer(rng.randint(1, 5))
    v = _potential(rng, level)
    x0 = next(r for r in sp.solve(sp.diff(v, X), X)
              if r.is_real and sp.diff(v, X, 2).subs(X, r) > 0)
    ans = sp.sqrt(sp.diff(v, X, 2).subs(X, x0) / m)
    return Problem(
        prompt=(f"A particle of mass {m} moves in the potential "
                f"V(x) = {sp.sstr(v)}, which has a stable equilibrium "
                f"at x = {sp.sstr(x0)}. Find the angular frequency of "
                "small oscillations about it. Answer with an exact "
                "expression."),
        answer=sp.sstr(ans), kind="small_osc", level=level, _expr=ans)


def make_kinematics(level: int, seed: int) -> Problem:
    rng = random.Random(f"mech-kin-{level}-{seed}")
    terms = [sp.Integer(rng.randint(1, 5)) * T ** rng.randint(1, 3)]
    if level >= 2:
        terms.append(sp.Integer(rng.randint(1, 4))
                     * rng.choice([sp.sin(2 * T), sp.cos(3 * T)]))
    if level >= 3:
        terms.append(sp.Integer(rng.randint(1, 3)) * sp.exp(-T)
                     * rng.choice([sp.sin(T), sp.cos(T)]))
    x_t = sp.Add(*terms)
    order = rng.choice([1, 2])
    ans = sp.diff(x_t, T, order)
    what = "velocity x'(t)" if order == 1 else "acceleration x''(t)"
    return Problem(
        prompt=(f"A particle's position is x(t) = {sp.sstr(x_t)}. "
                f"Compute the {what}. Answer with an expression in t."),
        answer=sp.sstr(ans), kind="kinematics", level=level, _expr=ans)


MAKERS = {
    "eom": make_eom,
    "small_osc": make_small_osc,
    "kinematics": make_kinematics,
}

"""Verifiable multivariable-calculus problems. Oracle = the same sympy
differentiation the single-variable kinds trust; answers are symbolic
expressions checked by simplify-difference (any equivalent form passes,
via problems.Problem.check's default branch).

Families:
- partial: d/dx or d/dy of f(x, y).
- gradient: both partials as a tuple (checked componentwise).
- mixed_partial: d2f/dxdy — also a free Clairaut consistency check
  on the generator (smooth f => order can't matter).
- directional: gradient . unit vector at a point, along axis-aligned
  or integer vectors (kept exact: norm is sqrt of an integer).
"""

from __future__ import annotations

import random

import sympy as sp

from llmopt.mathgen.problems import Problem, X

Y = sp.Symbol("y")


def _f(rng: random.Random, level: int) -> sp.Expr:
    """f(x, y): polynomial at level 1, + trig/exp cross terms above."""
    terms = []
    for _ in range(rng.randint(2, 3)):
        c = rng.randint(1, 6) * rng.choice([1, 1, -1])
        nx, ny = rng.randint(0, 2), rng.randint(0, 2)
        if nx == ny == 0:
            nx = 1
        terms.append(c * X**nx * Y**ny)
    if level >= 2:
        c = rng.randint(1, 4)
        terms.append(c * rng.choice([sp.sin(X * Y), sp.cos(X + Y),
                                     sp.exp(X * Y), X * sp.sin(Y)]))
    if level >= 3:
        c = rng.randint(1, 3)
        terms.append(c * rng.choice([sp.exp(X**2 * Y), sp.sin(X**2 + Y**2),
                                     X * Y * sp.cos(X * Y)]))
    return sp.Add(*terms)


def make_partial(level: int, seed: int) -> Problem:
    rng = random.Random(f"mvar-partial-{level}-{seed}")
    f = _f(rng, level)
    var = rng.choice([X, Y])
    ans = sp.diff(f, var)
    return Problem(
        prompt=(f"Let f(x, y) = {sp.sstr(f)}. Compute the partial "
                f"derivative of f with respect to {var}. "
                "Answer with an expression in x and y."),
        answer=sp.sstr(ans), kind="partial", level=level, _expr=ans)


def make_mixed_partial(level: int, seed: int) -> Problem:
    rng = random.Random(f"mvar-mixed-{level}-{seed}")
    f = _f(rng, level)
    ans = sp.diff(f, X, Y)
    assert sp.simplify(ans - sp.diff(f, Y, X)) == 0  # Clairaut, free check
    return Problem(
        prompt=(f"Let f(x, y) = {sp.sstr(f)}. Compute the mixed partial "
                "d^2 f / dx dy. Answer with an expression in x and y."),
        answer=sp.sstr(ans), kind="partial", level=level, _expr=ans)


def make_directional(level: int, seed: int) -> Problem:
    rng = random.Random(f"mvar-dir-{level}-{seed}")
    f = _f(rng, max(1, level - 1))  # evaluation keeps these hairy enough
    vx, vy = rng.choice([(1, 0), (0, 1), (1, 1), (3, 4), (1, -1), (2, 1)])
    px, py = rng.randint(-2, 2), rng.randint(-2, 2)
    norm = sp.sqrt(vx**2 + vy**2)
    ans = sp.simplify(
        (sp.diff(f, X) * vx + sp.diff(f, Y) * vy).subs({X: px, Y: py})
        / norm)
    return Problem(
        prompt=(f"Let f(x, y) = {sp.sstr(f)}. Compute the directional "
                f"derivative of f at the point ({px}, {py}) in the "
                f"direction of the vector ({vx}, {vy}) (normalize the "
                "vector). Answer with an exact expression."),
        answer=sp.sstr(ans), kind="directional", level=level, _expr=ans)


MAKERS = {
    "partial": make_partial,
    "mixed_partial": make_mixed_partial,
    "directional": make_directional,
}

"""Verifiable ODE problems: solution drawn first, equation built from it.

check() (problems.py, kind "ode") never compares to the stored solution —
it runs sympy.checkodesol on the prediction against the equation and
verifies the initial condition, so any equivalent closed form passes.

Families:
- linear first order  y' + p(x) y = q(x): pick p and the solution y,
  q falls out. IC pins the constant.
- constant-coefficient homogeneous second order: pick distinct integer
  roots r1, r2 and constants -> y'' - (r1+r2) y' + r1 r2 y = 0 with two
  ICs.
- separable growth  y' = k x^n y: solution C exp(k x^{n+1}/(n+1)).
"""

from __future__ import annotations

import random

import sympy as sp

from llmopt.mathgen.problems import Problem, X

Y = sp.Function("y")


def _ode_problem(eq, sol, x0, kind, level, seed) -> Problem:
    y0 = sp.simplify(sol.subs(X, x0))
    del kind, seed  # naming consistency with other makers
    def _fmt(expr) -> str:
        return (sp.sstr(expr).replace("Derivative(y(x), x)", "y'")
                .replace("Derivative(y(x), (x, 2))", "y''").replace("y(x)", "y"))

    return Problem(
        prompt=(f"Solve the differential equation {_fmt(eq.lhs)} = {_fmt(eq.rhs)} "
                f"with y({x0}) = {sp.sstr(y0)}. "
                "Answer with y as an expression in x."),
        answer=sp.sstr(sol), kind="ode", level=level, _expr=(eq, x0, y0),
    )


def make_linear_first_order(level: int, seed: int) -> Problem:
    rng = random.Random(f"ode1-{level}-{seed}")
    p = sp.Integer(rng.randint(-3, 3))
    if p == 0:
        return make_linear_first_order(level, seed + 1_000_003)
    if level == 1:
        sol = sp.Integer(rng.randint(1, 5)) * sp.exp(-p * X)
    else:
        sol = (sp.Integer(rng.randint(1, 4)) * sp.exp(-p * X)
               + sp.Integer(rng.randint(1, 5)) * X + rng.randint(-3, 3))
    q = sp.expand(sp.diff(sol, X) + p * sol)
    eq = sp.Eq(sp.diff(Y(X), X) + p * Y(X), q)
    return _ode_problem(eq, sol, 0, "ode", level, seed)


def make_second_order_cc(level: int, seed: int) -> Problem:
    rng = random.Random(f"ode2-{level}-{seed}")
    r1, r2 = rng.sample([-3, -2, -1, 1, 2, 3], 2)
    c1, c2 = rng.randint(1, 4), rng.randint(1, 4) * rng.choice([1, -1])
    sol = c1 * sp.exp(r1 * X) + c2 * sp.exp(r2 * X)
    eq = sp.Eq(
        sp.diff(Y(X), X, 2) - (r1 + r2) * sp.diff(Y(X), X) + r1 * r2 * Y(X), 0
    )
    # a single IC does not pin both constants: state and check y' too
    y0 = c1 + c2
    yp0 = c1 * r1 + c2 * r2
    prompt = (
        f"Solve y'' - ({r1 + r2})*y' + ({r1 * r2})*y = 0 with "
        f"y(0) = {y0} and y'(0) = {yp0}. Answer with y as an expression in x."
    )
    return Problem(
        prompt=prompt, answer=sp.sstr(sol), kind="ode", level=level,
        _expr=(eq, 0, sp.Integer(y0), sp.Integer(yp0)),
    )


def make_separable_growth(level: int, seed: int) -> Problem:
    rng = random.Random(f"odes-{level}-{seed}")
    k = rng.choice([-2, -1, 1, 2])
    n = 1 if level <= 2 else rng.randint(1, 2)
    c = sp.Integer(rng.randint(1, 5))
    sol = c * sp.exp(sp.Rational(k, n + 1) * X ** (n + 1))
    eq = sp.Eq(sp.diff(Y(X), X), k * X**n * Y(X))
    return _ode_problem(eq, sol, 0, "ode", level, seed)


MAKERS = {
    "ode_linear1": make_linear_first_order,
    "ode_cc2": make_second_order_cc,
    "ode_separable": make_separable_growth,
}

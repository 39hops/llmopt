"""Verifiable calculus problem generator (sympy-grounded).

The point: training/eval data whose correctness is machine-checked, not
judged. Every problem carries a symbolic answer and a check() that
accepts any algebraically equivalent form — `2*x + 2` == `2*(x + 1)` —
so the metric is mathematical correctness, not string match.

Tasks:
- differentiate: d/dx of a sampled expression (answer: sympy diff).
- integrate: antiderivative. Sampled in reverse — draw F, present
  F' as the integrand — so every problem has an elementary answer.
  check() differentiates the prediction: any valid antiderivative
  passes, +C included.
- limit: rational 0/0 forms built by planting a common (x - a) factor;
  answer via factor cancellation (checked against sympy.limit).

Difficulty levels: 1 = polynomials; 2 = + sin/cos/exp; 3 = + products
and compositions (chain rule territory). Deterministic per seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

X = sp.Symbol("x")
_PARSE_LOCALS = {
    "x": X, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
    "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
    "e": sp.E, "C": sp.Symbol("C"),
}


def parse_answer(text: str):
    """Parse a model-produced expression. Returns None on garbage."""
    text = text.strip().rstrip(".")
    if "=" in text:  # tolerate "F(x) = ..." style
        text = text.split("=")[-1]
    try:
        return parse_expr(text, local_dict=_PARSE_LOCALS, evaluate=True)
    except Exception:
        return None


@dataclass(frozen=True)
class Problem:
    prompt: str
    answer: str          # canonical answer text (for training targets)
    kind: str            # differentiate | integrate | limit
    level: int
    _expr: object = field(compare=False, repr=False)  # task-specific payload

    def check(self, prediction: str) -> bool:
        pred = parse_answer(prediction)
        if pred is None:
            return False
        try:
            if self.kind == "integrate":
                # any antiderivative is correct: F'(pred) must equal integrand
                pred = pred.subs(sp.Symbol("C"), 0)
                return sp.simplify(sp.diff(pred, X) - self._expr) == 0
            return sp.simplify(pred - self._expr) == 0
        except Exception:
            return False


def _atom(rng: random.Random, level: int):
    c = rng.randint(1, 9) * rng.choice([1, 1, 1, -1])
    n = rng.randint(1, 3 if level == 1 else 5)
    choices = [c * X**n, c * X**n, c * X, sp.Integer(c)]
    if level >= 2:
        choices += [c * sp.sin(X), c * sp.cos(X), c * sp.exp(X), c * sp.log(X)]
    return rng.choice(choices)


def _expression(rng: random.Random, level: int):
    if level <= 2:
        return sum(_atom(rng, level) for _ in range(rng.randint(2, 4)))
    kind = rng.choice(["product", "compose", "mixed"])
    if kind == "product":
        return _atom(rng, 2) * _atom(rng, 2)
    if kind == "compose":
        inner = (rng.randint(1, 5) * X ** rng.randint(1, 3)
                 + rng.randint(-4, 4) * X + rng.randint(0, 5))
        outer = rng.choice([sp.sin, sp.cos, sp.exp])
        return outer(inner)
    return _atom(rng, 2) * _atom(rng, 2) + _atom(rng, 2)


def make_differentiate(level: int, seed: int) -> Problem:
    rng = random.Random(f"diff-{level}-{seed}")  # string seed: stable across processes
    f = _expression(rng, level)
    ans = sp.diff(f, X)
    return Problem(
        prompt=f"Differentiate with respect to x: {sp.sstr(f)}",
        answer=sp.sstr(ans), kind="differentiate", level=level, _expr=ans,
    )


def make_integrate(level: int, seed: int) -> Problem:
    rng = random.Random(f"int-{level}-{seed}")
    F = _expression(rng, level)          # draw the antiderivative...
    integrand = sp.simplify(sp.diff(F, X))  # ...present its derivative
    if integrand == 0:  # constant F: degenerate, retry deterministically
        return make_integrate(level, seed + 1_000_003)
    return Problem(
        prompt=f"Find an antiderivative of: {sp.sstr(integrand)}",
        answer=sp.sstr(F), kind="integrate", level=level, _expr=integrand,
    )


def make_limit(level: int, seed: int) -> Problem:
    rng = random.Random(f"lim-{level}-{seed}")
    a = rng.randint(-3, 3)
    q = _expression(rng, min(level, 2))
    s_ = rng.randint(1, 4) * X + rng.randint(1, 5)
    if s_.subs(X, a) == 0 or q.subs(X, a) is sp.nan:
        return make_limit(level, seed + 1_000_003)
    num = sp.expand((X - a) * q)
    den = sp.expand((X - a) * s_)
    ans = sp.simplify(q.subs(X, a) / s_.subs(X, a))
    return Problem(
        prompt=f"Evaluate the limit as x approaches {a} of ({sp.sstr(num)}) / ({sp.sstr(den)})",
        answer=sp.sstr(ans), kind="limit", level=level, _expr=ans,
    )


def make_second_derivative(level: int, seed: int) -> Problem:
    rng = random.Random(f"diff2-{level}-{seed}")
    f = _expression(rng, level)
    ans = sp.diff(f, X, 2)
    return Problem(
        prompt=f"Find the second derivative with respect to x of: {sp.sstr(f)}",
        answer=sp.sstr(ans), kind="second_derivative", level=level, _expr=ans,
    )


def make_definite_integral(level: int, seed: int) -> Problem:
    """Reverse-constructed like make_integrate; answer F(b) - F(a) is a
    closed-form constant, checked symbolically."""
    rng = random.Random(f"defint-{level}-{seed}")
    F = _expression(rng, min(level, 2))
    if any(f.func == sp.log for f in sp.preorder_traversal(F)):
        return make_definite_integral(level, seed + 1_000_003)  # keep domain safe
    a = rng.randint(0, 2)
    b = a + rng.randint(1, 3)
    integrand = sp.simplify(sp.diff(F, X))
    if integrand == 0:
        return make_definite_integral(level, seed + 1_000_003)
    ans = sp.simplify(F.subs(X, b) - F.subs(X, a))
    return Problem(
        prompt=(f"Evaluate the definite integral of {sp.sstr(integrand)} "
                f"with respect to x from {a} to {b}"),
        answer=sp.sstr(ans), kind="definite_integral", level=level, _expr=ans,
    )


def make_tangent_line(level: int, seed: int) -> Problem:
    """Tangent line to f at x = a; answer is a linear expression in x."""
    rng = random.Random(f"tan-{level}-{seed}")
    f = _expression(rng, min(level, 2))
    if any(fn.func == sp.log for fn in sp.preorder_traversal(f)):
        return make_tangent_line(level, seed + 1_000_003)
    a = rng.randint(-2, 2)
    slope = sp.diff(f, X).subs(X, a)
    ans = sp.expand(f.subs(X, a) + slope * (X - a))
    return Problem(
        prompt=(f"Find the tangent line to f(x) = {sp.sstr(f)} at x = {a}. "
                "Give the line as an expression in x."),
        answer=sp.sstr(ans), kind="tangent_line", level=level, _expr=ans,
    )


_MAKERS = {
    "differentiate": make_differentiate,
    "integrate": make_integrate,
    "limit": make_limit,
    "second_derivative": make_second_derivative,
    "definite_integral": make_definite_integral,
    "tangent_line": make_tangent_line,
}

ALL_KINDS = tuple(_MAKERS)


def make_dataset(
    n: int, *, kinds=("differentiate", "integrate", "limit"),
    levels=(1, 2, 3), seed: int = 0, exclude: frozenset[str] = frozenset(),
) -> list[Problem]:
    """n unique problems cycling kinds x levels, deterministic per seed.

    Prompts are deduplicated, and any prompt in ``exclude`` is skipped —
    generate the eval set first, then pass its prompts here so the train
    set provably contains no eval problem (random seeds alone don't
    guarantee it: the low-difficulty space is small enough to collide).
    """
    out = []
    seen = set(exclude)
    i = 0
    while len(out) < n:
        kind = kinds[i % len(kinds)]
        level = levels[(i // len(kinds)) % len(levels)]
        p = _MAKERS[kind](level, seed * 1_000_000 + i)
        i += 1
        if p.prompt in seen:
            continue
        seen.add(p.prompt)
        out.append(p)
    return out

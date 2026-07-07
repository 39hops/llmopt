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
    "e": sp.E, "C": sp.Symbol("C"), "Matrix": sp.Matrix, "I": sp.I,
    "i": sp.I,
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
        if "Answer:" in prediction:
            prediction = prediction.rsplit("Answer:", 1)[1].strip().splitlines()[0]
        pred = parse_answer(prediction)
        if pred is None:
            return False
        try:
            if self.kind == "integrate":
                # any antiderivative is correct: F'(pred) must equal integrand
                pred = pred.subs(sp.Symbol("C"), 0)
                return sp.simplify(sp.diff(pred, X) - self._expr) == 0
            if self.kind == "eigenvalues":
                # multiset equality, any order
                got = pred if isinstance(pred, (tuple, list)) else (pred,)
                return sorted(sp.sympify(g) for g in got) == self._expr
            if self.kind == "matrix_inverse":
                # verify, don't compare: A @ prediction must be identity
                got = sp.Matrix(pred)
                a = self._expr
                return got.shape == a.shape and sp.simplify(a * got) == sp.eye(a.rows)
            if self.kind == "ode":
                # verify against the equation + initial conditions, so any
                # equivalent closed form passes
                eq, x0, y0, *rest = self._expr
                y = sp.Function("y")
                if not bool(sp.checkodesol(eq, sp.Eq(y(X), pred))[0]):
                    return False
                if sp.simplify(pred.subs(X, x0) - y0) != 0:
                    return False
                if rest:  # second-order: y'(x0) must match too
                    return sp.simplify(sp.diff(pred, X).subs(X, x0) - rest[0]) == 0
                return True
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
    if level >= 4:
        return _expression_l4(rng)
    kind = rng.choice(["product", "compose", "mixed"])
    if kind == "product":
        return _atom(rng, 2) * _atom(rng, 2)
    if kind == "compose":
        inner = (rng.randint(1, 5) * X ** rng.randint(1, 3)
                 + rng.randint(-4, 4) * X + rng.randint(0, 5))
        outer = rng.choice([sp.sin, sp.cos, sp.exp])
        return outer(inner)
    return _atom(rng, 2) * _atom(rng, 2) + _atom(rng, 2)


def _expression_l4(rng: random.Random):
    """Level 4 (expert-iteration frontier): depth-3 compositions,
    composed products, chained-u-sub shapes, and sums of two. The
    space must stay wide — collision guard in tests/test_mathgen_l4.py
    (repo scar tissue: two real contamination incidents)."""
    def poly():
        while True:  # x-terms can cancel (3x - 3x + c): redraw
            p = (rng.randint(1, 5) * X ** rng.randint(1, 3)
                 + rng.randint(-4, 4) * X + rng.randint(0, 5))
            if p.has(X):
                return p

    def deep():
        outer = rng.choice([sp.sin, sp.cos, sp.exp])
        mid = rng.choice([sp.sin, sp.cos, sp.sqrt])
        return rng.randint(1, 9) * outer(poly() + mid(rng.randint(1, 9) * X))

    def composed_product():
        outer = rng.choice([sp.sin, sp.cos, sp.exp])
        return _atom(rng, 2) * outer(poly()) * rng.randint(1, 9)

    def chained():
        g = poly()
        fn = rng.choice([sp.sin, sp.cos, sp.exp])
        return rng.randint(1, 9) * sp.diff(g, X) * fn(g) ** rng.randint(1, 4)

    kind = rng.choice(["deep", "composed_product", "chained", "sum2"])
    if kind == "deep":
        return deep()
    if kind == "composed_product":
        return composed_product()
    if kind == "chained":
        return chained()
    parts = [deep(), composed_product(), chained()]
    rng.shuffle(parts)
    return parts[0] + parts[1]


def _cexpr(rng: random.Random, level: int):
    """Complex-coefficient expression (spec: mathgen-expansion Part A).
    Level 1: Gaussian-integer-coefficient polynomials; 2: + e^{ikx}
    atoms; 3: products/compositions."""
    def gauss():
        while True:
            z = rng.randint(-4, 4) + rng.randint(-4, 4) * sp.I
            if z != 0:
                return z

    def catom():
        choices = [gauss() * X ** rng.randint(1, 4), gauss() * X]
        if level >= 2:
            choices += [gauss() * sp.exp(sp.I * rng.randint(1, 5) * X),
                        gauss() * sp.exp((rng.randint(-2, 2)
                                          + rng.randint(1, 3) * sp.I) * X)]
        return rng.choice(choices)

    if level <= 2:
        return sum(catom() for _ in range(rng.randint(2, 3)))
    return catom() * catom() + catom()


def make_cdiff(level: int, seed: int) -> Problem:
    rng = random.Random(f"cdiff-{level}-{seed}")  # string seed
    f = _cexpr(rng, level)
    ans = sp.diff(f, X)
    return Problem(
        prompt=f"Differentiate with respect to x (I is the imaginary "
               f"unit): {sp.sstr(f)}",
        answer=sp.sstr(ans), kind="cdiff", level=level, _expr=ans,
    )


def make_cint(level: int, seed: int) -> Problem:
    rng = random.Random(f"cint-{level}-{seed}")
    F = _cexpr(rng, level)
    integrand = sp.simplify(sp.diff(F, X))
    if integrand == 0:
        return make_cint(level, seed + 1_000_003)
    # kind="integrate" so check() verifies by differentiating: any
    # antiderivative (mod constant, complex constants included) passes
    return Problem(
        prompt=f"Find an antiderivative (I is the imaginary unit) "
               f"of: {sp.sstr(integrand)}",
        answer=sp.sstr(F), kind="integrate", level=level, _expr=integrand,
    )


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


def _limit_parts(level: int, seed: int):
    rng = random.Random(f"lim-{level}-{seed}")
    a = rng.randint(-3, 3)
    q = _expression(rng, min(level, 2))
    s_ = rng.randint(1, 4) * X + rng.randint(1, 5)
    if s_.subs(X, a) == 0 or q.subs(X, a) is sp.nan:
        return _limit_parts(level, seed + 1_000_003)
    num = sp.expand((X - a) * q)
    den = sp.expand((X - a) * s_)
    ans = sp.simplify(sp.Rational(1, 1) * q.subs(X, a) / s_.subs(X, a))
    return a, q, s_, num, den, ans


def make_limit(level: int, seed: int) -> Problem:
    a, q, s_, num, den, ans = _limit_parts(level, seed)
    return Problem(
        prompt=f"Evaluate the limit as x approaches {a} of ({sp.sstr(num)}) / ({sp.sstr(den)})",
        answer=sp.sstr(ans), kind="limit", level=level, _expr=ans,
    )


def make_limit_traced(level: int, seed: int) -> Problem:
    """Same limits, but the target is the worked factor/cancel/substitute
    trace ending in 'Answer: <value>' — the generator knows its own
    construction, so correct step decompositions are free. check() scores
    only the final answer line (the steps are scaffolding, the metric
    stays symbolic)."""
    a, q, s_, num, den, ans = _limit_parts(level, seed)
    trace = "\n".join([
        f"Step 1: factor the numerator: {sp.sstr(num)} = (x - ({a}))*({sp.sstr(q)})",
        f"Step 2: factor the denominator: {sp.sstr(den)} = (x - ({a}))*({sp.sstr(s_)})",
        f"Step 3: cancel the common factor (x - ({a})): "
        f"the limit equals ({sp.sstr(q)})/({sp.sstr(s_)}) at x = {a}",
        f"Step 4: substitute x = {a}: ({sp.sstr(q.subs(X, a))})/({sp.sstr(s_.subs(X, a))})",
        f"Answer: {sp.sstr(ans)}",
    ])
    return Problem(
        prompt=(f"Evaluate the limit as x approaches {a} of "
                f"({sp.sstr(num)}) / ({sp.sstr(den)}). Work step by step, "
                "then give the final line as 'Answer: <value>'."),
        answer=trace, kind="limit_traced", level=level, _expr=ans,
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
    "limit_traced": make_limit_traced,
    "second_derivative": make_second_derivative,
    "definite_integral": make_definite_integral,
    "tangent_line": make_tangent_line,
}

ALL_KINDS = tuple(_MAKERS)


def _resolve_maker(kind: str):
    """Makers from sibling modules (linalg, odes) register lazily — avoids
    an import cycle since they build on Problem from this module."""
    if kind not in _MAKERS:
        from llmopt.mathgen import linalg, odes

        _MAKERS.update(linalg.MAKERS)
        _MAKERS.update(odes.MAKERS)
    return _MAKERS[kind]


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
        p = _resolve_maker(kind)(level, seed * 1_000_000 + i)
        i += 1
        if p.prompt in seen:
            continue
        seen.add(p.prompt)
        out.append(p)
    return out

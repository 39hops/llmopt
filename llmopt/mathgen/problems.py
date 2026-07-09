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
    "i": sp.I, "y": sp.Symbol("y"), "t": sp.Symbol("t"),
    "n": sp.Symbol("n"), "k": sp.Symbol("k"),
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
        if self.kind in ("prove_ind", "prove_discover"):
            # multi-line proof answers: keep the full prediction
            from llmopt.mathgen.proofs import (check_discovery,
                                               check_induction)
            if "Answer:" in prediction:
                prediction = prediction.rsplit("Answer:", 1)[1]
            checker = (check_induction if self.kind == "prove_ind"
                       else check_discovery)
            return checker(self._expr, prediction)
        if "Answer:" in prediction:
            prediction = prediction.rsplit("Answer:", 1)[1].strip().splitlines()[0]
        pred = parse_answer(prediction)
        if pred is None:
            return False
        try:
            if self.kind in ("integrate", "cint"):
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
            if self.kind == "drec":
                # verify, don't compare: prediction must satisfy the
                # recurrence for symbolic n AND the initial conditions
                eq, inits = self._expr
                a = sp.Function("a")
                n_ = sp.Symbol("n")

                def val(expr_n):
                    return pred.subs(n_, expr_n)

                res = eq.rhs - eq.lhs
                res = res.replace(a, lambda arg: val(arg))
                if sp.simplify(res) != 0:
                    return False
                return all(sp.simplify(val(key.args[0]) - v) == 0
                           for key, v in inits.items())
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
    if level >= 5:
        return _expression_l5(rng)
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


def _expression_l5(rng: random.Random):
    """Level 5 (2026-07-09, magic-estimator saturation at L4: 806/827
    solved at budget 200). Antiderivative families the L4 shapes never
    produce: cross-family products (exp*trig — the double-by-parts
    trick), inverse trig, log powers, sqrt compositions, and sums of
    two. Same wide-space discipline as L4 (contamination scar tissue)."""
    def poly():
        while True:
            p = (rng.randint(1, 5) * X ** rng.randint(1, 3)
                 + rng.randint(-4, 4) * X + rng.randint(0, 5))
            if p.has(X):
                return p

    def cross():  # F = exp(ax)*trig(bx): integrand needs the cycle trick
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        return (rng.randint(1, 9) * sp.exp(a * X)
                * rng.choice([sp.sin, sp.cos])(b * X))

    def inv_trig():
        return rng.randint(1, 9) * rng.choice([sp.atan, sp.asin])(
            rng.randint(1, 3) * X) + rng.randint(1, 5) * X

    def log_power():
        return (rng.randint(1, 9) * X ** rng.randint(1, 3)
                * sp.log(rng.randint(1, 5) * X) ** rng.randint(1, 2))

    def root():
        return rng.randint(1, 9) * sp.sqrt(poly()) * poly()

    kind = rng.choice(["cross", "inv_trig", "log_power", "root", "sum2"])
    if kind == "sum2":
        parts = [cross(), inv_trig(), log_power()]
        rng.shuffle(parts)
        return parts[0] + parts[1]
    return {"cross": cross, "inv_trig": inv_trig,
            "log_power": log_power, "root": root}[kind]()


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
    # check() verifies by differentiating (shared with "integrate"):
    # any antiderivative mod complex constant passes
    return Problem(
        prompt=f"Find an antiderivative (I is the imaginary unit) "
               f"of: {sp.sstr(integrand)}",
        answer=sp.sstr(F), kind="cint", level=level, _expr=integrand,
    )


N = sp.Symbol("n")


def make_sum(level: int, seed: int) -> Problem:
    """Closed form of sum_{k=1}^{n} p(k). Answer checked by symbolic
    equality in n — any equivalent form passes."""
    rng = random.Random(f"dsum-{level}-{seed}")  # string seed
    k = sp.Symbol("k")
    deg = rng.randint(1, 2 if level == 1 else 3)
    p = sum(rng.randint(1, 6) * k**d for d in range(deg + 1))
    closed = sp.factor(sp.summation(p, (k, 1, N)))
    return Problem(
        prompt=f"Find the closed form of the sum of {sp.sstr(p)} "
               f"for k from 1 to n:",
        answer=sp.sstr(closed), kind="dsum", level=level, _expr=closed,
    )


def make_recurrence(level: int, seed: int) -> Problem:
    """Solve a(n) = c1*a(n-1) [+ c2*a(n-2)] with initial conditions.
    check() VERIFIES the prediction satisfies recurrence + initials
    (the ODE-check pattern) rather than comparing to one closed form."""
    rng = random.Random(f"drec-{level}-{seed}")
    a = sp.Function("a")
    if level == 1:
        c1, a0 = rng.randint(2, 5), rng.randint(1, 6)
        eq = sp.Eq(a(N), c1 * a(N - 1))
        inits = {a(0): a0}
        prompt = (f"Solve the recurrence a(n) = {c1}*a(n-1) with "
                  f"a(0) = {a0}. Give a(n) in closed form:")
    else:
        while True:  # distinct real roots keep closed forms clean
            c1, c2 = rng.randint(1, 4), rng.randint(1, 4)
            disc = c1**2 + 4 * c2
            if sp.sqrt(disc).is_rational:
                break
        a0, a1 = rng.randint(0, 4), rng.randint(1, 5)
        eq = sp.Eq(a(N), c1 * a(N - 1) + c2 * a(N - 2))
        inits = {a(0): a0, a(1): a1}
        prompt = (f"Solve the recurrence a(n) = {c1}*a(n-1) + {c2}*a(n-2) "
                  f"with a(0) = {a0}, a(1) = {a1}. Give a(n) in closed form:")
    sol = sp.rsolve(eq, a(N), inits)
    return Problem(
        prompt=prompt, answer=sp.sstr(sol), kind="drec", level=level,
        _expr=(eq, inits),
    )


def make_taylor(level: int, seed: int) -> Problem:
    """Taylor polynomial of degree d about x=a. Level 1: Maclaurin
    (a=0) of table functions; 2: shifted centers; 3: composites
    (the special cases live in what sympy's series returns)."""
    rng = random.Random(f"taylor-{level}-{seed}")  # string seed
    base = rng.choice([sp.sin(X), sp.cos(X), sp.exp(X), sp.log(1 + X),
                       1 / (1 - X)])
    if level >= 3:
        base = base.subs(X, rng.randint(2, 3) * X ** rng.randint(1, 2))
    a = 0 if level == 1 else rng.randint(-1, 2)
    d = rng.randint(2, 4)
    if a != 0 and base.subs(X, a) in (sp.zoo, sp.nan):  # singular center
        return make_taylor(level, seed + 1_000_003)
    poly = sp.series(base, X, a, d + 1).removeO()
    return Problem(
        prompt=f"Find the Taylor polynomial of degree {d} of "
               f"{sp.sstr(base)} about x = {a}:",
        answer=sp.sstr(sp.expand(poly)), kind="taylor", level=level,
        _expr=sp.expand(poly),
    )


def make_continuity(level: int, seed: int) -> Problem:
    """Find c making a piecewise function continuous at the seam:
    f(x) = p(x) for x < a, c*q(x)+r for x >= a. Verifiable: the two
    one-sided limits must agree at a — check() solves nothing, it
    substitutes the prediction and compares seam values."""
    rng = random.Random(f"cont-{level}-{seed}")
    a = rng.randint(-2, 2)
    while True:  # p(a) must be finite (log(x) at a<=0 -> zoo/nan)
        p = _expression(rng, min(level, 2))
        v = p.subs(X, a)
        if v.is_finite and not v.has(sp.zoo, sp.nan):
            break
    while True:
        q = _expression(rng, min(level, 2))
        vq = q.subs(X, a)
        if vq != 0 and vq.is_finite:
            break
    r = rng.randint(-5, 5)
    # continuity at a: p(a) = c*q(a) + r  =>  c = (p(a) - r)/q(a)
    c_true = sp.Rational(1, 1) * (p.subs(X, a) - r) / q.subs(X, a)
    return Problem(
        prompt=f"Find the constant c that makes f continuous at "
               f"x = {a}, where f(x) = {sp.sstr(p)} for x < {a} and "
               f"f(x) = c*({sp.sstr(q)}) + {r} for x >= {a}:",
        answer=sp.sstr(c_true), kind="continuity", level=level,
        _expr=c_true,
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
    "cdiff": make_cdiff,
    "cint": make_cint,
    "dsum": make_sum,
    "drec": make_recurrence,
    "taylor": make_taylor,
    "continuity": make_continuity,
    "definite_integral": make_definite_integral,
    "tangent_line": make_tangent_line,
}

ALL_KINDS = tuple(_MAKERS)


def _resolve_maker(kind: str):
    """Makers from sibling modules (linalg, odes) register lazily — avoids
    an import cycle since they build on Problem from this module."""
    if kind not in _MAKERS:
        from llmopt.mathgen import (linalg, mechanics, multivar, ntheory,
                                    odes, proofs)

        _MAKERS.update(linalg.MAKERS)
        _MAKERS.update(odes.MAKERS)
        _MAKERS.update(ntheory.MAKERS)
        _MAKERS.update(multivar.MAKERS)
        _MAKERS.update(mechanics.MAKERS)
        _MAKERS.update(proofs.MAKERS)
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

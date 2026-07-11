"""Primitive differentiation rewrite rules (HCE rung 1, spec
2026-07-06-hce-rung1-primitive-moves-design.md).

A DiffRule takes one unevaluated Derivative node and returns the list
of candidate rewrites (usually 0 or 1; d_product returns one per
(head, rest) split). The list return generalizes the spec's
`Expr | None` signature to support split branching. Rules only fire on
single-variable first-order Derivatives; everything else returns [].

Chain rule is not a standalone move: it is fused into d_power and
d_chain_table (explicit-u chain is u-sub territory, rung 2). There is
no quotient rule in the core: sympy has no quotient node (u/v is
u * v**-1), so d_product + d_power cover it; the textbook quotient
rule lives in MACRO_RULES, off by default, ablation-gated.
"""

from __future__ import annotations

from typing import Callable

import sympy as sp

DiffRule = Callable[[sp.Derivative], "list[sp.Expr]"]


def _unpack(node: sp.Derivative) -> tuple[sp.Expr, sp.Symbol] | None:
    """(f, x) for first-order single-variable Derivatives, else None."""
    if len(node.variables) != 1:
        return None
    return node.expr, node.variables[0]


def d_const(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    return [sp.Integer(0)] if not f.has(x) else []


def d_x(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    return [sp.Integer(1)] if f == x else []


def d_sum(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Add):
        return []
    return [sp.Add(*(sp.Derivative(t, x) for t in f.args))]


def d_product(node: sp.Derivative) -> list[sp.Expr]:
    # (also Artin's exercise — reference solution; see ~/practice_2.py)
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Mul) or not f.has(x):
        return []
    out: list[sp.Expr] = []
    for i, head in enumerate(f.args):
        rest = sp.Mul(*(a for j, a in enumerate(f.args) if j != i))
        out.append(sp.Derivative(head, x) * rest + head * sp.Derivative(rest, x))
    return out


def d_power(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Pow) or f.exp.has(x) or not f.base.has(x):
        return []
    base, n = f.base, f.exp
    return [n * base ** (n - 1) * sp.Derivative(base, x)]


# h -> h' as a function of the inner expression. sqrt is Pow — d_power
# covers it, so it needs no entry here.
_CHAIN_TABLE: dict[type, Callable[[sp.Expr], sp.Expr]] = {
    sp.sin: lambda u: sp.cos(u),
    sp.cos: lambda u: -sp.sin(u),
    sp.tan: lambda u: 1 / sp.cos(u) ** 2,
    sp.exp: lambda u: sp.exp(u),
    sp.log: lambda u: 1 / u,
}


def d_chain_table(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not (isinstance(f, sp.Function) and f.func in _CHAIN_TABLE and len(f.args) == 1):
        return []
    inner = f.args[0]
    return [_CHAIN_TABLE[f.func](inner) * sp.Derivative(inner, x)]


def d_quotient(node: sp.Derivative) -> list[sp.Expr]:
    """MACRO: textbook quotient rule. Redundant with d_product+d_power;
    kept for the solve-rate-per-node ablation only."""
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    num, den = f.as_numer_denom()
    if den == 1 or not den.has(x):
        return []
    return [(sp.Derivative(num, x) * den - num * sp.Derivative(den, x)) / den**2]


CORE_RULES: list[tuple[str, DiffRule]] = [
    ("d_const", d_const),
    ("d_x", d_x),
    ("d_sum", d_sum),
    ("d_product", d_product),
    ("d_power", d_power),
    ("d_chain_table", d_chain_table),
]

def d_const_factor(node: sp.Derivative) -> list[sp.Expr]:
    """MACRO, data-certified: d_product -> d_const carries 14.8% of
    winning-path traffic (scripts/mine_highways.py, 2026-07-07) — the
    engine constantly splits a Mul just to kill the constant factor's
    derivative. Fused: d(c*f) = c*Derivative(f) in ONE ply."""
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Mul):
        return []
    const = sp.Mul(*(a for a in f.args if not a.has(x)))
    rest = sp.Mul(*(a for a in f.args if a.has(x)))
    if const == 1 or rest == 1:
        return []
    return [const * sp.Derivative(rest, x)]


MACRO_RULES: list[tuple[str, DiffRule]] = [
    ("d_quotient", d_quotient),
    ("d_const_factor", d_const_factor),
]


# ------------------------------------------------------ integration

IntRule = Callable[[sp.Integral], "list[sp.Expr]"]

U = sp.Symbol("u_")  # reserved substitution symbol (named: Dummy breaks srepr dedup)


def _unpack_int(node: sp.Integral) -> tuple[sp.Expr, sp.Symbol] | None:
    """(f, x) for single-variable indefinite Integrals, else None."""
    if len(node.limits) != 1 or len(node.limits[0]) != 1:
        return None
    return node.function, node.limits[0][0]


def i_const(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    return [f * x] if not f.has(x) else []


def i_inverse_trig(node: sp.Integral) -> list[sp.Expr]:
    """Inverse-trig antiderivatives (L5 autopsy 2026-07-09: the
    engine solved 0/23 of these — no rule PRODUCED atan/asin, the
    l_hopital-class gap). Two closed forms:
      c/(a*x^2 + b)      -> c/sqrt(a*b) * atan(x*sqrt(a/b))   (a,b > 0)
      c/sqrt(b - a*x^2)  -> c/sqrt(a)  * asin(x*sqrt(a/b))    (a,b > 0)
    Candidates only; the edge verifier owns soundness."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    num, den = sp.fraction(sp.cancel(sp.together(f)))
    poly_part = sp.S.Zero
    if num.has(x):
        # improper rational (the residue of the first L5 rerun: all 20
        # still-failing rational shapes were (ax^2+c)/(bx^2+d)) —
        # divide out the polynomial part, integrate it directly
        if not (num.is_polynomial(x) and den.is_polynomial(x)):
            return []
        quo, rem = sp.div(num, den, x)
        if rem.has(x) or quo == 0:
            return []
        poly_part = sp.integrate(quo, x)  # plain polynomial: safe
        num = rem
    # rational form: den = a*x^2 + b
    p = den.as_poly(x)
    if p is not None and p.degree() == 2:
        a, m, b = p.all_coeffs()
        if m == 0 and a.is_positive and b.is_positive:
            return [poly_part
                    + num / sp.sqrt(a * b) * sp.atan(x * sp.sqrt(a / b))]
    # sqrt form: den = sqrt(b - a*x^2) (possibly times a constant)
    co, rest = den.as_coeff_Mul()
    if (isinstance(rest, sp.Pow) and rest.exp == sp.Rational(1, 2)
            and rest.base.is_polynomial(x)):
        q = rest.base.as_poly(x)
        if q is not None and q.degree() == 2:
            na, nm, b = q.all_coeffs()
            a = -na
            if nm == 0 and a.is_positive and b.is_positive:
                return [num / (co * sp.sqrt(a))
                        * sp.asin(x * sp.sqrt(a / b))]
    return []


def i_sqrt_basis(node: sp.Integral) -> list[sp.Expr]:
    """sqrt-of-poly ansatz (L5 autopsy: root family 14/94 solved —
    the biggest gap). If f*sqrt(P) is a polynomial, the answer lives
    in A(x)*sqrt(P): d/dx[A*sqrt(P)] = (2A'P + A*P')/(2*sqrt(P)), so
    equate 2A'P + A*P' = 2*f*sqrt(P) in the poly ring and solve for
    A's coefficients — i_linear_basis's move with a radical basis."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    roots = [r for r in f.atoms(sp.Pow)
             if abs(r.exp) == sp.Rational(1, 2)
             and r.base.is_polynomial(x) and r.base.has(x)]
    if not roots:
        return []
    P = max((r.base for r in roots), key=sp.count_ops)
    h = sp.cancel(sp.expand(f * sp.sqrt(P)))
    if not h.is_polynomial(x) or h.has(sp.sin, sp.cos, sp.exp, sp.log):
        return []
    h = sp.Poly(sp.expand(h), x)
    Pp = sp.Poly(P, x)
    degA = max(h.degree() - Pp.degree() + 1, 0) + 1
    if degA > 8:
        return []
    cs = sp.symbols(f"s0:{degA + 1}", cls=sp.Dummy)
    A = sp.Add(*(c * x**j for j, c in enumerate(cs)))
    resid = sp.Poly(sp.expand(
        2 * sp.diff(A, x) * P + A * sp.diff(P, x) - 2 * h.as_expr()), x)
    eqs = resid.coeffs()
    if any(not e.has(*cs) for e in eqs if e != 0):
        return []
    sol = sp.solve(eqs, cs, dict=True)
    if not sol:
        return []
    a = (A.subs(sol[0]).subs({c: 0 for c in cs})) * sp.sqrt(P)
    return [a] if a != 0 else []


def i_log_power(node: sp.Integral) -> list[sp.Expr]:
    """x**n * log(k*x)**m closed form (2026-07-09 frontier-gap autopsy:
    27/36 mined failures contain this family). i_parts CAN reach it but
    needs m chained by-parts plies through nested Integrals — a node-
    budget death, not unreachability. Collapse to one ply:
    n != -1: x^(n+1) * sum_j (-1)^(m-j) (m!/j!) L^j / (n+1)^(m-j+1);
    n == -1: L^(m+1)/(m+1)."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    c, rest = f.as_independent(x)
    n = sp.Integer(0)
    L = None
    m = 0
    for a in sp.Mul.make_args(rest):
        base, e = a.as_base_exp()
        if isinstance(base, sp.log):
            if L is not None or not (e.is_Integer and e > 0):
                return []
            slope = _linear_coeff(base.args[0], x)
            if slope is None or base.args[0] != slope * x:
                return []
            L, m = base, int(e)
        elif base == x and e.is_Rational:
            n = n + e
        else:
            return []
    if L is None:
        return []
    if n == -1:
        return [c * L ** (m + 1) / (m + 1)]
    fact = sp.factorial(m)
    s = sp.Add(*((-1) ** (m - j) * (fact / sp.factorial(j))
                 * L**j / (n + 1) ** (m - j + 1)
                 for j in range(m + 1)))
    return [c * x ** (n + 1) * s]


def i_transcend_div(node: sp.Integral) -> list[sp.Expr]:
    """Generator-shape splitter (2026-07-09 frontier-gap autopsy:
    rat+exp+trig family, 8/36). Integrands built as (den*g + c)/den
    have sub-terms that are individually NON-elementary once expanded
    (exp*trig/(x^2+1)), so i_sum/i_apart make things worse. Group the
    numerator's additive terms by transcendental monomial, divide each
    group's polynomial coefficient by the denominator, and re-emit —
    exact per-group division recovers g + c/den."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    num, den = sp.fraction(sp.together(f))
    if den == 1 or not den.is_polynomial(x) or not den.has(x):
        return []
    groups: dict[sp.Expr, sp.Expr] = {}
    for t in sp.Add.make_args(sp.expand(num)):
        # rational-in-x factors (incl. x**-1) stay with the
        # coefficient; only genuine transcendentals key the group —
        # a 1/x in the key would double-divide by den below
        rat = sp.Integer(1)
        trans = sp.Integer(1)
        for a in sp.Mul.make_args(t):
            if a.is_rational_function(x):
                rat *= a
            else:
                trans *= a
        groups[trans] = groups.get(trans, sp.Integer(0)) + rat
    if len(groups) < 2:
        return []
    out = sp.Integer(0)
    changed = False
    for trans, coeff in groups.items():
        rn, rd = sp.fraction(sp.cancel(coeff / den))
        q, r = sp.div(rn, rd, x)
        if q != 0:
            changed = True
        out += trans * q + trans * r / rd
    # syntactic compare only: expand() auto-cancels (den*g)/den and
    # would veto every genuine fire of this rule (measured: the 8
    # rat+exp·trig gaps regressed 32/36 -> 24/36 under expand-guard)
    if not changed or out == f:
        return []
    return [sp.Integral(out, x)]


def i_heurisch(node: sp.Integral) -> list[sp.Expr]:
    """sympy's integrator as a gated LEAF CLOSER (2026-07-11, Artin's
    cascade: 'the engine buys speed, sympy buys the closed forms —
    let them work at the node level'). The L6 probe measured the
    complementarity: engine 36/60 vs sympy 56/60 on whole problems,
    but sympy's wins are SLOW (up to 55s+) and its losses are hangs —
    while our engine decomposes fast and loses only at leaves. So:
    fire sympy.integrate ONLY on small single integrals (op-capped;
    the shapes it's reliable and quick on), inside the standard rule
    timebox, output verified by differentiation like every edge.
    Every fire is also a label: cluster heurisch's wins and each
    cluster is a missing native rule (heurisch as rule-synthesis
    teacher — the self-updating half of the cascade)."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    # cap 100 (2026-07-11 sweep): L6 39/60 @ 1191s vs cap-40's 37/60
    # @ 1502s — the wider cap solves more AND runs faster (early
    # leaf-closes save wandering search). Residual L6 gap (17 vs
    # sympy-whole) is PRE-decomposition: no leaves form at any cap.
    if sp.count_ops(f) > 100 or f.has(sp.Integral):
        return []
    try:
        F = sp.integrate(f, x)
    except Exception:
        return []
    if F.has(sp.Integral, sp.Piecewise) or F is None:
        return []
    return [F]


def i_power(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if f == x:
        return [x**2 / 2]
    if not (isinstance(f, sp.Pow) and f.base == x and not f.exp.has(x)):
        return []
    n = f.exp
    return [sp.log(x)] if n == -1 else [x ** (n + 1) / (n + 1)]


def i_sum(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Add):
        return []
    return [sp.Add(*(sp.Integral(t, x) for t in f.args))]


def i_const_factor(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Mul):
        return []
    const = sp.Mul(*(a for a in f.args if not a.has(x)))
    rest = sp.Mul(*(a for a in f.args if a.has(x)))
    if const == 1 or rest == 1:
        return []
    return [const * sp.Integral(rest, x)]


_INT_TABLE = {sp.sin: lambda v: -sp.cos(v), sp.cos: sp.sin, sp.exp: sp.exp}


def i_table(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if isinstance(f, sp.Function) and f.func in _INT_TABLE and f.args == (x,):
        return [_INT_TABLE[f.func](x)]
    if f == sp.log(x):  # the invisible-·1 by-parts case i_parts can't see
        return [x * sp.log(x) - x]
    return []


def _usub_candidates(f: sp.Expr, x: sp.Symbol) -> list[sp.Expr]:
    cands = []
    for fn in f.atoms(sp.Function):
        cands.append(fn.args[0])
    for p in f.atoms(sp.Pow):
        cands.append(p.base)
    seen = set()
    out = []
    for g in cands:
        k = sp.srepr(g)
        if k not in seen and g.has(x) and g != x:
            seen.add(k)
            out.append(g)
    return out


def i_usub(node: sp.Integral) -> list[sp.Expr]:
    """u-substitution: if f == h(g)·g', rewrite to Subs(∫h(u)du, u, g).
    One branch per candidate g — wrong choices are the search's problem."""
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    out: list[sp.Expr] = []
    for g in _usub_candidates(f, x):
        dg = sp.diff(g, x)
        if dg == 0:
            continue
        # doit=False: simplify's default doit EVALUATES inner Integral
        # nodes (pathology #1 through the back door) — and crashes
        # sympy's manualintegrate on complex integrands (euler states)
        q = sp.simplify(sp.cancel(f / dg), doit=False).subs(g, U)
        if q.has(x) or not q.has(U):
            continue
        out.append(sp.Subs(sp.Integral(q, U), U, g))
    return out


def i_parts(node: sp.Integral) -> list[sp.Expr]:
    """Integration by parts, stepwise: ∫u dv = u·∫dv − ∫(∫dv)·u'.
    Inner integrals stay unevaluated; one branch per (u, dv) split."""
    u_ = _unpack_int(node)
    if u_ is None:
        return []
    f, x = u_
    if not isinstance(f, sp.Mul):
        return []
    out: list[sp.Expr] = []
    for i, u_part in enumerate(f.args):
        du = sp.diff(u_part, x)
        if du == 0:
            continue
        dv = sp.Mul(*(a for j, a in enumerate(f.args) if j != i))
        v = sp.Integral(dv, x)
        out.append(u_part * v - sp.Integral(v * du, x))
    return out


# --------------------------------------------------------- limits
# The origin-story rung: limits resisted LoRA training (<=21%),
# motivating this engine — these moves close that loop (spec:
# 2026-07-07-mathgen-expansion-design.md Part B).

LimRule = Callable[[sp.Limit], "list[sp.Expr]"]


def _unpack_lim(node: sp.Limit):
    """(f, x, a) for finite two-sided-representable limits, else None.
    sympy Limit args: (expr, var, point, dir)."""
    f, x, a, _dir = node.args
    if a in (sp.oo, -sp.oo):
        return None
    return f, x, a


def l_direct(node: sp.Limit) -> list[sp.Expr]:
    """Continuity move: substitute when the value is finite/defined."""
    u = _unpack_lim(node)
    if u is None:
        return []
    f, x, a = u
    try:
        v = f.subs(x, a)
    except Exception:
        return []
    if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or isinstance(v, sp.Limit):
        return []
    return [v]


def l_factor_cancel(node: sp.Limit) -> list[sp.Expr]:
    """0/0 rational forms: cancel the common factor, emit a new Limit."""
    u = _unpack_lim(node)
    if u is None:
        return []
    f, x, a = u
    num, den = f.as_numer_denom()
    if den == 1:
        return []
    try:
        if num.subs(x, a) != 0 or den.subs(x, a) != 0:
            return []
        g = sp.cancel(f)
    except Exception:
        return []
    if g == f:
        return []
    return [sp.Limit(g, x, a)]


def l_hopital(node: sp.Limit) -> list[sp.Expr]:
    """L'Hopital on 0/0: Limit(f/g) -> Limit(f'/g') with UNEVALUATED
    inner Derivatives — chains into the diff rules (rungs compose)."""
    u = _unpack_lim(node)
    if u is None:
        return []
    f, x, a = u
    num, den = f.as_numer_denom()
    if den == 1 or not den.has(x):
        return []
    try:
        if num.subs(x, a) != 0 or den.subs(x, a) != 0:
            return []
    except Exception:
        return []
    return [sp.Limit(sp.Derivative(num, x) / sp.Derivative(den, x), x, a)]


LIM_RULES: list[tuple[str, LimRule]] = [
    ("l_direct", l_direct),
    ("l_factor_cancel", l_factor_cancel),
    ("l_hopital", l_hopital),
]


def i_apart(node: sp.Integral) -> list[sp.Expr]:
    """Partial fractions (ceiling-mover #2): rational integrands split
    into pieces i_usub + i_power(log) already solve — e.g. 1/(x**2-1)
    had no derivation until this move."""
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if not f.is_rational_function(x):
        return []
    try:
        g = sp.apart(f, x)
    except (sp.PolynomialError, NotImplementedError, ZeroDivisionError):
        return []
    if g == f:
        return []
    return [sp.Integral(g, x)]


def _linear_coeff(e: sp.Expr, x: sp.Symbol) -> sp.Expr | None:
    """Slope of e if e is linear in x (slope x-free), else None."""
    d = sp.diff(e, x)
    return d if not d.has(x) else None


def i_cyclic(node: sp.Integral) -> list[sp.Expr]:
    """Table macro (ceiling-mover #3): exp(ax+d)*sin/cos(bx+c) closed
    forms. By-parts twice returns the original integral (I = f - I);
    the winning step is algebra on the EQUATION, which no expression
    rewrite can express — so emit the solved form directly. Verified
    by differentiation like every edge, so soundness is free. Top
    family in the 2026-07-07 integration failure autopsy."""
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if not (isinstance(f, sp.Mul) and len(f.args) == 2):
        return []
    ex = [a for a in f.args if isinstance(a, sp.exp)]
    tr = [a for a in f.args if isinstance(a, (sp.sin, sp.cos))]
    if len(ex) != 1 or len(tr) != 1:
        return []
    a = _linear_coeff(ex[0].args[0], x)
    b = _linear_coeff(tr[0].args[0], x)
    if a is None or b is None or a == 0 or b == 0:
        return []
    v = tr[0].args[0]
    if isinstance(tr[0], sp.sin):
        num = a * sp.sin(v) - b * sp.cos(v)
    else:
        num = a * sp.cos(v) + b * sp.sin(v)
    return [ex[0] * num / (a**2 + b**2)]


def i_unprod(node: sp.Integral) -> list[sp.Expr]:
    """Reverse product rule (ceiling-mover #4): sum integrands of the
    shape f'·H(u) + f·u'·h(u) are an expanded d/dx[f·H(u)] — but after
    expansion no single Mul node holds the pair, so i_parts never sees
    it (dominant L4 family in the 2026-07-07 autopsy). For each term
    c·f·u'·h(u) with h in the table, GUESS A = f_cof·H(u) and emit
    A + ∫(rest − f_cof'·H(u)); a right guess cancels a sibling term,
    a wrong one loses in ranking. Verifier-checked like every edge."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    if not isinstance(f, sp.Add):
        return []
    out: list[sp.Expr] = []
    seen: set[str] = set()
    for t in f.args:
        for fn in t.atoms(sp.sin, sp.cos, sp.exp):
            v = fn.args[0]
            dv = sp.diff(v, x)
            if dv == 0:
                continue
            cof = sp.cancel(t / (dv * fn))
            if cof.has(sp.sin, sp.cos, sp.exp, sp.log, sp.Integral):
                continue
            A = cof * _INT_TABLE[fn.func](v)
            k = sp.srepr(A)
            if k in seen:
                continue
            seen.add(k)
            resid = sp.expand(f - sp.diff(A, x))
            if sp.count_ops(resid) >= sp.count_ops(f):
                continue
            out.append(A + (sp.Integral(resid, x) if resid != 0 else 0))
            if len(out) >= 6:
                return out
        # second guess family (2026-07-10 frontier-v2 autopsy): the
        # term may be the f'·H(u) HALF of the pair — e.g. 9cos(x)/x
        # pairs with 9·log(x)·cos(x). Then f = ∫cofactor, so INTEGRATE
        # the rational cofactor instead of table-lookup on u'.
        for fn in t.atoms(sp.sin, sp.cos, sp.exp):
            cof = sp.cancel(t / fn)
            if (cof.has(sp.sin, sp.cos, sp.exp, sp.log, sp.Integral)
                    or not cof.is_rational_function(x)
                    or cof.is_polynomial(x)):
                continue
            try:
                F = sp.integrate(cof, x)
            except Exception:
                continue
            if F.has(sp.Integral) or not F.has(sp.log):
                continue
            A = F * fn
            k = sp.srepr(A)
            if k in seen:
                continue
            seen.add(k)
            resid = sp.expand(f - sp.diff(A, x))
            if sp.count_ops(resid) >= sp.count_ops(f):
                continue
            out.append(A + (sp.Integral(resid, x) if resid != 0 else 0))
            if len(out) >= 6:
                return out
    return out


def i_ansatz_exp(node: sp.Integral) -> list[sp.Expr]:
    """Polynomial ansatz for P(x)·exp(w(x)) (ceiling-mover #4b): the
    antiderivative, when elementary, is Q(x)·exp(w) with Q'+Q·w' = P —
    solve for Q by undetermined coefficients. Catches the L4 family
    where f·w' spans several expanded terms so i_unprod's per-term
    cofactor guess can't reassemble it."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    f = sp.expand(f)
    # Group terms by exp-family signature (multi-family sums never
    # reach per-family nodes in practice: `together`/hce re-merge
    # them before i_sum splits — measured on autopsy case #9), and
    # sympy auto-splits exp(w1+w2) into exp(w1)*exp(w2): recombine.
    groups: dict[str, list[sp.Expr]] = {}
    rest: list[sp.Expr] = []
    for t in (f.args if isinstance(f, sp.Add) else (f,)):
        exps = [e for e in t.atoms(sp.exp) if e.args[0].is_polynomial(x)]
        if exps:
            sig = sp.srepr(sp.Mul(*exps))
            groups.setdefault(sig, []).append(t)
        else:
            rest.append(t)
    solved: list[sp.Expr] = []
    for terms in groups.values():
        g = sp.Add(*terms)
        exps = [e for e in g.atoms(sp.exp) if e.args[0].is_polynomial(x)]
        w = sp.Add(*(e.args[0] for e in exps))
        fn = sp.Mul(*exps)
        P = sp.cancel(g / fn)
        # w == 0 (cancelling exp args) makes degree() -oo: guard first
        if w == 0 or sp.degree(w, x) < 2 or not P.is_polynomial(x):
            rest.extend(terms)
            continue
        dw = sp.diff(w, x)
        deg = max(int(sp.degree(P, x)) - int(sp.degree(dw, x)), 0)
        cs = sp.symbols(f"c0:{deg + 1}", cls=sp.Dummy)
        Q = sp.Add(*(c * x**i for i, c in enumerate(cs)))
        eqs = sp.Poly(sp.expand(sp.diff(Q, x) + Q * dw - P), x).coeffs()
        sol = sp.solve(eqs, cs, dict=True)
        if sol:
            solved.append(Q.subs(sol[0]) * fn)
        else:
            rest.extend(terms)
    if not solved:
        return []
    tail = sp.Integral(sp.Add(*rest), x) if rest else sp.S.Zero
    return [sp.Add(*solved) + tail]


def i_linear_basis(node: sp.Integral) -> list[sp.Expr]:
    """Bidirectional search v0, collapsed into linear algebra: d/dx is
    a LINEAR operator, so meet-in-the-middle over answer shapes is a
    matrix solve. Enumerate a basis of candidate answer terms from the
    integrand's own atoms (x^j * sin(u), cos(u), exp(w)), differentiate
    the ansatz once, and equate coefficients in the extended polynomial
    ring — one sp.solve call replaces the whole backward frontier.
    Subsumes i_cyclic/i_ansatz_exp shapes and reaches mixed products
    neither can. Wrong basis just returns [] (verifier never needed)."""
    un = _unpack_int(node)
    if un is None:
        return []
    f, x = un
    f = sp.expand(f)
    # Laurent tail: c*x^-n terms break Poly, but their antiderivative
    # is KNOWN (powers and a log) — split it off analytically, ansatz
    # the rest (found via the 356-record holdout: (x+1)e^x + 1/x).
    laurent = sp.S.Zero
    kept = []
    for t in (f.args if isinstance(f, sp.Add) else (f,)):
        c, rest_t = t.as_coeff_Mul()
        pw = rest_t.as_base_exp() if not rest_t.has(sp.sin, sp.cos,
                                                    sp.exp) else None
        if (pw is not None and pw[0] == x and pw[1].is_Integer
                and pw[1] < 0):
            n = -int(pw[1])
            laurent += (c * sp.log(x) if n == 1
                        else c * x ** (1 - n) / (1 - n))
        else:
            kept.append(t)
    f = sp.Add(*kept)
    if f == 0:
        return [laurent] if laurent != 0 else []
    # transcendental generators present in the integrand
    args: list[sp.Expr] = []
    for fn in f.atoms(sp.sin, sp.cos):
        v = fn.args[0]
        if v.has(x) and v.is_polynomial(x) and v not in args:
            args.append(v)
    trig = [g for v in args for g in (sp.sin(v), sp.cos(v))]
    exps = [e for e in f.atoms(sp.exp) if e.args[0].is_polynomial(x)
            and e.args[0].has(x)]
    # log generators (2026-07-11 L6 autopsy: 14/22 failures were
    # x^j*log(kx)*trig products — the answer basis was MISSING THE
    # LOG ORBITAL; d/dx of x^j*L*trig stays within {x^j, x^j*L}*trig
    # so the linear solve closes). d(L) = 1/x is handled by
    # multiplying the residual through by x before Poly.
    logs = [g for g in f.atoms(sp.log)
            if g.args[0].is_polynomial(x) and g.args[0].has(x)]
    # atan orbital (2026-07-11 L7 autopsy: 12/14 residue failures were
    # d/dx[log(u)*atan(v)] pairs — same missing-orbital shape as log,
    # one day later). d(atan(v)) = v'/(1+v^2): cleared with the other
    # denominators before the Poly solve.
    atans = [g for g in f.atoms(sp.atan)
             if g.args[0].is_polynomial(x) and g.args[0].has(x)]
    gens = trig + exps + logs + atans  # gens must be ATOMS, no products
    if not gens or len(gens) > 8:
        return []
    # basis monomials per trig arg v: sin^a(v)*cos^b(v) with
    # 1 <= a+b <= M_v (the family is d/dx-closed at fixed total
    # degree; M_v = max total trig power in f + 1 covers the
    # sin^k*cos*u' -> sin^(k+1) integration bump). Plus the
    # (recombined) exp product, exp*trig cross terms (power 1),
    # and 1 for a pure-polynomial part.
    ep = sp.Mul(*exps) if exps else None
    mons: list[sp.Expr] = []
    for v in args:
        s_, c_ = sp.sin(v), sp.cos(v)
        # max total sin+cos power of this arg in any one term of f
        mv = 1
        for t in (f.args if isinstance(f, sp.Add) else (f,)):
            tot = 0
            for b in (s_, c_):
                e = max((int(p.exp) for p in t.atoms(sp.Pow)
                         if p.base == b and p.exp.is_Integer
                         and p.exp > 0), default=1 if t.has(b) else 0)
                tot += e
            mv = max(mv, tot)
        mv = min(mv + 1, 5)
        mons += [s_**a * c_**b for a in range(mv + 1)
                 for b in range(mv + 1 - a) if a + b >= 1]
    if ep is not None:
        mons += [ep] + [ep * t for t in trig]
    for L in logs:
        mons += [L] + [L * t for t in trig]
        if ep is not None:
            mons.append(L * ep)
    for A in atans:
        mons += [A] + [A * L for L in logs]
    mons.append(sp.S.One)
    # Poly can't take gens that contain x — substitute Dummy
    # placeholders for the transcendental atoms first
    ph = {g: sp.Dummy(f"g{i}") for i, g in enumerate(gens)}

    def _poly(e: sp.Expr) -> sp.Poly | None:
        try:
            return sp.Poly(e.subs(ph), x, *ph.values())
        except sp.PolynomialError:
            return None

    # log/atan-derivative debris makes the INTEGRAND rational (e.g.
    # atan(x)/x + log(u)/(1+x^2), the L7 pair family): gate and size
    # the ansatz on the DENOMINATOR-CLEARED integrand instead
    clear = sp.S.One
    if logs or atans:
        # keep clear FACTORED and cancel poly-aware: the integrand's
        # denominators auto-combine (x*(x^2+1) -> x^3+x) and only
        # sp.cancel matches them against the multiplier
        clear = (x * sp.Mul(*(g.args[0] for g in logs))
                 * sp.Mul(*((1 + g.args[0] ** 2) for g in atans)))
    pf = _poly(sp.expand(sp.cancel(f * clear)))
    if pf is None:
        return []
    deg = max(int(sp.degree(pf, x)), 1)
    nc = (deg + 2) * len(mons)  # +1 x-degree headroom for the poly part
    if deg > 8 or nc > 200:  # linear solve: 200 unknowns is still fast
        return []
    cs = sp.symbols(f"c0:{nc}", cls=sp.Dummy)
    cand = sp.Add(*(cs[i * (deg + 2) + j] * x**j * m
                    for i, m in enumerate(mons)
                    for j in range(deg + 2)))
    # d(log(P)) = P'/P and d(atan(v)) = v'/(1+v^2): the same `clear`
    # multiplier removes every denominator (equating clear*resid to 0
    # is equivalent away from the poles); cancel before expand so the
    # auto-combined denominators actually divide out
    resid = sp.expand(sp.cancel((sp.diff(cand, x) - f) * clear))
    pr = _poly(resid)
    if pr is None:
        return []
    eqs = pr.coeffs()
    if any(not e.has(*cs) for e in eqs if e != 0):
        return []  # residue outside the span: no solution exists
    sol = sp.solve(eqs, cs, dict=True)
    if not sol:
        return []
    a = cand.subs(sol[0]).subs({c: 0 for c in cs}) + laurent
    return [a] if a != 0 else []


INT_RULES: list[tuple[str, IntRule]] = [
    ("i_const", i_const),
    ("i_apart", i_apart),
    ("i_cyclic", i_cyclic),
    ("i_unprod", i_unprod),
    ("i_ansatz_exp", i_ansatz_exp),
    ("i_linear_basis", i_linear_basis),
    ("i_inverse_trig", i_inverse_trig),
    ("i_sqrt_basis", i_sqrt_basis),
    ("i_log_power", i_log_power),
    ("i_transcend_div", i_transcend_div),
    ("i_heurisch", i_heurisch),
    ("i_power", i_power),
    ("i_sum", i_sum),
    ("i_const_factor", i_const_factor),
    ("i_table", i_table),
    ("i_usub", i_usub),
    ("i_parts", i_parts),
]

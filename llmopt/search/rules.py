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
        q = sp.simplify(sp.cancel(f / dg)).subs(g, U)
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
    # sympy auto-splits exp(w1+w2) into exp(w1)*exp(w2): recombine
    exps = [e for e in f.atoms(sp.exp) if e.args[0].is_polynomial(x)]
    if not exps:
        return []
    w = sp.Add(*(e.args[0] for e in exps))
    if int(sp.degree(w, x)) < 2:
        return []
    fn = sp.Mul(*exps)
    P = sp.cancel(f / fn)
    if not P.is_polynomial(x):
        return []
    dw = sp.diff(w, x)
    deg = max(int(sp.degree(P, x)) - int(sp.degree(dw, x)), 0)
    cs = sp.symbols(f"c0:{deg + 1}", cls=sp.Dummy)
    Q = sp.Add(*(c * x**i for i, c in enumerate(cs)))
    eqs = sp.Poly(sp.expand(sp.diff(Q, x) + Q * dw - P), x).coeffs()
    sol = sp.solve(eqs, cs, dict=True)
    return [Q.subs(sol[0]) * fn] if sol else []


INT_RULES: list[tuple[str, IntRule]] = [
    ("i_const", i_const),
    ("i_apart", i_apart),
    ("i_cyclic", i_cyclic),
    ("i_unprod", i_unprod),
    ("i_ansatz_exp", i_ansatz_exp),
    ("i_power", i_power),
    ("i_sum", i_sum),
    ("i_const_factor", i_const_factor),
    ("i_table", i_table),
    ("i_usub", i_usub),
    ("i_parts", i_parts),
]

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

MACRO_RULES: list[tuple[str, DiffRule]] = [
    ("d_quotient", d_quotient),
]

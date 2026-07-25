"""Prefix (Polish) serialization for closed-system expressions —
native-transformer rung 1 (spec 2026-07-25-native-transformer).

Same atoms as MathTokenizer, no paren bookkeeping inside
expressions: `7*x` -> `*7x`. n-ary Add/Mul are left-folded to
binary so the stream parses without arity counts; function heads
reuse the existing "sin(" -class atoms (the '(' is part of the
atom, never a standalone token). Separators are MINIMAL: a space
is paid only where two number tokens would otherwise merge —
every other boundary is recovered by the lexer, so the sequence
length actually drops vs infix (the rung-1 target). Round-trip
law: from_prefix(to_prefix(e)) == e structurally for every corpus
expression (tested against gen-4).
"""
from __future__ import annotations

import re

import sympy as sp

_FUNC_HEADS = {sp.sin: "sin(", sp.cos: "cos(", sp.tan: "tan(",
               sp.exp: "exp(", sp.log: "log(", sp.atan: "atan(",
               sp.asin: "asin(", sp.sqrt: "sqrt("}
_HEAD_FUNCS = {v: k for k, v in _FUNC_HEADS.items()}
_BINARY = {"+", "*", "**", "/"}


def to_prefix(e: sp.Expr) -> str:
    toks = _walk(e)
    out = [toks[0]]
    for prev, cur in zip(toks, toks[1:]):
        # ambiguous flattened boundaries: number|number and *|*
        # (adjacent Mul folds would re-lex as one ** token)
        if (prev[-1] in "0123456789." and (cur[0] in "0123456789."
                                           or cur[0] == "-")) \
                or (prev[-1] == "*" and cur[0] == "*"):
            out.append(" ")
        out.append(cur)
    return "".join(out)


def _fold(op: str, args) -> list[str]:
    out: list[str] = []
    for _ in range(len(args) - 1):
        out.append(op)
    for a in args:
        out.extend(_walk(a))
    return out


def _walk(e: sp.Expr) -> list[str]:
    if isinstance(e, sp.Integral):
        toks = ["Integral("]
        toks.extend(_walk(e.function))
        toks.extend(str(v) for v in e.variables)
        return toks
    if isinstance(e, sp.Rational) and not isinstance(e, sp.Integer):
        return ["/", str(e.p), str(e.q)]
    if e is sp.pi:
        return ["pi"]
    if e is sp.E:
        return ["E"]
    if isinstance(e, (sp.Integer, sp.Float, sp.Symbol)):
        return [str(e)]
    if isinstance(e, sp.Pow):
        if e.exp == sp.Rational(1, 2):
            return ["sqrt(", *_walk(e.base)]
        return ["**", *_walk(e.base), *_walk(e.exp)]
    if isinstance(e, sp.Add):
        return _fold("+", e.as_ordered_terms())
    if isinstance(e, sp.Mul):
        return _fold("*", e.as_ordered_factors())
    if type(e) in _FUNC_HEADS:
        return [_FUNC_HEADS[type(e)], *_walk(e.args[0])]
    raise ValueError(f"prefix: unsupported node {type(e).__name__}: {e}")


_LEX = re.compile(
    "|".join([r"Integral\(",
              "|".join(re.escape(h) for h in
                       sorted(_HEAD_FUNCS, key=len, reverse=True)),
              r"\*\*", r"[+*/]", r"-?\d+(?:\.\d+)?", r"pi|E|x|t"]))


def _lex(s: str) -> list[str]:
    toks, i = [], 0
    for m in _LEX.finditer(s):
        if s[i:m.start()].strip():
            raise ValueError(f"prefix: unlexable {s[i:m.start()]!r} in {s!r}")
        toks.append(m.group())
        i = m.end()
    if s[i:].strip():
        raise ValueError(f"prefix: unlexable tail {s[i:]!r} in {s!r}")
    return toks


def from_prefix(s: str) -> sp.Expr:
    e, rest = _parse(_lex(s))
    if rest:
        raise ValueError(f"prefix: {len(rest)} trailing tokens: {rest[:6]}")
    return e


_SYMS = {"x": sp.Symbol("x"), "t": sp.Symbol("t"),
         "pi": sp.pi, "E": sp.E}


def _parse(toks: list[str]) -> tuple[sp.Expr, list[str]]:
    if not toks:
        raise ValueError("prefix: empty stream")
    t, rest = toks[0], toks[1:]
    if t in _BINARY:
        a, rest = _parse(rest)
        b, rest = _parse(rest)
        if t == "+":
            return sp.Add(a, b), rest
        if t == "*":
            return sp.Mul(a, b), rest
        if t == "**":
            return sp.Pow(a, b), rest
        if a.is_Integer and b.is_Integer:
            return sp.Rational(a, b), rest
        return a / b, rest
    if t == "Integral(":
        f, rest = _parse(rest)
        if not rest or rest[0] not in _SYMS:
            raise ValueError("prefix: Integral( needs a variable token")
        return sp.Integral(f, _SYMS[rest[0]]), rest[1:]
    if t in _HEAD_FUNCS:
        a, rest = _parse(rest)
        return _HEAD_FUNCS[t](a), rest
    if t in _SYMS:
        return _SYMS[t], rest
    try:
        return (sp.Float(t) if "." in t else sp.Integer(t)), rest
    except ValueError:
        raise ValueError(f"prefix: unknown token {t!r}") from None

"""Structural features for the NNUE eval (spec:
2026-07-07-nnue-eval-design.md). Cheap, deterministic, pure — the
NNUE lesson is cheap features + tiny net. State.plies is deliberately
absent: probes restart fresh, so history cannot affect solvability
and would leak the training label."""

from __future__ import annotations

import math

import sympy as sp

UNSOLVED = (sp.Derivative, sp.Integral, sp.Limit)

N_FEATURES = 20


def _depth(e: sp.Basic) -> int:
    if not e.args:
        return 1
    return 1 + max(_depth(a) for a in e.args)


def featurize(expr: sp.Expr) -> list[float]:
    ops = float(sp.count_ops(expr))
    unsolved = list(expr.atoms(*UNSOLVED))
    unsolved_ops = [float(sp.count_ops(u)) for u in unsolved]
    trig = sum(1 for _ in expr.atoms(sp.sin, sp.cos, sp.tan))
    integrals = list(expr.atoms(sp.Integral))
    max_limits = max((len(i.limits) for i in integrals), default=0)
    n_nodes = sum(1 for _ in sp.preorder_traversal(expr))
    sum_uops = sum(unsolved_ops)
    return [
        ops,
        float(len(integrals)),
        float(len(list(expr.atoms(sp.Derivative)))),
        float(len(list(expr.atoms(sp.Subs)))),
        float(len(list(expr.atoms(sp.Add)))),
        float(len(list(expr.atoms(sp.Mul)))),
        float(len(list(expr.atoms(sp.Pow)))),
        float(trig),
        float(len(list(expr.atoms(sp.exp)))),
        float(len(list(expr.atoms(sp.log)))),
        float(len(unsolved)),
        max(unsolved_ops, default=0.0),
        float(sum_uops),
        float(max_limits),
        float(_depth(expr)),
        float(len(expr.free_symbols)),
        float(n_nodes),
        0.0 if unsolved else 1.0,
        max(ops - sum_uops, 0.0),
        math.log2(1.0 + ops),
    ]

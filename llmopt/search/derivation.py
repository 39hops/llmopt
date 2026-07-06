"""Derivation search: Stockfish-for-math foundations (roadmap #1).

Chess mapping: a State is a position (expression + goal kind); a Move
is a sympy rewrite applied to the whole expression; the move layer only
ever emits *legal* moves — sympy performs the rewrite, so unlike a
model-proposed edit, a generated successor cannot be mathematically
wrong, only unhelpful. The HCE (hand-crafted evaluation, chess sense)
scores states; beam search minimizes it.

This module is deliberately model-free: enumerated moves + HCE + beam
is the rung-0 engine. The fine-tuned model slots in later as a move
*proposer* (ordering/pruning the enumeration), exactly like a policy
net in front of a classical searcher.

HCE v0 (lower = better):
- tree size (sympy count_ops) — solving usually shrinks things
- unsolved-operator count: nodes a finished answer cannot contain
  (Derivative, Integral, Limit) weigh heavily
- small step penalty per ply to prefer short derivations
Calibration harness (does HCE correlate with solve rate?) lives in the
bench script, per the roadmap.

Rung 1 (2026-07-06): doit_one is gone from the move set. Moves are
(rule, Derivative-node) pairs from search.rules — the search composes
the derivation and sympy is demoted to per-edge verifier (verify_edge).
Algebra cleanup moves are kept trimmed (no simplify: an algebra
mega-move that collapses plies). doit remains available to the
verifier and to tests as ground truth. Differentiation only; Integral/
Limit states are honestly unsolvable until rung 2. Spec:
docs/superpowers/specs/2026-07-06-hce-rung1-primitive-moves-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import sympy as sp

from llmopt.search.rules import CORE_RULES, MACRO_RULES

UNSOLVED = (sp.Derivative, sp.Integral, sp.Limit)


@dataclass(frozen=True)
class State:
    expr: sp.Expr
    plies: int = 0
    history: tuple[str, ...] = field(default_factory=tuple)

    def key(self) -> str:
        return sp.srepr(self.expr)


def is_solved(state: State) -> bool:
    return not state.expr.has(*UNSOLVED)


def hce(state: State) -> float:
    """Hand-crafted evaluation, v0. Lower is better."""
    unsolved = sum(1 for _ in state.expr.atoms(*UNSOLVED))
    return 100.0 * unsolved + float(sp.count_ops(state.expr)) + 0.1 * state.plies


# ------------------------------------------------------------- moves

ALGEBRA_MOVES: list[tuple[str, Callable[[sp.Expr], sp.Expr]]] = [
    ("expand", sp.expand),
    ("factor", sp.factor),
    ("cancel", sp.cancel),
    ("together", sp.together),
    ("trigsimp", sp.trigsimp),
    ("powsimp", sp.powsimp),
]


def verify_edge(parent: sp.Expr, child: sp.Expr) -> bool:
    """Oracle check: a legal move preserves the value. doit() is the
    complete solver — too strong as a mover (rung 0's mistake), exactly
    right as a verifier."""
    try:
        return sp.simplify(parent.doit() - child.doit()) == 0
    except Exception:
        return False


def successors(
    state: State, *, use_macros: bool = False
) -> Iterator[tuple[str, State]]:
    """Legal, non-identity, sympy-verified successor states. Rule moves
    target one Derivative node ((rule, node) pairs — real branching);
    algebra moves rewrite the whole expression."""
    seen = {state.key()}
    rules = CORE_RULES + MACRO_RULES if use_macros else CORE_RULES

    def emit(name: str, new_expr: sp.Expr) -> Iterator[tuple[str, State]]:
        child = State(new_expr, state.plies + 1, state.history + (name,))
        if child.key() not in seen and verify_edge(state.expr, new_expr):
            seen.add(child.key())
            yield name, child

    for node in sorted(state.expr.atoms(sp.Derivative), key=sp.count_ops):
        for rule_name, rule in rules:
            for rewrite in rule(node):
                label = f"{rule_name}@{sp.sstr(node)}"
                yield from emit(label, state.expr.xreplace({node: rewrite}))
    for name, fn in ALGEBRA_MOVES:
        try:
            new = fn(state.expr)
        except Exception:
            continue
        yield from emit(name, new)


# ------------------------------------------------------------- search

@dataclass
class SearchResult:
    solved: bool
    state: State
    nodes: int


def beam_search(
    expr: sp.Expr,
    *,
    width: int = 8,
    max_plies: int = 12,
    max_nodes: int | None = None,
    use_macros: bool = False,
    trace: list[State] | None = None,
) -> SearchResult:
    """Minimize hce over the rewrite tree. Returns the best solved
    state found, else the best-evaluated state at exhaustion."""
    root = State(expr)
    if is_solved(root):
        return SearchResult(True, root, 1)
    beam = [root]
    best_solved: State | None = None
    visited = {root.key()}
    nodes = 1
    for _ in range(max_plies):
        candidates: list[State] = []
        for s in beam:
            for _, child in successors(s, use_macros=use_macros):
                if child.key() in visited:
                    continue
                visited.add(child.key())
                if trace is not None:
                    trace.append(child)
                nodes += 1
                if max_nodes is not None and nodes >= max_nodes:
                    candidates.append(child)
                    break
                if is_solved(child) and (
                    best_solved is None or hce(child) < hce(best_solved)
                ):
                    best_solved = child
                candidates.append(child)
            if max_nodes is not None and nodes >= max_nodes:
                break
        if max_nodes is not None and nodes >= max_nodes:
            break
        if not candidates:
            break
        candidates.sort(key=hce)
        beam = candidates[:width]
        # a solved state that also tops the beam won't improve: stop
        if best_solved is not None and hce(best_solved) <= hce(beam[0]):
            break
    if best_solved is not None:
        return SearchResult(True, best_solved, nodes)
    return SearchResult(False, min(beam, key=hce), nodes)

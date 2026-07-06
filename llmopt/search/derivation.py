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

KNOWN LIMITATION (deliberate, rung 0): doit_one delegates to sympy's
own solver, which is complete for this domain — so beam_search solves
everything in ~1 ply and the search is degenerate. The chassis (State,
legality-by-construction, HCE, beam) is the deliverable here. Rung 1
must replace doit_one with PRIMITIVE rules (power rule, product-rule
split, u-substitution as individual rewrites) so the search does the
reasoning and sympy only verifies steps; that is also exactly where
the model plugs in as a move proposer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import sympy as sp

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

Move = tuple[str, Callable[[sp.Expr], sp.Expr]]

MOVES: list[Move] = [
    ("doit_one", lambda e: _doit_one(e)),
    ("expand", sp.expand),
    ("factor", sp.factor),
    ("cancel", sp.cancel),
    ("together", sp.together),
    ("apart", lambda e: _apart_safe(e)),
    ("trigsimp", sp.trigsimp),
    ("powsimp", sp.powsimp),
    ("simplify", sp.simplify),
]


def _doit_one(e: sp.Expr) -> sp.Expr:
    """Evaluate ONE unsolved node (innermost first) — the single-step
    version of .doit(), so a derivation is a sequence of visible steps
    instead of one opaque jump."""
    nodes = sorted(e.atoms(*UNSOLVED), key=sp.count_ops)
    if not nodes:
        return e
    target = nodes[0]
    return e.xreplace({target: target.doit(deep=False)})


def _apart_safe(e: sp.Expr) -> sp.Expr:
    frees = e.free_symbols
    if len(frees) != 1:
        return e
    try:
        return sp.apart(e, *frees)
    except (sp.PolynomialError, NotImplementedError):
        return e


def successors(state: State) -> Iterator[tuple[str, State]]:
    """Legal, non-identity successor states. Every rewrite is performed
    by sympy, so successors are equal to the parent by construction —
    the oracle is embedded in move generation."""
    seen = {state.key()}
    for name, fn in MOVES:
        try:
            new = fn(state.expr)
        except Exception:
            continue
        child = State(new, state.plies + 1, state.history + (name,))
        if child.key() not in seen:
            seen.add(child.key())
            yield name, child


# ------------------------------------------------------------- search

@dataclass
class SearchResult:
    solved: bool
    state: State
    nodes: int


def beam_search(expr: sp.Expr, *, width: int = 8, max_plies: int = 12) -> SearchResult:
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
            for _, child in successors(s):
                if child.key() in visited:
                    continue
                visited.add(child.key())
                nodes += 1
                if is_solved(child) and (
                    best_solved is None or hce(child) < hce(best_solved)
                ):
                    best_solved = child
                candidates.append(child)
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

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
verifier and to tests as ground truth. Rung 2 adds integration rules
(u-sub via Subs carriers, stepwise by-parts); Limit states remain
unsolvable. Integral edges verify modulo additive constants. Specs:
docs/superpowers/specs/2026-07-06-hce-rung1-primitive-moves-design.md
docs/superpowers/specs/2026-07-07-rung2-integration-moves-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import sympy as sp

from llmopt.search.rules import CORE_RULES, INT_RULES, LIM_RULES, MACRO_RULES

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
    ("subs_eval", lambda e: _subs_eval(e)),
    ("euler", lambda e: _euler_rewrite(e)),
]


def _euler_rewrite(e: sp.Expr) -> sp.Expr:
    """The ceiling-mover (Artin's complex-numbers thread): rewrite trig
    to complex exponentials so integrals with no real-form derivation
    (e.g. sin^2) become exponential chains the EXISTING rules solve.
    Only fires when trig sits under an unsolved Integral (elsewhere it
    just bloats states for the HCE to bury)."""
    if not any(i.has(sp.sin, sp.cos, sp.tan) for i in e.atoms(sp.Integral)):
        return e
    return sp.expand(e.rewrite(sp.exp))


def _subs_eval(e: sp.Expr) -> sp.Expr:
    """Back-substitute solved Subs carriers (from i_usub) — a visible ply."""
    repl = {s: s.doit() for s in e.atoms(sp.Subs) if not s.expr.has(*UNSOLVED)}
    return e.xreplace(repl) if repl else e


def _is_zero(d: sp.Expr) -> bool:
    """Bounded zero-test for edge verification. simplify() can burn
    20+ minutes on 18-op exp/log/trig mixes (measured 2026-07-07), so:
    expand fast path -> deterministic multi-point numeric screen (30
    digits; identically-zero elementary expressions vanish at generic
    points with overwhelming probability) -> simplify only for the
    rare ambiguous leftovers. Terminal answers are still checked fully
    symbolically in tests/bench; this bounds *edge* cost only."""
    d = sp.expand(d)
    if d == 0:
        return True
    if d.has(sp.Integral, sp.Subs):
        # unsolved carriers survived structural cancellation (nested
        # i_parts residue). evalf here would run numerical QUADRATURE
        # (measured minutes per edge) and simplify is no better. Reject:
        # verification incompleteness only prunes a legal move — it can
        # never admit an illegal one, so soundness is preserved.
        return False
    frees = sorted(d.free_symbols, key=str)
    if frees:
        decided = True
        for k in range(3):
            subs = {
                v: sp.Float("0.7183") + sp.Rational(17 * (k + 1) + 5 * i, 100)
                for i, v in enumerate(frees)
            }
            try:
                val = complex(d.evalf(30, subs=subs))
            except Exception:
                decided = False
                break
            if abs(val) > 1e-15:
                return False
            if abs(val) > 1e-25:  # suspicious near-zero: escalate
                decided = False
                break
        if decided:
            return True
    return sp.simplify(d) == 0


def verify_edge(parent: sp.Expr, child: sp.Expr) -> bool:
    """Oracle check: a legal move preserves the value. Integral edges
    are verified modulo an additive constant (antiderivatives are an
    equivalence class); Derivative-only edges must match exactly.

    Integral edges are checked by DIFFERENTIATING the difference, not
    by doit(): d/dx of an unevaluated Integral is just its integrand,
    so the check never asks sympy to integrate — the first version did
    (one full integration per candidate edge) and was ~100x slower.
    Derivative of the difference vanishing for every free symbol is
    exactly equality-mod-constant."""
    try:
        if parent.has(sp.Integral):
            d = parent - child
            frees = parent.free_symbols | child.free_symbols
            return all(_is_zero(sp.diff(d, v).doit()) for v in frees)
        return _is_zero(parent.doit() - child.doit())
    except Exception:
        return False


def successors(
    state: State, *, use_macros: bool = False, verify_p: float = 1.0
) -> Iterator[tuple[str, State]]:
    """Legal, non-identity, sympy-verified successor states. Rule moves
    target one Derivative node ((rule, node) pairs — real branching);
    algebra moves rewrite the whole expression.

    verify_p < 1 samples which edges pay the oracle (deterministic in
    the child key, so runs reproduce). Soundness lives at the boundary:
    beam_search replays and FULLY verifies any winning path before
    reporting solved — sampling can only waste search, never emit a
    wrong answer (spec O2, 2026-07-07-engine-optimizations-design.md)."""
    seen = {state.key()}
    rules = CORE_RULES + MACRO_RULES if use_macros else CORE_RULES

    def emit(name: str, new_expr: sp.Expr) -> Iterator[tuple[str, State]]:
        child = State(new_expr, state.plies + 1, state.history + (name,))
        if child.key() in seen:
            return
        if verify_p >= 1.0 or (hash(child.key()) % 1000) < verify_p * 1000:
            if not verify_edge(state.expr, new_expr):
                return
        seen.add(child.key())
        yield name, child

    def _safe(rule, node) -> list:
        # a rule crashing deep in sympy (measured: i_usub's simplify
        # reached manualintegrate's non-real comparison on a euler
        # state) must cost one move, never the whole search
        try:
            return rule(node)
        except Exception:
            return []

    for node in sorted(state.expr.atoms(sp.Derivative), key=sp.count_ops):
        for rule_name, rule in rules:
            for rewrite in _safe(rule, node):
                label = f"{rule_name}@{sp.sstr(node)}"
                yield from emit(label, state.expr.xreplace({node: rewrite}))
    for node in sorted(state.expr.atoms(sp.Integral), key=sp.count_ops):
        # multi-limit integrals (sympy collapses ∫∫f at construction, e.g.
        # from a du=1 by-parts split): peel — apply rules to the innermost
        # limit, rewrap the rest. Rules themselves stay single-limit.
        nested = len(node.limits) > 1
        inner = sp.Integral(node.function, node.limits[0]) if nested else node
        for rule_name, rule in INT_RULES:
            for rewrite in _safe(rule, inner):
                new_node = (
                    sp.Integral(rewrite, *node.limits[1:]) if nested else rewrite
                )
                label = f"{rule_name}@{sp.sstr(inner)}"
                yield from emit(label, state.expr.xreplace({node: new_node}))
    for node in sorted(state.expr.atoms(sp.Limit), key=sp.count_ops):
        for rule_name, rule in LIM_RULES:
            for rewrite in _safe(rule, node):
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
    corrupted: bool = False  # a sampled-mode winning path failed full
    # verification and was discarded (honest-accounting counter)


def replay_verify(root: sp.Expr, history: tuple[str, ...]) -> bool:
    """Fully re-verify a winning path edge by edge (verify_p=1)."""
    cur = State(root)
    for chosen in history:
        for name, child in successors(cur, use_macros=True, verify_p=1.0):
            if name == chosen:
                cur = child
                break
        else:
            return False
    return True


def beam_search(
    expr: sp.Expr,
    *,
    width: int = 8,
    max_plies: int = 12,
    max_nodes: int | None = None,
    use_macros: bool = False,
    trace: list[State] | None = None,
    eval_fn: Callable[[State], float] = hce,
    proposer: Callable[
        [State, list[tuple[str, State]]], list[tuple[str, State]]
    ]
    | None = None,
    propose_k: int | Callable[..., int] | None = None,
    verify_p: float = 1.0,
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
            kids = list(successors(s, use_macros=use_macros,
                                   verify_p=verify_p))
            scores = None
            if proposer is not None:
                out = proposer(s, kids)
                if isinstance(out, tuple):
                    kids, scores = out
                else:
                    kids = out
            if propose_k is not None:
                k = (propose_k(s, kids, scores) if callable(propose_k)
                     else propose_k)
                kids = kids[:max(1, int(k))]
            for _, child in kids:
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
                    best_solved is None or eval_fn(child) < eval_fn(best_solved)
                ):
                    best_solved = child
                candidates.append(child)
            if max_nodes is not None and nodes >= max_nodes:
                break
        if max_nodes is not None and nodes >= max_nodes:
            break
        if not candidates:
            break
        candidates.sort(key=eval_fn)
        beam = candidates[:width]
        # a solved state that also tops the beam won't improve: stop
        if best_solved is not None and eval_fn(best_solved) <= eval_fn(beam[0]):
            break
    if best_solved is not None:
        if verify_p < 1.0 and not replay_verify(expr, best_solved.history):
            # sampled mode let a corrupted path through: discard, report
            return SearchResult(False, min(beam, key=eval_fn), nodes,
                                corrupted=True)
        return SearchResult(True, best_solved, nodes)
    return SearchResult(False, min(beam, key=eval_fn), nodes)

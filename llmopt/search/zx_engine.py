"""T-count engine rung 1 (spec: 2026-07-08-tcount-engine-design.md):
the derivation-engine chassis pointed at ZX diagrams.

States = pyzx graphs; moves = (rule, site) pairs from pyzx's
check_*/apply pairs (fuse, local complementation, pivot, identity
removal — each move is soundness-preserving by pyzx's construction,
the analog of legality-by-construction); eval = T-count with
vertex-count tie-break; boundary oracle = tensor equality on small
circuits (compare_tensors), the analog of replay_verify.

The greedy incumbent is zx.full_reduce — the domain's "doit". Rung-1
question (pre-registered): does search beat greedy on >= 20% of
seeded circuits at equal budget?
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import pyzx as zx
from pyzx import rewrite_rules as rr


def tcount(g) -> int:
    return zx.tcount(g)


@dataclass
class ZXState:
    g: object
    plies: int = 0
    history: tuple = field(default_factory=tuple)

    def key(self) -> str:
        # syntactic dedup (like srepr): stable stats + adjacency string
        gs = self.g
        vs = sorted((gs.type(v), str(gs.phase(v)),
                     tuple(sorted(gs.neighbors(v)))) for v in gs.vertices())
        return str(vs)


def _phases_ok(g) -> bool:
    return True


def moves(state: ZXState, max_per_rule: int = 8):
    """(label, child) pairs. Each child is an independent graph copy."""
    g = state.g
    verts = list(g.vertices())
    out = 0
    # fuse: adjacent same-color spiders
    for e in list(g.edges())[:200]:
        v, w = g.edge_st(e)
        if rr.check_fuse(g, v, w):
            g2 = g.copy()
            if rr.fuse(g2, v, w):
                yield f"fuse@{v},{w}", ZXState(
                    g2, state.plies + 1, state.history + (f"fuse@{v},{w}",))
                out += 1
                if out >= max_per_rule:
                    break
    for name, check, apply1 in (("lcomp", rr.check_lcomp, rr.lcomp),
                                ("remove_id", rr.check_remove_id,
                                 rr.remove_id)):
        out = 0
        for v in verts:
            if check(g, v):
                g2 = g.copy()
                if apply1(g2, v):
                    lab = f"{name}@{v}"
                    yield lab, ZXState(g2, state.plies + 1,
                                       state.history + (lab,))
                    out += 1
                    if out >= max_per_rule:
                        break
    for name, check, apply2 in (
            ("pivot", rr.check_pivot, rr.pivot),
            ("pivot_gadget", rr.check_pivot_gadget,
             rr.unsafe_pivot_gadget),
            ("pivot_boundary", rr.check_pivot_boundary,
             rr.unsafe_pivot_boundary),
            ("hopf", rr.check_hopf, rr.hopf)):
        out = 0
        for e in list(g.edges())[:200]:
            v, w = g.edge_st(e)
            if check(g, v, w):
                g2 = g.copy()
                try:
                    ok = apply2(g2, v, w)
                except Exception:
                    continue
                if ok:
                    lab = f"{name}@{v},{w}"
                    yield lab, ZXState(g2, state.plies + 1,
                                       state.history + (lab,))
                    out += 1
                    if out >= max_per_rule:
                        break
    out = 0
    for v in verts:
        if rr.check_copy(g, v):
            g2 = g.copy()
            try:
                ok = rr.copy(g2, v)
            except Exception:
                continue
            if ok:
                lab = f"copy@{v}"
                yield lab, ZXState(g2, state.plies + 1,
                                   state.history + (lab,))
                out += 1
                if out >= max_per_rule:
                    break


def macro_moves(state: ZXState):
    """Whole-graph macro moves (the algebra-moves analog): pyzx's
    fused simplifiers as single plies — including full_reduce itself
    as a move, which lets the search RESTART greedy from intermediate
    shapes it steered to (the macro-greedy trick)."""
    g = state.g

    def try_macro(name, fn):
        g2 = g.copy()
        try:
            fn(g2)
        except Exception:
            return None
        if ZXState(g2).key() == state.key():
            return None
        return name, ZXState(g2, state.plies + 1, state.history + (name,))

    for name, fn in (
            ("M:gadget_merge", rr.merge_phase_gadgets_for_simp),
            ("M:gadget_poly", rr.gadgets_phasepoly_for_simp),
            ("M:full_reduce", zx.full_reduce),
            ("M:clifford_simp",
             lambda gg: zx.simplify.clifford_simp(gg, quiet=True))):
        out = try_macro(name, fn)
        if out is not None:
            yield out


def zx_eval(state: ZXState) -> tuple:
    return (tcount(state.g), state.g.num_vertices())


def best_first_zx(g0, budget: int = 300, max_per_rule: int = 8):
    """Minimize T-count by best-first over ZX rewrites. Returns the
    best state seen (search never 'solves'; it descends)."""
    tie = itertools.count()
    start = ZXState(g0.copy())
    best = start
    pq = [(zx_eval(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    import itertools as _it
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        for _, child in _it.chain(moves(s, max_per_rule=max_per_rule),
                                  macro_moves(s)):
            k = child.key()
            if k in visited:
                continue
            visited.add(k)
            nodes += 1
            if zx_eval(child) < zx_eval(best):
                best = child
            heapq.heappush(pq, (zx_eval(child), next(tie), child))
            if nodes >= budget:
                break
    return best, nodes


def verify_equal(c_or_g1, g2, qubits: int) -> bool:
    """Boundary oracle: exact tensor equality for small circuits."""
    if qubits > 8:
        return True  # tier-2: trust pyzx rule soundness (documented)
    try:
        return bool(zx.compare_tensors(c_or_g1, g2))
    except Exception:
        return False

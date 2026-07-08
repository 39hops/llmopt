"""Commutator-structure pruning (partial-order reduction, imported
from model checking). Local rewrites on DISJOINT nodes commute: the
search currently generates both orderings of every such pair and lets
the transposition table eat the duplicate — paying full sympy price
for the twin first. Canonical-order pruning refuses to GENERATE the
non-canonical ordering: skip move m at state s when

  (1) m was already legal at s's grandparent — i.e. the previous move
      didn't touch m's target node (the commutation certificate: the
      target node string appears in the grandparent expression), and
  (2) m sorts before the previous move in a fixed total order (the
      twin path, ordered canonically, is the one we keep), and
  (3) both moves are LOCAL (algebra moves are global: never pruned).

Sound for results (verifier unchanged); the race measures whether the
saved generation cost converts to solves at a fixed wall. Arms: eager
vs pruned, bf-nnue + markov top-3, int L3/L4.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import random
import signal

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import (ALGEBRA_MOVES, State, is_solved,
                                      successors)
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")
WALL = 240
GLOBAL_MOVES = {name for name, _ in ALGEBRA_MOVES}


class _Timeout(BaseException):
    pass


class NnueEval(torch.nn.Module):
    # NOTE: mirrors scripts/train_nnue.py NnueEval (scripts aren't a
    # package); keep the two definitions identical.
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_nnue(path: str):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    net = NnueEval()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mean, std = ck["mean"], ck["std"]

    def h(state: State) -> float:
        v = torch.tensor([featurize(state.expr)], dtype=torch.float32)
        with torch.no_grad():
            return float(net((v - mean) / std))

    return h


def make_move_filter(state: State, grand_expr):
    """Canonical-order filter for expanding `state` (see module doc)."""
    if grand_expr is None or not state.history:
        return None
    prev = state.history[-1]
    if prev.split("@")[0] in GLOBAL_MOVES:
        return None  # previous move was global: no certificate

    def keep(label: str) -> bool:
        rule = label.split("@")[0]
        if rule in GLOBAL_MOVES or label >= prev:
            return True
        target = label.split("@", 1)[1]
        # commutation certificate: target existed untouched at the
        # grandparent (string containment on sstr is conservative)
        return target not in grand_expr
    return keep


def best_first(root, budget, prop, h, prune):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    grand: dict[str, str | None] = {start.key(): None}
    parent_sstr: dict[str, str] = {start.key(): sp.sstr(start.expr)}
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        mf = (make_move_filter(s, grand.get(s.key()))
              if prune else None)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1,
                                       move_filter=mf)))[:3]
        for _, child in kids:
            if child.key() in visited:
                continue
            visited.add(child.key())
            grand[child.key()] = parent_sstr[s.key()]
            parent_sstr[child.key()] = sp.sstr(child.expr)
            nodes += 1
            if is_solved(child):
                return child
            heapq.heappush(pq, (h(child), next(tie), child))
            if nodes >= budget:
                break
    return None


def main(n: int, budget: int) -> None:
    prop = MarkovPrior.load().proposer()
    h = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# commutator pruning — bf-nnue+markov3, budget={budget}, "
          f"n={n}/cell, wall={WALL}s")
    print(f"{'lvl':>3} {'arm':>7} {'solved':>7} {'timeouts':>9}")
    for level in (3, 4):
        for cfg, prune in (("eager", False), ("pruned", True)):
            rng = random.Random(f"proposer-race-int-{level}-0")
            ok = to = 0
            for _ in range(n):
                while True:
                    g = sp.simplify(sp.diff(_expression(rng, level), X))
                    if g != 0:
                        break
                signal.alarm(WALL)
                try:
                    sol = best_first(sp.Integral(g, X), budget, prop, h,
                                     prune)
                    ok += (sol is not None
                           and sp.simplify(sp.diff(sol.expr, X) - g) == 0)
                except _Timeout:
                    to += 1
                finally:
                    signal.alarm(0)
            print(f"{level:>3} {cfg:>7} {ok:>4}/{n:<2} {to:>6}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--rule-wall", type=float, default=None)
    a = ap.parse_args()
    if a.rule_wall is not None:
        import llmopt.search.derivation as _d
        _d.RULE_WALL = a.rule_wall
    main(a.n, a.budget)

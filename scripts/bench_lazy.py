"""Lazy expansion vs the L4 total-work wall. The profile said there is
no single stall: timeouts are death by a thousand sympy calls — every
node pays ALL ~20 rules, then the prior keeps 3. But the prior ranks by
RULE NAME, known before any work: consult it first, apply rules one at
a time in prior order, stop at k children. Same selection, a fraction
of the sympy. Arms: eager (incumbent) vs lazy, bf-nnue, int L4.
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
from llmopt.search.derivation import State, is_solved, successors
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import N_FEATURES, featurize
from llmopt.search.rules import (CORE_RULES, INT_RULES, LIM_RULES,
                                 MACRO_RULES)
from llmopt.search.derivation import ALGEBRA_MOVES

X = sp.Symbol("x")
WALL = 240
ALL_RULE_NAMES = [n for n, _ in
                  CORE_RULES + MACRO_RULES + INT_RULES + LIM_RULES
                  + ALGEBRA_MOVES]


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


def make_rankers():
    prior = MarkovPrior.load()
    med = (sorted(prior.unigram.values())[len(prior.unigram) // 2]
           if prior.unigram else 1)

    def rule_rank(state: State) -> list[str]:
        prev = state.history[-1].split("@")[0] if state.history else None
        table = prior.bigram.get(prev) if prev else None

        def s(r: str) -> float:
            return ((table.get(r, 0) if table else 0)
                    + 0.01 * prior.unigram.get(r, med))

        return sorted(ALL_RULE_NAMES, key=lambda r: -s(r))

    return prior.proposer(), rule_rank


def eager_children(s, prop, k=3):
    return prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:k]


def lazy_children(s, rule_rank, k=3):
    out = []
    for name in rule_rank(s):
        for nc in successors(s, use_macros=True, verify_p=0.1,
                             only_rules={name}):
            out.append(nc)
            if len(out) >= k:
                return out
    return out


def best_first(root, budget, expand, h):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        for _, child in expand(s):
            if child.key() in visited:
                continue
            visited.add(child.key())
            nodes += 1
            if is_solved(child):
                return child
            heapq.heappush(pq, (h(child), next(tie), child))
            if nodes >= budget:
                break
    return None


def main(n: int, budget: int) -> None:
    prop, rule_rank = make_rankers()
    h = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    arms = {
        "eager": lambda s: eager_children(s, prop),
        "lazy": lambda s: lazy_children(s, rule_rank),
    }
    print(f"# lazy expansion vs total-work wall — bf-nnue, int L4, "
          f"budget={budget}, n={n}/arm, wall={WALL}s")
    print(f"{'arm':>6} {'solved':>7} {'timeouts':>9}")
    for cfg, expand in arms.items():
        rng = random.Random("proposer-race-int-4-0")
        ok = to = 0
        for _ in range(n):
            while True:
                g = sp.simplify(sp.diff(_expression(rng, 4), X))
                if g != 0:
                    break
            signal.alarm(WALL)
            try:
                sol = best_first(sp.Integral(g, X), budget, expand, h)
                ok += (sol is not None
                       and sp.simplify(sp.diff(sol.expr, X) - g) == 0)
            except _Timeout:
                to += 1
            finally:
                signal.alarm(0)
        print(f"{cfg:>6} {ok:>4}/{n:<2} {to:>6}", flush=True)


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

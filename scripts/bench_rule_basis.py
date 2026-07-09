"""The universal-gate-set question (Artin, from Toffoli universality):
what is the minimal rule basis that still generates our derivations?
Leave-one-out ablation of every INT rule from the champion structural
config (bf-nnue + markov3): a rule whose removal costs nothing is a
non-generator (a dead gate — cf. d_quotient); the survivors are the
domain's gate set. Runs the full-rules arm first as the paired
baseline (methodology rule: one run, one machine state).
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
from llmopt.search.derivation import ALGEBRA_MOVES, State, is_solved, successors
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import N_FEATURES, featurize
from llmopt.search.rules import CORE_RULES, INT_RULES, LIM_RULES, MACRO_RULES

X = sp.Symbol("x")
WALL = 180
ALL_NAMES = set(n for n, _ in CORE_RULES + MACRO_RULES + INT_RULES
                + LIM_RULES + ALGEBRA_MOVES)
INT_NAMES = [n for n, _ in INT_RULES]


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


def best_first(root, budget, prop, h, only_rules):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1,
                                       only_rules=only_rules)))[:3]
        for _, child in kids:
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


def _root(rng, level):
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X), g


def main(n: int, budget: int) -> None:
    prop = MarkovPrior.load().proposer()
    h = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    arms = [("FULL", None)] + [(f"-{r}", ALL_NAMES - {r})
                               for r in INT_NAMES]
    print(f"# rule-basis ablation (leave-one-out) — bf-nnue, int L2-4, "
          f"n={n}/level, budget={budget}")
    print(f"{'arm':>16} {'L2':>6} {'L3':>6} {'L4':>6} {'total':>6}")
    for name, only in arms:
        cells = []
        for level in (2, 3, 4):
            rng = random.Random(f"proposer-race-int-{level}-0")
            ok = 0
            for _ in range(n):
                root, truth = _root(rng, level)
                signal.alarm(WALL)
                try:
                    sol = best_first(root, budget, prop, h, only)
                    ok += (sol is not None and sp.simplify(
                        sp.diff(sol.expr, X) - truth) == 0)
                except _Timeout:
                    pass
                finally:
                    signal.alarm(0)
            cells.append(ok)
        print(f"{name:>16} {cells[0]:>4}/{n:<1} {cells[1]:>4}/{n:<1} "
              f"{cells[2]:>4}/{n:<1} {sum(cells):>6}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget", type=int, default=200)
    a = ap.parse_args()
    main(a.n, a.budget)

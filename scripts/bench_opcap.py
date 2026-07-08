"""Cheap-simplify budgets (autopsy rung 4 candidate): the remaining
int L4 failures are 10/11 WALL timeouts — expression-size economics,
not missing operators. Lever: size-cap pruning — children whose
count_ops exceeds cap are discarded before their sympy costs are paid.
Arms: no cap / 300 / 150. Reports solves AND timeout counts per arm.
bf-nnue + markov3 (the champion structural config), int L4, budget 400.
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

X = sp.Symbol("x")
WALL = 240


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


def best_first(root, budget, prop, h, cap):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:3]
        for _, child in kids:
            if child.key() in visited:
                continue
            visited.add(child.key())
            if cap and sp.count_ops(child.expr) > cap:
                continue  # blow-up branch: refuse to pay its sympy bill
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
    arms = {"nocap": 0, "cap300": 300, "cap150": 150}
    print(f"# size-cap pruning vs L4 timeouts — bf-nnue, budget={budget}, "
          f"n={n}/arm, wall={WALL}s")
    print(f"{'arm':>8} {'solved':>7} {'timeouts':>9}")
    for cfg, cap in arms.items():
        rng = random.Random("proposer-race-int-4-0")
        ok = to = 0
        for _ in range(n):
            root, truth = _root(rng, 4)
            signal.alarm(WALL)
            try:
                sol = best_first(root, budget, prop, h, cap)
                ok += (sol is not None
                       and sp.simplify(sp.diff(sol.expr, X) - truth) == 0)
            except _Timeout:
                to += 1
            finally:
                signal.alarm(0)
        print(f"{cfg:>8} {ok:>4}/{n:<2} {to:>6}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--budget", type=int, default=400)
    a = ap.parse_args()
    main(a.n, a.budget)

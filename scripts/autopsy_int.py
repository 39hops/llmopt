"""Failure autopsy for integration: run the best structural engine
(bf + NNUE h + markov top-3) at a GENEROUS budget on int L3/L4, and
dump every failure — the root integrand plus the best (lowest-h)
state the search died on. Both prior ceiling-movers (euler, i_apart)
came from reading one failing problem; this reads all of them.
Classification of the dump chooses the next rules; frequencies first,
code second.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import signal

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, is_solved, successors
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


def markov():
    # the engine's prior, WITH median smoothing for unseen rules —
    # a local unsmoothed copy silently guillotined every new mover
    # (measured: autopsy #3 reran the same cyclic failure as #2)
    from llmopt.search.engine import MarkovPrior
    return MarkovPrior.load().proposer()


def best_first(root, budget, prop, h):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start, start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    best = (h(start), start)
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:3]
        for _, child in kids:
            if child.key() in visited:
                continue
            visited.add(child.key())
            nodes += 1
            if is_solved(child):
                return child, child
            hc = h(child)
            if hc < best[0]:
                best = (hc, child)
            heapq.heappush(pq, (hc, next(tie), child))
            if nodes >= budget:
                break
    return None, best[1]


def main(n: int, budget: int) -> None:
    h = load_nnue("checkpoints/nnue_eval.pt")
    prop = markov()
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# int failure autopsy — bf-nnue+markov3, budget={budget}, "
          f"n={n}/level")
    fails = []
    for level in (3, 4):
        ok = 0
        rng = random.Random(f"proposer-race-int-{level}-0")
        for i in range(n):
            while True:
                g = sp.simplify(sp.diff(_expression(rng, level), X))
                if g != 0:
                    break
            root = sp.Integral(g, X)
            signal.alarm(WALL)
            try:
                sol, best = best_first(root, budget, prop, h)
                if (sol is not None
                        and sp.simplify(sp.diff(sol.expr, X) - g) == 0):
                    ok += 1
                    continue
            except _Timeout:
                best = None
            finally:
                signal.alarm(0)
            fails.append((level, i, g, best))
            print(f"\nFAIL int L{level} #{i}")
            print(f"  integrand: {g}")
            if best is not None:
                print(f"  died at:   {best.expr}")
                print(f"  history:   {' > '.join(best.history) or '(root)'}")
            else:
                print("  died at:   WALL timeout")
        print(f"\nint L{level}: {ok}/{n} solved, {n - ok} failures", flush=True)
    print(f"\nTOTAL FAILURES: {len(fails)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--budget", type=int, default=400)
    a = ap.parse_args()
    main(a.n, a.budget)

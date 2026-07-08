"""Path-integral-inspired eval (physics night 3): Feynman's sum over
histories says the classical path emerges by CONSTRUCTIVE INTERFERENCE
— many nearby paths agree there. Best-first is the classical limit
(one extremal path) and throws the interference data away: the
transposition table already counts how many distinct derivation
orderings arrive at each state. Amplitude = arrival multiplicity.
Arms: bf-nnue (incumbent) vs bf-nnue with h' = h - w*log2(1+arrivals)
(re-scored on re-arrival; a state many derivations converge on is a
natural waypoint). Paired arms, one run (the methodology rule).
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


def best_first(root, budget, prop, h, interference_w=0.0):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    import math
    pq = [(h(start), next(tie), start)]
    arrivals: dict[str, int] = {start.key(): 1}
    in_queue: dict[str, float] = {}  # key -> base h, for re-scoring
    nodes = 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        if s.key() not in arrivals:
            continue
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:3]
        for _, child in kids:
            k = child.key()
            if k in arrivals:
                arrivals[k] += 1
                if interference_w and k in in_queue:
                    # constructive interference: re-push with a bonus;
                    # stale entries are skipped via the arrivals check
                    hb = in_queue[k] - interference_w * math.log2(
                        1 + arrivals[k])
                    heapq.heappush(pq, (hb, next(tie), child))
                continue
            arrivals[k] = 1
            nodes += 1
            if is_solved(child):
                return child
            hb = h(child)
            in_queue[k] = hb
            heapq.heappush(pq, (hb, next(tie), child))
            if nodes >= budget:
                break
    return None


def _root(rng, level, kind):
    if kind == "diff":
        f = _expression(rng, level)
        return sp.Derivative(f, X), sp.diff(f, X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X), g


def _check(kind, expr, truth):
    if kind == "diff":
        return sp.simplify(expr - truth) == 0
    return sp.simplify(sp.diff(expr, X) - truth) == 0


def main(n: int, budget: int) -> None:
    prop = MarkovPrior.load().proposer()
    h = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    arms = {"classical": 0.0, "interf-w1": 1.0, "interf-w3": 3.0}
    print(f"# interference eval — arrival multiplicity as amplitude; "
          f"bf-nnue, n={n}/cell, budget={budget}")
    print(f"{'kind':>4} {'lvl':>3}" + "".join(f" {c:>10}" for c in arms))
    tot = {c: 0 for c in arms}
    for kind, levels in (("diff", (3, 4)), ("int", (3, 4))):
        for level in levels:
            row = {}
            for cfg, w in arms.items():
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        sol = best_first(root, budget, prop, h,
                                         interference_w=w)
                        ok += (sol is not None
                               and _check(kind, sol.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                row[cfg] = ok
                tot[cfg] += ok
            print(f"{kind:>4} {level:>3}"
                  + "".join(f" {row[c]:>7}/{n:<2}" for c in arms),
                  flush=True)
    print("TOTALS: " + "  ".join(f"{c}: {t}" for c, t in tot.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budget", type=int, default=200)
    a = ap.parse_args()
    main(a.n, a.budget)

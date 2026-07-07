"""Best-first h-race: structural h vs NNUE h, plus a no-dedup ablation.

Best-first made the search h-dominated (pop order IS the eval), so
NNUE's held-out rho advantage (+0.937 vs +0.72) should finally pay
more than the beam photo-finish (93 v 92). The nodedup arm isolates
the transposition table's share of best-first's 104-v-91 win over the
beam (bench_bestfirst.py had `visited` on; the beam has no such set).
All arms g=0 (greedy won the lambda sweep). Same seeds as prior races.
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
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")
WALL = 120
UNSOLVED = (sp.Derivative, sp.Integral, sp.Limit)


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


def h_struct(state: State) -> float:
    return (100.0 * sum(1 for _ in state.expr.atoms(*UNSOLVED))
            + float(sp.count_ops(state.expr)))


def markov():
    import json
    d = json.load(open("checkpoints/markov_prior.json"))
    uni, bi = d["unigram"], d["bigram"]

    def prop(state, children):
        prev = state.history[-1].split("@")[0] if state.history else None
        t = bi.get(prev) if prev else None

        def s(n):
            r = n.split("@")[0]
            return (t.get(r, 0) if t else 0) + 0.01 * uni.get(r, 0)

        return sorted(children, key=lambda c: -s(c[0]))

    return prop


def best_first(root, budget, prop, h, dedup=True):
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
            if dedup:
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


def main(n: int) -> None:
    prop = markov()
    h_nnue = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    arms = {"bf-struct": (h_struct, True),
            "bf-nnue": (h_nnue, True),
            "bf-nodedup": (h_struct, False)}
    print(f"# best-first h-race + dedup ablation — g=0, markov top-3, "
          f"n={n}/cell")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6}" +
          "".join(f" {c:>10}" for c in arms))
    totals = {c: 0 for c in arms}
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in (25, 50):
                row = {}
                for cfg, (h, dedup) in arms.items():
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            sol = best_first(root, budget, prop, h, dedup)
                            ok += (sol is not None
                                   and _check(kind, sol.expr, truth))
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[cfg] = ok
                    totals[cfg] += ok
                print(f"{kind:>4} {level:>3} {budget:>6}" +
                      "".join(f" {row[c]:>7}/{n:<2}" for c in arms),
                      flush=True)
    print("TOTALS: " + "  ".join(f"{c}: {t}" for c, t in totals.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    main(ap.parse_args().n)

"""Computation = cooling, measured (notes/physics-night section 16):
greedy best-first is a T->0 quench; annealing theory says a finite
temperature schedule escapes local minima a quench commits to — and we
HAVE measured local minima (the L4 wall-timeouts where the search
marries a blow-up branch). Metropolis-flavored best-first: pop from
the frontier by Boltzmann weight exp(-(h - h_min)/T) instead of argmin,
T decaying linearly to 0 over the node budget (quench at the end).
Arms: greedy (T=0 incumbent) vs anneal at T0 in {1, 5, 25}.
NNUE h is the energy. Same seeds as every race.
"""

from __future__ import annotations

import argparse
import math
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


def anneal_search(root, budget, prop, h, t0, seed):
    """t0=0 -> exact greedy best-first (the incumbent)."""
    rng = random.Random(seed)
    start = State(root)
    if is_solved(start):
        return start
    frontier = [(h(start), start)]
    visited, nodes = {start.key()}, 1
    while frontier and nodes < budget:
        temp = t0 * max(0.0, 1.0 - nodes / budget)  # linear cool, end quench
        if temp <= 1e-9:
            i = min(range(len(frontier)), key=lambda j: frontier[j][0])
        else:
            hmin = min(hh for hh, _ in frontier)
            ws = [math.exp(-(hh - hmin) / temp) for hh, _ in frontier]
            i = rng.choices(range(len(frontier)), weights=ws)[0]
        _, s = frontier.pop(i)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:3]
        for _, child in kids:
            if child.key() in visited:
                continue
            visited.add(child.key())
            nodes += 1
            if is_solved(child):
                return child
            frontier.append((h(child), child))
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
    prop = MarkovPrior.load().proposer()
    h = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    arms = {"greedy": 0.0, "T0=1": 1.0, "T0=5": 5.0, "T0=25": 25.0}
    print(f"# annealed best-first — NNUE h as energy, linear cool, "
          f"n={n}/cell")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6}"
          + "".join(f" {c:>8}" for c in arms))
    tot = {c: 0 for c in arms}
    for kind, levels in (("diff", (3, 4)), ("int", (3, 4))):
        for level in levels:
            for budget in (100, 400):
                row = {}
                for cfg, t0 in arms.items():
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    for p in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            sol = anneal_search(
                                root, budget, prop, h, t0,
                                f"anneal-{cfg}-{kind}-{level}-{p}")
                            ok += (sol is not None
                                   and _check(kind, sol.expr, truth))
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[cfg] = ok
                    tot[cfg] += ok
                print(f"{kind:>4} {level:>3} {budget:>6}"
                      + "".join(f" {row[c]:>5}/{n:<2}" for c in arms),
                      flush=True)
    print("TOTALS: " + "  ".join(f"{c}: {t}" for c, t in tot.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    main(ap.parse_args().n)

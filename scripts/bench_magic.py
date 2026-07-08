"""The magic detector (physics night 3: Liouville 1835 as the
Gottesman-Knill of integration). sympy's Risch implementation can
PROVE an integrand non-elementary in ~10ms on our death-state shapes.
A state carrying a certified non-elementary Integral node is dead
WITHIN OUR OPERATOR CLOSURE (no rule merges integral nodes, so split
non-elementary siblings can never recombine — the mathematical
loophole is closed by the move set). Pruning it is a theorem per cut,
not a heuristic.

Arms: classical bf-nnue vs magic-pruned. Cells where death states
live: diff/int L4 (+L3 control). Prune stats printed per cell.
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
from llmopt.search.derivation import State, _timeboxed, is_solved, successors
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")
WALL = 240
_VERDICT_CACHE: dict[str, bool] = {}  # integrand srepr -> is_dead


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


def _risch_dead(integrand: sp.Expr) -> bool:
    """True only on a POSITIVE non-elementarity certificate."""
    from sympy.integrals.risch import NonElementaryIntegral, risch_integrate
    try:
        r = risch_integrate(integrand, X)
        return isinstance(r, NonElementaryIntegral) or bool(
            r.has(NonElementaryIntegral))
    except Exception:
        return False  # no verdict -> never prune


def is_dead(state: State) -> bool:
    for node in state.expr.atoms(sp.Integral):
        if len(node.limits) != 1 or len(node.limits[0]) != 1:
            continue
        if node.limits[0][0] != X:
            continue  # u_-substitution carriers: skip
        k = sp.srepr(node.function)
        if k not in _VERDICT_CACHE:
            _VERDICT_CACHE[k] = _timeboxed(
                _risch_dead, node.function, default=False)
        if _VERDICT_CACHE[k]:
            return True
    return False


def best_first(root, budget, prop, h, magic):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start, 0
    pq = [(h(start), next(tie), start)]
    visited, nodes, pruned = {start.key()}, 1, 0
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = prop(s, list(successors(s, use_macros=True, verify_p=0.1)))[:3]
        for _, child in kids:
            if child.key() in visited:
                continue
            visited.add(child.key())
            if magic and is_dead(child):
                pruned += 1
                continue  # theorem per cut: this branch can never close
            nodes += 1
            if is_solved(child):
                return child, pruned
            heapq.heappush(pq, (h(child), next(tie), child))
            if nodes >= budget:
                break
    return None, pruned


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
    print(f"# magic detector — Risch-certified pruning; bf-nnue, "
          f"n={n}/cell, budget={budget}, wall={WALL}s")
    print(f"{'kind':>4} {'lvl':>3} {'classical':>10} {'magic':>8} "
          f"{'cuts':>6}")
    tot = {"classical": 0, "magic": 0}
    for kind, levels in (("diff", (3, 4)), ("int", (3, 4))):
        for level in levels:
            row, cuts = {}, 0
            for cfg, magic in (("classical", False), ("magic", True)):
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        sol, pr = best_first(root, budget, prop, h, magic)
                        cuts += pr if magic else 0
                        ok += (sol is not None
                               and _check(kind, sol.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                row[cfg] = ok
                tot[cfg] += ok
            print(f"{kind:>4} {level:>3} {row['classical']:>7}/{n:<2} "
                  f"{row['magic']:>5}/{n:<2} {cuts:>6}", flush=True)
    print(f"TOTALS: classical: {tot['classical']}  magic: {tot['magic']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budget", type=int, default=200)
    a = ap.parse_args()
    main(a.n, a.budget)

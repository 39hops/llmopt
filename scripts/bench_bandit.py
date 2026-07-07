"""Strategy-portfolio bandit: UCB1 over engine configs, one bandit per
problem class (kind, level). The measured complementarity that motivates
it: int L3 tight-budget prefers flat markov top-3 while everything else
prefers NNUE best-first — no single champion wins every cell. Compare:
each fixed arm, the bandit (online, no oracle), and the per-cell oracle
(upper bound). Bandit must beat the best fixed arm to earn its slot.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
import signal

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, is_solved, successors
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
    d = json.load(open("checkpoints/markov_prior.json"))
    uni, bi = d["unigram"], d["bigram"]
    med = sorted(uni.values())[len(uni) // 2] if uni else 1

    def prop(state, children):
        prev = state.history[-1].split("@")[0] if state.history else None
        t = bi.get(prev) if prev else None

        def s(n):
            r = n.split("@")[0]
            return (t.get(r, 0) if t else 0) + 0.01 * uni.get(r, med)

        return sorted(children, key=lambda c: -s(c[0]))

    return prop


def best_first(root, budget, prop, h):
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
            nodes += 1
            if is_solved(child):
                return child
            heapq.heappush(pq, (h(child), next(tie), child))
            if nodes >= budget:
                break
    return None


def make_arms():
    prop = markov()
    h_nnue = load_nnue("checkpoints/nnue_eval.pt")

    def beam_arm(root, budget):
        r = beam_search(root, width=2, max_plies=20, max_nodes=budget,
                        proposer=prop, propose_k=3, verify_p=0.1,
                        use_macros=True)
        return r.state if r.solved else None

    return {
        "beam-mk3": beam_arm,
        "bf-struct": lambda root, b: best_first(root, b, prop, h_struct),
        "bf-nnue": lambda root, b: best_first(root, b, prop, h_nnue),
    }


def ucb_pick(stats: dict, t: int) -> str:
    for name, (nplays, _) in stats.items():
        if nplays == 0:
            return name
    return max(stats, key=lambda a: stats[a][1] / stats[a][0]
               + math.sqrt(2 * math.log(t) / stats[a][0]))


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
    arms = make_arms()
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    names = list(arms)
    print(f"# strategy bandit — UCB1 per (kind, level), n={n}/cell, "
          f"budget={budget}")
    print(f"{'kind':>4} {'lvl':>3}" + "".join(f" {c:>10}" for c in names)
          + f" {'bandit':>8} {'picks':>22}")
    tot = {c: 0 for c in names} | {"bandit": 0, "oracle": 0}
    for kind in ("diff", "int"):
        for level in (2, 3):
            # fixed arms: full replay per arm, same seeds
            row = {}
            for cfg in names:
                rng = random.Random(f"bandit-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        sol = arms[cfg](root, budget)
                        ok += (sol is not None
                               and _check(kind, sol.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                row[cfg] = ok
                tot[cfg] += ok
            # bandit: one online pass, same seeds
            rng = random.Random(f"bandit-{kind}-{level}-0")
            stats = {c: (0, 0.0) for c in names}
            bok, picks = 0, {c: 0 for c in names}
            for t in range(1, n + 1):
                root, truth = _root(rng, level, kind)
                a = ucb_pick(stats, t)
                picks[a] += 1
                signal.alarm(WALL)
                try:
                    sol = arms[a](root, budget)
                    win = (sol is not None
                           and _check(kind, sol.expr, truth))
                except _Timeout:
                    win = False
                finally:
                    signal.alarm(0)
                bok += win
                np_, rw = stats[a]
                stats[a] = (np_ + 1, rw + win)
            tot["bandit"] += bok
            tot["oracle"] += max(row.values())
            pk = " ".join(f"{c.split('-')[1]}:{picks[c]}" for c in names)
            print(f"{kind:>4} {level:>3}"
                  + "".join(f" {row[c]:>7}/{n:<2}" for c in names)
                  + f" {bok:>5}/{n:<2} {pk:>22}", flush=True)
    print("TOTALS: " + "  ".join(f"{k}: {v}" for k, v in tot.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()
    main(a.n, a.budget)

"""Shared bench harness helpers (spec 2026-08-12 §4.2), adopted
verbatim from the six identical script copies (bench_control lineage
for _root/_check; train_nnue/autopsy_int lineage for NnueEval +
load_nnue). Divergent copies stay in place untouched:
gen_proposer_data.py (_root without the int arm's simplify loop) and
train_nnue.py (single-value _root).

_root returns (unevaluated problem, ground truth); _check oracles a
candidate against the truth (sympy symbolic equivalence, never string
match). load_nnue returns a State -> float heuristic closure.
"""
from __future__ import annotations

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")


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


class NnueEval(torch.nn.Module):
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

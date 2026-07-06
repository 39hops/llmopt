"""The NNUE race: eval_fn=hce vs eval_fn=nnue inside the SAME search,
held-out problems, fixed node budgets. Solve rate is the score —
never training loss. Spec: 2026-07-07-nnue-eval-design.md."""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, hce
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")
WALL = 120


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


def load_eval(path: str):
    ck = torch.load(path, weights_only=True)
    net = NnueEval()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mean, std = ck["mean"], ck["std"]

    def nnue_eval(state: State) -> float:
        v = torch.tensor([featurize(state.expr)], dtype=torch.float32)
        with torch.no_grad():
            return float(net((v - mean) / std))

    return nnue_eval


def _root(rng, level, kind):
    if kind == "diff":
        f = _expression(rng, level)
        return sp.Derivative(f, X), sp.diff(f, X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X), g


def _check(kind, result_expr, truth):
    if kind == "diff":
        return sp.simplify(result_expr - truth) == 0
    return sp.simplify(sp.diff(result_expr, X) - truth) == 0


def main(n: int, budgets: list[int], ckpt: str) -> None:
    nnue = load_eval(ckpt)
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))
    print(f"# NNUE race — n={n}/cell, wall {WALL}s/search")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'hce':>7} {'nnue':>7}")
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                scores = {}
                for name, fn in (("hce", hce), ("nnue", nnue)):
                    rng = random.Random(f"nnue-race-{kind}-{level}-0")
                    ok = 0
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            r = beam_search(root, width=8, max_plies=20,
                                            max_nodes=budget, eval_fn=fn)
                            ok += r.solved and _check(kind, r.state.expr, truth)
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    scores[name] = ok
                print(f"{kind:>4} {level:>3} {budget:>6} "
                      f"{scores['hce']:>4}/{n:<2} {scores['nnue']:>4}/{n:<2}",
                      flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--ckpt", default="checkpoints/nnue_eval.pt")
    a = ap.parse_args()
    main(a.n, a.budgets, a.ckpt)

"""The proposer race: full enumeration vs model-proposed top-k vs
random-k control, under HCE and NNUE evals, held-out problems.
Solve rate at fixed node budget is the score; proposer inference time
is wall clock, reported separately. Spec:
2026-07-07-move-proposer-design.md."""

from __future__ import annotations

import argparse
import random
import signal
import time

import sympy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search, hce
from llmopt.search.features import N_FEATURES, featurize
from llmopt.search.proposer import hf_score_fn, make_proposer
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


class NnueEval(torch.nn.Module):  # mirrors train_nnue/bench_nnue
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_nnue(path="checkpoints/nnue_eval.pt"):
    ck = torch.load(path, weights_only=True)
    net = NnueEval()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mean, std = ck["mean"], ck["std"]

    def ev(state):
        v = torch.tensor([featurize(state.expr)], dtype=torch.float32)
        with torch.no_grad():
            return float(net((v - mean) / std))

    return ev


def load_proposer(ckpt="checkpoints/proposer_lora.pt"):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt, weights_only=True, map_location="cpu"), strict=False)
    assert not unexpected, unexpected
    model.eval()
    return make_proposer(hf_score_fn(model, tok, device))


def random_proposer(seed_tag: str):
    rng = random.Random(f"random-proposer-{seed_tag}")

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


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


def main(n: int, budgets: list[int]) -> None:
    nnue = load_nnue()
    model_prop = load_proposer()
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    configs = [
        ("full+hce", hce, None),
        ("full+nnue", nnue, None),
        ("rand3+hce", hce, "random"),
        ("prop3+hce", hce, model_prop),
        ("prop3+nnue", nnue, model_prop),
    ]
    print(f"# proposer race — n={n}/cell, wall {WALL}s/search, k=3")
    header = f"{'kind':>4} {'lvl':>3} {'budget':>6}" + "".join(
        f" {name:>16}" for name, _, _ in configs)
    print(header)
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                cells = []
                for name, ev, prop in configs:
                    if prop == "random":
                        prop = random_proposer(f"{kind}-{level}-{budget}")
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok, t0 = 0, time.time()
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            r = beam_search(
                                root, width=8, max_plies=20,
                                max_nodes=budget, eval_fn=ev,
                                proposer=prop,
                                propose_k=3 if prop else None)
                            ok += r.solved and _check(kind, r.state.expr, truth)
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    cells.append(f"{ok:>3}/{n:<2}({time.time() - t0:5.0f}s)")
                print(f"{kind:>4} {level:>3} {budget:>6} " +
                      " ".join(cells), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    a = ap.parse_args()
    main(a.n, a.budgets)

"""The fused-architecture race (Artin's integration, 2026-07-08): bf
with h = value head on the 0.5B trunk's hidden state, vs bf-nnue (20
hand features). Offline the trunk lost the ordering fight (+0.859 vs
+0.937), but offline rho has under-predicted search before. Honest
cost note printed per arm: the fused eval pays an LLM forward per
node — if it wins solves but loses wall, that's the verdict too.
Same cells/seeds as the 113/120 record races.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import random
import signal
import time

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, is_solved, successors
from llmopt.search.engine import MarkovPrior
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


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


def load_fused(v2: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.train.lora import apply_lora

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load("checkpoints/proposer_lora.pt", weights_only=True,
                   map_location="cpu"), strict=False)
    model.eval()
    ck_path = ("checkpoints/value_head_v2.pt" if v2
               else "checkpoints/value_head.pt")
    ck = torch.load(ck_path, weights_only=True, map_location="cpu")
    if v2:  # v2 ships its own value-LoRA over the trunk
        model.load_state_dict(ck["lora"], strict=False)
    d_model = model.config.hidden_size
    head = torch.nn.Sequential(
        torch.nn.Linear(d_model, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1)).to(device).to(torch.float32)
    head.load_state_dict(ck["state_dict"])
    head.eval()

    def h(state: State) -> float:
        enc = tok(sp.sstr(state.expr), return_tensors="pt",
                  truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.model(input_ids=enc.input_ids)
            hid = out.last_hidden_state[0, -1].to(torch.float32)
            return float(head(hid))

    return h


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


def main(n: int, v2: bool = False) -> None:
    prop = MarkovPrior.load().proposer()
    arms = {"bf-nnue": load_nnue("checkpoints/nnue_eval.pt"),
            "bf-fused": load_fused(v2=v2)}
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# fused (trunk hidden-state value) vs NNUE (hand features) "
          f"— bf + markov3, n={n}/cell, wall={WALL}s")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6}"
          + "".join(f" {c:>9}" for c in arms) + f" {'sec/prob':>18}")
    tot = {c: 0 for c in arms}
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in (25, 50):
                row, secs = {}, {}
                for cfg, h in arms.items():
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok, t0 = 0, time.monotonic()
                    for _ in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            sol = best_first(root, budget, prop, h)
                            ok += (sol is not None
                                   and _check(kind, sol.expr, truth))
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    row[cfg] = ok
                    secs[cfg] = (time.monotonic() - t0) / n
                    tot[cfg] += ok
                print(f"{kind:>4} {level:>3} {budget:>6}"
                      + "".join(f" {row[c]:>6}/{n:<2}" for c in arms)
                      + f"  {secs['bf-nnue']:>6.1f}s vs "
                        f"{secs['bf-fused']:>6.1f}s", flush=True)
    print("TOTALS: " + "  ".join(f"{c}: {t}" for c, t in tot.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    main(a.n, v2=a.v2)

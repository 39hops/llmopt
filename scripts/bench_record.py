"""The record attempt: every proven component in one search, first
time. Best-first frontier (beat the beam 103v91) + NNUE h (113/120)
+ markov ranking (choice is grammar) + LLM entropy-gated k (the +15
confidence premium behind 349/360) + magic pruning (Liouville,
replicated +1). Full 24-cell matrix, same seeds as every race.
Standing record: hybrid beam 349/360 (96.9%), n=30-confirmed 694/720.
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
from llmopt.search.magic import is_dead
from llmopt.search.proposer import entropy_k, hf_score_fn
from llmopt.train.lora import apply_lora

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


def load_score_fn():
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    return hf_score_fn(model, tok, device)


def record_search(root, budget, mk_prop, score_fn, k_policy, h):
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = mk_prop(s, list(successors(s, use_macros=True,
                                          verify_p=0.1)))
        if not kids:
            continue
        labels = [name for name, _ in kids]
        scores = score_fn(sp.sstr(s.expr), labels)
        k = k_policy(s, kids, scores)
        for _, child in kids[:max(1, int(k))]:
            if child.key() in visited:
                continue
            visited.add(child.key())
            if is_dead(child):
                continue  # Liouville cut
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
    mk_prop = MarkovPrior.load().proposer()
    score_fn = load_score_fn()
    k_policy = entropy_k(1, 3, temperature=0.1)
    h = load_nnue("checkpoints/nnue_eval.pt")
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# RECORD ATTEMPT — bf + nnue-h + markov-rank + LLM-gate + "
          f"magic; n={n}/cell; standing record 349/360")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'record':>8}")
    tot = 0
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in (25, 50, 100, 200):
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        sol = record_search(root, budget, mk_prop,
                                            score_fn, k_policy, h)
                        ok += (sol is not None
                               and _check(kind, sol.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                tot += ok
                print(f"{kind:>4} {level:>3} {budget:>6} {ok:>5}/{n:<2}",
                      flush=True)
    print(f"TOTAL: {tot}/360  (standing record: 349)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    main(ap.parse_args().n)

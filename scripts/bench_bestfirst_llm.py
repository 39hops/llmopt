"""The record attempt: best-first + NNUE h + entropy-gated 0.5B
confidence — the three winning components in one search for the first
time. Incumbent to beat: bf-nnue + markov top-3 = 113/120 on these
exact cells and seeds (bench_bestfirst_nnue.py). Only the new arm
runs; compare row-by-row against the recorded incumbent table.
"GPU buys confidence, not choice": the LLM's job here is k, not rank.
"""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, is_solved, successors
from llmopt.search.features import N_FEATURES, featurize
from llmopt.search.proposer import entropy_k, hf_score_fn, \
    make_scoring_proposer
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
INCUMBENT = {("diff", 2, 25): 13, ("diff", 2, 50): 15,
             ("diff", 3, 25): 14, ("diff", 3, 50): 15,
             ("int", 2, 25): 15, ("int", 2, 50): 15,
             ("int", 3, 25): 13, ("int", 3, 50): 13}


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


def load_llm():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load("checkpoints/proposer_lora.pt", weights_only=True,
                   map_location="cpu"), strict=False)
    model.eval()
    return hf_score_fn(model, tok, device)


def best_first_adaptive(root, budget, scoring_prop, k_policy, h):
    import heapq
    import itertools
    tie = itertools.count()
    start = State(root)
    if is_solved(start):
        return start
    pq = [(h(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = list(successors(s, use_macros=True, verify_p=0.1))
        if not kids:
            continue
        ranked, scores = scoring_prop(s, kids)
        k = k_policy(s, ranked, scores)
        for _, child in ranked[:max(1, int(k))]:
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
    h = load_nnue("checkpoints/nnue_eval.pt")
    scoring_prop = make_scoring_proposer(load_llm())
    k_policy = entropy_k(1, 3, temperature=0.1)
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# bf-nnue + adaptive-LLM (entropy_k(1,3,T=0.1)) — n={n}/cell; "
          f"incumbent = bf-nnue+markov3 (113/120)")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'bf-adapt':>9} "
          f"{'incumbent':>9}")
    tot, inc_tot = 0, 0
    for kind in ("diff", "int"):
        for level in (2, 3):
            for budget in (25, 50):
                rng = random.Random(f"proposer-race-{kind}-{level}-0")
                ok = 0
                for _ in range(n):
                    root, truth = _root(rng, level, kind)
                    signal.alarm(WALL)
                    try:
                        sol = best_first_adaptive(root, budget, scoring_prop,
                                                  k_policy, h)
                        ok += (sol is not None
                               and _check(kind, sol.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                inc = INCUMBENT[(kind, level, budget)]
                tot += ok
                inc_tot += inc
                print(f"{kind:>4} {level:>3} {budget:>6} {ok:>6}/{n:<2} "
                      f"{inc:>6}/{n:<2}", flush=True)
    print(f"TOTALS: bf-adapt: {tot}  incumbent: {inc_tot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    main(ap.parse_args().n)

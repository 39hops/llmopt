"""Adaptive-k race: entropy-gated branching vs the fixed strategies.
Pre-registered prediction (spec 2026-07-07-adaptive-k-design.md):
adaptive should match k1x3 on diff L2-3 AND match full on int L3 —
spending width exactly where the sweep showed width matters. Also
prints mean-k and an H histogram per cell: the null-check instrument
(if H doesn't localize, the confidence signal is the gap)."""

from __future__ import annotations

import argparse
import random
import signal
import statistics
import time

import sympy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search, hce
from llmopt.search.proposer import entropy_k, hf_score_fn, make_proposer, \
    make_scoring_proposer
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


def load_model():
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


def random_proposer(seed_tag: str):
    rng = random.Random(f"random-proposer-{seed_tag}")

    def prop(state, children):
        children = list(children)
        rng.shuffle(children)
        return children

    return prop


def restart_search(root, total_budget, restarts, seed, width=8):
    per = max(1, total_budget // restarts)
    for i in range(restarts):
        r = beam_search(root, width=width, max_plies=20, max_nodes=per,
                        proposer=random_proposer(f"{seed}-r{i}"),
                        propose_k=1)
        if r.solved:
            return r
    return r


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


def main(n: int, budgets: list[int], temperature: float = 1.0,
         configs: list[str] | None = None, width: int = 8) -> None:
    score_fn = load_model()
    scoring_prop = make_scoring_proposer(score_fn)
    fixed_prop = make_proposer(score_fn)
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    configs = configs or ["full", "k3prop", "k1x3", "adapt"]
    print(f"# adaptive-k race — n={n}/cell, wall {WALL}s/search, "
          f"entropy_k(1,6,T={temperature}), configs={configs}, width={width}")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'full':>7} {'k3prop':>7} "
          f"{'k1x3':>7} {'adapt':>7} {'mean-k':>7} {'H-deciles':>22}")
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            for budget in budgets:
                cells, mean_k, h_hist = {}, float("nan"), ""
                for name in configs:
                    rng = random.Random(f"proposer-race-{kind}-{level}-0")
                    ok = 0
                    ks_seen: list[int] = []
                    hs_seen: list[float] = []
                    for p in range(n):
                        root, truth = _root(rng, level, kind)
                        signal.alarm(WALL)
                        try:
                            if name == "full":
                                r = beam_search(root, width=width, max_plies=20,
                                                max_nodes=budget)
                            elif name == "k3prop":
                                r = beam_search(root, width=width, max_plies=20,
                                                max_nodes=budget,
                                                proposer=fixed_prop,
                                                propose_k=3)
                            elif name == "k1x3":
                                r = restart_search(root, budget, 3,
                                                   f"{kind}-{level}-{p}",
                                                   width=width)
                            else:
                                base_policy = entropy_k(
                                    1, 6, temperature=temperature)

                                def policy(s_, ranked, scores):
                                    import math
                                    k = base_policy(s_, ranked, scores)
                                    ks_seen.append(k)
                                    nsc = len(scores)
                                    if nsc > 1:
                                        m = max(scores)
                                        e = [math.exp(v - m) for v in scores]
                                        z = sum(e)
                                        ps = [q / z for q in e]
                                        h = -sum(q * math.log(q)
                                                 for q in ps if q > 0)
                                        hs_seen.append(h / math.log(nsc))
                                    return k

                                r = beam_search(root, width=width, max_plies=20,
                                                max_nodes=budget,
                                                proposer=scoring_prop,
                                                propose_k=policy)
                            ok += r.solved and _check(kind, r.state.expr, truth)
                        except _Timeout:
                            pass
                        finally:
                            signal.alarm(0)
                    cells[name] = ok
                    if name == "adapt" and ks_seen:
                        mean_k = statistics.mean(ks_seen)
                        deciles = [0] * 10
                        for h in hs_seen:
                            deciles[min(9, int(h * 10))] += 1
                        tot = max(1, sum(deciles))
                        h_hist = "".join(str(min(9, d * 10 // tot))
                                         for d in deciles)
                row = " ".join(f"{cells[c]:>4}/{n:<2}" for c in configs)
                print(f"{kind:>4} {level:>3} {budget:>6} {row} "
                      f"{mean_k:>7.2f} {h_hist:>22}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--configs", nargs="+", default=None,
                    choices=["full", "k3prop", "k1x3", "adapt"])
    ap.add_argument("--width", type=int, default=8)
    a = ap.parse_args()
    main(a.n, a.budgets, temperature=a.temperature, configs=a.configs,
         width=a.width)

"""Expert-iteration curve point: engine-r1 (original proposer ckpt) vs
engine-r2 (retrained on frontier harvest) on HELD-OUT L4 frontier-ish
problems, prop3+HCE, budgets 100/200/400. Also the regression guard:
quick L1-3 totals must stay within noise of r1's. Spec:
2026-07-07-expert-iteration-r2-design.md Task 3.

  python scripts/bench_frontier.py --r1 checkpoints/proposer_lora.pt \
      --r2 checkpoints/proposer_lora_r2.pt --n 20
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import sympy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search
from llmopt.search.proposer import hf_score_fn, make_proposer
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


def load_proposer(ckpt: str):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device)
    apply_lora(model, TARGETS, r=16, alpha=32)
    model.load_state_dict(
        torch.load(ckpt, weights_only=True, map_location="cpu"),
        strict=False)
    model.eval()
    return make_proposer(hf_score_fn(model, tok, device))


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


def main(r1: str, r2: str, n: int) -> None:
    harvest_roots = set(json.loads(
        Path("data/frontier_r1_roots.json").read_text()))
    signal.signal(signal.SIGALRM,
                  lambda s, f: (_ for _ in ()).throw(_Timeout()))
    props = {"r1": load_proposer(r1), "r2": load_proposer(r2)}
    print(f"# expert-iteration curve point — held-out L4, n={n}/cell, "
          f"prop3+HCE")
    print(f"{'kind':>4} {'budget':>6} {'r1':>7} {'r2':>7}")
    totals = {"r1": 0, "r2": 0}
    for kind in ("diff", "int"):
        for budget in (100, 200, 400):
            row = {}
            for tag, prop in props.items():
                rng = random.Random(f"eir2-eval-{kind}-4-0")  # held-out seed
                ok = drawn = 0
                while drawn < n:
                    root, truth = _root(rng, 4, kind)
                    if sp.srepr(root) in harvest_roots:
                        continue  # exclude guard vs training data
                    drawn += 1
                    signal.alarm(WALL)
                    try:
                        r = beam_search(root, width=8, max_plies=24,
                                        max_nodes=budget, proposer=prop,
                                        propose_k=3)
                        ok += r.solved and _check(kind, r.state.expr, truth)
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                row[tag] = ok
                totals[tag] += ok
            print(f"{kind:>4} {budget:>6} {row['r1']:>4}/{n:<2} "
                  f"{row['r2']:>4}/{n:<2}", flush=True)
    print(f"TOTALS: r1 {totals['r1']}  r2 {totals['r2']}  "
          f"(delta {totals['r2'] - totals['r1']:+d})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default="checkpoints/proposer_lora.pt")
    ap.add_argument("--r2", default="checkpoints/proposer_lora_r2.pt")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(a.r1, a.r2, a.n)

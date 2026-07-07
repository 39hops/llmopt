"""The hybrid cell (Artin's distillation question, 2026-07-07): markov
RANKS, the 0.5B GATES k. We measured LLM-ranks+LLM-gates (328/360) and
markov-ranks+fixed-k3 (316/360); markov-confidence was a null. If
ranking is grammar and the GPU's real contribution is confidence, the
hybrid should approach the champion at zero LLM ranking cost — the
strongest possible statement of "the GPU buys confidence, not choice."

Standard 24-cell matrix, same seeds as every prior race.
"""

from __future__ import annotations

import argparse
import random
import signal

import sympy as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search
from llmopt.search.engine import MarkovPrior
from llmopt.search.proposer import entropy_k, hf_score_fn
from llmopt.train.lora import apply_lora

X = sp.Symbol("x")
WALL = 300
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


class _Timeout(BaseException):
    pass


def load_score_fn():
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


def make_hybrid_proposer(score_fn):
    """Rank by the bigram dict; attach LLM scores IN MARKOV ORDER so
    entropy_k reads the model's confidence over the dict's ranking."""
    mk = MarkovPrior.load().proposer()

    def proposer(state, children):
        if not children:
            return children, []
        ranked = mk(state, children)
        labels = [name for name, _ in ranked]
        scores = score_fn(sp.sstr(state.expr), labels)
        return ranked, scores

    return proposer


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
    prop = make_hybrid_proposer(load_score_fn())
    k_policy = entropy_k(1, 3, temperature=0.1)
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    print(f"# hybrid: markov ranks + LLM gates (entropy_k(1,3,T=0.1)) "
          f"@ w2 — n={n}/cell; reference: adapt-LLM 328/360, markov3 316/360")
    print(f"{'kind':>4} {'lvl':>3} {'budget':>6} {'hybrid':>8}")
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
                        r = beam_search(root, width=2, max_plies=20,
                                        max_nodes=budget, proposer=prop,
                                        propose_k=k_policy, verify_p=0.1,
                                        use_macros=True)
                        ok += (r.solved
                               and _check(kind, r.state.expr, truth))
                    except _Timeout:
                        pass
                    finally:
                        signal.alarm(0)
                tot += ok
                print(f"{kind:>4} {level:>3} {budget:>6} {ok:>5}/{n:<2}",
                      flush=True)
    print(f"TOTAL: hybrid {tot}/360")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    main(ap.parse_args().n)

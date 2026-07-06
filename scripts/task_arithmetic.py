"""Task-arithmetic experiments 1-3 (see the 2026-07-06 spec).

Requires checkpoints/calculus_lora.pt from scripts/train_calculus.py.
Eval set is the SAME (N=300, seed 99) the training script used, so
numbers line up with its printed baseline/post-training accuracy.

1. scaling sweep  λ in {0, 0.5, 1.0, 1.5, 2.0} on the Instruct model
2. negation       λ = -1 (calculus should crater, perplexity should not)
3. transfer       λ in {0.5, 1.0} on the BASE (non-Instruct) model

General-ability control: perplexity over a fixed prose sample at every
λ. Math accuracy bought by wrecking the language model is a loss.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.evaluate import evaluate_model
from llmopt.mathgen.problems import make_dataset
from llmopt.train.task_vector import apply_task_vector, load_adapter

INSTRUCT = "Qwen/Qwen2.5-0.5B-Instruct"
BASE = "Qwen/Qwen2.5-0.5B"
ADAPTER = Path("checkpoints/calculus_lora.pt")
KINDS = ("differentiate", "integrate", "limit_traced")
N_EVAL = 300

PROSE = (
    "The lighthouse keeper kept a meticulous log. Each morning he noted "
    "the wind, the shipping traffic, and the state of the lamp. Years "
    "later, historians would treasure these small observations: they "
    "recorded a coastline, an economy, and a way of life that had "
    "otherwise vanished without remark."
)


def perplexity(model, tok) -> float:
    ids = tok(PROSE, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        loss = model(input_ids=ids, labels=ids).loss
    return float(torch.exp(loss))


def run(model, tok, problems, adapter, scale, label):
    undo = apply_task_vector(model, adapter, scale) if scale != 0 else lambda: None
    try:
        acc = evaluate_model(model, tok, problems, max_new_tokens=160)["overall"]
        ppl = perplexity(model, tok)
        print(f"  λ={scale:+4.1f}  {label:22s} acc {acc:6.1%}   ppl {ppl:7.2f}")
        return acc, ppl
    finally:
        undo()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "mps"
    adapter = load_adapter(ADAPTER)
    problems = make_dataset(N_EVAL, kinds=KINDS, seed=99)

    print("== instruct model: scaling sweep + negation ==")
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16).to(device)
    model.eval()
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0, -1.0):
        label = "negation" if lam < 0 else "scaling"
        run(model, tok, problems, adapter, lam, label)
    del model

    print("== base model: cross-model transfer ==")
    tok_b = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(device)
    base.eval()
    for lam in (0.0, 0.5, 1.0):
        run(base, tok_b, problems, adapter, lam, "transfer")


if __name__ == "__main__":
    main()

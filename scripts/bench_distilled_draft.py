"""Distilled-draft speculative decoding: accept rate + tok/s, real models.

Target Qwen2.5-3B, draft Qwen2.5-0.5B (shared vocab). Measure greedy
baseline and speculative before/after LoRA logit-KD of the draft toward
the target on the target's own greedy continuations (the deployment
distribution). Output must stay token-identical to target greedy.

Corpus prompts are held out from the benchmark prompt.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.speculative import generate_speculative
from llmopt.distill.logit_kd import distill_logits
from llmopt.train.lora import apply_lora
from scripts.bench_decoding import PROMPT, timed, vanilla_greedy

TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
N = 150
NUM_DRAFT = 5
CORPUS_PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Describe the water cycle in simple terms.",
    "What are the trade-offs of microservices?",
    "Summarize why the sky appears blue.",
    "How does public key cryptography work?",
    "Explain gradient descent to a beginner.",
    "What causes inflation in an economy?",
    "Describe how a compiler optimizes loops.",
]
CORPUS_NEW_TOKENS = 96
EPOCHS, LR = 3, 2e-5
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RANK = 16


def merge_all(model) -> None:
    """Fold every LoRALinear back into a plain Linear (zero overhead)."""
    from llmopt.train.lora import LoRALinear

    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                dev, dt = child.base.weight.device, child.base.weight.dtype
                setattr(module, name, child.merge().to(device=dev, dtype=dt))


def bench_spec(target, draft, ids, ref, label):
    (out, stats), t = timed(
        lambda: generate_speculative(
            target, draft, ids, max_new_tokens=N, num_draft=NUM_DRAFT
        )
    )
    eq = "OK" if out == ref else "DIVERGED"
    acc = stats["accepted"] / max(stats["drafted"], 1)
    print(
        f"{label:28s}: {N / t:6.1f} tok/s  accept={acc:.1%}"
        f"  target_passes={stats['target_passes']} equiv={eq}"
    )
    return acc


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float16).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT).input_ids

    (ref, _), t0 = timed(lambda: vanilla_greedy(target, ids, N))
    print(f"{'target greedy':28s}: {N / t0:6.1f} tok/s  (baseline)")
    acc0 = bench_spec(target, draft, ids, ref, "speculative (stock draft)")

    # corpus: target's own greedy continuations of held-out prompts
    print("\nbuilding KD corpus from target generations...")
    corpus = []
    for p in CORPUS_PROMPTS:
        pids = tok(p).input_ids
        seq, _ = vanilla_greedy(target, pids, CORPUS_NEW_TOKENS)
        corpus.append(seq)

    # distill in fp32 (fp16 Adam on the adapters NaN'd), bench back in fp16
    draft = draft.float()
    wrapped = apply_lora(draft, TARGETS, r=RANK, alpha=2 * RANK)
    n_train = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"LoRA on draft: {wrapped} linears, {n_train / 1e6:.1f}M trainable")
    draft.train()
    losses = distill_logits(draft, target, corpus, epochs=EPOCHS, lr=LR)
    print("KD loss per epoch:", " ".join(f"{x:.4f}" for x in losses))
    for p in draft.parameters():
        p.grad = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    merge_all(draft)
    draft = draft.half().eval()

    bench_spec(target, draft, ids, ref, "speculative (distilled)")
    print(f"stock accept rate was {acc0:.1%}")


if __name__ == "__main__":
    main()

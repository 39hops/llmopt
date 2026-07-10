"""Entropy-adaptive draft length vs fixed-k speculative decoding (3080).

Arms: fixed k in {3,5,8} (generate_speculative greedy) vs adaptive
(k_min=1, k_max=12) at an ent_stop sweep. Every arm's output is
asserted token-identical to target-only greedy — a mismatch is a bug,
not a data point.

Pre-registered bar: adaptive beats the BEST fixed k on tokens/s at
identical outputs. Secondary diagnostics: accepted/drafted ratio,
target passes (each one is a full weight read — the currency),
draft passes, early stops.

Usage (WSL 3080): python scripts/bench_adaptive_draft.py --max-new 200
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.speculative import generate_speculative
from llmopt.decoding.speculative_adaptive import generate_speculative_adaptive
from llmopt.eval.equivalence import assert_tokens_equal

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"

ARTICLE = (
    "The RTX 3080 uses the GA102 GPU with 8704 CUDA cores and 10 GB of GDDR6X "
    "memory on a 320-bit bus, giving 760 GB/s of memory bandwidth. The GA102 "
    "chip is fabricated on Samsung's 8 nm process. Memory bandwidth matters "
    "because autoregressive decoding is memory-bound: every generated token "
    "requires reading all model weights from memory. Speculative decoding "
    "amortizes those weight reads across several tokens per forward pass."
)

# Mixed regimes on purpose: grounded summary (easy for the draft),
# open-ended prose (hard), code (bursty: easy inside idioms, hard at
# decision points — where adaptive length should earn its keep).
PROMPTS = [
    f"Summarize the following in two sentences:\n\n{ARTICLE}\n\nSummary:",
    "Write a short story about a lighthouse keeper who discovers "
    "something unusual in the fog. Story:",
    "Write a Python function that parses a CSV file and returns the "
    "rows as dictionaries, with type inference for ints and floats.\n\n"
    "```python\n",
]


def vanilla_greedy(model, prompt_ids: list[int], max_new: int) -> list[int]:
    from llmopt.decoding.kv import valid_len

    tokens = list(prompt_ids)
    past = None
    with torch.inference_mode():
        for _ in range(max_new):
            fed = tokens[valid_len(past):]
            out = model(
                input_ids=torch.tensor([fed], device=model.device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            tokens.append(int(out.logits[0, -1].argmax()))
    return tokens


def timed(fn, repeats: int = 3):
    fn()  # warmup
    times, result = [], None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return result, min(times)


def main(max_new: int) -> None:
    tok = AutoTokenizer.from_pretrained(TARGET)
    tgt = AutoModelForCausalLM.from_pretrained(
        TARGET, torch_dtype=torch.float16, device_map="cuda")
    drf = AutoModelForCausalLM.from_pretrained(
        DRAFT, torch_dtype=torch.float16, device_map="cuda")
    tgt.eval(), drf.eval()

    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        ref = vanilla_greedy(tgt, ids, max_new)
        new_ref = len(ref) - len(ids)
        print(f"\n## prompt {pi} ({new_ref} new tokens)")

        for k in (3, 5, 8):
            (out, st), t = timed(lambda k=k: generate_speculative(
                tgt, drf, torch.tensor([ids]), max_new_tokens=max_new,
                num_draft=k))
            eq = assert_tokens_equal(ref, out)
            if not eq:
                raise AssertionError(f"fixed k={k} p{pi}: {eq.detail}")
            acc = st["accepted"] / max(st["drafted"], 1)
            print(f"fixed k={k}: {new_ref / t:6.1f} tok/s  "
                  f"acc {acc:.2f}  tgt-passes {st['target_passes']}  "
                  f"draft-passes {st['draft_passes']}")

        for ent in (1.0, 2.0, 3.0, 4.0):
            (out, st), t = timed(lambda e=ent: generate_speculative_adaptive(
                tgt, drf, torch.tensor([ids]), max_new_tokens=max_new,
                k_min=1, k_max=12, ent_stop=e))
            eq = assert_tokens_equal(ref, out)
            if not eq:
                raise AssertionError(f"adaptive e={ent} p{pi}: {eq.detail}")
            acc = st["accepted"] / max(st["drafted"], 1)
            print(f"adapt e={ent}: {new_ref / t:6.1f} tok/s  "
                  f"acc {acc:.2f}  tgt-passes {st['target_passes']}  "
                  f"draft-passes {st['draft_passes']}  "
                  f"stops {st['early_stops']}")
    print("\nbar: adaptive > best fixed-k tok/s at token-identical output")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=200)
    a = ap.parse_args()
    main(a.max_new)

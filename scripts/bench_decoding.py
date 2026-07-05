"""Benchmark vanilla greedy vs prompt-lookup vs speculative on real models.

Usage: python scripts/bench_decoding.py [--max-new 200]
Downloads Qwen2.5-0.5B + 1.5B on first run (~4 GB total).
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.prompt_lookup import generate_with_prompt_lookup
from llmopt.decoding.speculative import generate_speculative
from llmopt.eval.equivalence import assert_tokens_equal

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"

# summarization-style prompt: input-grounded output favors prompt-lookup
ARTICLE = (
    "The RTX 3080 uses the GA102 GPU with 8704 CUDA cores and 10 GB of GDDR6X "
    "memory on a 320-bit bus, giving 760 GB/s of memory bandwidth. The GA102 "
    "chip is fabricated on Samsung's 8 nm process. Memory bandwidth matters "
    "because autoregressive decoding is memory-bound: every generated token "
    "requires reading all model weights from memory. Speculative decoding "
    "amortizes those weight reads across several tokens per forward pass."
)
PROMPT = f"Summarize the following in two sentences:\n\n{ARTICLE}\n\nSummary:"


def vanilla_greedy(model, prompt_ids: list[int], max_new: int) -> tuple[list[int], int]:
    from llmopt.decoding.kv import valid_len

    tokens = list(prompt_ids)
    past = None
    with torch.inference_mode():
        for _ in range(max_new):
            fed = tokens[valid_len(past) :]
            out = model(
                input_ids=torch.tensor([fed], device=model.device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            tokens.append(int(out.logits[0, -1].argmax()))
    return tokens, max_new


def timed(fn, warmup: int = 1, repeats: int = 3):
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return result, sorted(times)[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float16).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float16).cuda().eval()

    prompt_ids = tok(PROMPT).input_ids
    n = args.max_new
    print(f"prompt {len(prompt_ids)} tokens, generating {n}\n")

    (ref, _), t_van = timed(lambda: vanilla_greedy(target, prompt_ids, n))
    print(f"vanilla greedy      : {n / t_van:7.1f} tok/s  ({t_van:.2f}s)")

    (out_pl, st_pl), t_pl = timed(
        lambda: generate_with_prompt_lookup(
            target, torch.tensor([prompt_ids]), max_new_tokens=n, num_draft=10
        )
    )
    eq = assert_tokens_equal(ref, out_pl)
    print(
        f"prompt-lookup       : {n / t_pl:7.1f} tok/s  ({t_pl:.2f}s)  "
        f"{st_pl['forward_passes']} passes, "
        f"accept {st_pl['accepted']}/{st_pl['drafted']}  "
        f"equiv={'OK' if eq else 'FAIL: ' + eq.detail}"
    )

    (out_sp, st_sp), t_sp = timed(
        lambda: generate_speculative(
            target, draft, torch.tensor([prompt_ids]), max_new_tokens=n, num_draft=5
        )
    )
    eq = assert_tokens_equal(ref, out_sp)
    print(
        f"speculative (0.5B)  : {n / t_sp:7.1f} tok/s  ({t_sp:.2f}s)  "
        f"{st_sp['target_passes']} target passes, "
        f"accept {st_sp['accepted']}/{st_sp['drafted']}  "
        f"equiv={'OK' if eq else 'FAIL: ' + eq.detail}"
    )

    print("\noutput:", tok.decode(ref[len(prompt_ids) :])[:300])


if __name__ == "__main__":
    main()

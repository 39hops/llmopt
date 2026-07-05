"""Radix prefix KV reuse on a real model: TTFT with a shared long prefix.

The RAG / system-prompt scenario: many requests share a long prefix
(document + instructions) and differ only in a short question. Cold
request pays full prefill; warm requests should pay only the divergent
suffix. Measures per-request prefill wall time through BatchEngine with
and without a prefix cache.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.cache.prefix_reuse import split_payload
from llmopt.cache.radix import RadixCache
from llmopt.decoding.batching import BatchEngine

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_NEW = 20

DOCUMENT = (
    "You are a precise assistant. Answer using only the report below.\n\n"
    "=== Q3 infrastructure report ===\n"
    + "\n".join(
        f"Section {i}: cluster {i} ran at {50 + i % 40}% utilization, "
        f"served {1000 + 37 * i} requests per second at p99 latency "
        f"{5 + (i * 7) % 90} ms, and logged {(i * 13) % 17} incidents."
        for i in range(80)
    )
    + "\n=== end of report ===\n\n"
)
QUESTIONS = [
    "Question: which cluster had the highest p99 latency?",
    "Question: how many incidents did cluster 40 log?",
    "Question: summarize utilization across clusters in one sentence.",
    "Question: what throughput did cluster 12 serve?",
]


def prefill_times(model, prompts, prefix_cache):
    """Run prompts sequentially through the engine, timing each request's
    prefill (submit -> first generated token)."""
    times = []
    for prompt in prompts:
        engine = BatchEngine(model, max_batch=1, chunk_size=512,
                             prefix_cache=prefix_cache)
        engine.submit(prompt, max_new_tokens=N_NEW)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while not (engine.running and engine.running[0].generated):
            engine.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        engine.run()  # finish decode (keeps engines comparable)
    return times


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    prompts = [tok(DOCUMENT + q).input_ids for q in QUESTIONS]
    shared = len(tok(DOCUMENT).input_ids)
    print(f"prefix ~{shared} tokens shared, prompts "
          f"{[len(p) for p in prompts]} tokens, {N_NEW} new tokens each\n")

    # warm the GPU + weights once, then measure
    prefill_times(model, prompts[:1], None)

    cold = prefill_times(model, prompts, None)
    cache = RadixCache(split_payload=split_payload)
    warm = prefill_times(model, prompts, cache)

    print(f"{'request':<44} {'no cache':>10} {'radix reuse':>12}")
    for q, c, w in zip(QUESTIONS, cold, warm):
        print(f"{q[10:52]:<44} {c * 1e3:8.1f}ms {w * 1e3:10.1f}ms  ({c / w:4.1f}x)")
    rest = sum(cold[1:]) / sum(warm[1:])
    print(f"\nwarm requests (2..{len(QUESTIONS)}): {rest:.1f}x faster TTFT")


if __name__ == "__main__":
    main()

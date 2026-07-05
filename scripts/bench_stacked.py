"""Full-stack benchmark: radix prefix reuse + prompt-lookup + CUDA graphs.

RAG-shaped workload (shared long document, short questions), end-to-end
wall time per request. Ladder:

1. static cache, eager, greedy      — baseline
2. + lookup + CUDA graphs (cold)    — decode accelerated, full prefill
3. + radix prefix reuse (warm)      — prefill collapses to the suffix

Run inside vcvars64 (see bench_compile.py docstring).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.stacked import StackedEngine
from scripts.bench_prefix_reuse import DOCUMENT, QUESTIONS
from scripts.bench_static import greedy_static

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_NEW = 100


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    prompts = [tok(DOCUMENT + q).input_ids for q in QUESTIONS]
    print(f"prompts {[len(p) for p in prompts]} tokens, {N_NEW} new each\n")

    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    # build graphs + autotune once at the real shapes (a different cache
    # max_len re-specializes the compiled step), excluded from timings
    engine = StackedEngine(model, compiled_step=compiled)
    engine.generate(prompts[0], max_new_tokens=N_NEW)
    engine = StackedEngine(model, compiled_step=compiled)  # fresh radix

    print(f"{'request':<10} {'eager greedy':>13} {'stack cold/warm':>16} {'speedup':>8}")
    for i, p in enumerate(prompts):
        ref, t_eager = timed(lambda: greedy_static(model, p, N_NEW))
        (out, stats), t_stack = timed(
            lambda: engine.generate(p, max_new_tokens=N_NEW)
        )
        tag = "cold" if stats["prefix_hit_tokens"] == 0 else "warm"
        eq = "OK" if out == ref else "DIVERGED"
        print(f"request {i} {t_eager * 1e3:11.0f}ms {t_stack * 1e3:9.0f}ms {tag} "
              f"{t_eager / t_stack:7.1f}x  equiv={eq} "
              f"(hit={stats['prefix_hit_tokens']}, passes={stats['forward_passes']})")


if __name__ == "__main__":
    main()

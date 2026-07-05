"""Static KV cache + CUDA graphs benchmark.

Fixed shapes are what make CUDA graphs legal: decode always feeds exactly one
token with an explicit cache_position into a preallocated StaticCache, so the
compiled graph replays without recompilation or buffer hazards.

Run inside vcvars64 (see bench_compile.py docstring).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from scripts.bench_decoding import PROMPT

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 150


def greedy_static(model, prompt_ids: list[int], max_new: int, *, compiled_step=None):
    """Greedy decode with StaticCache: prefill once, then 1-token steps."""
    device = model.device
    total = len(prompt_ids) + max_new
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=total,
        device=device, dtype=model.dtype,
    )
    tokens = list(prompt_ids)
    with torch.inference_mode():
        # prefill (dynamic length, eager is fine -- happens once)
        pos = torch.arange(len(tokens), device=device)
        out = model(
            input_ids=torch.tensor([tokens], device=device),
            past_key_values=cache, cache_position=pos, use_cache=True,
        )
        nxt = int(out.logits[0, -1].argmax())
        tokens.append(nxt)
        # decode: fixed-shape single-token steps
        step = model if compiled_step is None else compiled_step
        for i in range(max_new - 1):
            cp = torch.tensor([len(tokens) - 1], device=device)
            out = step(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=cache, cache_position=cp, use_cache=True,
            )
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
    return tokens


def bench(fn, warmup=1, repeats=3):
    for _ in range(warmup):
        fn()
    ts = []
    r = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return r, sorted(ts)[len(ts) // 2]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT).input_ids

    ref, t0 = bench(lambda: greedy_static(m, ids, N))
    print(f"static eager        : {N / t0:7.1f} tok/s")

    compiled = torch.compile(m, mode="reduce-overhead", fullgraph=True)
    out, t1 = bench(lambda: greedy_static(m, ids, N, compiled_step=compiled), warmup=2)
    match = "OK" if out == ref else "DIVERGED"
    print(f"static + cudagraphs : {N / t1:7.1f} tok/s  equiv={match}")


if __name__ == "__main__":
    main()

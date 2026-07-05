"""Sweep prompt-lookup hyperparams (max_ngram x num_draft) on MLX.

Apple-silicon counterpart of sweep_lookup.py: same generic decode loop
(llmopt.decoding.lookup_generic) over MLXBackend. No fixed shapes or
graph capture needed — mlx-lm caches grow on demand and trim on rewind.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from llmopt.backends.mlx_backend import MLXBackend
from llmopt.decoding.lookup_generic import generate_lookup
from scripts.bench_decoding import PROMPT

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
N = 150
NGRAMS = (2, 3, 4)
DRAFTS = (5, 10, 16)


def greedy_reference(model, ids: list[int], n: int) -> list[int]:
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    out = [int(mx.argmax(logits[0, -1]))]
    for _ in range(n - 1):
        logits = model(mx.array([[out[-1]]]), cache=cache)
        out.append(int(mx.argmax(logits[0, -1])))
    return out


def main() -> None:
    model, tok = load(MODEL)
    ids = tok.encode(PROMPT)

    t0 = time.perf_counter()
    ref = greedy_reference(model, ids, N)
    t_ref = time.perf_counter() - t0
    print(f"greedy reference: {N / t_ref:.1f} tok/s")

    print(f"{'ngram':>5} {'draft':>5} {'tok/s':>7} {'passes':>6} {'accept':>9} equiv")
    for num_draft in DRAFTS:
        for max_ngram in NGRAMS:
            t0 = time.perf_counter()
            out, st = generate_lookup(
                MLXBackend(model), ids, max_new_tokens=N,
                num_draft=num_draft, max_ngram=max_ngram,
            )
            t = time.perf_counter() - t0
            eq = out[len(ids):] == ref
            print(
                f"{max_ngram:>5} {num_draft:>5} {N / t:>7.1f} "
                f"{st['forward_passes']:>6} "
                f"{st['accepted']:>4}/{st['drafted']:<4} "
                f"{'OK' if eq else 'DIVERGED'}"
            )


if __name__ == "__main__":
    main()

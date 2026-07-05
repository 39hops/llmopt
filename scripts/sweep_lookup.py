"""Sweep prompt-lookup hyperparams (max_ngram x num_draft) on the
StaticCache + CUDA graphs path.

Each distinct num_draft is a new block shape [1, num_draft+1], so
reduce-overhead captures one extra CUDA graph per shape (paid in warmup).

Run inside vcvars64 (see bench_compile.py docstring).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.lookup_static import generate_lookup_static
from llmopt.eval.equivalence import assert_tokens_equal
from scripts.bench_decoding import PROMPT
from scripts.bench_static import bench, greedy_static

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 150
NGRAMS = (2, 3, 4)
DRAFTS = (5, 10, 16)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT).input_ids

    ref, _ = bench(lambda: greedy_static(m, ids, N), warmup=0, repeats=1)
    compiled = torch.compile(m, mode="reduce-overhead", fullgraph=True)

    print(f"{'ngram':>5} {'draft':>5} {'tok/s':>7} {'passes':>6} {'accept':>9} equiv")
    for num_draft in DRAFTS:  # outer loop: one graph capture per shape
        for max_ngram in NGRAMS:
            (out, st), t = bench(
                lambda: generate_lookup_static(
                    m, ids, max_new_tokens=N, num_draft=num_draft,
                    max_ngram=max_ngram, compiled_step=compiled,
                ),
                warmup=2,
            )
            eq = assert_tokens_equal(ref, out)
            print(
                f"{max_ngram:>5} {num_draft:>5} {N / t:>7.1f} "
                f"{st['forward_passes']:>6} "
                f"{st['accepted']:>4}/{st['drafted']:<4} "
                f"{'OK' if eq else 'DIVERGED'}"
            )


if __name__ == "__main__":
    main()

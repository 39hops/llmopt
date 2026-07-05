"""Prompt-lookup + StaticCache + CUDA graphs: the stacked benchmark.

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


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT).input_ids

    ref, t0 = bench(lambda: greedy_static(m, ids, N))
    print(f"static eager vanilla    : {N / t0:7.1f} tok/s")

    compiled = torch.compile(m, mode="reduce-overhead", fullgraph=True)

    outv, t1 = bench(lambda: greedy_static(m, ids, N, compiled_step=compiled), warmup=2)
    print(f"vanilla + cudagraphs    : {N / t1:7.1f} tok/s  "
          f"equiv={'OK' if outv == ref else 'DIVERGED'}")

    (out, st), t2 = bench(
        lambda: generate_lookup_static(
            m, ids, max_new_tokens=N, num_draft=10, compiled_step=compiled
        ),
        warmup=2,
    )
    eq = assert_tokens_equal(ref, out)
    print(
        f"lookup + cudagraphs     : {N / t2:7.1f} tok/s  "
        f"passes={st['forward_passes']} accept={st['accepted']}/{st['drafted']}  "
        f"equiv={'OK' if eq else 'near-tie? ' + eq.detail[:80]}"
    )


if __name__ == "__main__":
    main()

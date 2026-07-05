"""torch.compile impact benchmark: eager vs compiled vanilla vs compiled+lookup.

Run from a vcvars64 environment on Windows (inductor's cpp wrapper needs MSVC):
  cmd /c "call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat\" && python scripts/bench_compile.py"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.prompt_lookup import generate_with_prompt_lookup
from scripts.bench_decoding import PROMPT, timed, vanilla_greedy

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 150


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT).input_ids

    (ref, _), t0 = timed(lambda: vanilla_greedy(m, ids, N), warmup=1, repeats=3)
    print(f"eager vanilla       : {N / t0:7.1f} tok/s")

    # reduce-overhead (CUDA graphs) clashes with HF cache buffer reuse:
    # "accessing tensor output of CUDAGraphs that has been overwritten"
    m.forward = torch.compile(m.forward, dynamic=True)
    (ref2, _), t1 = timed(lambda: vanilla_greedy(m, ids, N), warmup=2, repeats=3)
    print(f"compiled vanilla    : {N / t1:7.1f} tok/s")

    (o, s), t2 = timed(
        lambda: generate_with_prompt_lookup(
            m, torch.tensor([ids]), max_new_tokens=N, num_draft=10
        ),
        warmup=2,
        repeats=3,
    )
    print(f"compiled + lookup   : {N / t2:7.1f} tok/s  passes={s['forward_passes']}")


if __name__ == "__main__":
    main()

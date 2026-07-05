"""Tree verify vs linear prompt-lookup benchmark.

Same prompt/model as bench_decoding.py so numbers are comparable: does
verifying several lookup candidates as a tree (one pass, 4D mask) beat
verifying the single best candidate linearly? Both must stay
token-identical to vanilla greedy.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.decoding.prompt_lookup import generate_with_prompt_lookup
from llmopt.decoding.tree_verify import generate_lookup_tree
from scripts.bench_decoding import PROMPT, timed, vanilla_greedy

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N = 150
NUM_DRAFT = 10


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT).input_ids

    (ref, _), t0 = timed(lambda: vanilla_greedy(m, ids, N))
    print(f"vanilla greedy               : {N / t0:6.1f} tok/s  (1.0x)")

    (out, stats), t1 = timed(
        lambda: generate_with_prompt_lookup(
            m, ids, max_new_tokens=N, num_draft=NUM_DRAFT
        )
    )
    eq = "OK" if out == ref else "DIVERGED"
    acc = stats["accepted"] / max(stats["drafted"], 1)
    print(
        f"linear lookup                : {N / t1:6.1f} tok/s  ({t0 / t1:.1f}x)"
        f"  passes={stats['forward_passes']} accept={acc:.0%} equiv={eq}"
    )

    for nc in (2, 4, 8):
        (out, stats), t2 = timed(
            lambda: generate_lookup_tree(
                m, ids, max_new_tokens=N, num_draft=NUM_DRAFT, num_candidates=nc
            )
        )
        eq = "OK" if out == ref else "DIVERGED"
        acc = stats["accepted"] / max(stats["drafted"], 1)
        print(
            f"tree verify (candidates={nc})  : {N / t2:6.1f} tok/s  ({t0 / t2:.1f}x)"
            f"  passes={stats['forward_passes']} accept={acc:.0%} equiv={eq}"
        )


if __name__ == "__main__":
    main()

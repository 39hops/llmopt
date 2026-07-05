"""Prompt-lookup decoding on a StaticCache with fixed-shape verify blocks,
CUDA-graph compatible.

Thin torch-specific wrapper: the algorithm lives in
llmopt.decoding.lookup_generic and the torch/StaticCache mechanics in
llmopt.backends.torch_static. See those modules for the correctness
argument (pad masking, cache rewind, fixed shapes).
"""

from __future__ import annotations

from llmopt.backends.torch_static import TorchStaticBackend
from llmopt.decoding.lookup_generic import generate_lookup


def generate_lookup_static(
    model,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 128,
    num_draft: int = 10,
    max_ngram: int = 3,
    compiled_step=None,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
):
    """Greedy prompt-lookup decoding with StaticCache + fixed verify blocks.

    compiled_step: optionally torch.compile(model, mode="reduce-overhead");
    prefill always runs the eager model (dynamic length, happens once).
    Returns (tokens, stats).
    """
    backend = TorchStaticBackend(model, compiled_step=compiled_step)
    return generate_lookup(
        backend, prompt_ids,
        max_new_tokens=max_new_tokens, num_draft=num_draft,
        max_ngram=max_ngram, pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

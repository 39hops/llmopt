"""Backend-agnostic prompt-lookup decoding with fixed verify blocks.

Same algorithm as lookup_static.generate_lookup_static, but all model /
cache / framework details live behind a DecodeBackend (llmopt.backends).
Only plain Python ints cross the boundary, so torch (CUDA graphs) and
MLX backends share this loop unchanged.
"""

from __future__ import annotations

from typing import Sequence

from llmopt.backends.base import DecodeBackend
from llmopt.decoding.prompt_lookup import find_ngram_continuation


def generate_lookup(
    backend: DecodeBackend,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int = 128,
    num_draft: int = 10,
    max_ngram: int = 3,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
    draft_fn=None,
):
    """Greedy prompt-lookup decoding over any DecodeBackend.

    ``draft_fn(tokens, num_draft) -> list[int]`` overrides the draft
    source (e.g. a retrieval datastore, see decoding/datastore.py);
    default is n-gram lookup within the context itself.

    Returns (tokens, stats).
    """
    block = num_draft + 1
    total = len(prompt_ids) + max_new_tokens + block  # slack for last block
    tokens = list(prompt_ids)
    stats = {
        "drafted": 0, "accepted": 0, "forward_passes": 0,
        "prompt_len": len(prompt_ids),
    }

    nxt = backend.begin(tokens, total)
    stats["forward_passes"] += 1
    tokens.append(nxt)
    produced = 1

    while produced < max_new_tokens:
        if draft_fn is not None:
            draft = draft_fn(tokens, num_draft)[:num_draft]
        else:
            draft = find_ngram_continuation(
                tokens, max_ngram=max_ngram, num_draft=num_draft
            )[:num_draft]
        stats["drafted"] += len(draft)
        n_real = 1 + len(draft)  # last accepted token + draft
        fed = [tokens[-1]] + draft + [pad_token_id] * (block - n_real)
        preds = backend.step_argmax(fed, len(tokens) - 1, n_real)
        stats["forward_passes"] += 1

        accepted = 0
        for j, d in enumerate(draft):
            if preds[j] == d:
                accepted += 1
            else:
                break
        stats["accepted"] += accepted
        new = (draft[:accepted] + [preds[accepted]])[: max_new_tokens - produced]
        tokens.extend(new)
        produced += len(new)
        # cache holds writes for the whole block (incl. rejected + pads);
        # only positions < len(tokens)-1 are valid (newest token was output,
        # never fed). Rewind the write pointer there.
        backend.rewind(len(tokens) - 1)
        if eos_token_id is not None and eos_token_id in new:
            idx = tokens.index(eos_token_id, len(tokens) - len(new))
            tokens = tokens[: idx + 1]
            break

    return tokens, stats

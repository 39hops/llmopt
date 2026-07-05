"""Prompt-lookup (n-gram) decoding: draft tokens by copying from the prompt.

Pure-Python matcher (no torch) plus a generate loop that verifies drafts
with a single target-model forward pass per step.
"""

from __future__ import annotations

from typing import Sequence


def find_ngram_continuation(
    context: Sequence[int],
    *,
    max_ngram: int = 3,
    min_ngram: int = 1,
    num_draft: int = 10,
) -> list[int]:
    """Find longest suffix n-gram of `context` that occurred earlier in it;
    return up to num_draft tokens that followed that earlier occurrence.

    Longest n first, and among equal n the most recent match wins
    (recent repetition is the best predictor of continuation).
    """
    ctx = list(context)
    n_total = len(ctx)
    for n in range(min(max_ngram, n_total - 1), min_ngram - 1, -1):
        suffix = ctx[n_total - n :]
        # scan right-to-left, excluding the suffix occurrence itself
        for start in range(n_total - n - 1, -1, -1):
            if ctx[start : start + n] == suffix:
                cont = ctx[start + n : start + n + num_draft]
                if cont:
                    return cont
    return []


def generate_with_prompt_lookup(
    model,
    input_ids,
    *,
    max_new_tokens: int = 128,
    num_draft: int = 10,
    max_ngram: int = 3,
    eos_token_id: int | None = None,
):
    """Greedy decoding with prompt-lookup drafts.

    Returns (tokens, stats) where stats has drafted/accepted counts and
    number of model forward passes (vs max_new_tokens for vanilla greedy).
    Correctness invariant: output is identical to vanilla greedy decoding.
    """
    import torch

    device = next(model.parameters()).device
    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {"drafted": 0, "accepted": 0, "forward_passes": 0}
    produced = 0

    with torch.inference_mode():
        while produced < max_new_tokens:
            draft = find_ngram_continuation(
                tokens, max_ngram=max_ngram, num_draft=num_draft
            )
            draft = draft[: max_new_tokens - produced - 1]
            stats["drafted"] += len(draft)

            # one forward over context + draft verifies all draft tokens
            ids = torch.tensor([tokens + draft], device=device)
            logits = model(input_ids=ids).logits[0]
            stats["forward_passes"] += 1
            preds = logits.argmax(-1).tolist()  # preds[t] = greedy next after pos t

            base = len(tokens) - 1
            accepted = 0
            for j, d in enumerate(draft):
                if preds[base + j] == d:
                    accepted += 1
                else:
                    break
            stats["accepted"] += accepted
            # accepted draft tokens + one free token from the verify pass
            new = draft[:accepted] + [preds[base + accepted]]
            tokens.extend(new)
            produced += len(new)
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

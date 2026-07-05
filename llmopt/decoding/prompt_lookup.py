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

    from llmopt.decoding.kv import crop, to_legacy, valid_len

    device = next(model.parameters()).device
    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {"drafted": 0, "accepted": 0, "forward_passes": 0, "prompt_len": len(tokens)}
    produced = 0
    past = None  # legacy KV tuple covering tokens[:valid_len(past)]

    with torch.inference_mode():
        while produced < max_new_tokens:
            draft = find_ngram_continuation(
                tokens, max_ngram=max_ngram, num_draft=num_draft
            )
            draft = draft[: max_new_tokens - produced - 1]
            stats["drafted"] += len(draft)

            # one forward over uncached tokens + draft verifies all drafts
            cached = valid_len(past)
            fed = tokens[cached:] + draft
            ids = torch.tensor([fed], device=device)
            out = model(input_ids=ids, past_key_values=past, use_cache=True)
            stats["forward_passes"] += 1
            past = to_legacy(out.past_key_values)
            preds = out.logits[0].argmax(-1).tolist()  # next-token per fed position

            # fed position i sits at absolute position cached + i;
            # prediction after tokens[-1] is at fed index len(tokens)-1-cached
            base = len(tokens) - 1 - cached
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
            # drop rejected draft positions from the cache; the bonus token
            # was never fed, so cache covers len(tokens) - 1 positions
            past = crop(past, len(tokens) - 1)
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

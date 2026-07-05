"""Prompt-lookup decoding on a StaticCache with fixed-shape verify blocks,
CUDA-graph compatible.

Every decode step feeds exactly `block` tokens: [last_token, draft..., pads].
Fixed shape means torch.compile(mode="reduce-overhead") captures one CUDA
graph and replays it for every step. Correctness relies on three properties:

1. Pads sit at the highest positions in the block; causal masking means no
   real query position ever attends a pad's K/V.
2. Pad logits are simply ignored.
3. After acceptance, the cache write pointer rewinds to the true sequence
   length, so the next block overwrites any stale (rejected/pad) slots.
   Slots beyond the current length are future positions -- causally masked
   (via the model-level cache_position we pass) until rewritten.

transformers 5.x StaticCache layers ignore the passed cache_position and
append at an internal `cumulative_length` counter, so rewinding means
resetting that counter tensor in-place (`fill_`), which is CUDA-graph-safe
because the tensor has a static address (mark_static_address in HF).

Output is greedy-equivalent modulo fp16 near-tie argmax flips (see README).
"""

from __future__ import annotations

from llmopt.decoding.prompt_lookup import find_ngram_continuation


def _rewind(cache, length: int) -> None:
    """Reset every layer's write pointer to `length` (in-place, graph-safe)."""
    for layer in cache.layers:
        cl = getattr(layer, "cumulative_length", None)
        if cl is None:
            continue
        if hasattr(cl, "fill_"):
            cl.fill_(length)
        else:  # plain int in some versions
            layer.cumulative_length = length


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
    import torch
    from transformers import StaticCache

    device = model.device
    block = num_draft + 1
    total = len(prompt_ids) + max_new_tokens + block  # slack for last block
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=total,
        device=device, dtype=model.dtype,
    )
    step = model if compiled_step is None else compiled_step

    tokens = list(prompt_ids)
    stats = {
        "drafted": 0, "accepted": 0, "forward_passes": 0,
        "prompt_len": len(prompt_ids),
    }
    produced = 0

    with torch.inference_mode():
        # prefill (eager, once)
        pos = torch.arange(len(tokens), device=device)
        out = model(
            input_ids=torch.tensor([tokens], device=device),
            past_key_values=cache, cache_position=pos, use_cache=True,
        )
        stats["forward_passes"] += 1
        nxt = int(out.logits[0, -1].argmax())
        tokens.append(nxt)
        produced += 1

        while produced < max_new_tokens:
            draft = find_ngram_continuation(
                tokens, max_ngram=max_ngram, num_draft=num_draft
            )
            draft = draft[: num_draft]
            stats["drafted"] += len(draft)
            n_real = 1 + len(draft)  # last accepted token + draft
            fed = [tokens[-1]] + draft + [pad_token_id] * (block - n_real)
            start = len(tokens) - 1  # absolute position of fed[0]
            cp = torch.arange(start, start + block, device=device)
            out = step(
                input_ids=torch.tensor([fed], device=device),
                past_key_values=cache, cache_position=cp, use_cache=True,
            )
            stats["forward_passes"] += 1
            # preds[j] = greedy next-token after fed[j]; pads ignored
            preds = out.logits[0, :n_real].argmax(-1).tolist()

            accepted = 0
            for j, d in enumerate(draft):
                if preds[j] == d:
                    accepted += 1
                else:
                    break
            stats["accepted"] += accepted
            new = draft[:accepted] + [preds[accepted]]
            new = new[: max_new_tokens - produced]
            tokens.extend(new)
            produced += len(new)
            # cache now holds writes for the whole block (incl. rejected +
            # pads); only positions < len(tokens)-1 are valid (the newest
            # token was output, never fed). Rewind the write pointer there.
            _rewind(cache, len(tokens) - 1)
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

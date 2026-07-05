"""LLMLingua-style prompt compression: shrink the prompt itself.

Prefill is compute-bound at length (see eval/roofline), so dropping
tokens is a direct cost cut — and attention is robust to losing
low-information tokens. A small scorer LM measures each token's
self-information -log2 p(token | prefix); tokens the LM finds
predictable carry little signal and are dropped first. Spans the caller
marks as protected (instructions, the question) always survive.

This is the coarse token-level LLMLingua idea, not the full iterative
budget controller; gist tokens (learned compression) stay on the
roadmap.
"""

from __future__ import annotations

from typing import Sequence


def token_self_information(model, ids: Sequence[int]) -> list[float]:
    """-log2 p(ids[i] | ids[:i]) per token, position 0 gets +inf (never
    predictable, never dropped)."""
    import math

    import torch

    with torch.inference_mode():
        logits = model(
            input_ids=torch.tensor([list(ids)], device=model.device)
        ).logits[0]
        logp = torch.log_softmax(logits[:-1], dim=-1)
        info = [-float(logp[i, ids[i + 1]]) / math.log(2) for i in range(len(ids) - 1)]
    return [float("inf")] + info


def compress_prompt(
    model,
    ids: Sequence[int],
    *,
    ratio: float = 0.5,
    protect: Sequence[tuple[int, int]] = (),
) -> tuple[list[int], dict]:
    """Keep the ceil(ratio * len) most informative tokens, in original
    order. ``protect`` spans [start, end) are always kept and don't
    consume budget priority (they're forced in first).

    Returns (compressed_ids, stats).
    """
    import math

    ids = list(ids)
    n = len(ids)
    budget = max(1, math.ceil(ratio * n))
    protected = set()
    for s, e in protect:
        protected |= set(range(s, min(e, n)))

    info = token_self_information(model, ids)
    keep = set(protected)
    for pos in sorted(range(n), key=lambda i: -info[i]):
        if len(keep) >= max(budget, len(protected)):
            break
        keep.add(pos)
    kept = sorted(keep)
    stats = {
        "original_len": n,
        "compressed_len": len(kept),
        "ratio": len(kept) / n,
        "dropped_mean_bits": (
            sum(info[i] for i in range(n) if i not in keep)
            / max(n - len(kept), 1)
        ),
        "kept_mean_bits": sum(info[i] for i in kept if info[i] != float("inf"))
        / max(len(kept) - 1, 1),
    }
    return [ids[i] for i in kept], stats

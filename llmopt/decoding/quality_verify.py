"""Quality-verified decoding: accept drafts by score, not bit-exactness.

Exact speculative decoding rejects a draft token the moment it differs
from the target argmax — even when the target thinks it's a perfectly
good token. Quality verification relaxes the criterion:

- "top_k": accept if the draft token is within the target's top-k.
- "logprob_margin": accept if target logprob of the draft token is
  within ``margin`` nats of the target's best token.

Output is no longer bit-identical to target greedy — it's a controlled
quality trade: every accepted token provably satisfies the criterion
under the *target* model, and the returned stats report the target
logprob of the emitted sequence so the trade is measurable (compare
against greedy with the same target; see eval/).
"""

from __future__ import annotations


def generate_quality_verify(
    target_model,
    draft_model,
    input_ids,
    *,
    max_new_tokens: int = 128,
    num_draft: int = 5,
    criterion: str = "top_k",
    top_k: int = 3,
    margin: float = 1.0,
    eos_token_id: int | None = None,
):
    """Draft with draft_model (greedy), verify by score. Returns
    (tokens, stats) — stats include target_logprob of emitted tokens."""
    import torch

    from llmopt.decoding.kv import crop, valid_len

    device = next(target_model.parameters()).device
    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {
        "drafted": 0, "accepted": 0, "target_passes": 0, "draft_passes": 0,
        "target_logprob": 0.0, "prompt_len": len(tokens),
    }
    t_past = d_past = None
    produced = 0

    def ok(logp_row, tok: int) -> bool:
        if criterion == "top_k":
            return tok in torch.topk(logp_row, top_k).indices
        if criterion == "logprob_margin":
            return float(logp_row.max() - logp_row[tok]) <= margin
        raise ValueError(criterion)

    with torch.inference_mode():
        while produced < max_new_tokens:
            k = min(num_draft, max_new_tokens - produced - 1)
            # draft k greedy tokens
            d_ctx = tokens[:]
            draft: list[int] = []
            for _ in range(k):
                fed = d_ctx[valid_len(d_past) :]
                out = draft_model(
                    input_ids=torch.tensor([fed], device=device),
                    past_key_values=d_past, use_cache=True,
                )
                d_past = out.past_key_values
                stats["draft_passes"] += 1
                nxt = int(out.logits[0, -1].argmax())
                draft.append(nxt)
                d_ctx.append(nxt)
            stats["drafted"] += len(draft)

            # one target pass scores everything
            cached = valid_len(t_past)
            fed = tokens[cached:] + draft
            out = target_model(
                input_ids=torch.tensor([fed], device=device),
                past_key_values=t_past, use_cache=True,
            )
            t_past = out.past_key_values
            stats["target_passes"] += 1
            logp = torch.log_softmax(out.logits[0], dim=-1)
            base = len(tokens) - 1 - cached

            accepted = 0
            for j, d in enumerate(draft):
                if ok(logp[base + j], d):
                    accepted += 1
                else:
                    break
            stats["accepted"] += accepted
            # on rejection (or draft exhausted): fall back to target argmax
            bonus = int(logp[base + accepted].argmax())
            new = draft[:accepted] + [bonus]
            for j, tok in enumerate(new):
                stats["target_logprob"] += float(logp[base + j, tok])
            tokens.extend(new)
            produced += len(new)
            t_past = crop(t_past, len(tokens) - 1)
            d_past = crop(d_past, min(valid_len(d_past), len(tokens) - 1))
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

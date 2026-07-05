"""Sampler-aware speculative verify: rejection sampling under a
*filtered* target distribution.

Standard speculative sampling preserves the raw target distribution.
But deployments sample from a processed one (temperature, top-k/p,
min-p...), and verifying against raw logits while serving filtered
samples silently changes the output distribution. Fix: apply the same
processor pipeline to both models' logits, then run the Leviathan
rejection scheme on the *filtered* probabilities — the emitted tokens
are then distributed exactly as direct filtered sampling from the
target.

Processors are the (logits, ctx) -> logits callables from
decoding/samplers.py.
"""

from __future__ import annotations


def generate_speculative_filtered(
    target_model,
    draft_model,
    input_ids,
    *,
    processors=(),
    max_new_tokens: int = 128,
    num_draft: int = 5,
    seed: int | None = None,
    eos_token_id: int | None = None,
):
    """Returns (tokens, stats). Output distribution == direct sampling
    from the processor-filtered target."""
    import torch

    from llmopt.decoding.kv import crop, valid_len

    device = next(target_model.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    def filt(logits, ctx):
        for proc in processors:
            logits = proc(logits, ctx)
        p = torch.softmax(logits, dim=-1)
        return torch.nan_to_num(p, nan=0.0)  # all--inf rows can't happen w/ sane pipelines

    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {
        "drafted": 0, "accepted": 0, "target_passes": 0, "draft_passes": 0,
        "prompt_len": len(tokens),
    }
    t_past = d_past = None
    produced = 0

    with torch.inference_mode():
        while produced < max_new_tokens:
            k = min(num_draft, max_new_tokens - produced - 1)
            d_ctx = tokens[:]
            draft, draft_probs = [], []
            for _ in range(k):
                fed = d_ctx[valid_len(d_past) :]
                out = draft_model(
                    input_ids=torch.tensor([fed], device=device),
                    past_key_values=d_past, use_cache=True,
                )
                d_past = out.past_key_values
                stats["draft_passes"] += 1
                q = filt(out.logits[0, -1], d_ctx)
                nxt = int(torch.multinomial(q, 1, generator=gen))
                draft_probs.append(q)
                draft.append(nxt)
                d_ctx.append(nxt)
            stats["drafted"] += len(draft)

            cached = valid_len(t_past)
            fed = tokens[cached:] + draft
            out = target_model(
                input_ids=torch.tensor([fed], device=device),
                past_key_values=t_past, use_cache=True,
            )
            t_past = out.past_key_values
            stats["target_passes"] += 1
            base = len(tokens) - 1 - cached

            accepted, new = 0, []
            for j, d in enumerate(draft):
                ctx_j = tokens + draft[:j]
                p = filt(out.logits[0, base + j], ctx_j)
                q = draft_probs[j]
                r = torch.rand((), device=device, generator=gen)
                if q[d] > 0 and r < (p[d] / q[d]).clamp(max=1.0):
                    accepted += 1
                else:
                    resid = (p - q).clamp(min=0)
                    if float(resid.sum()) <= 0:
                        resid = p
                    resid = resid / resid.sum()
                    new = draft[:accepted] + [
                        int(torch.multinomial(resid, 1, generator=gen))
                    ]
                    break
            else:
                p = filt(out.logits[0, base + len(draft)], tokens + draft)
                new = draft + [int(torch.multinomial(p, 1, generator=gen))]

            stats["accepted"] += accepted
            tokens.extend(new)
            produced += len(new)
            t_past = crop(t_past, len(tokens) - 1)
            d_past = crop(d_past, min(valid_len(d_past), len(tokens) - 1))
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

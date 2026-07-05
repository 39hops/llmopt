"""Speculative decoding: small draft model proposes, target verifies.

Greedy variant (deterministic, output identical to target-only greedy) and
sampling variant using the standard rejection scheme (Leviathan et al. 2023),
which preserves the target distribution exactly.

Both draft and target maintain KV caches across steps; rejected draft
positions are cropped after each verify pass.
"""

from __future__ import annotations


def generate_speculative(
    target_model,
    draft_model,
    input_ids,
    *,
    max_new_tokens: int = 128,
    num_draft: int = 5,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    seed: int | None = None,
):
    """Returns (tokens, stats). temperature=0 -> greedy verify (exact match
    with target greedy). temperature>0 -> rejection sampling (exact target
    distribution in expectation).
    """
    import torch

    from llmopt.decoding.kv import crop, to_legacy, valid_len

    device = next(target_model.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {
        "drafted": 0,
        "accepted": 0,
        "target_passes": 0,
        "draft_passes": 0,
        "prompt_len": len(tokens),
    }
    produced = 0
    t_past = None
    d_past = None

    def step(model, past, fed_tokens):
        ids = torch.tensor([fed_tokens], device=device)
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        return out.logits[0], to_legacy(out.past_key_values)

    with torch.inference_mode():
        while produced < max_new_tokens:
            k = min(num_draft, max_new_tokens - produced - 1)

            # draft k tokens autoregressively, reusing draft KV
            draft_tokens: list[int] = []
            draft_probs = []
            d_ctx = tokens[:]
            for _ in range(k):
                d_logits, d_past = step(
                    draft_model, d_past, d_ctx[valid_len(d_past) :]
                )
                stats["draft_passes"] += 1
                d_last = d_logits[-1]
                if temperature == 0.0:
                    nxt = int(d_last.argmax())
                else:
                    p = torch.softmax(d_last / temperature, dim=-1)
                    draft_probs.append(p)
                    nxt = int(torch.multinomial(p, 1, generator=gen))
                draft_tokens.append(nxt)
                d_ctx.append(nxt)
            stats["drafted"] += len(draft_tokens)

            # one target pass over uncached context + draft verifies all
            t_cached = valid_len(t_past)
            fed = tokens[t_cached:] + draft_tokens
            t_logits, t_past = step(target_model, t_past, fed)
            stats["target_passes"] += 1
            base = len(tokens) - 1 - t_cached  # fed index predicting draft[0]

            accepted = 0
            new: list[int] = []
            if temperature == 0.0:
                preds = t_logits.argmax(-1)
                for j, d in enumerate(draft_tokens):
                    if int(preds[base + j]) == d:
                        accepted += 1
                    else:
                        break
                new = draft_tokens[:accepted] + [int(preds[base + accepted])]
            else:
                for j, d in enumerate(draft_tokens):
                    q = draft_probs[j]
                    p = torch.softmax(t_logits[base + j] / temperature, dim=-1)
                    r = torch.rand((), device=device, generator=gen)
                    if r < (p[d] / q[d]).clamp(max=1.0):
                        accepted += 1
                    else:
                        # resample from residual max(p - q, 0)
                        resid = (p - q).clamp(min=0)
                        resid = resid / resid.sum()
                        new = draft_tokens[:accepted] + [
                            int(torch.multinomial(resid, 1, generator=gen))
                        ]
                        break
                else:
                    p = torch.softmax(
                        t_logits[base + len(draft_tokens)] / temperature, dim=-1
                    )
                    new = draft_tokens + [int(torch.multinomial(p, 1, generator=gen))]

            stats["accepted"] += accepted
            tokens.extend(new)
            produced += len(new)
            # caches cover context + all draft tokens; keep only what is now
            # part of `tokens`, minus the never-fed bonus token
            t_past = crop(t_past, len(tokens) - 1)
            d_past = crop(d_past, min(valid_len(d_past), len(tokens) - 1))
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

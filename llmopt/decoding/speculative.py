"""Speculative decoding: small draft model proposes, target verifies.

Greedy variant (deterministic, output identical to target-only greedy) and
sampling variant using the standard rejection scheme (Leviathan et al. 2023),
which preserves the target distribution exactly.
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

    Simple recompute-KV implementation: clear and correct, not peak-speed.
    KV-cache reuse is the radix_cache module's job.
    """
    import torch

    device = next(target_model.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {"drafted": 0, "accepted": 0, "target_passes": 0, "draft_passes": 0}
    produced = 0

    def probs_from(logits):
        if temperature == 0.0:
            return None
        return torch.softmax(logits / temperature, dim=-1)

    with torch.inference_mode():
        while produced < max_new_tokens:
            k = min(num_draft, max_new_tokens - produced - 1)

            # draft k tokens autoregressively (small model, cheap)
            draft_tokens: list[int] = []
            draft_probs = []
            d_ctx = tokens[:]
            for _ in range(k):
                d_logits = draft_model(
                    input_ids=torch.tensor([d_ctx], device=device)
                ).logits[0, -1]
                stats["draft_passes"] += 1
                if temperature == 0.0:
                    nxt = int(d_logits.argmax())
                else:
                    p = torch.softmax(d_logits / temperature, dim=-1)
                    draft_probs.append(p)
                    nxt = int(torch.multinomial(p, 1, generator=gen))
                draft_tokens.append(nxt)
                d_ctx.append(nxt)
            stats["drafted"] += len(draft_tokens)

            # one target pass verifies all draft positions
            ids = torch.tensor([tokens + draft_tokens], device=device)
            t_logits = target_model(input_ids=ids).logits[0]
            stats["target_passes"] += 1
            base = len(tokens) - 1

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
                    p = probs_from(t_logits[base + j])
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
                    p = probs_from(t_logits[base + len(draft_tokens)])
                    new = draft_tokens + [int(torch.multinomial(p, 1, generator=gen))]

            stats["accepted"] += accepted
            tokens.extend(new)
            produced += len(new)
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

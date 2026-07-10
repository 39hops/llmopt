"""Entropy-adaptive speculative decoding (2026-07-10, banked crossover).

The derivation-engine gate law ("imitation can't beat the teacher, but
it makes the teacher ~4x cheaper") is speculative decoding's thesis;
this ports the gate's deference refinement: the draft stops proposing
when its OWN next-token entropy spikes ("I don't recognize this
state"), and drafts deep while confident. Fixed-k wastes draft passes
after the first likely rejection and caps acceptance runs on easy
spans; entropy is free (logits already in hand).

Greedy-only (temperature 0): output must stay token-identical to
target-only greedy — same oracle as every decode in this repo.
"""

from __future__ import annotations


def generate_speculative_adaptive(
    target_model,
    draft_model,
    input_ids,
    *,
    max_new_tokens: int = 128,
    k_min: int = 1,
    k_max: int = 12,
    ent_stop: float = 2.5,
    eos_token_id: int | None = None,
):
    """Returns (tokens, stats). Drafts up to k_max tokens, stopping
    early once >= k_min are drafted and the draft's next-token entropy
    (bits) exceeds ent_stop. Verify pass identical to
    generate_speculative's greedy arm."""
    import torch

    from llmopt.decoding.kv import crop, to_legacy, valid_len

    device = next(target_model.parameters()).device
    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {
        "drafted": 0,
        "accepted": 0,
        "target_passes": 0,
        "draft_passes": 0,
        "early_stops": 0,
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
            cap = min(k_max, max_new_tokens - produced - 1)

            draft_tokens: list[int] = []
            d_ctx = tokens[:]
            for _ in range(cap):
                d_logits, d_past = step(
                    draft_model, d_past, d_ctx[valid_len(d_past):]
                )
                stats["draft_passes"] += 1
                d_last = d_logits[-1]
                # float32: in fp16 clamp_min(1e-9) underflows to 0 ->
                # log2(0) = -inf -> 0*-inf = nan -> the stop test is
                # never true (measured: 0 stops in 771 passes)
                p = torch.softmax(d_last.float(), dim=-1)
                ent = float(-(p * p.clamp_min(1e-9).log2()).sum())
                if ent > ent_stop and len(draft_tokens) >= k_min:
                    stats["early_stops"] += 1
                    break
                nxt = int(d_last.argmax())
                draft_tokens.append(nxt)
                d_ctx.append(nxt)
            stats["drafted"] += len(draft_tokens)

            t_cached = valid_len(t_past)
            fed = tokens[t_cached:] + draft_tokens
            t_logits, t_past = step(target_model, t_past, fed)
            stats["target_passes"] += 1
            base = len(tokens) - 1 - t_cached

            accepted = 0
            preds = t_logits.argmax(-1)
            for j, d in enumerate(draft_tokens):
                if int(preds[base + j]) == d:
                    accepted += 1
                else:
                    break
            new = draft_tokens[:accepted] + [int(preds[base + accepted])]

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

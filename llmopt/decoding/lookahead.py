"""Lookahead (Jacobi) decoding: fixed-point iteration on the sequence.

Greedy decoding solves x[t+1] = argmax p(.|x[:t+1]) sequentially. Jacobi
iteration instead guesses a whole window of future tokens and refines
all of them in one forward pass: feed [last, g1..gW], read the argmax
prediction at every position. Wherever the guess already equals the
prediction, the fixed point is reached and the token is *provably* the
greedy token (its entire prefix was correct) — accept it. The rest of
the predictions become the next iteration's guess, shifted into place.

No draft model, no n-gram source needed: on loops and repetitive spans
the iteration converges several tokens per pass; worst case it accepts
one token per pass like vanilla greedy (plus window overhead).
Output is token-identical to greedy decoding.
"""

from __future__ import annotations


def generate_lookahead(
    model,
    input_ids,
    *,
    max_new_tokens: int = 128,
    window: int = 8,
    eos_token_id: int | None = None,
):
    """Greedy Jacobi decoding. Returns (tokens, stats)."""
    import torch

    from llmopt.decoding.kv import crop, valid_len

    device = next(model.parameters()).device
    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {
        "forward_passes": 0, "accepted_extra": 0, "prompt_len": len(tokens),
    }

    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([tokens], device=device), use_cache=True
        )
        past = out.past_key_values
        stats["forward_passes"] += 1
        tokens.append(int(out.logits[0, -1].argmax()))
        produced = 1
        guess = [0] * window  # arbitrary init; correctness never depends on it

        while produced < max_new_tokens:
            w = min(window, max_new_tokens - produced - 1)
            fed = [tokens[-1]] + guess[:w]
            out = model(
                input_ids=torch.tensor([fed], device=device),
                past_key_values=past, use_cache=True,
            )
            past = out.past_key_values
            stats["forward_passes"] += 1
            preds = out.logits[0].argmax(-1).tolist()

            # preds[0] follows tokens[-1]: always the true greedy token.
            # guess[j] is confirmed iff it equals preds[j] (its prefix
            # then being fully greedy, by induction).
            new = [preds[0]]
            for j in range(w):
                if guess[j] == preds[j]:
                    new.append(preds[j + 1])
                else:
                    break
            stats["accepted_extra"] += len(new) - 1
            new = new[: max_new_tokens - produced]
            tokens.extend(new)
            produced += len(new)
            # refined guess: model's own predictions, shifted past what
            # we just accepted
            guess = (preds[len(new) :] + guess)[:window]
            past = crop(past, len(tokens) - 1)
            assert valid_len(past) == len(tokens) - 1
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

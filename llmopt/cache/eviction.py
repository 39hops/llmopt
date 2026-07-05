"""KV eviction policies: which cached positions survive a budget cut.

Policies are pure functions from (attention evidence, budget) to kept
position indices, sorted ascending — storage-agnostic, like the rest of
cache/. apply_eviction compacts an HF cache down to the kept indices.

- sliding_window: keep the most recent w positions.
- attention_sinks (StreamingLLM): first s positions + recent window;
  the softmax dumps mass on early tokens, so dropping them wrecks
  attention even when they carry no content.
- h2o (Heavy-Hitter Oracle): keep positions with the highest cumulative
  attention received, plus the recent window.
- snapkv: score prefix positions by attention received from the last
  `observe` queries (max over heads, summed over queries); keep top-k
  plus the observation window itself.
"""

from __future__ import annotations


def sliding_window(length: int, window: int) -> list[int]:
    return list(range(max(0, length - window), length))


def attention_sinks(length: int, *, sinks: int = 4, window: int = 128) -> list[int]:
    keep = set(range(min(sinks, length)))
    keep |= set(range(max(0, length - window), length))
    return sorted(keep)


def h2o(cum_scores, *, budget: int, window: int = 8) -> list[int]:
    """cum_scores: [length] cumulative attention each position received.
    Keeps `budget` heavy hitters overall, always including the last
    `window` positions."""
    import torch

    length = cum_scores.shape[0]
    keep = set(range(max(0, length - window), length))
    order = torch.argsort(cum_scores, descending=True).tolist()
    for pos in order:
        if len(keep) >= budget:
            break
        keep.add(pos)
    return sorted(keep)


def snapkv(attn, *, budget: int, observe: int = 8) -> list[int]:
    """attn: [heads, q_len, k_len] attention weights of a recent forward
    pass. Scores prefix positions by attention from the last `observe`
    queries (sum over those queries, max over heads); keeps top scoring
    prefix positions + the observation window."""
    import torch

    heads, q_len, k_len = attn.shape
    obs = min(observe, q_len)
    prefix_len = k_len - obs
    keep = set(range(prefix_len, k_len))  # observation window survives
    if prefix_len > 0 and budget > len(keep):
        scores = attn[:, q_len - obs :, :prefix_len].sum(1).amax(0)
        top = torch.argsort(scores, descending=True).tolist()
        for pos in top[: budget - len(keep)]:
            keep.add(pos)
    return sorted(keep)


def apply_eviction(past, keep: list[int]):
    """Compact an HF cache (5.x layers / 4.x DynamicCache / legacy
    tuples) down to the kept positions, in order. In-place per layer."""
    def take(t):
        return t[:, :, keep]

    if hasattr(past, "layers"):
        for layer in past.layers:
            layer.keys, layer.values = take(layer.keys), take(layer.values)
        return past
    if hasattr(past, "key_cache"):
        for i in range(len(past.key_cache)):
            past.key_cache[i] = take(past.key_cache[i])
            past.value_cache[i] = take(past.value_cache[i])
        return past
    return tuple((take(k), take(v)) for k, v in past)

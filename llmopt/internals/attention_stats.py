"""Attention pattern statistics: entropy and mean attended distance.

Entropy per (layer, head, query): near 0 = the head is a hard pointer
(copy/induction heads), near log(T) = diffuse mixing. Mean attended
distance separates local heads from long-range ones. Both feed
eviction-policy and pruning decisions: diffuse heads tolerate KV
eviction badly; pointer heads need their targets kept.
"""

from __future__ import annotations


def attention_stats(model, ids):
    """Returns dict of tensors, all [layers, heads, seq(query)]:
      entropy: attention entropy in nats (row-wise over keys)
      mean_distance: expected query-key distance under the attention
    """
    import torch

    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([list(ids)], device=model.device),
            output_attentions=True,
        )
        attn = torch.stack([a[0] for a in out.attentions])  # [L, H, q, k]
        eps = 1e-12
        entropy = -(attn * (attn + eps).log()).sum(-1)
        q = torch.arange(attn.shape[2], device=attn.device)
        k = torch.arange(attn.shape[3], device=attn.device)
        dist = (q[:, None] - k[None, :]).abs().float()  # [q, k]
        mean_distance = (attn * dist).sum(-1)
    return {"entropy": entropy, "mean_distance": mean_distance}

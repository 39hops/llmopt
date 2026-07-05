"""Logit lens: decode every layer's hidden state as if it were final.

Projects each layer's residual stream through the model's own final
norm + lm_head, showing where in depth the prediction forms. Early
layers predict syntax-ish continuations; the answer typically snaps
into place in the last third. Per-layer KL to the final distribution
quantifies it — useful for early-exit (LayerSkip) depth selection.
"""

from __future__ import annotations


def logit_lens(model, ids):
    """Per-layer next-token distributions for each position.

    Returns dict with:
      logits: [n_layers+1, seq, vocab] (embedding output first, final
              layer last; final row equals the model's real logits)
      kl_to_final: [n_layers+1, seq] KL(final || layer) in nats
    """
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([list(ids)], device=model.device),
            output_hidden_states=True,
        )
        norm = model.model.norm
        head = model.lm_head
        # hidden_states[-1] is already final-normed in HF; earlier entries
        # are raw residual streams that the lens must norm itself
        per_layer = torch.stack(
            [head(norm(h[0])) for h in out.hidden_states[:-1]]
            + [head(out.hidden_states[-1][0])]
        )  # [L+1, seq, vocab]
        final = F.log_softmax(per_layer[-1], dim=-1)
        layer_logp = F.log_softmax(per_layer, dim=-1)
        kl = (final.exp() * (final - layer_logp)).sum(-1)
    return {"logits": per_layer, "kl_to_final": kl}

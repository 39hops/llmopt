"""Self-speculative decoding (LayerSkip-style): draft with an early
exit of the *same* model, verify with the full model.

The draft is the target model with the top layers skipped — a shallow
copy sharing every weight, no extra memory beyond its own KV cache.
Verification uses the standard speculative loop, so output is exactly
full-model greedy regardless of how bad the early exit is; exit depth
only trades draft cost against accept rate.

Note: pretrained models aren't trained for early exit, so accept rates
are modest out of the box. LayerSkip (Elhoushi et al. 2024) recovers
them with layer-dropout + early-exit-loss training; distill/logit_kd
pointed at the early exit is the post-hoc equivalent.
"""

from __future__ import annotations

import copy


def early_exit_draft(model, exit_layer: int):
    """Shallow-copied view of a llama-style CausalLM that runs only the
    first ``exit_layer`` decoder layers (then final norm + lm_head).
    Shares all weights with ``model``; keeps its own KV cache when used
    as a draft.
    """
    import torch.nn as nn

    assert 0 < exit_layer < len(model.model.layers)
    draft = copy.copy(model)
    # nn.Module stores children in _modules; a plain copy.copy shares that
    # dict, so reassigning .layers would mutate the original model too
    draft._modules = dict(model._modules)
    draft.model = copy.copy(model.model)
    draft.model._modules = dict(model.model._modules)
    draft.model.layers = nn.ModuleList(model.model.layers[:exit_layer])
    draft.config = copy.copy(model.config)
    draft.config.num_hidden_layers = exit_layer
    return draft


def generate_self_speculative(
    model,
    input_ids,
    *,
    exit_layer: int,
    max_new_tokens: int = 128,
    num_draft: int = 5,
    eos_token_id: int | None = None,
):
    """Greedy self-speculative decoding; token-identical to full-model
    greedy. Returns (tokens, stats)."""
    from llmopt.decoding.speculative import generate_speculative

    return generate_speculative(
        model,
        early_exit_draft(model, exit_layer),
        input_ids,
        max_new_tokens=max_new_tokens,
        num_draft=num_draft,
        eos_token_id=eos_token_id,
    )

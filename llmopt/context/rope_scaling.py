"""RoPE scaling: run a model past its trained context window.

RoPE encodes position as rotations at frequencies inv_freq[j] =
base^(-2j/dim). Extending context means slowing rotations so unseen
absolute positions land inside the trained rotation range:

- Position interpolation (PI): divide all frequencies by the scale —
  simple, but crushes the fast (local-detail) frequencies too.
- NTK-aware: raise the base instead — high frequencies barely move
  (local detail preserved), low frequencies stretch the most.
- YaRN: interpolate per frequency — full PI for wavelengths longer than
  the trained window, untouched for short ones, linear ramp between;
  plus a logits temperature on attention.

apply_rope_scaling patches a llama-style HF model's rotary embedding
in place.
"""

from __future__ import annotations

import math


def pi_inv_freq(inv_freq, scale: float):
    return inv_freq / scale


def ntk_inv_freq(inv_freq, scale: float, dim: int):
    """Equivalent to base *= scale^(dim/(dim-2)); inv_freq[j] =
    base^(-2j/dim) so each frequency scales by scale^(-2j/(dim-2))."""
    import torch

    j = torch.arange(inv_freq.shape[0], dtype=inv_freq.dtype)
    return inv_freq * scale ** (-2.0 * j / (dim - 2))


def yarn_inv_freq(
    inv_freq,
    scale: float,
    orig_max_pos: int,
    *,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
):
    """Per-frequency interpolation (YaRN). A frequency completing more
    than beta_fast rotations over the trained window is untouched;
    fewer than beta_slow gets full PI; a linear ramp interpolates."""
    import torch

    rotations = inv_freq * orig_max_pos / (2 * math.pi)
    t = (rotations - beta_slow) / (beta_fast - beta_slow)
    ramp = 1 - t.clamp(0, 1)  # 1 -> full interpolation, 0 -> none
    return inv_freq * ((1 - ramp) + ramp / scale)


def yarn_attention_temperature(scale: float) -> float:
    """YaRN scales attention logits by ~ (1 + 0.1 ln s) to compensate
    entropy growth with context."""
    return 0.1 * math.log(scale) + 1.0


def apply_rope_scaling(
    model, *, method: str = "yarn", scale: float = 4.0
) -> None:
    """Patch a llama-style model's rotary inv_freq in place and lift its
    max_position_embeddings by ``scale``."""
    rot = model.model.rotary_emb
    orig_max = model.config.max_position_embeddings
    dim = rot.inv_freq.shape[0] * 2
    if method == "pi":
        new = pi_inv_freq(rot.inv_freq, scale)
    elif method == "ntk":
        new = ntk_inv_freq(rot.inv_freq, scale, dim)
    elif method == "yarn":
        new = yarn_inv_freq(rot.inv_freq, scale, orig_max)
    else:
        raise ValueError(f"unknown method {method!r}")
    rot.inv_freq.copy_(new)
    if hasattr(rot, "original_inv_freq"):
        rot.original_inv_freq.copy_(new)
    model.config.max_position_embeddings = int(orig_max * scale)

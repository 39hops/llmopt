"""Source guards for scratch/qwen_metal_tower.py's two library-facing
assumptions. No model, no GPU.

1. The mlx-lm GatedDeltaNet still normalizes q and k with the two lines
   the tower overrides (rms_norm with eps 1e-6 and the 1/sqrt(dk) fold);
   if the library changes them, the override must be re-derived against
   the CPU oracle rather than silently applied.
2. The mlx-lm Qwen3.5 sanitizer still shifts the five zero-centered norm
   families by +1 and moves the conv1d axis, which the tower reproduces
   when loading raw tensors.
"""
import inspect

import pytest

qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")


def test_gated_delta_net_normalization_lines_are_the_ones_overridden():
    src = inspect.getsource(qwen3_5.GatedDeltaNet.__call__)
    assert "q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)" in src
    assert "k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)" in src


def test_sanitizer_conventions_reproduced_by_the_tower():
    src = inspect.getsource(qwen3_5.TextModel.sanitize)
    for sfx in (".input_layernorm.weight", ".post_attention_layernorm.weight",
                "model.norm.weight", ".q_norm.weight", ".k_norm.weight"):
        assert sfx in src
    assert "v + 1.0" in src and "moveaxis(2, 1)" in src
    tower = open("scratch/qwen_metal_tower.py").read()
    assert "np.moveaxis(arr, 2, 1)" in tower and "arr + 1.0" in tower
    for sfx in (".input_layernorm.weight", ".post_attention_layernorm.weight",
                "model.norm.weight", ".q_norm.weight", ".k_norm.weight"):
        assert sfx in tower


def test_exact_l2norm_equals_hf_form_not_rms_norm_form():
    """The two forms differ by eps * dk inside the root; show it on a
    small vector so the override is not mistaken for a no-op."""
    mx = pytest.importorskip("mlx.core")
    x = mx.array([[1e-3, 2e-3, 0.0, 0.0]], dtype=mx.float32)
    hf = x * mx.rsqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    inv = 4 ** -0.5
    mlx_form = inv * mx.fast.rms_norm(x, None, 1e-6)
    assert float(mx.abs(hf - mlx_form).max()) > 1e-3

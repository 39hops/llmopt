"""Triton kernels vs pure-torch references (CUDA only)."""

import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton kernels", allow_module_level=True)

from llmopt.kernels.triton_kernels import (
    attention_decode,
    flash_attention,
    rmsnorm,
    rope,
    swiglu,
)


def _cuda(*shapes, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(s, device="cuda", generator=g) for s in shapes]


def _rmsnorm_ref(x, w, eps=1e-6):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def test_rmsnorm_matches_reference():
    x, w = _cuda((17, 384), (384,))  # dim not a power of 2 exercises masking
    torch.testing.assert_close(rmsnorm(x, w), _rmsnorm_ref(x, w), atol=1e-5, rtol=1e-5)


def test_swiglu_matches_reference():
    g, u = _cuda((33, 128), (33, 128), seed=1)
    ref = (g * torch.sigmoid(g)) * u
    torch.testing.assert_close(swiglu(g, u), ref, atol=1e-5, rtol=1e-5)


def _rope_ref(x, pos, base=10000.0):
    n, h, d = x.shape
    pair = torch.arange(d // 2, device=x.device, dtype=torch.float32)
    theta = (1.0 / base) ** (2.0 * pair / d)
    ang = pos[:, None, None].float() * theta[None, None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def test_rope_matches_reference():
    (x,) = _cuda((5, 3, 64), seed=2)
    pos = torch.tensor([0, 1, 2, 100, 1000], device="cuda")
    torch.testing.assert_close(rope(x, pos), _rope_ref(x, pos), atol=1e-4, rtol=1e-4)


def test_attention_decode_matches_softmax():
    t, dim = 500, 64  # T >> BLOCK_T exercises the split-K merge
    q, k, v = _cuda((dim,), (t, dim), (t, dim), seed=3)
    ref = torch.softmax((k @ q) / dim**0.5, dim=0) @ v
    torch.testing.assert_close(attention_decode(q, k, v), ref, atol=1e-4, rtol=1e-4)


def test_attention_decode_extreme_scores_stable():
    # online softmax must survive score ranges that overflow naive exp
    dim = 32
    q = torch.ones(dim, device="cuda") * 6.0
    k = torch.cat([torch.ones(1, dim, device="cuda") * 10,
                   -torch.ones(63, dim, device="cuda") * 10])
    (v,) = _cuda((64, dim), seed=4)
    out = attention_decode(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, v[0], atol=1e-3, rtol=1e-3)


# tl.dot uses tf32 tensor cores on fp32 inputs (~1e-3 abs error vs ieee);
# that's the intended fast path, so flash tests use a tf32-sized tolerance.
_FLASH_TOL = dict(atol=5e-3, rtol=0.0)


def test_flash_attention_matches_sdpa_causal():
    heads, t, dim = 4, 300, 64  # t not a multiple of block size
    q, k, v = _cuda((heads, t, dim), (heads, t, dim), (heads, t, dim), seed=5)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(flash_attention(q, k, v), ref, **_FLASH_TOL)


def test_flash_attention_noncausal():
    heads, t, dim = 2, 129, 64
    q, k, v = _cuda((heads, t, dim), (heads, t, dim), (heads, t, dim), seed=6)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(flash_attention(q, k, v, causal=False), ref, **_FLASH_TOL)


def test_flash_attention_cross_lengths_causal():
    # tq < tk (decode-with-context shape): last query sees all keys
    heads, tq, tk, dim = 2, 65, 200, 64
    q, k, v = _cuda((heads, tq, dim), (heads, tk, dim), (heads, tk, dim), seed=7)
    mask = torch.ones(tq, tk, device="cuda", dtype=torch.bool).tril(diagonal=tk - tq)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(flash_attention(q, k, v), ref, **_FLASH_TOL)

"""Metal kernels vs pure-MLX references (Apple silicon only)."""

import pytest

mx = pytest.importorskip("mlx.core")

from llmopt.kernels.metal import attention_decode, rmsnorm, rope, swiglu


def _rmsnorm_ref(x, w, eps=1e-6):
    return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + eps) * w


def test_rmsnorm_matches_reference():
    mx.random.seed(0)
    x = mx.random.normal((17, 384))  # dim > TG exercises the strided loop
    w = mx.random.normal((384,))
    assert mx.allclose(rmsnorm(x, w), _rmsnorm_ref(x, w), atol=1e-5)


def test_swiglu_matches_reference():
    g = mx.random.normal((33, 128))
    u = mx.random.normal((33, 128))
    ref = (g * mx.sigmoid(g)) * u
    assert mx.allclose(swiglu(g, u), ref, atol=1e-5)


def _rope_ref(x, pos, base=10000.0):
    n, h, d = x.shape
    pair = mx.arange(d // 2)
    theta = (1.0 / base) ** (2.0 * pair / d)
    ang = pos[:, None, None] * theta[None, None, :]
    c, s = mx.cos(ang), mx.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = mx.zeros_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def test_rope_matches_reference():
    mx.random.seed(1)
    x = mx.random.normal((5, 3, 64))
    pos = mx.array([0, 1, 2, 100, 1000])
    assert mx.allclose(rope(x, pos), _rope_ref(x, pos.astype(mx.float32)), atol=1e-4)


def test_attention_decode_matches_softmax():
    mx.random.seed(2)
    t, dim = 500, 64  # T >> threads exercises the online rescaling
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    scores = (k @ q) / dim**0.5
    ref = mx.softmax(scores) @ v
    assert mx.allclose(attention_decode(q, k, v), ref, atol=1e-4)


def test_attention_decode_extreme_scores_stable():
    # online softmax must survive score ranges that overflow naive exp
    dim = 32
    q = mx.ones((dim,)) * 6.0
    k = mx.concatenate([mx.ones((1, dim)) * 10, -mx.ones((63, dim)) * 10])
    v = mx.random.normal((64, dim))
    out = attention_decode(q, k, v)
    assert mx.isfinite(out).all()
    assert mx.allclose(out, v[0], atol=1e-3)  # winner takes all

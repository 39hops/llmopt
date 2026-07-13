"""Metal kernels vs pure-MLX references (Apple silicon only)."""

import pytest

mx = pytest.importorskip("mlx.core")

from llmopt.kernels.metal import (
    attention_decode,
    attention_decode_gqa,
    attention_decode_splitk,
    rmsnorm,
    rope,
    swiglu,
)


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


def _decode_ref(q, k, v):
    return mx.softmax((k @ q) / q.shape[0] ** 0.5) @ v


def test_splitk_matches_softmax():
    mx.random.seed(3)
    t, dim = 3000, 64  # several chunks + a ragged final chunk
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    assert mx.allclose(attention_decode_splitk(q, k, v), _decode_ref(q, k, v), atol=1e-4)


@pytest.mark.parametrize("t", [1, 7, 512, 513])
def test_splitk_boundary_lengths(t):
    mx.random.seed(4)
    dim = 32
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    assert mx.allclose(attention_decode_splitk(q, k, v), _decode_ref(q, k, v), atol=1e-4)


def test_splitk_fp16():
    mx.random.seed(5)
    t, dim = 2000, 128
    q = mx.random.normal((dim,)).astype(mx.float16)
    k = mx.random.normal((t, dim)).astype(mx.float16)
    v = mx.random.normal((t, dim)).astype(mx.float16)
    ref = _decode_ref(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32))
    out = attention_decode_splitk(q, k, v)
    assert out.dtype == mx.float16
    assert mx.allclose(out.astype(mx.float32), ref, atol=2e-3)


def test_splitk_extreme_scores_stable():
    dim = 32
    q = mx.ones((dim,)) * 6.0
    k = mx.concatenate([mx.ones((1, dim)) * 10, -mx.ones((1100, dim)) * 10])
    v = mx.random.normal((1101, dim))
    out = attention_decode_splitk(q, k, v)
    assert mx.isfinite(out).all()
    assert mx.allclose(out, v[0], atol=1e-3)


def _gqa_ref(q, k, v):
    h, dim = q.shape
    kvh = k.shape[1]
    group = h // kvh
    outs = []
    for i in range(h):
        ki, vi = k[:, i // group, :], v[:, i // group, :]
        outs.append(mx.softmax((ki @ q[i]) / dim**0.5) @ vi)
    return mx.stack(outs)


@pytest.mark.parametrize("h,kvh", [(8, 8), (8, 2), (4, 1)])
def test_gqa_matches_reference(h, kvh):
    mx.random.seed(6)
    t, dim = 1500, 64
    q = mx.random.normal((h, dim))
    k = mx.random.normal((t, kvh, dim))
    v = mx.random.normal((t, kvh, dim))
    assert mx.allclose(attention_decode_gqa(q, k, v), _gqa_ref(q, k, v), atol=1e-4)


def test_gqa_single_head_matches_splitk():
    mx.random.seed(7)
    t, dim = 700, 32
    q = mx.random.normal((dim,))
    k = mx.random.normal((t, dim))
    v = mx.random.normal((t, dim))
    a = attention_decode_gqa(q[None], k[:, None], v[:, None])[0]
    b = attention_decode_splitk(q, k, v)
    assert mx.allclose(a, b, atol=1e-5)


def _dequant_ref(packed, scales, mins, gs=128):
    lo, hi = packed & 0x0F, packed >> 4
    n, d2 = packed.shape
    q = mx.zeros((n, d2 * 2), dtype=mx.float32)
    q[:, 0::2], q[:, 1::2] = lo, hi
    g = q.reshape(n, -1, gs)
    return (g * scales[..., None].astype(mx.float32)
            + mins[..., None].astype(mx.float32)).reshape(n, d2 * 2)


def test_int4_gemv_matches_dequant_matmul():
    from llmopt.kernels.metal import int4_gemv, quantize_pack_int4
    mx.random.seed(11)
    n, d = 512, 1024
    w = mx.random.normal((n, d))
    x = mx.random.normal((d,)).astype(mx.float16)
    packed, sc, mn = quantize_pack_int4(w)
    ref = x.astype(mx.float32) @ _dequant_ref(packed, sc, mn).T
    got = int4_gemv(x, packed, sc, mn)
    assert mx.abs(got.astype(mx.float32) - ref).max() < 0.05 * mx.abs(ref).mean()


def test_int4_gemv_awq_channel_scale_fold():
    # awq_lite fold: quantize w*s, feed x/s. The kernel must compute
    # exactly the dequant-reference of the SCALED weights — the fold's
    # accuracy story (error moves off important channels) is measured
    # in scripts/bench_quant_schemes.py with REAL calibration scales;
    # a random s here would test noise, not the mechanism (a random
    # scale uncorrelated with x makes quantization WORSE — measured).
    from llmopt.kernels.metal import int4_gemv, quantize_pack_int4
    mx.random.seed(12)
    n, d = 256, 512
    w = mx.random.normal((n, d))
    x = mx.random.normal((d,)).astype(mx.float16)
    s = (mx.abs(mx.random.normal((d,))) + 0.5)
    packed, sc, mn = quantize_pack_int4(w, channel_scale=s)
    xs = (x.astype(mx.float32) / s).astype(mx.float16)
    got = int4_gemv(xs, packed, sc, mn)
    ref = xs.astype(mx.float32) @ _dequant_ref(packed, sc, mn).T
    assert mx.abs(got.astype(mx.float32) - ref).max() < 0.05 * mx.abs(ref).mean()


@pytest.mark.parametrize("t,causal", [(96, True), (128, False), (200, True)])
def test_flash_prefill_matches_sdpa(t, causal):
    from llmopt.kernels.metal import flash_prefill
    mx.random.seed(13)
    h, d = 2, 64
    q = mx.random.normal((h, t, d)).astype(mx.float16)
    k = mx.random.normal((h, t, d)).astype(mx.float16)
    v = mx.random.normal((h, t, d)).astype(mx.float16)
    ref = mx.fast.scaled_dot_product_attention(
        q[None], k[None], v[None], scale=1 / d**0.5,
        mask="causal" if causal else None)[0]
    got = flash_prefill(q, k, v, causal=causal)
    assert mx.abs(got.astype(mx.float32)
                  - ref.astype(mx.float32)).max() < 5e-3

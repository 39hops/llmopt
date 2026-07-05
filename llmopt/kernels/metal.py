"""Readable Metal kernels via mx.fast.metal_kernel (Apple silicon).

Each kernel is the fused version of an op the roofline model says is
memory-bound: the win is one pass over the data instead of several,
not extra FLOPs. Sources are written for readability — the classic
patterns visible in plain Metal:

- rmsnorm: one threadgroup per row; strided sum-of-squares, shared-
  memory tree reduction, then normalize+scale in the same pass.
- swiglu: silu(gate) * up fused elementwise (one read of each input,
  one write, vs three kernel launches unfused).
- rope: thread per (position, head, pair) rotating (x[2p], x[2p+1]).
- attention_decode: single-query attention with the flash-style online
  softmax — each thread streams a stride of keys keeping a running
  (max, sumexp, weighted-V accumulator); partials merge at the end.
  Never materializes the [T] score vector.

All kernels are verified elementwise against pure-MLX references in
tests; benchmark with scripts/bench_metal_kernels.py. Measured (M3 Pro,
4096x4096): rmsnorm 1.7x over unfused ops (Apple's mx.fast.rms_norm is
another 2.3x — hand-tuned simdgroup reductions); swiglu 1.6x.
attention_decode *loses* to naive softmax@V: a single 32-thread
threadgroup demonstrates the online-softmax algorithm but can't match a
GEMV that saturates the GPU — parallelizing across the whole device is
where flash implementations earn their complexity.
"""

from __future__ import annotations

import mlx.core as mx

_TG = 256  # threadgroup size for row kernels

_RMSNORM_SRC = """
    constexpr uint TG = 256;
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float buf[TG];

    const device T* xr = x + row * (uint)DIM;

    // strided sum of squares: thread tid covers tid, tid+TG, ...
    float acc = 0.0f;
    for (uint j = tid; j < (uint)DIM; j += TG) {
        float v = (float)xr[j];
        acc += v * v;
    }
    buf[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // tree reduction in shared memory
    for (uint s = TG / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = metal::rsqrt(buf[0] / (float)DIM + eps[0]);
    for (uint j = tid; j < (uint)DIM; j += TG)
        out[row * (uint)DIM + j] = (T)((float)xr[j] * scale * (float)w[j]);
"""

_SWIGLU_SRC = """
    uint i = thread_position_in_grid.x;
    float g = (float)gate[i];
    float silu = g / (1.0f + metal::exp(-g));
    out[i] = (T)(silu * (float)up[i]);
"""

_ROPE_SRC = """
    // one thread per rotated pair; layout x: [n, heads, DIM]
    uint i = thread_position_in_grid.x;
    uint half_dim = (uint)DIM / 2;
    uint pair = i % half_dim;
    uint nh = i / half_dim;          // flattened (token, head)
    uint tok = nh / (uint)HEADS;

    float theta = metal::pow(1.0f / base[0],
                             2.0f * (float)pair / (float)DIM);
    float angle = (float)pos[tok] * theta;
    float c = metal::cos(angle), s = metal::sin(angle);

    uint base_idx = nh * (uint)DIM;
    float x0 = (float)x[base_idx + 2 * pair];
    float x1 = (float)x[base_idx + 2 * pair + 1];
    out[base_idx + 2 * pair]     = (T)(x0 * c - x1 * s);
    out[base_idx + 2 * pair + 1] = (T)(x0 * s + x1 * c);
"""

_ATTN_DECODE_SRC = """
    // single query q[DIM] vs K,V [T, DIM]; TG threads, one threadgroup.
    // flash pattern: stream keys, keep running (m, l, acc[DIM]),
    // rescale acc whenever the running max moves.
    constexpr uint TG = 32;
    uint tid = thread_position_in_threadgroup.x;

    float m = -INFINITY;             // running max
    float l = 0.0f;                  // running sum of exp
    float acc[DIM];                  // running sum of exp * V
    for (uint d = 0; d < (uint)DIM; d++) acc[d] = 0.0f;

    for (uint t = tid; t < (uint)TLEN; t += TG) {
        float score = 0.0f;
        for (uint d = 0; d < (uint)DIM; d++)
            score += (float)q[d] * (float)k[t * (uint)DIM + d];
        score *= scale[0];

        float m_new = metal::max(m, score);
        float corr = metal::exp(m - m_new);     // rescale old state
        float p = metal::exp(score - m_new);
        l = l * corr + p;
        for (uint d = 0; d < (uint)DIM; d++)
            acc[d] = acc[d] * corr + p * (float)v[t * (uint)DIM + d];
        m = m_new;
    }

    // merge per-thread partials (same online rule, serially by thread 0)
    threadgroup float sm[TG], sl[TG], sacc[TG * (uint)DIM];
    sm[tid] = m; sl[tid] = l;
    for (uint d = 0; d < (uint)DIM; d++) sacc[tid * (uint)DIM + d] = acc[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float M = sm[0], L = sl[0];
        for (uint w_ = 1; w_ < TG; w_++) {
            float M_new = metal::max(M, sm[w_]);
            float c0 = metal::exp(M - M_new), c1 = metal::exp(sm[w_] - M_new);
            L = L * c0 + sl[w_] * c1;
            for (uint d = 0; d < (uint)DIM; d++)
                sacc[d] = sacc[d] * c0 + sacc[w_ * (uint)DIM + d] * c1;
            M = M_new;
        }
        for (uint d = 0; d < (uint)DIM; d++) out[d] = (T)(sacc[d] / L);
    }
"""


def _kernel(name, src, inputs):
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=["out"], source=src
    )


_rmsnorm = _kernel("llmopt_rmsnorm", _RMSNORM_SRC, ["x", "w", "eps"])
_swiglu = _kernel("llmopt_swiglu", _SWIGLU_SRC, ["gate", "up"])
_rope = _kernel("llmopt_rope", _ROPE_SRC, ["x", "pos", "base"])
_attn = _kernel("llmopt_attn_decode", _ATTN_DECODE_SRC, ["q", "k", "v", "scale"])


def rmsnorm(x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
    """x: [rows, dim], w: [dim]."""
    rows, dim = x.shape
    (out,) = _rmsnorm(
        inputs=[x, w, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("DIM", dim)],
        grid=(rows * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    (out,) = _swiglu(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(min(_TG, gate.size), 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )
    return out


def rope(x: mx.array, positions: mx.array, base: float = 10000.0) -> mx.array:
    """x: [n, heads, dim] (dim even), positions: [n] int32."""
    n, heads, dim = x.shape
    (out,) = _rope(
        inputs=[x, positions.astype(mx.float32), mx.array([base], dtype=mx.float32)],
        template=[("T", x.dtype), ("DIM", dim), ("HEADS", heads)],
        grid=(n * heads * dim // 2, 1, 1),
        threadgroup=(min(_TG, n * heads * dim // 2), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


def attention_decode(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Single-query attention: q [dim], k/v [T, dim] -> [dim]."""
    t, dim = k.shape
    scale = 1.0 / (dim ** 0.5)
    (out,) = _attn(
        inputs=[q, k, v, mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype), ("DIM", dim), ("TLEN", t)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(dim,)],
        output_dtypes=[q.dtype],
    )
    return out

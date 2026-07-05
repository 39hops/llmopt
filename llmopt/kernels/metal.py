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


_LOG2E = 1.4426950408889634  # softmax via exp2: exp(x) == exp2(x * log2(e))

_ATTN_PARTIAL_SRC = """
    // split-K phase 1: one threadgroup per BLOCK_T-key chunk.
    // online softmax in exp2 domain; writes float32 partials
    // (m, l, acc[DIM]) for the merge kernel. Mirrors the Triton
    // _attn_decode_kernel in kernels/triton_kernels.py.
    constexpr uint TG = 256;
    uint chunk = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint t0 = chunk * (uint)BLOCK_T;

    // scores for this chunk into shared memory (thread tid takes
    // keys t0+tid, t0+tid+TG, ...), tracking a per-thread max.
    threadgroup float sc[(uint)BLOCK_T];
    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float s = -INFINITY;
        uint t = t0 + j;
        if (t < (uint)TLEN) {
            s = 0.0f;
            for (uint d = 0; d < (uint)DIM; d++)
                s += (float)q[d] * (float)k[t * (uint)DIM + d];
            s *= scale2[0];
        }
        sc[j] = s;
        local_max = metal::max(local_max, s);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // p = exp2(score - m) in place; l = sum(p) by tree reduction.
    // padding keys carry score -inf; skip them explicitly so an
    // all-padding tail can't produce exp2(-inf - -inf) = NaN.
    float local_sum = 0.0f;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float p = (sc[j] == -INFINITY) ? 0.0f : metal::exp2(sc[j] - m);
        sc[j] = p;
        local_sum += p;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] += red[tid + r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) { m_out[chunk] = m; l_out[chunk] = red[0]; }

    // acc[d] = sum_j p[j] * V[t0+j][d]; threads stride over d so each
    // V element is read exactly once by exactly one thread.
    uint chunk_len = metal::min((uint)BLOCK_T, (uint)TLEN - t0);
    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint j = 0; j < chunk_len; j++)
            a += sc[j] * (float)v[(t0 + j) * (uint)DIM + d];
        acc_out[chunk * (uint)DIM + d] = a;
    }
"""

_ATTN_MERGE_SRC = """
    // split-K phase 2: one threadgroup merges C partials.
    constexpr uint TG = 256;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint c = tid; c < (uint)C; c += TG)
        local_max = metal::max(local_max, m_in[c]);
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float M = red[0];

    // corr_c = exp2(m_c - M) rescales chunk c's partials to the global
    // max. Precompute once into shared memory (parallel over c) instead
    // of re-evaluating exp2 inside the per-d loop below.
    threadgroup float corr[(uint)C];
    float local_l = 0.0f;
    for (uint c = tid; c < (uint)C; c += TG) {
        corr[c] = metal::exp2(m_in[c] - M);
        local_l += corr[c] * l_in[c];
    }
    red[tid] = local_l;                 // L = sum_c corr_c * l_c
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] += red[tid + r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float L = red[0];

    // one axis per stride: threads own output dims, chunks are serial.
    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint c = 0; c < (uint)C; c++)
            a += corr[c] * acc_in[c * (uint)DIM + d];
        out[d] = (T)(a / L);
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
_attn_partial = mx.fast.metal_kernel(
    name="llmopt_attn_decode_partial",
    input_names=["q", "k", "v", "scale2"],
    output_names=["m_out", "l_out", "acc_out"],
    source=_ATTN_PARTIAL_SRC,
)
_attn_merge = _kernel(
    "llmopt_attn_decode_merge", _ATTN_MERGE_SRC, ["m_in", "l_in", "acc_in"]
)


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


def attention_decode_splitk(
    q: mx.array, k: mx.array, v: mx.array, block_t: int = 512
) -> mx.array:
    """Single-query attention, split-K: q [dim], k/v [T, dim] -> [dim].

    Phase 1 gives each threadgroup a BLOCK_T chunk of keys (parallel
    across the device); phase 2 merges the per-chunk (m, l, acc)
    partials. Partials are float32 regardless of input dtype.
    """
    t, dim = k.shape
    nchunks = (t + block_t - 1) // block_t
    scale2 = mx.array([_LOG2E / dim**0.5], dtype=mx.float32)
    m, l, acc = _attn_partial(
        inputs=[q, k, v, scale2],
        template=[("T", q.dtype), ("DIM", dim), ("TLEN", t), ("BLOCK_T", block_t)],
        grid=(nchunks * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(nchunks,), (nchunks,), (nchunks, dim)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _attn_merge(
        inputs=[m, l, acc],
        template=[("T", q.dtype), ("DIM", dim), ("C", nchunks)],
        grid=(_TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(dim,)],
        output_dtypes=[q.dtype],
    )
    return out

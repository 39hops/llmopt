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

- attention_decode_splitk: the fix for the above — split-K across the
  device. Phase 1: one threadgroup per BLOCK_T key chunk writes float32
  partials (m, l, acc); phase 2 merges with corr_c = exp2(m_c - M).
  Softmax runs in the exp2 domain (log2e folded into the scale).

All kernels are verified elementwise against pure-MLX references in
tests; benchmark with scripts/bench_metal_kernels.py. Measured (M3 Pro,
36GB; earlier numbers here were wrong — the old bench harness timed
lazy graph construction, not the GPU): rmsnorm (4096x4096) 3.0x over
unfused and on par with mx.fast.rms_norm; swiglu 2.2x. attention decode
dim=128: v1 (32-thread) loses to naive softmax@V everywhere, 10x at
T=32768. split-K (block_t=256) loses to naive at T=2048 (250 vs 197 us,
launch overhead), wins from T~8192, and at T=32768 is 1.8x over naive
and ties mx.fast.scaled_dot_product_attention (440 vs 443 us).
attention_decode_gqa (H=32, KVH=8) beats a per-head naive loop ~3x at
every T (T=32768: 2322 vs 6910 us).

int4_gemv (2026-07-11, the Cerebras-riff rung): fused int4
dequant-GEMV, practice_7 packing (Artin: adjacent nibbles, group 128),
awq_lite channel scales foldable at pack time. Three measured
versions: v1 (256-thread tree reduction, byte loads) 0.47-0.70x vs
mx.quantized_matmul — the reduction overhead swamped the tiny per-row
work; v2 (one simdgroup per row, simd_sum, uint32 loads) 0.75-1.00x;
v3 (uint2 + half4 vector loads, dot-product MACs) beats mx_q4 1.11x
at D=4096/N=14336 and 2.80x over fp16, ties at D=2048 (0.94x), loses
at D=896 (0.72x) — small shapes are overhead-bound (fp16 GEMV itself
only reaches 40 GB/s there) and mx_q4's tuned dispatch wins the
launch game. Honest split: win big, lose small, the 4x roofline still
not saturated — group/threadgroup sweeps are the config-estimator
rung's training data.

Negative result worth keeping: restructuring the GQA kernel for
group-shared K/V reads (grid (chunks, KVH), one threadgroup computing
all GROUP query heads per K row read — the Triton/CUDA lesson) was
consistently ~10% SLOWER here, even with per-head reductions done by
parallel thread sections. Apple silicon's shared last-level cache
already dedupes the duplicate K reads across query heads within a
group, so the "saved" DRAM traffic was never being paid; the grouped
kernel just loses grid parallelism (KVH vs H threadgroups) and adds
register pressure. CUDA lessons about redundant global reads don't
transfer 1:1 to the M-series memory hierarchy.
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


_GQA_PARTIAL_SRC = """
    // split-K + GQA: grid (chunks, q_heads). q head h reads kv head
    // h / (H / KVH); K/V layout [T, KVH, DIM].
    constexpr uint TG = 256;
    uint chunk = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint t0 = chunk * (uint)BLOCK_T;
    uint kvh = h / ((uint)H / (uint)KVH);
    uint krow = (uint)KVH * (uint)DIM;          // stride of one timestep
    const device T* qh = q + h * (uint)DIM;

    threadgroup float sc[(uint)BLOCK_T];
    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint j = tid; j < (uint)BLOCK_T; j += TG) {
        float s = -INFINITY;
        uint t = t0 + j;
        if (t < (uint)TLEN) {
            s = 0.0f;
            const device T* kt = k + t * krow + kvh * (uint)DIM;
            for (uint d = 0; d < (uint)DIM; d++)
                s += (float)qh[d] * (float)kt[d];
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

    uint pidx = h * (uint)C + chunk;            // partials laid out [H, C]
    if (tid == 0) { m_out[pidx] = m; l_out[pidx] = red[0]; }

    uint chunk_len = metal::min((uint)BLOCK_T, (uint)TLEN - t0);
    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint j = 0; j < chunk_len; j++)
            a += sc[j] * (float)v[(t0 + j) * krow + kvh * (uint)DIM + d];
        acc_out[pidx * (uint)DIM + d] = a;
    }
"""

_GQA_MERGE_SRC = """
    // grid (1, q_heads): merge C partials per head; layout [H, C].
    constexpr uint TG = 256;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint base = h * (uint)C;

    threadgroup float red[TG];
    float local_max = -INFINITY;
    for (uint c = tid; c < (uint)C; c += TG)
        local_max = metal::max(local_max, m_in[base + c]);
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] = metal::max(red[tid], red[tid + r]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float M = red[0];

    threadgroup float corr[(uint)C];
    float local_l = 0.0f;
    for (uint c = tid; c < (uint)C; c += TG) {
        corr[c] = metal::exp2(m_in[base + c] - M);
        local_l += corr[c] * l_in[base + c];
    }
    red[tid] = local_l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = TG / 2; r > 0; r >>= 1) {
        if (tid < r) red[tid] += red[tid + r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float L = red[0];

    for (uint d = tid; d < (uint)DIM; d += TG) {
        float a = 0.0f;
        for (uint c = 0; c < (uint)C; c++)
            a += corr[c] * acc_in[(base + c) * (uint)DIM + d];
        out[h * (uint)DIM + d] = (T)(a / L);
    }
"""


_INT4_GEMV_SRC = """
    // Fused int4 dequant-GEMV: y[row] = sum_d x[d] * W[row, d], with W
    // streamed as packed nibbles and dequantized IN REGISTERS — the
    // fp16 weights never exist in memory (the Cerebras lesson at our
    // scale: decode speed = weight bytes moved / bandwidth). Packing is
    // the practice_7 adjacent scheme (Artin): byte j of a row holds
    // weights d=2j (low nibble) and d=2j+1 (high), so one uint8 load
    // feeds two MACs — 1 load + mask + shift, never straddling bytes.
    // GS=128 (Artin's call: metadata nearly free) => a byte's two
    // weights always share a (scale, min) pair.
    // v2 after the first bench (v1: 256-thread tree reduction, byte
    // loads — 0.47-0.70x vs mx.quantized_matmul): one SIMDGROUP per
    // row (simd_sum, no shared memory, no barriers) and uint32 loads
    // (8 weights per load; at GS=128 a uint's 8 weights never straddle
    // a group, so one (scale, min) fetch serves all 8 MACs).
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;

    const device uint32_t* wr =
        (const device uint32_t*)(wq + row * (uint)D2);
    const device T* sr = sc + row * (uint)NG;
    const device T* mr = mn + row * (uint)NG;
    uint nu = (uint)D2 / 4;

    float acc = 0.0f;
    for (uint j = lane; j < nu; j += SG) {
        uint32_t b = wr[j];
        uint d0 = j * 8;                 // little-endian nibble t = d0+t
        uint g = d0 / (uint)GS;
        float s = (float)sr[g];
        float m = (float)mr[g];
        for (uint t = 0; t < 8; t++)
            acc += (float)x[d0 + t]
                 * ((float)((b >> (4 * t)) & 0xFu) * s + m);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) out[row] = (T)acc;
"""

_INT4_GEMV_V3_SRC = """
    // v3: uint2 weight loads (16 weights/iter) + half4 activation
    // loads. At GS=128 a uint2's 16 weights share one (scale, min).
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;

    const device uint2* wr =
        (const device uint2*)(wq + row * (uint)D2);
    const device T* sr = sc + row * (uint)NG;
    const device T* mr = mn + row * (uint)NG;
    uint nu = (uint)D2 / 8;

    float acc = 0.0f;
    for (uint j = lane; j < nu; j += SG) {
        uint2 b = wr[j];
        uint d0 = j * 16;
        uint g = d0 / (uint)GS;
        float s = (float)sr[g];
        float m = (float)mr[g];
        const device half4* x4 = (const device half4*)(x + d0);
        float4 xa = (float4)x4[0], xb = (float4)x4[1];
        float4 xc = (float4)x4[2], xd = (float4)x4[3];
        float4 wa = float4((float)((b.x >>  0) & 0xFu),
                           (float)((b.x >>  4) & 0xFu),
                           (float)((b.x >>  8) & 0xFu),
                           (float)((b.x >> 12) & 0xFu)) * s + m;
        float4 wb = float4((float)((b.x >> 16) & 0xFu),
                           (float)((b.x >> 20) & 0xFu),
                           (float)((b.x >> 24) & 0xFu),
                           (float)((b.x >> 28) & 0xFu)) * s + m;
        float4 wc = float4((float)((b.y >>  0) & 0xFu),
                           (float)((b.y >>  4) & 0xFu),
                           (float)((b.y >>  8) & 0xFu),
                           (float)((b.y >> 12) & 0xFu)) * s + m;
        float4 wd = float4((float)((b.y >> 16) & 0xFu),
                           (float)((b.y >> 20) & 0xFu),
                           (float)((b.y >> 24) & 0xFu),
                           (float)((b.y >> 28) & 0xFu)) * s + m;
        acc += metal::dot(xa, wa) + metal::dot(xb, wb)
             + metal::dot(xc, wc) + metal::dot(xd, wd);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) out[row] = (T)acc;
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
_gqa_partial = mx.fast.metal_kernel(
    name="llmopt_attn_gqa_partial",
    input_names=["q", "k", "v", "scale2"],
    output_names=["m_out", "l_out", "acc_out"],
    source=_GQA_PARTIAL_SRC,
)
_gqa_merge = _kernel(
    "llmopt_attn_gqa_merge", _GQA_MERGE_SRC, ["m_in", "l_in", "acc_in"]
)
_int4_gemv = _kernel(
    "llmopt_int4_gemv", _INT4_GEMV_SRC, ["x", "wq", "sc", "mn"]
)
_int4_gemv_v3 = _kernel(
    "llmopt_int4_gemv_v3", _INT4_GEMV_V3_SRC, ["x", "wq", "sc", "mn"]
)


def quantize_pack_int4(w: mx.array, group_size: int = 128,
                       channel_scale: mx.array | None = None):
    """Quantize + pack fp weights for int4_gemv.

    w: [N, D]; groups of `group_size` along D share a per-group
    fp16 (scale, min) — uniform min/max affine, the function-space
    race's base grid. channel_scale [D] folds awq_lite in: pass
    mean|x_c|**0.5 from calibration and w is quantized as w*s; the
    caller then feeds int4_gemv x/s (fold it into the previous op —
    the function computed is identical, the ERROR moves off the
    high-activation channels; measured 8.07% vs 10.06% on real Qwen).
    Returns (packed [N, D//2] uint8, scales [N, D//gs] f16,
    mins [N, D//gs] f16).
    """
    n, d = w.shape
    assert d % group_size == 0 and group_size % 2 == 0
    w = w.astype(mx.float32)
    if channel_scale is not None:
        w = w * channel_scale.astype(mx.float32)
    g = w.reshape(n, d // group_size, group_size)
    mn = g.min(axis=2)
    sc = (g.max(axis=2) - mn) / 15.0
    sc = mx.where(sc == 0, mx.ones_like(sc), sc)
    q = mx.clip(mx.round((g - mn[..., None]) / sc[..., None]), 0, 15)
    q = q.reshape(n, d).astype(mx.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(mx.uint8)  # practice_7
    return packed, sc.astype(mx.float16), mn.astype(mx.float16)


def int4_gemv(x: mx.array, packed: mx.array, scales: mx.array,
              mins: mx.array, group_size: int = 128) -> mx.array:
    """y = x @ W.T for int4-packed W: x [D] fp16 -> y [N] fp16."""
    n, d2 = packed.shape
    ng = scales.shape[1]
    kern = _int4_gemv_v3 if d2 % 8 == 0 else _int4_gemv
    (out,) = kern(
        inputs=[x, packed, scales, mins],
        template=[("T", x.dtype), ("D2", d2), ("NG", ng),
                  ("GS", group_size)],
        grid=(n * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )
    return out


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
    q: mx.array, k: mx.array, v: mx.array, block_t: int = 256
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


def attention_decode_gqa(
    q: mx.array, k: mx.array, v: mx.array, block_t: int = 256
) -> mx.array:
    """GQA decode attention: q [H, dim], k/v [T, KVH, dim] -> [H, dim].

    Same split-K structure as attention_decode_splitk with a 2D grid
    (key chunks x query heads); query head h attends to kv head
    h // (H // KVH).
    """
    h, dim = q.shape
    t, kvh, _ = k.shape
    assert h % kvh == 0, f"q heads {h} not a multiple of kv heads {kvh}"
    nchunks = (t + block_t - 1) // block_t
    scale2 = mx.array([_LOG2E / dim**0.5], dtype=mx.float32)
    tmpl = [("T", q.dtype), ("DIM", dim), ("TLEN", t), ("BLOCK_T", block_t),
            ("H", h), ("KVH", kvh), ("C", nchunks)]
    m, l, acc = _gqa_partial(
        inputs=[q, k, v, scale2],
        template=tmpl,
        grid=(nchunks * _TG, h, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(h, nchunks), (h, nchunks), (h, nchunks, dim)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _gqa_merge(
        inputs=[m, l, acc],
        template=[("T", q.dtype), ("DIM", dim), ("C", nchunks), ("H", h)],
        grid=(_TG, h, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(h, dim)],
        output_dtypes=[q.dtype],
    )
    return out

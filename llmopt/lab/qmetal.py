"""Fused Metal decode-and-multiply for the frozen WHOLE-0T payloads
(Apple silicon, mx.fast.metal_kernel). Companion of llmopt.lab.qcuda.

The on-disk formats are those of llmopt.lab.qcodec and are not changed:
  w4 : exps u8[nb] ++ codebook fp16[256,4] ++ idxs u8[n/4]; one index
       byte selects FOUR consecutive weights from the codebook; a block
       of 128 consecutive weights shares one power-of-two scale
       2^(exp-127).
  s16: exps u8[nb] ++ levels fp16[16] ++ codes u8[n/2]; HIGH nibble is
       the EVEN element; same block-128 power-of-two scale.
Neither is affine 4-bit quantization: there is no zero point and the
codebook / level table is per tensor.

Kernels never materialize a decoded matrix. One simdgroup (32 lanes)
owns one output row; each lane streams 32-bit words of the row's
payload (w4: 4 index bytes = 16 weights; s16: 4 code bytes = 8
weights), rebuilds the block scale from the exponent byte exactly as
qcodec does (2^(e-127) in float32: bit construction (e << 23) for
1 <= e <= 254, the exact subnormal constant for e == 0, +inf for
e == 255), multiplies against the matching slice of x, and
accumulates in float32; simd_sum reduces the row. The scale multiplies
the partial dot of a word, not the individual weights, and blocks with
exponent below 31 (scale under 2^-96) or at 191 and above (scale 2^64
and above, except the infinite 255) go through second and third
accumulators whose remaining factors 2^-96 and 2^64 are applied once per
row, so no intermediate is subnormal or infinite unless the true row
value is; the GPU flushes subnormals to zero. Four documented deviations
from decode-then-multiply remain: outputs whose TRUE value is below the
float32 normal range (about 1.2e-38) can read as zero; weights the
oracle itself decodes to subnormal floats (exponent bytes 0 and 1)
carry the oracle's subnormal rounding, which the fused path does not
reproduce; a row whose float32 oracle overflows on one weight can read
finite here because the scale is applied to the word partial (the tests
judge such rows against the exact float64 law); and exponent byte 255
gives inf or nan depending on the signs that meet, while the oracle gives
its own non-finite value (both non-finite, not necessarily the same one).
The tests pin all four. Arm C exponent bytes span 111 to 128, so the
second and third accumulators never fire on the real payload. A 32-bit word never
crosses a 128-element block, so one exponent serves a whole word.
Multi-token (prefill) use runs the same decode once per row against a
tile of up to TT activations.

Parity regime (tests/test_qmetal.py): |got - ref| <= 1e-5 |ref| + 2e-6
times the row's absolute dot sum_c |W_rc x_c| (float32 accumulation
against a float64 oracle; the second term is the cancellation
allowance) plus one subnormal spacing per oracle-subnormal weight, on the
committed golden payloads, random payloads with exponent edges, and
adversarial codebooks (including non-finite entries, which must
propagate exactly as the oracle propagates them). Bit exactness is not
expected for a different accumulation order.
"""
from __future__ import annotations

import numpy as np

from llmopt.lab.qcodec import BLOCK, expected_len

try:  # pragma: no cover - environment dependent
    import mlx.core as mx
    HAVE_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAVE_MLX = False

_NEED = "llmopt.lab.qmetal needs mlx on Apple silicon (the Mac lab venv)."


def require_mlx() -> None:
    if not HAVE_MLX:
        raise RuntimeError(_NEED)


_HEADER = """
// Block scale from the exponent byte, exactly qcodec's exp2(e - 127) in
// float32 for E_LO <= e < E_HI: bit construction (e << 23); e == 255 is
// +inf and stays on the direct path so that inf propagates as the oracle
// propagates it. Blocks with e < E_LO (scale below 2^-96) and blocks
// with E_HI <= e < 255 (scale 2^64 and above) go through a second and
// third accumulator with partial scales 2^(e-31) and 2^(e-191), both
// normal floats, and the remaining 2^-96 or 2^64 is applied once per
// row at the end. The GPU flushes subnormal intermediates to zero and a
// word's partial dot times a huge scale could overflow where the
// elementwise oracle does not; with the split, an intermediate is
// subnormal or infinite only when the true row value is.
constant uint E_LO = 31u;
constant uint E_HI = 191u;
constant float LO_REST = 1.2621774483536189e-29f;   // 2^-96 exactly
constant float HI_REST = 1.8446744073709552e19f;    // 2^64 exactly
inline uint q_class(uint e) { return e < E_LO ? 1u : ((e >= E_HI && e != 255u) ? 2u : 0u); }
inline float q_scale(uint e, uint cls) {
    if (cls == 1u) return as_type<float>((e + 96u) << 23);
    if (cls == 2u) return as_type<float>((e - 64u) << 23);
    return as_type<float>(e << 23);
}
"""

# y[row, t] = sum_c W[row, c] * x[t, c]; TT activations per launch.
_W4_SRC = """
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const uint C4 = (uint)C / 4;                 // index bytes per row
    const device uint* ir = (const device uint*)(idx + row * C4);
    const device half4* cb4 = (const device half4*)cb;
    const uint nwords = C4 / 4;                  // 16 weights per word
    const uint blocks_per_row = (uint)C / 128;
    float acc[TT], lo[TT], hi[TT];
    for (uint t = 0; t < (uint)TT; t++) { acc[t] = 0.0f; lo[t] = 0.0f; hi[t] = 0.0f; }
    for (uint j = lane; j < nwords; j += SG) {
        uint word = ir[j];
        uint c0 = j * 16;
        uint e = (uint)exps[row * blocks_per_row + c0 / 128];
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        // codebook values first, the block scale on the word's partial dot
        float4 w0 = (float4)cb4[(word      ) & 0xFFu];
        float4 w1 = (float4)cb4[(word >>  8) & 0xFFu];
        float4 w2 = (float4)cb4[(word >> 16) & 0xFFu];
        float4 w3 = (float4)cb4[(word >> 24) & 0xFFu];
        for (uint t = 0; t < (uint)TT; t++) {
            const device float4* x4 = (const device float4*)(x + t * (uint)C + c0);
            float d = (metal::dot(x4[0], w0) + metal::dot(x4[1], w1)
                     + metal::dot(x4[2], w2) + metal::dot(x4[3], w3)) * s;
            if (cls == 0u) acc[t] += d; else if (cls == 1u) lo[t] += d; else hi[t] += d;
        }
    }
    for (uint t = 0; t < (uint)TT; t++) {
        float r = metal::simd_sum(acc[t]);
        float rl = metal::simd_sum(lo[t]);
        float rh = metal::simd_sum(hi[t]);
        if (lane == 0) out[t * (uint)R + row] = (r + rl * LO_REST) + rh * HI_REST;
    }
"""

_S16_SRC = """
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const uint C2 = (uint)C / 2;                 // code bytes per row
    const device uint* cr = (const device uint*)(codes + row * C2);
    const uint nwords = C2 / 4;                  // 8 weights per word
    const uint blocks_per_row = (uint)C / 128;
    float lvf[16];
    for (uint i = 0; i < 16u; i++) lvf[i] = (float)lv[i];
    float acc[TT], lo[TT], hi[TT];
    for (uint t = 0; t < (uint)TT; t++) { acc[t] = 0.0f; lo[t] = 0.0f; hi[t] = 0.0f; }
    for (uint j = lane; j < nwords; j += SG) {
        uint word = cr[j];
        uint c0 = j * 8;
        uint e = (uint)exps[row * blocks_per_row + c0 / 128];
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        // byte b holds elements 2b (HIGH nibble) and 2b+1 (LOW nibble)
        uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu;
        uint b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
        float4 w0 = float4(lvf[b0 >> 4], lvf[b0 & 0xFu],
                           lvf[b1 >> 4], lvf[b1 & 0xFu]);
        float4 w1 = float4(lvf[b2 >> 4], lvf[b2 & 0xFu],
                           lvf[b3 >> 4], lvf[b3 & 0xFu]);
        for (uint t = 0; t < (uint)TT; t++) {
            const device float4* x4 = (const device float4*)(x + t * (uint)C + c0);
            float d = (metal::dot(x4[0], w0) + metal::dot(x4[1], w1)) * s;
            if (cls == 0u) acc[t] += d; else if (cls == 1u) lo[t] += d; else hi[t] += d;
        }
    }
    for (uint t = 0; t < (uint)TT; t++) {
        float r = metal::simd_sum(acc[t]);
        float rl = metal::simd_sum(lo[t]);
        float rh = metal::simd_sum(hi[t]);
        if (lane == 0) out[t * (uint)R + row] = (r + rl * LO_REST) + rh * HI_REST;
    }
"""



# ---- v2: threadgroup-shared codebook / level table, 128-bit payload
# ---- loads (64 w4 weights or 32 s16 weights per lane per iteration),
# ---- ROWS simdgroups (rows) per threadgroup. Same arithmetic law.
_W4_V2_SRC = """
    constexpr uint SG = 32;
    uint sg = thread_position_in_threadgroup.x / SG;     // row within the threadgroup
    uint lane = thread_position_in_threadgroup.x % SG;
    uint row = threadgroup_position_in_grid.x * (uint)ROWS + sg;
    threadgroup half4 cbs[256];
    for (uint i = thread_position_in_threadgroup.x; i < 256u; i += (uint)ROWS * SG)
        cbs[i] = ((const device half4*)cb)[i];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (row >= (uint)R) return;
    const uint C4 = (uint)C / 4;
    const device uint4* ir = (const device uint4*)(idx + row * C4);
    const uint nq = C4 / 16;                             // 64 weights per uint4
    const uint blocks_per_row = (uint)C / 128;
    float acc[TT], lo[TT], hi[TT];
    for (uint t = 0; t < (uint)TT; t++) { acc[t] = 0.0f; lo[t] = 0.0f; hi[t] = 0.0f; }
    for (uint j = lane; j < nq; j += SG) {
        uint4 q = ir[j];
        uint c0 = j * 64;
        uint e = (uint)exps[row * blocks_per_row + c0 / 128];
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        float d[TT];
        for (uint t = 0; t < (uint)TT; t++) d[t] = 0.0f;
        uint words[4] = {q.x, q.y, q.z, q.w};
        for (uint w = 0; w < 4u; w++) {
            uint word = words[w];
            float4 w0 = (float4)cbs[(word      ) & 0xFFu];
            float4 w1 = (float4)cbs[(word >>  8) & 0xFFu];
            float4 w2 = (float4)cbs[(word >> 16) & 0xFFu];
            float4 w3 = (float4)cbs[(word >> 24) & 0xFFu];
            for (uint t = 0; t < (uint)TT; t++) {
                const device float4* x4 = (const device float4*)(x + t * (uint)C + c0 + w * 16);
                d[t] += metal::dot(x4[0], w0) + metal::dot(x4[1], w1)
                      + metal::dot(x4[2], w2) + metal::dot(x4[3], w3);
            }
        }
        for (uint t = 0; t < (uint)TT; t++) {
            float v = d[t] * s;
            if (cls == 0u) acc[t] += v; else if (cls == 1u) lo[t] += v; else hi[t] += v;
        }
    }
    for (uint t = 0; t < (uint)TT; t++) {
        float r = metal::simd_sum(acc[t]);
        float rl = metal::simd_sum(lo[t]);
        float rh = metal::simd_sum(hi[t]);
        if (lane == 0) out[t * (uint)R + row] = (r + rl * LO_REST) + rh * HI_REST;
    }
"""

_S16_V2_SRC = """
    constexpr uint SG = 32;
    uint sg = thread_position_in_threadgroup.x / SG;
    uint lane = thread_position_in_threadgroup.x % SG;
    uint row = threadgroup_position_in_grid.x * (uint)ROWS + sg;
    float lvf[16];
    for (uint i = 0; i < 16u; i++) lvf[i] = (float)lv[i];
    if (row >= (uint)R) return;
    const uint C2 = (uint)C / 2;
    const device uint4* cr = (const device uint4*)(codes + row * C2);
    const uint nq = C2 / 16;                             // 32 weights per uint4
    const uint blocks_per_row = (uint)C / 128;
    float acc[TT], lo[TT], hi[TT];
    for (uint t = 0; t < (uint)TT; t++) { acc[t] = 0.0f; lo[t] = 0.0f; hi[t] = 0.0f; }
    for (uint j = lane; j < nq; j += SG) {
        uint4 q = cr[j];
        uint c0 = j * 32;
        uint e = (uint)exps[row * blocks_per_row + c0 / 128];
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        float d[TT];
        for (uint t = 0; t < (uint)TT; t++) d[t] = 0.0f;
        uint words[4] = {q.x, q.y, q.z, q.w};
        for (uint w = 0; w < 4u; w++) {
            uint word = words[w];
            uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu;
            uint b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
            float4 w0 = float4(lvf[b0 >> 4], lvf[b0 & 0xFu], lvf[b1 >> 4], lvf[b1 & 0xFu]);
            float4 w1 = float4(lvf[b2 >> 4], lvf[b2 & 0xFu], lvf[b3 >> 4], lvf[b3 & 0xFu]);
            for (uint t = 0; t < (uint)TT; t++) {
                const device float4* x4 = (const device float4*)(x + t * (uint)C + c0 + w * 8);
                d[t] += metal::dot(x4[0], w0) + metal::dot(x4[1], w1);
            }
        }
        for (uint t = 0; t < (uint)TT; t++) {
            float v = d[t] * s;
            if (cls == 0u) acc[t] += v; else if (cls == 1u) lo[t] += v; else hi[t] += v;
        }
    }
    for (uint t = 0; t < (uint)TT; t++) {
        float r = metal::simd_sum(acc[t]);
        float rl = metal::simd_sum(lo[t]);
        float rh = metal::simd_sum(hi[t]);
        if (lane == 0) out[t * (uint)R + row] = (r + rl * LO_REST) + rh * HI_REST;
    }
"""

_W4_V3_SRC = """
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const uint C4 = (uint)C / 4;                 // index bytes per row
    const device uint* ir = (const device uint*)(idx + row * C4);
    threadgroup half4 cb4[256];               // the 2 KB codebook, loaded once per row
    for (uint i = lane; i < 256u; i += SG) cb4[i] = ((const device half4*)cb)[i];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const uint nwords = C4 / 4;                  // 16 weights per word
    const uint blocks_per_row = (uint)C / 128;
    float acc[TT], lo[TT], hi[TT];
    for (uint t = 0; t < (uint)TT; t++) { acc[t] = 0.0f; lo[t] = 0.0f; hi[t] = 0.0f; }
    for (uint j = lane; j < nwords; j += SG) {
        uint word = ir[j];
        uint c0 = j * 16;
        uint e = (uint)exps[row * blocks_per_row + c0 / 128];
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        // codebook values first, the block scale on the word's partial dot
        float4 w0 = (float4)cb4[(word      ) & 0xFFu];
        float4 w1 = (float4)cb4[(word >>  8) & 0xFFu];
        float4 w2 = (float4)cb4[(word >> 16) & 0xFFu];
        float4 w3 = (float4)cb4[(word >> 24) & 0xFFu];
        for (uint t = 0; t < (uint)TT; t++) {
            const device float4* x4 = (const device float4*)(x + t * (uint)C + c0);
            float d = (metal::dot(x4[0], w0) + metal::dot(x4[1], w1)
                     + metal::dot(x4[2], w2) + metal::dot(x4[3], w3)) * s;
            if (cls == 0u) acc[t] += d; else if (cls == 1u) lo[t] += d; else hi[t] += d;
        }
    }
    for (uint t = 0; t < (uint)TT; t++) {
        float r = metal::simd_sum(acc[t]);
        float rl = metal::simd_sum(lo[t]);
        float rh = metal::simd_sum(hi[t]);
        if (lane == 0) out[t * (uint)R + row] = (r + rl * LO_REST) + rh * HI_REST;
    }
"""


VERSION = 3       # kernel generation used by the resident objects (1, 2 or 3)
ROWS_V2 = 4       # rows (simdgroups) per threadgroup in v2

_kernels: dict = {}


def _kernel(name: str, version: int):
    key = (name, version)
    if key not in _kernels:
        require_mlx()
        srcs = {("w4", 1): _W4_SRC, ("w4", 2): _W4_V2_SRC, ("w4", 3): _W4_V3_SRC,
                ("s16", 1): _S16_SRC, ("s16", 2): _S16_V2_SRC, ("s16", 3): _S16_SRC}
        inputs = ["x", "idx", "cb", "exps"] if name == "w4" else ["x", "codes", "lv", "exps"]
        _kernels[key] = mx.fast.metal_kernel(
            name=f"llmopt_q_{name}_gemv_v{version}", input_names=inputs,
            output_names=["out"], header=_HEADER, source=srcs[key])
    return _kernels[key]


TILE = 8  # activations per launch in the multi-token path


def _check_shape(codec: str, shape, buf_len: int, version: int = None):
    if version is None:
        version = VERSION
    if len(shape) != 2:
        raise ValueError(f"{codec} expects a 2-d shape, got {shape}")
    R, C = int(shape[0]), int(shape[1])
    if C % BLOCK or (codec == "w4" and C % 16) or (codec == "s16" and C % 8):
        raise ValueError(f"{codec} row width {C} must be a multiple of 128")
    if version == 2 and codec == "w4" and C % 256:
        raise ValueError(f"w4 v2 kernel needs a row width multiple of 256, got {C}")
    if buf_len != expected_len(codec, shape):
        raise ValueError(f"{codec} payload {buf_len} != {expected_len(codec, shape)}")
    return R, C


class _Resident:
    """Payload parts as device arrays; shared gemv / gemm driver."""
    codec = ""
    version = VERSION

    def _run(self, x) -> "mx.array":
        R, C = self.shape
        x = x if isinstance(x, mx.array) else mx.array(np.asarray(x))
        if x.dtype != mx.float32:
            x = x.astype(mx.float32)
        squeeze = x.ndim == 1
        x2 = x.reshape(1, -1) if squeeze else x.reshape(-1, C)
        if x2.shape[1] != C:
            raise ValueError(f"x width {x2.shape[1]} != C {C}")
        T = x2.shape[0]
        outs = []
        for t0 in range(0, T, TILE):
            xt = x2[t0:t0 + TILE]
            tt = xt.shape[0]
            # the compiled tile is the smallest of 1, 2, 4, 8 that holds tt:
            # a single-token GEMV must not pay for eight activation rows
            tile = 1 if tt == 1 else 2 if tt == 2 else 4 if tt <= 4 else TILE
            if tt < tile:
                xt = mx.concatenate([xt, mx.zeros((tile - tt, C), mx.float32)], axis=0)
            version = self.version
            if version == 2:
                rows = ROWS_V2
                ntg = (R + rows - 1) // rows
                grid, tg = (ntg * rows * 32, 1, 1), (rows * 32, 1, 1)
                template = [("C", C), ("R", R), ("TT", tile), ("ROWS", rows)]
            else:
                grid, tg = (R * 32, 1, 1), (32, 1, 1)
                template = [("C", C), ("R", R), ("TT", tile)]
            (o,) = _kernel(self.codec, version)(
                inputs=[xt, *self._parts()], template=template,
                grid=grid, threadgroup=tg,
                output_shapes=[(tile, R)], output_dtypes=[mx.float32])
            outs.append(o[:tt])
        y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return y[0] if squeeze else y

    def gemv(self, x):
        """y = W x for one activation vector x[C] -> y[R] (float32)."""
        return self._run(x)

    def gemm(self, x):
        """y = x W^T for x[T, C] -> y[T, R] (float32); same decode per row,
        TILE activations per launch."""
        return self._run(x)

    def payload_bytes(self) -> int:
        return sum(int(p.nbytes) for p in self._parts())


class W4Metal(_Resident):
    codec = "w4"

    def __init__(self, buf: bytes, shape):
        require_mlx()
        R, C = _check_shape("w4", shape, len(buf), self.version)
        n = R * C
        nb = n // BLOCK
        self.shape = (R, C)
        self.exps = mx.array(np.frombuffer(buf, np.uint8, nb, 0))
        self.cb = mx.array(np.frombuffer(buf, np.float16, 1024, nb))
        self.idx = mx.array(np.frombuffer(buf, np.uint8, n // 4, nb + 2048))
        mx.eval(self.exps, self.cb, self.idx)

    def _parts(self):
        return (self.idx, self.cb, self.exps)


class S16Metal(_Resident):
    codec = "s16"

    def __init__(self, buf: bytes, shape):
        require_mlx()
        R, C = _check_shape("s16", shape, len(buf), self.version)
        n = R * C
        nb = n // BLOCK
        self.shape = (R, C)
        self.exps = mx.array(np.frombuffer(buf, np.uint8, nb, 0))
        self.lv = mx.array(np.frombuffer(buf, np.float16, 16, nb))
        self.codes = mx.array(np.frombuffer(buf, np.uint8, n // 2, nb + 32))
        mx.eval(self.exps, self.lv, self.codes)

    def _parts(self):
        return (self.codes, self.lv, self.exps)


def resident(buf: bytes, entry: dict):
    """Build the resident form for a manifest entry of codec w4 or s16."""
    if entry["codec"] == "w4":
        return W4Metal(buf, entry["shape"])
    if entry["codec"] == "s16":
        return S16Metal(buf, entry["shape"])
    raise ValueError(f"no fused kernel for codec {entry['codec']!r}")

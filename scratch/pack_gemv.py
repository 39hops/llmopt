"""PACKED CRYSTAL C2 (pre-reg 2026-07-29 night): dequant-fused
sigma-pack GEMV. Runtime format: int8 codes (byte-aligned twin of
the 5-bit disk pack) + ONE fp scale per tensor (1/q_t). Kernel in
int4_gemv v3 style: one simdgroup per row, char4 weight + half4
activation loads, simd_sum, scale once per row. Correctness v fp
reference, then bench v fp16 GEMV at crystal AND large shapes;
bandwidth-model prediction printed next to measured. mx.eval every
timed iteration (lazy-graph scar). __main__-guarded.
"""
import math
import sys
import time

sys.path.insert(0, ".")
import mlx.core as mx  # noqa: E402

_SRC = """
    // crystal8 GEMV: y[row] = scale * sum_d x[d] * codes[row, d]
    // codes int8, one fp scale per tensor. 32 lanes stride the row
    // in 4-code chunks: char4 weight load + half4 activation load.
    constexpr uint LANES = 32;
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const device char4* w4 =
        (const device char4*)(wq + (size_t)row * D);
    float acc = 0.0f;
    for (uint i = lane; i < D / 4; i += LANES) {
        const char4 c = w4[i];
        const device half4* xv = (const device half4*)(x + i * 4);
        const half4 xh = *xv;
        acc += (float)xh.x * (float)c.x + (float)xh.y * (float)c.y
             + (float)xh.z * (float)c.z + (float)xh.w * (float)c.w;
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) out[row] = (T)(acc * scale[0]);
"""

_kern = mx.fast.metal_kernel(
    name="llmopt_crystal8_gemv", input_names=["x", "wq", "scale"],
    output_names=["out"], source=_SRC)


def pack8(w):
    """fp [N, D] -> (codes int8, scale). sigma-law: q = ceil(2/std),
    codes = round(w * q) (int8 range checked)."""
    s = float(mx.array(w).astype(mx.float32).std())
    q = math.ceil(2.0 / s)
    codes = mx.round(w.astype(mx.float32) * q)
    assert float(codes.abs().max()) < 127, "codes exceed int8"
    return codes.astype(mx.int8), mx.array([1.0 / q], dtype=mx.float32)


def crystal8_gemv(x, codes, scale):
    n, d = codes.shape
    (out,) = _kern(
        inputs=[x, codes, scale], template=[("T", x.dtype), ("D", d)],
        grid=(n * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n,)], output_dtypes=[x.dtype])
    return out


def bench(fn, iters=200):
    for _ in range(20):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())  # eval EVERY iteration — the lazy-graph scar
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    mx.random.seed(0)
    shapes = [(224, 56), (256, 64), (1536, 384),
              (8192, 2048), (14336, 4096)]
    print(f"{'N x D':>14} {'fp16 us':>9} {'crystal8 us':>12} "
          f"{'speedup':>8} {'bw-model':>9} {'max err':>9}")
    for n, d in shapes:
        w = mx.random.normal((n, d)) * 0.19
        x = mx.random.normal((d,)).astype(mx.float16)
        codes, scale = pack8(w)
        wh = w.astype(mx.float16)
        mx.eval(codes, scale, wh, x)
        wdq = codes.astype(mx.float32) * scale  # kernel-exact reference
        ref = (x.astype(mx.float32) @ wdq.T).astype(mx.float16)
        out = crystal8_gemv(x, codes, scale)
        err = float((out.astype(mx.float32)
                     - ref.astype(mx.float32)).abs().max())
        rel = err / float(ref.abs().max())
        t_fp = bench(lambda: x @ wh.T)
        t_c8 = bench(lambda: crystal8_gemv(x, codes, scale))
        # bandwidth model: weights dominate; 2 B/wt -> 1 B/wt
        print(f"{n:>6} x {d:<5} {t_fp:9.1f} {t_c8:12.1f} "
              f"{t_fp / t_c8:7.2f}x {'2.00x':>9} {rel:9.1e}",
              flush=True)


if __name__ == "__main__":
    main()

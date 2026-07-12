"""Fused int4 dequant-GEMV vs the incumbents, decode shapes (M=1).

Lanes: fp16 GEMV (x @ w.T), mx.quantized_matmul (MLX's own int4,
group 64 — its tuned config), and llmopt int4_gemv (practice_7
packing, group 128, awq-foldable). Decode is bandwidth-bound: the
roofline win for int4 over fp16 is ~4x; MLX's own kernel measured
1.73-2.80x on this machine, so the open question is how much of the
gap a simple hand-rolled kernel recovers. mx.eval every timed
iteration (the lazy-graph lesson — see kernels/metal.py docstring).
"""
import time

import mlx.core as mx

from llmopt.kernels.metal import int4_gemv, quantize_pack_int4


def bench(f, it=200, warmup=20):
    for _ in range(warmup):
        mx.eval(f())
    t0 = time.perf_counter()
    for _ in range(it):
        mx.eval(f())
    return (time.perf_counter() - t0) / it * 1e6


def main():
    print(f"{'shape':16s} {'fp16':>9s} {'mx_q4':>9s} {'ours':>9s}"
          f"  {'ours vs fp16':>12s} {'vs mx_q4':>9s}")
    for d, n in [(896, 4864), (2048, 8192), (4096, 14336)]:
        mx.random.seed(0)
        w = mx.random.normal((n, d)).astype(mx.float16)
        x1 = mx.random.normal((1, d)).astype(mx.float16)
        xf = x1[0]
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        packed, sc, mn = quantize_pack_int4(w)
        mx.eval(w, x1, wq, s, b, packed, sc, mn)
        t_fp = bench(lambda: x1 @ w.T)
        t_mq = bench(lambda: mx.quantized_matmul(
            x1, wq, s, b, transpose=True, group_size=64, bits=4))
        t_us = bench(lambda: int4_gemv(xf, packed, sc, mn))
        gb = d * n * 2 / 1e9
        print(f"D={d:<5d} N={n:<6d} {t_fp:8.0f}u {t_mq:8.0f}u "
              f"{t_us:8.0f}u  {t_fp/t_us:11.2f}x {t_mq/t_us:8.2f}x"
              f"   (fp16 {gb/t_fp*1e6:.0f} GB/s)")
    print("bar: beat mx.quantized_matmul at decode shapes, or record "
          "the loss next to attention_decode's")


if __name__ == "__main__":
    main()

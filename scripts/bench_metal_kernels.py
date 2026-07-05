"""Benchmark llmopt Metal kernels vs unfused MLX ops and mx.fast.*

Readable-kernel reality check: we expect to beat *unfused* op chains
(fusion saves memory passes on memory-bound ops) and lose some margin
to Apple's hand-tuned mx.fast versions.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from llmopt.kernels.metal import attention_decode, rmsnorm, swiglu


def bench(fn, *args, repeats=200):
    for _ in range(20):
        mx.eval(fn(*args))
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    mx.eval(out)
    return (time.perf_counter() - t0) / repeats * 1e6  # us


def main() -> None:
    rows, dim = 4096, 4096
    x = mx.random.normal((rows, dim))
    w = mx.random.normal((dim,))
    unfused_rms = lambda: x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
    print(f"rmsnorm {rows}x{dim}:")
    print(f"  unfused mlx ops   {bench(unfused_rms):8.1f} us")
    print(f"  llmopt metal      {bench(rmsnorm, x, w):8.1f} us")
    print(f"  mx.fast.rms_norm  {bench(lambda: mx.fast.rms_norm(x, w, 1e-6)):8.1f} us")

    g, u = mx.random.normal((rows, dim)), mx.random.normal((rows, dim))
    unfused_swiglu = lambda: (g * mx.sigmoid(g)) * u
    print(f"swiglu {rows}x{dim}:")
    print(f"  unfused mlx ops   {bench(unfused_swiglu):8.1f} us")
    print(f"  llmopt metal      {bench(swiglu, g, u):8.1f} us")

    t, hd = 8192, 128
    q, k, v = (mx.random.normal(s) for s in ((hd,), (t, hd), (t, hd)))
    naive = lambda: mx.softmax((k @ q) / hd**0.5) @ v
    print(f"attention decode T={t} dim={hd}:")
    print(f"  naive softmax     {bench(naive):8.1f} us")
    print(f"  llmopt metal      {bench(attention_decode, q, k, v):8.1f} us")


if __name__ == "__main__":
    main()

"""Benchmark llmopt Triton kernels vs unfused torch ops and torch SDPA.

Same reality check as bench_metal_kernels.py: fused kernels should beat
unfused op chains on memory-bound ops; flash_attention is compared against
torch's scaled_dot_product_attention (cuDNN/flash backends).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from llmopt.kernels.triton_kernels import (
    attention_decode,
    flash_attention,
    rmsnorm,
    swiglu,
)


def bench(fn, *args, repeats=200):
    for _ in range(20):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e6  # us


def main() -> None:
    dev, dt = "cuda", torch.float16
    rows, dim = 4096, 4096
    x = torch.randn(rows, dim, device=dev, dtype=dt)
    w = torch.randn(dim, device=dev, dtype=dt)
    unfused_rms = lambda: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w
    print(f"rmsnorm {rows}x{dim} fp16:")
    print(f"  unfused torch ops {bench(unfused_rms):8.1f} us")
    print(f"  llmopt triton     {bench(rmsnorm, x, w):8.1f} us")
    rms_mod = torch.nn.RMSNorm(dim, eps=1e-6, device=dev, dtype=dt)
    print(f"  torch.nn.RMSNorm  {bench(rms_mod, x):8.1f} us")

    g = torch.randn(rows, dim, device=dev, dtype=dt)
    u = torch.randn(rows, dim, device=dev, dtype=dt)
    unfused_swiglu = lambda: (g * torch.sigmoid(g)) * u
    print(f"swiglu {rows}x{dim} fp16:")
    print(f"  unfused torch ops {bench(unfused_swiglu):8.1f} us")
    print(f"  llmopt triton     {bench(swiglu, g, u):8.1f} us")

    t, hd = 8192, 128
    q = torch.randn(hd, device=dev, dtype=dt)
    k = torch.randn(t, hd, device=dev, dtype=dt)
    v = torch.randn(t, hd, device=dev, dtype=dt)
    naive = lambda: torch.softmax((k @ q) / hd**0.5, dim=0) @ v
    print(f"attention decode T={t} dim={hd} fp16:")
    print(f"  naive softmax     {bench(naive):8.1f} us")
    print(f"  llmopt triton     {bench(attention_decode, q, k, v):8.1f} us")

    heads, tq, hd = 32, 2048, 128
    q = torch.randn(heads, tq, hd, device=dev, dtype=dt)
    k = torch.randn(heads, tq, hd, device=dev, dtype=dt)
    v = torch.randn(heads, tq, hd, device=dev, dtype=dt)
    # 4D input so SDPA can pick a fused backend (3D falls back to math)
    q4, k4, v4 = q[None], k[None], v[None]
    sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    print(f"flash attention heads={heads} T={tq} dim={hd} fp16 causal:")
    print(f"  torch SDPA (fused){bench(sdpa):8.1f} us")
    print(f"  llmopt triton     {bench(flash_attention, q, k, v):8.1f} us")


if __name__ == "__main__":
    main()

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
    paged_attention,
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


def bench_paged() -> None:
    # batched decode over a paged pool: kernel reads through block tables
    # in place; baseline must first gather blocks into contiguous KV.
    dev, dt = "cuda", torch.float16
    batch, t, bs = 8, 4096, 16
    kv_heads, group, hd = 8, 4, 128
    nb = t // bs
    k_pool = torch.randn(batch * nb, bs, kv_heads, hd, device=dev, dtype=dt)
    v_pool = torch.randn(batch * nb, bs, kv_heads, hd, device=dev, dtype=dt)
    perm = torch.randperm(batch * nb, device=dev)  # scattered physical blocks
    bt = perm.reshape(batch, nb).to(torch.int32)
    lens = torch.full((batch,), t, dtype=torch.int32, device=dev)
    q = torch.randn(batch, kv_heads * group, hd, device=dev, dtype=dt)

    def gather_sdpa():
        k = k_pool[bt.long()].reshape(batch, t, kv_heads, hd).transpose(1, 2)
        v = v_pool[bt.long()].reshape(batch, t, kv_heads, hd).transpose(1, 2)
        # GQA: the `group` queries per KV head sit on SDPA's Tq axis
        return torch.nn.functional.scaled_dot_product_attention(
            q.view(batch, kv_heads, group, hd), k, v,
        )

    print(f"paged attention B={batch} T={t} kv_heads={kv_heads} group={group} dim={hd} fp16:")
    print(f"  gather + SDPA     {bench(gather_sdpa):8.1f} us")
    print(f"  llmopt triton     {bench(paged_attention, q, k_pool, v_pool, bt, lens):8.1f} us")


if __name__ == "__main__":
    main()
    bench_paged()

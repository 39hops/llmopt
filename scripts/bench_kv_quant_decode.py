"""Quantized-KV decode attention: does the roofline ~4x show up?

Decode attention is HBM-bound on the K/V read. fp16 K/V moves 4*T*dim
bytes per call; fused int8 dequant halves that, packed int4 quarters it
(plus 8 bytes/token of fp32 scales). Compares against the fp16 split-K
kernel and the honest loser: dequantize-to-fp32-then-attend, which reads
MORE than fp16. Accuracy is reported as max |error| vs the exact-KV
output so the speed claim carries its quality cost.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from llmopt.kernels.triton_kernels import (
    attention_decode,
    attention_decode_quant,
    quantize_kv_rows,
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
    dim = 128
    for t in (8192, 65536, 262144):
        q = torch.randn(dim, device=dev, dtype=dt)
        k = torch.randn(t, dim, device=dev, dtype=dt)
        v = torch.randn(t, dim, device=dev, dtype=dt)
        exact = attention_decode(q, k, v)
        us_fp16 = bench(attention_decode, q, k, v)
        gb = 4 * t * dim / 1e9  # K+V fp16 bytes per call
        print(f"decode T={t} dim={dim}:")
        print(f"  fp16 KV (triton)   {us_fp16:8.1f} us  (1.0x, {gb / (us_fp16 * 1e-6):5.0f} GB/s)")

        for bits in (8, 4):
            kc, ks = quantize_kv_rows(k, bits)
            vc, vs = quantize_kv_rows(v, bits)
            out = attention_decode_quant(q, kc, ks, vc, vs, bits)
            err = (out.float() - exact.float()).abs().max()
            # dequant-then-attend baseline (the naive loser: extra HBM trip)
            deq = lambda: attention_decode(
                q,
                (kc.float() * ks[:, None] if bits == 8 else _unpack(kc, ks)).to(dt),
                (vc.float() * vs[:, None] if bits == 8 else _unpack(vc, vs)).to(dt),
            )
            us_q = bench(attention_decode_quant, q, kc, ks, vc, vs, bits)
            us_d = bench(deq)
            bytes_q = (t * dim // (1 if bits == 8 else 2) * 2 + 8 * t) / 1e9
            print(
                f"  int{bits} fused         {us_q:8.1f} us  ({us_fp16 / us_q:.1f}x,"
                f" {bytes_q / (us_q * 1e-6):5.0f} GB/s)  max|err|={err:.4f}"
            )
            print(f"  int{bits} dequant+attn  {us_d:8.1f} us  ({us_fp16 / us_d:.1f}x)")


def _unpack(codes, scale):
    lo = (codes & 0xF).to(torch.int8) - 7
    hi = (codes >> 4).to(torch.int8) - 7
    full = torch.stack([lo, hi], dim=-1).reshape(codes.shape[0], -1)
    return full.float() * scale[:, None]


if __name__ == "__main__":
    main()

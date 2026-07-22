"""Error-compensated TF32 birth (Markidis 2018 / Ootomo-Yokota 2022
style): every Linear matmul runs as 3 TF32 tensor-core products
(hi*hi + hi*lo + lo*hi) instead of 1 fp32 CUDA-core product —
~fp32 accuracy (CPU-verified: 1.15e-4 vs fp32's 1.01e-4 max err,
raw TF32 7.8e-2) at tensor-core throughput. Both forward and
backward compensated. Parity arm 4: gate + wall decide adoption;
pre-registered honestly — 3 TF32 matmuls may net SLOWER than 1
fp32 on GA102, the arm measures it.

    TF32X3 birth: python scripts/train_tf32x3.py --gen4 ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


def _split(x: torch.Tensor):
    # TF32 keeps 10 mantissa bits; rounding happens inside cuBLAS,
    # so the hi part is what TF32 will represent exactly.
    i = x.contiguous().view(torch.int32)
    hi = (i & ~((1 << 13) - 1)).view(torch.float32)
    return hi, x - hi


def _mm3(a, b):
    ah, al = _split(a)
    bh, bl = _split(b)
    return ah @ bh + ah @ bl + al @ bh


class _CompLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        y = _mm3(x.reshape(-1, x.shape[-1]), w.t())
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        gx = _mm3(g2, w).reshape_as(x)
        gw = _mm3(g2.t().contiguous(), x2)
        return gx, gw


class TF32x3Linear(nn.Linear):
    def forward(self, x):
        y = _CompLinear.apply(x, self.weight)
        return y + self.bias if self.bias is not None else y


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # the point
    nn.Linear = TF32x3Linear
    # reuse the standard trainer CLI wholesale under the override
    exec(open(Path(__file__).parent / "train_mathnative.py").read())

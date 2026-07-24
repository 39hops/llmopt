"""Ozaki rung 2a-v3 (3080): three lifts on the v2 crossover.
(A) fp16 TENSOR CORES as exact integer units (s=8 slices exact in
    fp16's 11-bit mantissa; fp32 accumulate; 2x TF32 rate on Ampere).
(B) recombination bottleneck fix: per-diagonal partial sums carried
    as fp32 (exact: diagonal sums of s=6 int products stay < 2^24
    within a block-diagonal), converted to fp64 ONCE per block.
(C) ZERO-ROUNDING OUTPUT: double-double (two-float64) accumulation
    via elementwise two-sum on GPU, spot-verified against exact
    big-integer arithmetic — deviation must be 0, not small.
"""
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
dev = "cuda"
N = 2048
g = torch.Generator().manual_seed(1)
A = (torch.randn(N, N, generator=g) * 0.05).float()
B = (torch.randn(N, N, generator=g) * 0.05).float()
ref = (A.double() @ B.double())
scale = ref.abs().max().item()

def slices_of(F, s):
    out, R = [], F
    while R.abs().max() > 0:
        Q = torch.round(R * 2.0**s)
        R = R * 2.0**s - Q
        out.append(Q)
    return out

def prep(M, s, block, side):
    out = []
    for b0 in range(0, N, block):
        Mb = (M[b0:b0+block, :] if side == "B"
              else M[:, b0:b0+block]).to(dev).double()
        dim = 0 if side == "B" else 1
        e = (Mb.abs().amax(dim, keepdim=True) + 1e-300
             ).log2().floor() + 1
        out.append((e, slices_of(Mb * 2.0**-e, s)))
    return out

def run(Bmat, Aprep, s, block, tri=None, mm="fp32", dd=False):
    C = torch.zeros(N, N, dtype=torch.float64, device=dev)
    Cl = torch.zeros_like(C) if dd else None
    Bprep = prep(Bmat, s, block, "B")
    for (ea, Asl), (eb, Bsl) in zip(Aprep, Bprep):
        diag = [None] * (len(Asl) + len(Bsl) - 1)
        for i, Ai in enumerate(Asl):
            Ax = (Ai.char() if mm == "int8" else
                  Ai.half() if mm == "fp16" else Ai.float())
            for j, Bj in enumerate(Bsl):
                if tri is not None and i + j >= tri:
                    continue
                if mm == "int8":
                    p = torch._int_mm(Ax, Bj.char()).float()
                elif mm == "fp16":
                    p = (Ax @ Bj.half()).float()   # fp32 accumulate?
                else:
                    p = Ax @ Bj.float()
                if dd:              # exact: two-sum each scaled pair
                    t = p.double() * 2.0**(-s*(i+j+2)) * 2.0**ea * 2.0**eb
                    snew = C + t
                    bb = snew - C
                    r = (C - (snew - bb)) + (t - bb)
                    C, Cl = snew, Cl + r
                    continue
                d = i + j
                diag[d] = p if diag[d] is None else diag[d] + p
        if dd:
            continue
        part = torch.zeros(N, N, dtype=torch.float64, device=dev)
        for d, P in enumerate(diag):
            if P is not None:
                part += P.double() * 2.0**(-s * (d + 2))
        C = C + part * 2.0**ea * 2.0**eb
    return (C, Cl) if dd else C

def bench(name, fn, n=3):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    C = out[0] if isinstance(out, tuple) else out
    err = (C.double().cpu() - ref).abs().max().item() / scale
    print(f"  {name:34s} {dt*1e3:8.1f} ms   err {err:.3e}", flush=True)
    return out

Ad = A.to(dev)
A64, B64 = Ad.double(), B.to(dev).double()
bench("native fp64 matmul", lambda: A64 @ B64)

# (A) fp16 tensor cores. NOTE: cublas fp16 gemm may accumulate in
# fp16 (ruinous) or fp32 depending on backend — the ERROR COLUMN is
# the audit: ~1e-16 means fp32 accumulate held, garbage means fp16.
AP8h = prep(A, 8, 256, "A")
bench("v3 s=8 FP16TC blk256 full", lambda: run(B, AP8h, 8, 256, mm="fp16"))
bench("v3 s=8 FP16TC blk256 tri<4", lambda: run(B, AP8h, 8, 256, tri=4, mm="fp16"))

# (B) int8 with fp32-diagonal carry is already what v2 does inside a
# block; the remaining fp64 cost is per-block conversion. Row-wide
# block minimizes it — rerun with bigger N-block already done in v2;
# here: int8 + dd output (C) — the zero-rounding claim.
AP6 = prep(A, 6, N, "A")
Cdd, Cl = bench("v3 s=6 INT8 full + DD output",
                lambda: run(B, AP6, 6, N, mm="int8", dd=True))

# (C) spot-verify DD output against exact big-int arithmetic
from fractions import Fraction
import numpy as np
An = A.numpy().astype(np.float64); Bn = B.numpy().astype(np.float64)
def to_int(M):
    e = np.frexp(M)[1]
    sh = int(24 - e.min())
    return np.round(M * 2.0**sh).astype(np.int64).astype(object), sh
IA, sa = to_int(An); IB, sb = to_int(Bn)
worst = 0
Ch = Cdd.cpu().numpy(); Cln = Cl.cpu().numpy()
for i in range(0, N, 511):
    for j in range(0, N, 511):
        true = int(sum(IA[i, t] * IB[t, j] for t in range(N)))
        got = (Fraction(Ch[i, j]) + Fraction(Cln[i, j])) * (1 << (sa + sb))
        worst = max(worst, abs(got - true))
print(f"  DD-output vs exact big-int: max deviation {worst} "
      f"(0 = rounding-free end to end)", flush=True)

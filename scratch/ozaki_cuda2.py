"""Ozaki rung 2a-v2 (3080): lift the wall floor with the three named
fixes. (1) WEIGHT slices amortized (static in inference/metabolism —
timed loop slices only the activation side); (2) recombination
grouped per (i+j) diagonal — fp64 elementwise falls 36 -> ~13 ops
per block; (3) int8 tensor cores (torch._int_mm, int32 accumulate:
exact at s=6 with row-wide blocks, products*N <= 2^25 << 2^31).
Same error scoring vs CPU fp64 reference as v1."""
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
N = 2048
g = torch.Generator().manual_seed(1)
A = (torch.randn(N, N, generator=g) * 0.05).float()   # "weights"
B = (torch.randn(N, N, generator=g) * 0.05).float()   # "activations"
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
    """block-align + slice; returns per-block (exp, [slices])"""
    out = []
    for b0 in range(0, M.shape[0 if side == "B" else 1], block):
        Mb = (M[b0:b0+block, :] if side == "B"
              else M[:, b0:b0+block]).to(dev).double()
        dim = 0 if side == "B" else 1
        e = (Mb.abs().amax(dim, keepdim=True) + 1e-300
             ).log2().floor() + 1
        out.append((e, slices_of(Mb * 2.0**-e, s)))
    return out

def sliced_v2(Bmat, Aprep, s, block, tri=None, int8=False):
    C = torch.zeros(N, N, dtype=torch.float64, device=dev)
    Bprep = prep(Bmat, s, block, "B")
    for (ea, Asl), (eb, Bsl) in zip(Aprep, Bprep):
        kmax = len(Asl) + len(Bsl) - 1
        diag = [None] * kmax
        for i, Ai in enumerate(Asl):
            Af = Ai.char() if int8 else Ai.float()
            for j, Bj in enumerate(Bsl):
                if tri is not None and i + j >= tri:
                    continue
                if int8:
                    p = torch._int_mm(Af, Bj.char()).float()
                else:
                    p = Af @ Bj.float()
                d = i + j
                diag[d] = p if diag[d] is None else diag[d] + p
        part = torch.zeros(N, N, dtype=torch.float64, device=dev)
        for d, P in enumerate(diag):
            if P is not None:
                part += P.double() * 2.0**(-s * (d + 2))
        C += part * 2.0**ea * 2.0**eb
    return C

def bench(name, fn, n=3):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        C = fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    err = (C.double().cpu() - ref).abs().max().item() / scale
    print(f"  {name:34s} {dt*1e3:8.1f} ms   err {err:.3e}", flush=True)

Ad, Bd = A.to(dev), B.to(dev)
A64, B64 = Ad.double(), Bd.double()
bench("native fp64 matmul", lambda: A64 @ B64)

AP8 = prep(A, 8, 256, "A")           # amortized: outside the timer
bench("v2 s=8 fp32acc blk256 full",
      lambda: sliced_v2(B, AP8, 8, 256))
bench("v2 s=8 fp32acc blk256 tri<4",
      lambda: sliced_v2(B, AP8, 8, 256, tri=4))

try:
    AP6 = prep(A, 6, N, "A")         # int8: row-wide blocks, s=6
    bench("v2 s=6 INT8 blkrow full",
          lambda: sliced_v2(B, AP6, 6, N, int8=True))
    bench("v2 s=6 INT8 blkrow tri<5",
          lambda: sliced_v2(B, AP6, 6, N, tri=5, int8=True))
except Exception as e:
    print(f"  int8 path unavailable: {type(e).__name__}: {e}")

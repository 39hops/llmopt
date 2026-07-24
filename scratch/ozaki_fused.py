"""Fused Ozaki recombination kernel (Triton, cuda). The measured
bottleneck: 36-64 separate elementwise passes (scale + two-sum per
slice-pair), each a full N^2 fp64 round-trip. This kernel does the
whole recombination in ONE pass: per element, loop pairs in
registers, two-sum locally, write hi/lo once. Exactness preserved
(every op identical, just fused). Race: v2 int8 full-exact with
looped recombination vs fused; bar = beat native fp64's ~41 ms."""
import time
import torch
import triton
import triton.language as tl

dev = "cuda"
N = 2048

@triton.jit
def recombine_kernel(P, SC, EA, EB, HI, LO, n_pairs, NN, NCOL,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NN
    row = offs // NCOL
    col = offs % NCOL
    fa = tl.load(EA + row, mask=mask, other=1.0)   # 2^ea (fp64)
    fb = tl.load(EB + col, mask=mask, other=1.0)   # 2^eb (fp64)
    hi = tl.zeros((BLOCK,), dtype=tl.float64)
    lo = tl.zeros((BLOCK,), dtype=tl.float64)
    for k in range(n_pairs):
        p = tl.load(P + k * NN + offs, mask=mask, other=0)
        s = tl.load(SC + k)
        t = p.to(tl.float64) * s * fa * fb
        snew = hi + t                               # two-sum (exact)
        bb = snew - hi
        lo = lo + (hi - (snew - bb)) + (t - bb)
        hi = snew
    tl.store(HI + offs, hi, mask=mask)
    tl.store(LO + offs, lo, mask=mask)

def slices_of(F, s):
    out, R = [], F
    while R.abs().max() > 0:
        Q = torch.round(R * 2.0**s)
        R = R * 2.0**s - Q
        out.append(Q)
    return out

def prep(M, s, dim):
    Md = M.to(dev).double()
    e = (Md.abs().amax(dim, keepdim=True) + 1e-300).log2().floor() + 1
    return e, [q.char() for q in slices_of(Md * 2.0**-e, s)]

def exact_fused(A, B, s=6):
    ea, Asl = prep(A, s, 1)
    eb, Bsl = prep(B, s, 0)
    pairs, scales = [], []
    for i, Ai in enumerate(Asl):
        for j, Bj in enumerate(Bsl):
            pairs.append(torch._int_mm(Ai, Bj))
            scales.append(2.0**(-s * (i + j + 2)))
    P = torch.stack(pairs)                          # [k, N, N] int32
    SC = torch.tensor(scales, dtype=torch.float64, device=dev)
    FA = (2.0**ea).squeeze(1).contiguous()
    FB = (2.0**eb).squeeze(0).contiguous()
    HI = torch.empty(N, N, dtype=torch.float64, device=dev)
    LO = torch.empty_like(HI)
    NN = N * N
    BLOCK = 256
    recombine_kernel[(triton.cdiv(NN, BLOCK),)](
        P.view(len(pairs), -1), SC, FA, FB,
        HI.view(-1), LO.view(-1), len(pairs), NN, N, BLOCK=BLOCK)
    return HI, LO

def exact_looped(A, B, s=6):
    ea, Asl = prep(A, s, 1)
    eb, Bsl = prep(B, s, 0)
    hi = torch.zeros(N, N, dtype=torch.float64, device=dev)
    lo = torch.zeros_like(hi)
    fa, fb = 2.0**ea, 2.0**eb
    for i, Ai in enumerate(Asl):
        for j, Bj in enumerate(Bsl):
            t = (torch._int_mm(Ai, Bj).double()
                 * 2.0**(-s*(i+j+2)) * fa * fb)
            snew = hi + t
            bb = snew - hi
            lo = lo + (hi - (snew - bb)) + (t - bb)
            hi = snew
    return hi, lo

g = torch.Generator().manual_seed(1)
A = (torch.randn(N, N, generator=g) * 0.05).float()
B = (torch.randn(N, N, generator=g) * 0.05).float()
ref = (A.double() @ B.double())
scale = ref.abs().max().item()

def bench(name, fn, n=3):
    out = fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    err = ((out[0] + out[1]).cpu() - ref).abs().max().item() / scale
    print(f"  {name:30s} {dt*1e3:8.1f} ms   err-vs-fp64ref {err:.3e}",
          flush=True)
    return out

A64, B64 = A.to(dev).double(), B.to(dev).double()
t0 = time.time()
for _ in range(3):
    C = A64 @ B64
torch.cuda.synchronize()
print(f"  {'native fp64':30s} {(time.time()-t0)/3*1e3:8.1f} ms",
      flush=True)
h1, l1 = bench("looped exact + DD (v3 style)", lambda: exact_looped(A, B))
h2, l2 = bench("FUSED exact + DD (one pass)", lambda: exact_fused(A, B))
print(f"  fused == looped bitwise: "
      f"{torch.equal(h1, h2) and torch.equal(l1, l2)}", flush=True)

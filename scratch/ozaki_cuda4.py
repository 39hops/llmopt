"""Ozaki rung v4 (3080): two escalations past the v3 crossover.
(A) RNS-GEMM (Chinese Remainder Theorem): integers represented by
    residues mod k small primes — multiplication is CHANNEL-LOCAL
    (k matmuls, NO cross products, no carries) vs slicing's k^2.
    Reconstruction: Garner mixed-radix digits (all mod-p arithmetic
    exact in fp64), assembled into double-double with 26-bit-split
    radix constants (every elementwise product exact by construction).
(B) fp64-INPUT exact matmul via int8 slicing — the product of two
    fp64 matrices carries ~106+ bits of true detail: fp128-grade
    linear algebra on a gaming card, spot-verified vs big integers.
"""
import time
import torch

dev = "cuda"
N = 2048
g = torch.Generator().manual_seed(1)

# ---------- (A) RNS-GEMM on fp32-derived integers ----------
# fixed-point integers: |a| < 2^24, products*N < 2^59 -> M > 2^60
PRIMES = [127, 113, 109, 107, 103, 101, 97, 89, 83, 79]  # ~2^67
A32 = (torch.randn(N, N, generator=g) * 0.05).float()
B32 = (torch.randn(N, N, generator=g) * 0.05).float()

def to_fixed(M):
    e = torch.frexp(M.double())[1]
    sh = int(24 - e.min().item())
    I = torch.round(M.double() * 2.0**sh)
    return I, sh
IA, sa = to_fixed(A32)
IB, sb = to_fixed(B32)

def rns_gemm(IA, IB):
    """returns dd (hi, lo) of the EXACT integer product matrix"""
    IAd, IBd = IA.to(dev), IB.to(dev)
    residues = []
    for p in PRIMES:
        ra = torch.remainder(IAd, p).char()          # [0, p) fits int8
        rb = torch.remainder(IBd, p).char()
        acc = torch.zeros(N, N, dtype=torch.int32, device=dev)
        # int8 products up to 126^2=15876; chunk K so sums stay < 2^31
        CH = 8192 // 64                              # 2^14 * 2^7 rows safe
        for k0 in range(0, N, 2048):                 # 15876*2048 < 2^25
            acc += torch._int_mm(ra[:, k0:k0+2048], rb[k0:k0+2048, :])
        residues.append(torch.remainder(acc, p).double())
    # Garner: mixed-radix digits d_i (exact fp64 mod-p arithmetic)
    k = len(PRIMES)
    inv = [[pow(PRIMES[j], -1, PRIMES[i]) for j in range(i)]
           for i in range(k)]
    digits = [residues[0]]
    for i in range(1, k):
        v = residues[i]
        for j in range(i):
            v = torch.remainder((v - digits[j]) * inv[i][j], PRIMES[i])
        digits.append(v)                             # all < p_i, exact
    # assemble sum d_i * R_i, R_i = prod_{j<i} p_j, via 26-bit splits
    hi = torch.zeros(N, N, dtype=torch.float64, device=dev)
    lo = torch.zeros_like(hi)
    R = 1
    M = 1
    for p in PRIMES:
        M *= p
    for i in range(k):
        for part in _split26(R):                     # exact d*part
            t = digits[i] * float(part)
            s = hi + t; bb = s - hi
            lo = lo + (hi - (s - bb)) + (t - bb)
            hi = s
        R *= PRIMES[i]
    # values are mod M in [0, M); map to signed via v > M/2 -> v - M
    half = M / 2
    over = (hi > half)
    for part in _split26(M):
        t = torch.where(over, torch.full_like(hi, -float(part)),
                        torch.zeros_like(hi))
        s = hi + t; bb = s - hi
        lo = lo + (hi - (s - bb)) + (t - bb)
        hi = s
    return hi, lo

def _split26(x):
    """split python int into exact <=26-bit*2^shift fp64 chunks"""
    out, sh = [], 0
    while x:
        c = x & ((1 << 26) - 1)
        if c: out.append(float(c * (1 << sh)))
        x >>= 26; sh += 26
    return out or [0.0]

t0 = time.time(); hi, lo = rns_gemm(IA, IB); torch.cuda.synchronize()
t_rns = time.time() - t0
print(f"[A] RNS-GEMM ({len(PRIMES)} channels): {t_rns*1e3:.1f} ms",
      flush=True)
from fractions import Fraction
import numpy as np
IAn = IA.numpy().astype(object); IBn = IB.numpy().astype(object)
hin, lon = hi.cpu().numpy(), lo.cpu().numpy()
worst = 0
for i in range(0, N, 511):
    for j in range(0, N, 511):
        true = int(sum(int(IAn[i, t]) * int(IBn[t, j]) for t in range(N)))
        got = Fraction(hin[i, j]) + Fraction(lon[i, j])
        worst = max(worst, abs(got - true))
print(f"[A] RNS vs exact big-int: max deviation {worst} (0 = exact)",
      flush=True)

# ---------- (B) fp64-input exact product (fp128-grade) ----------
A64 = (torch.randn(N, N, generator=g, dtype=torch.float64) * 0.05)
B64 = (torch.randn(N, N, generator=g, dtype=torch.float64) * 0.05)
def slices_of(F, s):
    out, R = [], F
    while R.abs().max() > 0:
        Q = torch.round(R * 2.0**s); R = R * 2.0**s - Q
        out.append(Q)
    return out
def exact64(A, B, s=6):
    Ad, Bd = A.to(dev), B.to(dev)
    ea = (Ad.abs().amax(1, keepdim=True) + 1e-300).log2().floor() + 1
    eb = (Bd.abs().amax(0, keepdim=True) + 1e-300).log2().floor() + 1
    Asl = slices_of(Ad * 2.0**-ea, s); Bsl = slices_of(Bd * 2.0**-eb, s)
    hi = torch.zeros(N, N, dtype=torch.float64, device=dev)
    lo = torch.zeros_like(hi)
    for i, Ai in enumerate(Asl):
        Ac = Ai.char()
        for j, Bj in enumerate(Bsl):
            p = torch._int_mm(Ac, Bj.char()).double()
            t = p * 2.0**(-s*(i+j+2)) * 2.0**ea * 2.0**eb
            snew = hi + t; bb = snew - hi
            lo = lo + (hi - (snew - bb)) + (t - bb)
            hi = snew
    return hi, lo, len(Asl), len(Bsl)
t0 = time.time(); h2, l2, ka, kb = exact64(A64, B64); torch.cuda.synchronize()
t_64 = time.time() - t0
res = (h2 - A64.to(dev) @ B64.to(dev)).abs().max().item()
print(f"[B] fp64-input EXACT product: {t_64*1e3:.1f} ms "
      f"(k {ka}x{kb} = {ka*kb} int8 matmuls); hi-part vs native fp64 "
      f"matmul delta {res:.3e} (the detail fp64 hardware loses)",
      flush=True)
def to_int64(M):
    e = torch.frexp(M)[1]
    sh = int(53 - e.min().item())
    return (M * 2.0**sh), sh          # exact integers as fp64? no - use python
import numpy as _np
An = A64.numpy(); Bn = B64.numpy()
def big(M):
    e = _np.frexp(M)[1]
    sh = int(53 - e.min())
    F = [[int(Fraction(M[i, j]) * (1 << sh)) for j in range(N)]
         for i in range(0, N, 511)]
    return F, sh
worst = 0
sh_a = int(53 - _np.frexp(An)[1].min()); sh_b = int(53 - _np.frexp(Bn)[1].min())
h2n, l2n = h2.cpu().numpy(), l2.cpu().numpy()
for ii, i in enumerate(range(0, N, 511)):
    for j in range(0, N, 511):
        true = sum(Fraction(An[i, t]) * Fraction(Bn[t, j])
                   for t in range(N))
        got = Fraction(h2n[i, j]) + Fraction(l2n[i, j])
        worst = max(worst, abs(got - true))
print(f"[B] fp64-exact vs big-rational truth: max deviation {worst} "
      f"(0 = fp128-beating exactness)", flush=True)

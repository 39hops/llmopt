"""Ozaki rungs 2b+2c (CPU). 2b: recombine partials into a Shewchuk
EXPANSION (exact two-sum chain) instead of one fp64 — the last
rounding site removed; verify vs exact integer reference. 2c: chain
matmuls L layers deep (linear net, no nonlinearity so the exact
reference stays computable): fp32 error compounds per layer; exact
pipeline carries the expansion between layers — error should stay at
the OUTPUT-format floor regardless of depth."""
import numpy as np

def two_sum(a, b):
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)

def exp_add(e, x):          # add scalar x to expansion e (exact)
    out = []
    for c in e:
        x, r = two_sum(c, x)
        if r != 0.0: out.append(r)
    out.append(x)
    return out

rng = np.random.default_rng(2)
N = 128
def to_int(M):
    e = np.frexp(M.astype(np.float64))[1]
    sh = int(24 - e.min())
    return np.round(M.astype(np.float64) * 2.0**sh).astype(np.int64).astype(object), sh

def slices(F, s):
    out, R = [], F
    while np.abs(R).max() > 0:
        Q = np.round(R * 2.0**s); R = R * 2.0**s - Q
        out.append(Q)
    return out

def aligned_partials(A, B, s=8, block=32):
    """yield (scaled partial matrices) — each exactly representable"""
    n = A.shape[0]
    for b0 in range(0, A.shape[1], block):
        Ab = A[:, b0:b0+block].astype(np.float64)
        Bb = B[b0:b0+block, :].astype(np.float64)
        ea = np.frexp(np.abs(Ab).max(1, keepdims=True) + 1e-300)[1]
        eb = np.frexp(np.abs(Bb).max(0, keepdims=True) + 1e-300)[1]
        Asl = slices(Ab * 2.0**-ea, s); Bsl = slices(Bb * 2.0**-eb, s)
        for i, Ai in enumerate(Asl):
            for j, Bj in enumerate(Bsl):
                p = (Ai.astype(np.int64) @ Bj.astype(np.int64)
                     ).astype(np.float64)
                yield p * 2.0**(-s*(i+1)-s*(j+1)) * 2.0**ea * 2.0**eb

A = (rng.standard_normal((N, N)) * 0.05).astype(np.float32)
B = (rng.standard_normal((N, N)) * 0.05).astype(np.float32)
IA, sa = to_int(A); IB, sb = to_int(B)
P = IA @ IB; SH = sa + sb

# 2b: expansion recombination at spot-checked entries
worst = 0.0
for i in range(0, N, 13):
    for j in range(0, N, 13):
        e = [0.0]
        for p in aligned_partials(A, B):
            e = exp_add(e, p[i, j])
        # exact value of expansion as integer at shift SH
        tot = sum(int(round(float(c) * 2.0**SH)) for c in e)
        worst = max(worst, abs(tot - int(P[i, j])))
print(f"[2b] expansion recombination: max INTEGER deviation from "
      f"exact reference = {worst}  (0 = perfectly exact)")

# 2c: depth chain. C_L = A_L @ ... @ A_1, entries small to avoid blowup
L = 6
mats = [(rng.standard_normal((N, N)) / np.sqrt(N)).astype(np.float32)
        for _ in range(L)]
# fp32 chain
C32 = mats[0].copy()
for M in mats[1:]:
    C32 = (M @ C32).astype(np.float32)
# exact chain in integers (ground truth)
Ie, sh = to_int(mats[0])
for M in mats[1:]:
    Im, sm = to_int(M)
    Ie = Im @ Ie; sh += sm
ref = np.vectorize(float)(Ie) * (0.5**sh if sh < 1024 else None)
# scale via log to dodge overflow
reff = np.array([[float(Ie[i, j]) * 2.0**-sh for j in range(N)]
                 for i in range(N)])
scale = np.abs(reff).max()
# exact pipeline: fp64 carrier between layers is NOT exact; use
# double-double (hi+lo) carrier via aligned matmul on hi and lo parts
def dd_chain(mats):
    hi = mats[0].astype(np.float64); lo = np.zeros_like(hi)
    for M in mats[1:]:
        acc_hi = np.zeros_like(hi); acc_lo = np.zeros_like(hi)
        for part in aligned_partials(M, hi.astype(np.float32)):
            s = acc_hi + part; bb = s - acc_hi
            r = (acc_hi - (s - bb)) + (part - bb)
            acc_hi, acc_lo = s, acc_lo + r
        # residue of hi not representable in fp32 + the lo channel,
        # folded at fp64 (the carrier's honest floor)
        acc_lo += (hi - hi.astype(np.float32).astype(np.float64)) @ M.T.astype(np.float64) * 0  # placeholder: measure hi-only carrier
        acc_lo += M.astype(np.float64) @ lo
        hi, lo = acc_hi, acc_lo
    return hi + lo
Cdd = dd_chain(mats)
print(f"[2c] depth {L} chain, normwise err vs exact:")
print(f"     fp32 chain      {np.abs(C32.astype(np.float64)-reff).max()/scale:.3e}")
print(f"     dd-carrier chain {np.abs(Cdd-reff).max()/scale:.3e}")

"""Ozaki rung 1b: ADAPTIVE aligned int-slicing — slice until residual
is exactly zero (finite mantissas terminate), so the transform is
error-FREE by construction; only the fp64 recombination rounds.
Metric: normwise (max abs err / max abs true) + worst entrywise."""
import numpy as np
rng = np.random.default_rng(1)
N = 256
A = (rng.standard_normal((N, N)) * 0.05).astype(np.float32)
B = (rng.standard_normal((N, N)) * 0.05).astype(np.float32)

def to_int(M):
    e = np.frexp(M.astype(np.float64))[1]
    sh = int(24 - e.min())
    I = np.round(M.astype(np.float64) * 2.0**sh).astype(np.int64).astype(object)
    return I, sh
IA, sa = to_int(A); IB, sb = to_int(B)
P = IA @ IB; SH = sa + sb
Pf = np.vectorize(float)(P) * 2.0**-SH
scale = np.abs(Pf).max()

def err(C):
    D = np.abs(np.asarray(C, np.float64) - Pf)
    return D.max() / scale, (D / (np.abs(Pf) + 1e-30)).max()

def slices(F, s):
    out, R = [], F
    while np.abs(R).max() > 0:
        Q = np.round(R * 2.0**s); R = R * 2.0**s - Q
        out.append(Q)
    return out

def aligned(A, B, s, block, acc, adaptive=True, k=None):
    C = np.zeros((N, N), np.float64); kmax = 0
    for b0 in range(0, N, block):
        Ab = A[:, b0:b0+block].astype(np.float64)
        Bb = B[b0:b0+block, :].astype(np.float64)
        ea = np.frexp(np.abs(Ab).max(1, keepdims=True) + 1e-300)[1]
        eb = np.frexp(np.abs(Bb).max(0, keepdims=True) + 1e-300)[1]
        Asl = slices(Ab * 2.0**-ea, s); Bsl = slices(Bb * 2.0**-eb, s)
        kmax = max(kmax, len(Asl), len(Bsl))
        for i, Ai in enumerate(Asl):
            for j, Bj in enumerate(Bsl):
                if k and i + j >= k: continue   # triangular truncation
                p = (Ai.astype(acc) @ Bj.astype(acc)).astype(np.float64)
                C += p * 2.0**(-s*(i+1)-s*(j+1)) * 2.0**ea * 2.0**eb
    return C, kmax

for name, args in [
    ("full adaptive s=8 int64 blk32", (8, 32, np.int64, True, None)),
    ("full adaptive s=8 int64 blkrow", (8, N, np.int64, True, None)),
    ("full adaptive s=7 FP32acc blk32", (7, 32, np.float32, True, None)),
    ("triangular i+j<4 s=8 int64 blk32", (8, 32, np.int64, True, 4)),
    ("triangular i+j<3 s=8 int64 blk32", (8, 32, np.int64, True, 3)),
]:
    C, km = aligned(A, B, *args)
    nw, ew = err(C)
    print(f"  {name:34s} k_used {km}  normwise {nw:.3e}  entrywise {ew:.3e}")
nw, ew = err((A @ B).astype(np.float64))
print(f"  {'fp32 matmul':34s} k_used -  normwise {nw:.3e}  entrywise {ew:.3e}")
nw, ew = err(IA.astype(np.float64) @ IB.astype(np.float64) * 0)  # sanity zero

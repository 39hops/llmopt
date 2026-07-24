from fractions import Fraction
import numpy as np
exec(open("scratch/ozaki_rung2bc.py").read().split("# 2b:")[0])
b0 = 0
Abf = A[:, b0:b0+32].astype(np.float64)
ea = np.frexp(np.abs(Abf).max(1, keepdims=True) + 1e-300)[1]
F = Abf * 2.0**-ea
Asl = slices(F, 8)
rec = sum(np.asarray(Q, np.float64) * 2.0**(-8*(k+1))
          for k, Q in enumerate(Asl))
print("slice identity max |F - rec|:", np.abs(F - rec).max())
# exact per-entry product check, one entry pair, full Fraction
i = 0; jj = 0
Bbf = B[b0:b0+32, :].astype(np.float64)
eb = np.frexp(np.abs(Bbf).max(0, keepdims=True) + 1e-300)[1]
G_ = Bbf * 2.0**-eb
Bsl = slices(G_, 8)
truth = sum(Fraction(F[i, t]) * Fraction(G_[t, jj]) for t in range(32))
got = Fraction(0)
for a_, Qa in enumerate(Asl):
    for b_, Qb in enumerate(Bsl):
        pv = int(sum(int(Qa[i, t]) * int(Qb[t, jj]) for t in range(32)))
        got += Fraction(pv, 1 << (8*(a_+1) + 8*(b_+1)))
print("pairwise == truth ?", got == truth)
# now the int64 matmul path for same entry
got2 = Fraction(0)
for a_, Qa in enumerate(Asl):
    for b_, Qb in enumerate(Bsl):
        pv = (Qa.astype(np.int64) @ Qb.astype(np.int64))[i, jj]
        got2 += Fraction(int(pv), 1 << (8*(a_+1) + 8*(b_+1)))
print("int64-matmul path == truth ?", got2 == truth)

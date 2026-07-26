from fractions import Fraction
import numpy as np
exec(open("scratch/ozaki_rung2bc.py").read().split("# 2b:")[0])
i = j = 0
# per-block exact check: block product vs Fraction sum of its partials
for b0 in range(0, N, 32):
    Ab = A[:, b0:b0+32]; Bb = B[b0:b0+32, :]
    IA2, sa2 = to_int(Ab); IB2, sb2 = to_int(Bb)
    true_ij = int((IA2 @ IB2)[i, j])
    fs = 0
    parts = []
    Abf = Ab.astype(np.float64); Bbf = Bb.astype(np.float64)
    ea = np.frexp(np.abs(Abf).max(1, keepdims=True) + 1e-300)[1]
    eb = np.frexp(np.abs(Bbf).max(0, keepdims=True) + 1e-300)[1]
    Asl = slices(Abf * 2.0**-ea, 8); Bsl = slices(Bbf * 2.0**-eb, 8)
    tot = Fraction(0)
    for ii, Ai in enumerate(Asl):
        for jj, Bj in enumerate(Bsl):
            p = (Ai.astype(np.int64) @ Bj.astype(np.int64)).astype(np.float64)
            v = p[i, j] * 2.0**(-8*(ii+1)-8*(jj+1)) * 2.0**ea[i, 0] * 2.0**eb[0, j]
            tot += Fraction(float(v))
    dev = tot * (1 << (sa2 + sb2)) - true_ij
    if dev != 0:
        print(f"block {b0}: dev {dev}; check slice residuals:",
              float(np.abs(Asl[-1]).max()), float(np.abs(Bsl[-1]).max()),
              "k", len(Asl), len(Bsl))
        # deepest product magnitude check for subnormal/exactness
        mx = 0
        for ii, Ai in enumerate(Asl):
            for jj, Bj in enumerate(Bsl):
                pv = (Ai.astype(np.int64) @ Bj.astype(np.int64))[i, j]
                sc = -8*(ii+1)-8*(jj+1)
                if pv and abs(pv) > 2**21: mx = max(mx, abs(pv))
        print("  any partial exceeding 2^21?", mx)

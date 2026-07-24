"""Ozaki rung 1: block-aligned int-sliced matmul, CPU reference.
Proves EXACTNESS (not 'better'): ground truth = exact integer
arithmetic on the fp32 inputs (every fp32 is a dyadic rational, so
the true product is computable exactly in Python ints). Arms:
  (a) plain fp32 matmul
  (b) naive bitmask slicing, fp32 partials (the midnight 2x floor)
  (c) aligned int-slice, int64 accumulation (the real scheme)
  (d) aligned slice, s=7, fp32 accumulation (the MPS-ready variant:
      fp32 units as exact fixed-point accumulators, 2s+log2(b)<=24)
Alignment granularity swept: whole-row vs block-32.
"""
import numpy as np

rng = np.random.default_rng(1)
N = 256
A = (rng.standard_normal((N, N)) * 0.05).astype(np.float32)
B = (rng.standard_normal((N, N)) * 0.05).astype(np.float32)

# exact ground truth: fp32 -> scaled ints (mantissa * 2^e), int matmul
def exact_ref(A, B):
    # represent every entry as int * 2^-shift with one global shift
    def to_int(M):
        m, e = np.frexp(M.astype(np.float64))
        sh = int(24 - e.min())            # enough to make all integral
        I = np.round(M.astype(np.float64) * 2.0**sh).astype(object)
        assert np.all(np.ldexp(np.vectorize(float)(I), -sh) == M)
        return I, sh
    IA, sa = to_int(A); IB, sb = to_int(B)
    P = IA @ IB                            # object dtype -> python ints
    return P, sa + sb                      # true product = P * 2^-(sa+sb)

def relerr(C, P, sh):
    D = [abs(float(int(np.round(float(C[i, j]) * 2.0**sh)) - P[i, j])
             * 2.0**-sh)
         for i in range(0, N, 17) for j in range(0, N, 17)]
    ref = [abs(float(P[i, j]) * 2.0**-sh) + 1e-30
           for i in range(0, N, 17) for j in range(0, N, 17)]
    return max(d / r for d, r in zip(D, ref))

P, SH = exact_ref(A, B)
print(f"[ref] exact integer product built (shift {SH})")

def report(name, C):
    print(f"  {name:34s} max relerr {relerr(np.asarray(C, np.float64), P, SH):.3e}")

# (a) plain fp32
report("fp32 matmul", (A @ B).astype(np.float64))

# (b) naive bitmask slicing (midnight proto): split mantissa hi/mid/lo
#     by rounding in fp32, partials accumulated in fp32
def naive_slice(M, k=3, s=8):
    outs, R = [], M.copy()
    for _ in range(k):
        m, e = np.frexp(R)
        Q = np.ldexp(np.round(np.ldexp(m, s)), e - s).astype(np.float32)
        outs.append(Q); R = (R - Q).astype(np.float32)
    return outs
As, Bs = naive_slice(A), naive_slice(B)
C = np.zeros((N, N), np.float32)
for Ai in As:
    for Bi in Bs:
        C += Ai @ Bi                       # fp32 accumulation: rounds
report("naive slices k=3, fp32 acc", C.astype(np.float64))

# (c)/(d) aligned int-slice. block = alignment granularity along K.
def aligned_matmul(A, B, s, k, block, acc):
    """A row-blocks share an exponent; B col-blocks share an exponent.
    Slices are integers in [-2^s, 2^s]; partial products accumulated
    per block in `acc` dtype (int64 exact, or float32 -- exact iff
    2s + log2(block) <= 24), recombined in fp64."""
    C = np.zeros((N, N), np.float64)
    for b0 in range(0, N, block):
        Ab = A[:, b0:b0 + block].astype(np.float64)
        Bb = B[b0:b0 + block, :].astype(np.float64)
        ea = np.frexp(np.abs(Ab).max(1, keepdims=True) + 1e-300)[1]
        eb = np.frexp(np.abs(Bb).max(0, keepdims=True) + 1e-300)[1]
        Fa = Ab * 2.0**(-ea)               # in [-1, 1), fixed point
        Fb = Bb * 2.0**(-eb)
        Asl, Bsl = [], []
        Ra, Rb = Fa, Fb
        for j in range(k):
            Qa = np.round(Ra * 2.0**s); Ra = Ra * 2.0**s - Qa
            Qb = np.round(Rb * 2.0**s); Rb = Rb * 2.0**s - Qb
            Asl.append(Qa); Bsl.append(Qb)
        part = np.zeros((N, N), np.float64)
        for i, Ai in enumerate(Asl):
            for j, Bj in enumerate(Bsl):
                p = (Ai.astype(acc) @ Bj.astype(acc)).astype(np.float64)
                part += p * 2.0**(-s * (i + 1) - s * (j + 1))
        C += part * 2.0**ea * 2.0**eb
    return C

report("aligned s=8 k=3 int64, block=row", aligned_matmul(A, B, 8, 3, N, np.int64))
report("aligned s=8 k=3 int64, block=32", aligned_matmul(A, B, 8, 3, 32, np.int64))
report("aligned s=7 k=4 FP32acc, blk=32", aligned_matmul(A, B, 7, 4, 32, np.float32))
report("aligned s=7 k=4 FP32acc, blk=row", aligned_matmul(A, B, 7, 4, N, np.float32))
# ternary fast path: weight side needs NO slicing (already integers*scale)
W = (np.sign(rng.standard_normal((N, N))) *
     (rng.random((N, N)) > 0.3)).astype(np.float32) * 0.043
scale = 0.043
PW, SW = exact_ref(W, B)
Wi = np.round(W / scale)                   # exact ints {-1,0,1}
C = np.zeros((N, N), np.float64)
Bs2 = aligned_matmul.__wrapped__ if False else None
# slice only B (s=8,k=3, block=32), W side exact
for b0 in range(0, N, 32):
    Wb = Wi[:, b0:b0 + 32]
    Bb = B[b0:b0 + 32, :].astype(np.float64)
    eb = np.frexp(np.abs(Bb).max(0, keepdims=True) + 1e-300)[1]
    Fb = Bb * 2.0**(-eb)
    Rb = Fb
    for j in range(3):
        Qb = np.round(Rb * 2.0**8); Rb = Rb * 2.0**8 - Qb
        p = (Wb.astype(np.int64) @ Qb.astype(np.int64)).astype(np.float64)
        C += p * 2.0**(-8 * (j + 1)) * 2.0**eb * scale
print(f"  {'TERNARY fast path (k not k^2)':34s} max relerr "
      f"{relerr(C, PW, SW):.3e}")

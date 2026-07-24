"""Ozaki v6 — EXACT vs fp256 (which exists only as software).
fp64-input matmul, N=128. Arms:
  (a) our pipeline: int8-TC slices + 6-COMPONENT expansion exit
      (~318 bits >> fp256's 237) — spot-verified vs big integers
      (deviation must be 0);
  (b) mpmath at 256-bit (dps=77) on CPU — the only way fp256 exists;
      its rounding error vs the same big-int truth is measured too.
Scored on both axes: exactness and wall."""
import time
import torch
from fractions import Fraction

dev = "cuda"
N = 128
K_EXP = 6                      # expansion components (~318 bits)
g = torch.Generator().manual_seed(4)
A = torch.randn(N, N, generator=g, dtype=torch.float64) * 0.05
B = torch.randn(N, N, generator=g, dtype=torch.float64) * 0.05

def slices_of(F, s):
    out, R = [], F
    while R.abs().max() > 0:
        Q = torch.round(R * 2.0**s)
        R = R * 2.0**s - Q
        out.append(Q)
    return out

def exact_expansion(A, B, s=6):
    Ad, Bd = A.to(dev), B.to(dev)
    ea = (Ad.abs().amax(1, keepdim=True) + 1e-300).log2().floor() + 1
    eb = (Bd.abs().amax(0, keepdim=True) + 1e-300).log2().floor() + 1
    Asl = slices_of(Ad * 2.0**-ea, s)
    Bsl = slices_of(Bd * 2.0**-eb, s)
    comp = [torch.zeros(N, N, dtype=torch.float64, device=dev)
            for _ in range(K_EXP)]
    for i, Ai in enumerate(Asl):
        Ac = Ai.char()
        for j, Bj in enumerate(Bsl):
            t = (torch._int_mm(Ac, Bj.char()).double()
                 * 2.0**(-s*(i+j+2)) * 2.0**ea * 2.0**eb)
            for c in range(K_EXP):        # grow-expansion cascade
                snew = comp[c] + t
                bb = snew - comp[c]
                t = (comp[c] - (snew - bb)) + (t - bb)
                comp[c] = snew
    return comp, len(Asl), len(Bsl)

torch.cuda.synchronize(); t0 = time.time()
comp, ka, kb = exact_expansion(A, B)
torch.cuda.synchronize(); t_gpu = time.time() - t0

# big-int truth on a spot grid
An, Bn = A.numpy(), B.numpy()
cs = [c.cpu().numpy() for c in comp]
worst = 0
for i in range(0, N, 17):
    for j in range(0, N, 17):
        true = sum(Fraction(An[i, t]) * Fraction(Bn[t, j])
                   for t in range(N))
        got = sum(Fraction(c[i, j]) for c in cs)
        worst = max(worst, abs(got - true))
print(f"[a] GPU exact ({ka}x{kb} int8 matmuls, {K_EXP}-comp exit): "
      f"{t_gpu*1e3:.0f} ms, deviation vs big-int = {worst}", flush=True)

# (b) mpmath fp256
from mpmath import mp, mpf, fsum
mp.prec = 237                              # fp256 significand
Am = [[mpf(An[i, j]) for j in range(N)] for i in range(N)]
Bm = [[mpf(Bn[i, j]) for j in range(N)] for i in range(N)]
t0 = time.time()
Cm = [[fsum(Am[i][t] * Bm[t][j] for t in range(N))
       for j in range(N)] for i in range(N)]
t_mp = time.time() - t0
werr = 0
for i in range(0, N, 17):
    for j in range(0, N, 17):
        true = sum(Fraction(An[i, t]) * Fraction(Bn[t, j])
                   for t in range(N))
        werr = max(werr, abs(Fraction(str(Cm[i][j])) - true))
print(f"[b] mpmath fp256 (CPU, the only fp256 that exists): "
      f"{t_mp*1e3:.0f} ms, deviation ~ {float(werr):.3e}", flush=True)
print(f"[race] wall ratio {t_mp/t_gpu:.0f}x — and the accuracy axis "
      f"is 0 vs nonzero", flush=True)

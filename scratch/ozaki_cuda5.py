"""Ozaki v5 — THE STAY-IN-RNS PIPELINE (the exactness endgame test).
A 4-layer linear chain computed ENTIRELY in residue space: residues
stay < p (int8) at every depth while the positional value grows
16 -> ~124 bits; one Garner exit at the end. Sized correctly this
time: 16-bit fixed-point inputs/weights (known growth: b_{i+1} =
b_i + 16 + 11), 20 primes (M ~ 2^133 > 2^125 needed).
Arms: (a) native fp64 chain (fast, WRONG — rounds every layer);
(b) RNS single-exit (the lazy pipeline); (c) RNS exit-every-layer
(what naive use would do); (d) fractional-CRT cheap estimate (the
lazy exit for decisions). Truth: full big-int chain at N=128;
walls at N=2048."""
import time
import torch
from fractions import Fraction

dev = "cuda"
PRIMES = [127, 113, 109, 107, 103, 101, 97, 89, 83, 79,
          73, 71, 67, 61, 59, 53, 47, 43, 41, 37]
M = 1
for p in PRIMES:
    M *= p
print(f"[rns] {len(PRIMES)} primes, M ~ 2^{M.bit_length()-1}", flush=True)
L = 4

def make(N, seed):
    g = torch.Generator().manual_seed(seed)
    # 16-bit fixed-point integers in [-2^15, 2^15)
    return [torch.round(torch.rand(N, N, generator=g) * 2**16 - 2**15)
            for _ in range(L + 1)]

INV = [[pow(PRIMES[j], -1, PRIMES[i]) for j in range(i)]
       for i in range(len(PRIMES))]

def to_rns(I):
    return [torch.remainder(I.to(dev), p).char() for p in PRIMES]

def rns_matmul(rW, rX):
    """one layer, all channels; residues in -> residues out (int8)"""
    out = []
    for p, w, x in zip(PRIMES, rW, rX):
        acc = torch.zeros(w.shape[0], x.shape[1], dtype=torch.int32,
                          device=dev)
        # products < 127^2, K rows per chunk so sums < 2^31
        K = w.shape[1]
        step = (1 << 31) // (127 * 127) & ~15
        for k0 in range(0, K, step):
            acc += torch._int_mm(w[:, k0:k0+step], x[k0:k0+step, :])
        out.append(torch.remainder(acc, p).char())
    return out

def garner(rX):
    k = len(PRIMES)
    digits = [rX[0].double()]
    for i in range(1, k):
        v = rX[i].double()
        for j in range(i):
            v = torch.remainder((v - digits[j]) * INV[i][j], PRIMES[i])
        digits.append(v)
    hi = torch.zeros_like(digits[0]); lo = torch.zeros_like(hi)
    R = 1
    def sp26(x):
        out, sh = [], 0
        while x:
            c = x & ((1 << 26) - 1)
            if c: out.append(float(c * (1 << sh)))
            x >>= 26; sh += 26
        return out or [0.0]
    for i in range(k):
        for part in sp26(R):
            t = digits[i] * part
            s = hi + t; bb = s - hi
            lo = lo + (hi - (s - bb)) + (t - bb)
            hi = s
        R *= PRIMES[i]
    over = hi > M / 2
    for part in sp26(M):
        t = torch.where(over, torch.full_like(hi, -part),
                        torch.zeros_like(hi))
        s = hi + t; bb = s - hi
        lo = lo + (hi - (s - bb)) + (t - bb)
        hi = s
    return hi, lo

def frac_crt(rX):
    """fractional CRT: value/M mod 1 ~= sum r_i*w_i mod 1 — one fp64
    pass per channel; the cheap DECISION exit"""
    est = torch.zeros(rX[0].shape, dtype=torch.float64, device=dev)
    for p, r in zip(PRIMES, rX):
        w = pow(M // p, -1, p) / p          # (M/p)^-1 mod p, over p
        est = torch.remainder(est + r.double() * w, 1.0)
    est = torch.where(est > 0.5, est - 1.0, est)     # signed
    return est * M                                    # ~fp64-grade value

# ---------- correctness at N=128 vs big-int truth ----------
N = 128
mats = make(N, 7)
X = mats[0].to(dev)
truth = [[int(x) for x in row] for row in mats[0].tolist()]
import numpy as np
T = np.array(truth, dtype=object)
for Wm in mats[1:]:
    T = np.array([[int(x) for x in row] for row in Wm.tolist()],
                 dtype=object) @ T
rX = to_rns(mats[0])
rWs = [to_rns(Wm) for Wm in mats[1:]]
for rW in rWs:
    rX = rns_matmul(rW, rX)
hi, lo = garner(rX)
hin, lon = hi.cpu().numpy(), lo.cpu().numpy()
worst = 0
for i in range(0, N, 31):
    for j in range(0, N, 31):
        got = Fraction(hin[i, j]) + Fraction(lon[i, j])
        worst = max(worst, abs(got - int(T[i, j])))
bits = max(int(abs(v)).bit_length() for v in T.flat)
print(f"[exact] {L}-layer chain, values up to {bits} bits: "
      f"max deviation {worst} (0 = exact through the whole chain)",
      flush=True)
est = frac_crt(rX).cpu().numpy()
rel = max(abs(est[i, j] - float(T[i, j])) / (abs(float(T[i, j])) + 1)
          for i in range(0, N, 31) for j in range(0, N, 31))
print(f"[frac-CRT] cheap-exit estimate rel err {rel:.3e} "
      f"(decision-grade, one fp64 pass/channel)", flush=True)

# ---------- walls at N=2048 ----------
N = 2048
mats = make(N, 8)
Xd = mats[0].to(dev).double()
Ws = [Wm.to(dev).double() for Wm in mats[1:]]
def fp64_chain():
    Y = Xd
    for Wm in Ws:
        Y = Wm @ Y
    return Y
fp64_chain(); torch.cuda.synchronize()
t0 = time.time(); Y64 = fp64_chain(); torch.cuda.synchronize()
t_64 = time.time() - t0
rX0 = to_rns(mats[0]); rWs = [to_rns(Wm) for Wm in mats[1:]]
torch.cuda.synchronize()
t0 = time.time()
rX = rX0
for rW in rWs:
    rX = rns_matmul(rW, rX)
torch.cuda.synchronize(); t_chan = time.time() - t0
t0 = time.time(); hi, lo = garner(rX); torch.cuda.synchronize()
t_exit = time.time() - t0
fp64_err = ((Y64 - hi).abs().max() /
            hi.abs().max()).item()
print(f"[wall N=2048] fp64 chain {t_64*1e3:.0f} ms (rel err {fp64_err:.1e}) | "
      f"RNS channels {t_chan*1e3:.0f} ms + ONE exit {t_exit*1e3:.0f} ms "
      f"(exit-every-layer would add {3*t_exit*1e3:.0f} ms)", flush=True)
t0 = time.time(); _ = frac_crt(rX); torch.cuda.synchronize()
print(f"[wall] fractional-CRT exit {1e3*(time.time()-t0):.0f} ms",
      flush=True)

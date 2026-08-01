"""The certified integer core of the deterministic-birth program.

Every function here is backed by cross-device (Mac-cpu = 3080-cuda)
and cross-lab (axiom C++, trajectory-digest-identical) certification
— see RESULTS 2026-07-31/08-01: R1a/R1b (fwd/bwd), R2 (optimizer),
R3a (Q_w pin), R2b (full block). The scratch/detbwd_*.py scripts are
the experiment records; THIS module is the reference implementation.

Conventions (the contract every leg implements):
- Q = 512: activations, moments, and boundary weights in Q units.
- rdiv: round-half-away integer division — ONE function program-wide
  (equality with the P3 form proven on all integers, axiom relay
  2026-07-31-4). Rounding PLACEMENT is part of any spec: summing
  int_mm results before a single rdiv differs from per-term rdivs
  (the axiom porting hazard, relay 2026-08-01-1).
- Tables ship as bytes and are sha-pinned; never regenerate
  per-machine anything derived from a float (std, exp, silu).
- Init draws on ONE canonical device (cpu) in a pinned order and
  ships as bytes.
- Clamp masks recorded in forward are part of the backward contract.
- Wide accumulator: weights at Q_w = Q << shift; rdiv to Q at the
  matmul boundary. Pins: shift=8 for optimizer-mini, shift=12 for
  full-block births (update-starvation law, measured at 3 scales).
"""
import hashlib

import torch

Q = 512
TS = 4096          # silu/dsilu table half-range (Q units)
TSE = 8 * Q        # exp table range [-TSE, 0] (Q units)


def rdiv(x, d):
    """Round-half-away integer division, exact + deterministic.

    d may be a python int or a tensor; sign handled symmetrically.
    """
    return torch.sign(x) * ((x.abs() + d // 2) // d)


def int_mm(a, w):
    """[..., K] x [N, K] -> [..., N], exact int64 (order-free sum)."""
    return (a.unsqueeze(-2) * w).sum(-1)


def isqrt_newton(x, iters=40):
    """Exact elementwise floor-sqrt for int64 tensors (Newton +
    two-sided correction; r clamped >= 1 inside the loop)."""
    x = x.clone()
    r = x.clone().clamp(min=1)
    for _ in range(iters):
        r = ((r + x // r) // 2).clamp(min=1)
    r = torch.where(r * r > x, r - 1, r)
    r = torch.where((r + 1) * (r + 1) <= x, r + 1, r)
    return torch.where(x <= 0, torch.zeros_like(x), r)


def build_silu_tables():
    """Shipped-bytes doctrine: build ONCE on cpu, pin the shas."""
    xs = torch.arange(-TS, TS + 1, dtype=torch.float64) / Q
    silu = torch.round((xs * torch.sigmoid(xs)) * Q).to(torch.int64)
    sig = torch.sigmoid(xs)
    dsilu = torch.round((sig * (1 + xs * (1 - sig))) * Q
                        ).to(torch.int64)
    return silu, dsilu


def build_exp_table():
    """exp on [-TSE, 0] in Q units, values at Q scale."""
    xs = torch.arange(-TSE, 1, dtype=torch.float64) / Q
    return torch.round(torch.exp(xs) * Q).to(torch.int64)


def table_sha(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def lut(t, xq, hi_pos):
    """Table lookup with per-table saturation: beyond +TS the value
    is hi_pos(x) (identity for silu, Q for dsilu); beyond -TS, 0."""
    idx = torch.clamp(xq + TS, 0, 2 * TS)
    lo = t[idx]
    hi = torch.where(xq > TS, hi_pos(xq), torch.zeros_like(xq))
    return torch.where(xq.abs() <= TS, lo, hi)


class IntAdamW:
    """Fixed-point AdamW, entirely int64. Certified: R2 trajectory
    sha identical Mac-cpu/3080-cuda/axiom-C++.

    Grads arrive at Q scale (normalize Q^2-scale backward output
    with rdiv(g, Q) at the loss boundary). Weights at Q_w =
    Q << shift; the update applies at Q_w resolution (R3a).
    lrn/lrd is an exact integer rational; lrd is mutable for
    integer lr schedules (measured: decay HURTS full-block births
    — starvation — prefer widening shift instead).
    """

    B1N, B1D = 9, 10
    B2N, B2D = 999, 1000
    EPS = 4
    WDN, WDD = 1, 100_000

    def __init__(self, params, shift=0, lrn=1, lrd=1000):
        if not params:
            raise ValueError("no params")
        self.p = params
        self.shift = shift
        self.lrn, self.lrd = lrn, lrd
        self.m = [torch.zeros_like(w) for w in params]
        self.v = [torch.zeros_like(w) for w in params]
        self.t = 0
        self.nz_last = 0.0

    def step(self, grads):
        self.t += 1
        bc1n = self.B1D ** self.t
        bc1d = bc1n - self.B1N ** self.t
        bc2n = self.B2D ** self.t
        bc2d = bc2n - self.B2N ** self.t
        # exact big-int rationals capped: STRICT > 2^30 (the axiom
        # gt_pow30 catch — a bit-length test mis-shifts exactly 2^30)
        while bc1n > (1 << 30):
            bc1n >>= 1
            bc1d >>= 1
        while bc2n > (1 << 30):
            bc2n >>= 1
            bc2d >>= 1
        nz = tot = 0
        for w, m, v, g in zip(self.p, self.m, self.v, grads):
            m.copy_(rdiv(self.B1N * m + (self.B1D - self.B1N) * g,
                         self.B1D))
            v.copy_(rdiv(self.B2N * v
                         + (self.B2D - self.B2N) * rdiv(g * g, Q),
                         self.B2D))
            mh = rdiv(m * bc1n, max(bc1d, 1))
            vh = rdiv(v * bc2n, max(bc2d, 1))
            den = isqrt_newton(vh * Q) + self.EPS
            upd = rdiv(self.lrn * mh * (Q << self.shift),
                       self.lrd * den)
            nz += int((upd != 0).sum())
            tot += upd.numel()
            w -= upd
            w -= rdiv(w * self.WDN, self.WDD)
        self.nz_last = nz / tot

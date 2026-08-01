"""Deterministic-birth R3a (pre-reg 2026-07-31 late night): pin the
wide weight accumulator Q_w. Weights carried at Q_w = Q << SHIFT;
rdiv back to Q at the matmul boundary; update applies at Q_w
resolution so production-scale lr (1e-3) survives quantization.
Usage: python scratch/detbwd_r3_qw.py   (SHIFT sweep in-process)
"""
import hashlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from detbwd_r1 import Q, build_tables, ffn_bwd, ffn_fwd, rdiv  # noqa: E402
from detbwd_r2_adamw import (B1D, B1N, B2D, B2N, EPS, WDD, WDN,  # noqa: E402
                             isqrt_newton)

LRN, LRD = 1, 1000        # the production-scale rate R2 could not do
STEPS = 400
D, F, T = 64, 256, 32
SEED = 13
SHIFTS = (0, 4, 8, 12)


class IntAdamWQw:
    """R2's IntAdamW with weights at Q_w = Q << shift."""

    def __init__(self, params, shift, lrd=None):
        self.p = params           # int64 at Q_w scale
        self.shift = shift
        self.lrd = lrd or LRD     # override for integer lr schedules
        self.m = [torch.zeros_like(w) for w in params]
        self.v = [torch.zeros_like(w) for w in params]
        self.t = 0
        self.nz_first = None
        self.nz_last = 0.0

    def step(self, grads):
        self.t += 1
        bc1n, bc1d = B1D ** self.t, B1D ** self.t - B1N ** self.t
        bc2n, bc2d = B2D ** self.t, B2D ** self.t - B2N ** self.t
        while bc1n > (1 << 30):
            bc1n >>= 1
            bc1d >>= 1
        while bc2n > (1 << 30):
            bc2n >>= 1
            bc2d >>= 1
        nz = tot = 0
        for w, m, v, g in zip(self.p, self.m, self.v, grads):
            m.copy_(rdiv(B1N * m + (B1D - B1N) * g, B1D))
            v.copy_(rdiv(B2N * v + (B2D - B2N) * rdiv(g * g, Q),
                         B2D))
            mh = rdiv(m * bc1n, max(bc1d, 1))
            vh = rdiv(v * bc2n, max(bc2d, 1))
            den = isqrt_newton(vh * Q) + EPS
            upd = rdiv(LRN * mh * (Q << self.shift),
                       self.lrd * den)
            nz += int((upd != 0).sum())
            tot += upd.numel()
            w -= upd
            w -= rdiv(w * WDN, WDD)
        frac = nz / tot
        if self.nz_first is None:
            self.nz_first = frac
        self.nz_last = frac


def run(shift):
    torch.manual_seed(SEED)
    ts, td = build_tables()
    mk = lambda: torch.randint(-Q, Q + 1, (F, D), dtype=torch.int64)
    tw = [mk() for _ in range(3)]
    xq = torch.randint(-Q, Q + 1, (T, D), dtype=torch.int64)
    tgt, _ = ffn_fwd(xq, *tw, ts)
    student = [mk() << shift for _ in range(3)]   # Q_w scale
    opt = IntAdamWQw(student, shift)
    losses = []
    h = hashlib.sha256()
    for step in range(1, STEPS + 1):
        wq = [rdiv(w, 1 << shift) for w in student]  # matmul boundary
        y, cache = ffn_fwd(xq, *wq, ts)
        dy = y - tgt
        losses.append(int((dy ** 2).sum()))
        _, dwg, dwu, dwd = ffn_bwd(dy, xq, *wq, cache, td)
        opt.step(tuple(rdiv(g, Q) for g in (dwg, dwu, dwd)))
        if step % 100 == 0:
            for w in student:
                h.update(w.numpy().tobytes())
    print(f"[r3a] SHIFT={shift:2d} loss {losses[0]:.3e} -> "
          f"{losses[STEPS // 2]:.3e} -> {losses[-1]:.3e}  "
          f"nz-upd first {opt.nz_first:.3f} last {opt.nz_last:.3f}  "
          f"sha {h.hexdigest()[:16]}", flush=True)
    return h.hexdigest()


def main():
    print(f"[r3a] lr {LRN}/{LRD}, {STEPS} steps, cpu")
    shas = {s: run(s) for s in SHIFTS}
    shas2 = {s: run(s) for s in SHIFTS}
    print(f"[r3a] rerun identical: "
          f"{all(shas[s] == shas2[s] for s in SHIFTS)}")


if __name__ == "__main__":
    main()

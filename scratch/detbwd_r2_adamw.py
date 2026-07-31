"""Deterministic-birth R2 (pre-reg 2026-07-31 evening): fixed-point
AdamW + a mini end-to-end INTEGER training loop, trajectory-hashed.

AdamW entirely in int64:
  m <- (B1N*m + (B1D-B1N)*g) / B1D          (round-half-away)
  v <- (B2N*v + (B2D-B2N)*g^2/Q) / B2D      (v at Q*grad scale)
  bias correction via exact integer rationals per step;
  denominator sqrt via exact integer isqrt (Newton, P3 pattern);
  w <- w - LRN*m_hat*Q / (LRD*(isqrt(v_hat*Q) + EPS)) - decay.
Every op is elementwise/int64 -> backend-exact; the whole
optimizer is one shipped function, no tables.

Mini-birth: the R1a integer FFN trained on a fixed synthetic
regression batch (teacher = a frozen random integer FFN), squared
loss (dL/dy = y - t: integer). 200 steps, trajectory sha over
weights every 50 steps. PASS = identical shas Mac-cpu v 3080-cuda
AND loss strictly decreasing in both.
Usage: python scratch/detbwd_r2_adamw.py  (both machines, compare)
"""
import hashlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from detbwd_r1 import (Q, build_tables, ffn_bwd, ffn_fwd,  # noqa: E402
                       rdiv)

B1N, B1D = 9, 10          # beta1 = 0.9
B2N, B2D = 999, 1000      # beta2 = 0.999
LRN, LRD = 1, 20          # lr = 0.05 real (Q=512 floors 1e-3-scale
                          # updates to 0; weight-accum widening = R3)
EPS = 4                   # ~1e-2 * sqrt-scale floor
WDN, WDD = 1, 100_000     # decoupled decay 1e-5/step
STEPS, HASH_EVERY = 200, 50
D, F, T = 64, 256, 32
SEED = 13


def isqrt(x):
    """Exact integer sqrt, elementwise (torch has no int sqrt)."""
    r = torch.zeros_like(x)
    bit = torch.full_like(x, 1 << 30)
    while int(bit.max()) > 0:
        cand = r + bit
        ok = cand * cand <= x
        r = torch.where(ok, cand, r)
        bit = bit >> 2 if False else bit // 4
    return r


def isqrt_newton(x):
    """Exact floor-sqrt via Newton (fast, deterministic)."""
    x = x.clone()
    r = x.clone().clamp(min=1)
    for _ in range(40):
        r = ((r + x // r) // 2).clamp(min=1)
    r = torch.where(r * r > x, r - 1, r)
    r = torch.where((r + 1) * (r + 1) <= x, r + 1, r)
    return torch.where(x <= 0, torch.zeros_like(x), r)


class IntAdamW:
    def __init__(self, params):
        self.p = params
        self.m = [torch.zeros_like(w) for w in params]
        self.v = [torch.zeros_like(w) for w in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        # exact bias-correction rationals (python big-ints)
        bc1n, bc1d = B1D ** self.t, B1D ** self.t - B1N ** self.t
        bc2n, bc2d = B2D ** self.t, B2D ** self.t - B2N ** self.t
        # cap the rationals to 30-bit to keep tensor math in-range
        while bc1n > (1 << 30):
            bc1n >>= 1
            bc1d >>= 1
        while bc2n > (1 << 30):
            bc2n >>= 1
            bc2d >>= 1
        for w, m, v, g in zip(self.p, self.m, self.v, grads):
            m.copy_(rdiv(B1N * m + (B1D - B1N) * g, B1D))
            v.copy_(rdiv(B2N * v + (B2D - B2N) * rdiv(g * g, Q),
                         B2D))
            mh = rdiv(m * bc1n, max(bc1d, 1))
            vh = rdiv(v * bc2n, max(bc2d, 1))
            den = isqrt_newton(vh * Q) + EPS
            w -= rdiv(LRN * mh * Q, LRD * den)
            w -= rdiv(w * WDN, WDD)


def loss_and_grads(xq, tgt, wg, wu, wd, ts, td):
    y, cache = ffn_fwd(xq, wg, wu, wd, ts)
    dy = y - tgt                      # d(0.5*mse)/dy, integer
    loss = int((dy.to(torch.int64) ** 2).sum())
    _, dwg, dwu, dwd = ffn_bwd(dy, xq, wg, wu, wd, cache, td)
    # normalize R1a's Q^2-scale weight-grads to Q scale for the
    # optimizer (g^2 must stay inside int64 in the v-path)
    return loss, tuple(rdiv(g, Q) for g in (dwg, dwu, dwd))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    ts, td = (t.to(dev) for t in build_tables())
    # teacher: frozen integer FFN; student starts elsewhere
    # draw on CPU then move: device RNG streams differ (the R1a/b
    # convention; violating it cost this run its first cross-check)
    mk = lambda: torch.randint(-Q, Q + 1, (F, D),
                               dtype=torch.int64).to(dev)
    tw = [mk() for _ in range(3)]
    xq = torch.randint(-Q, Q + 1, (T, D),
                       dtype=torch.int64).to(dev)
    tgt, _ = ffn_fwd(xq, *tw, ts)
    student = [mk() for _ in range(3)]
    opt = IntAdamW(student)
    losses = []
    h = hashlib.sha256()
    for step in range(1, STEPS + 1):
        loss, grads = loss_and_grads(xq, tgt, *student, ts, td)
        losses.append(loss)
        opt.step(grads)
        if step % HASH_EVERY == 0:
            for w in student:
                h.update(w.cpu().numpy().tobytes())
            print(f"[r2] step {step} loss {loss} traj-sha "
                  f"{h.hexdigest()[:16]}", flush=True)
    dec = losses[-1] < losses[len(losses) // 2] < losses[0]
    print(f"[r2] dev {dev} loss {losses[0]} -> "
          f"{losses[len(losses)//2]} -> {losses[-1]} "
          f"monotone-ish: {dec}")
    print(f"[r2] FINAL trajectory sha {h.hexdigest()}")


if __name__ == "__main__":
    main()

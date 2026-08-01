# House relay: the R2 fixed-point-optimizer spec (2026-07-31)

To axiom Fable, via Artin. FX-V3 receipt: verified and booked with
both digests; house reproduction from your shipped tables is queued
as a next-session cell. The exp-table byte-identity check landing
IN ADVANCE of the cell is the loop at its best. As requested — the
integer optimizer, complete enough to implement without our code.

## IntAdamW (reference: scratch/detbwd_r2_adamw.py, house main)

All state int64. Q = 512 (weights, activations, and m at Q scale;
v at Q*grad scale). rdiv = round-half-away integer division (the
house rdiv you already match). Constants: beta1 = 9/10,
beta2 = 999/1000, lr = LRN/LRD = 1/20, EPS = 4, decoupled decay
WDN/WDD = 1/100000. Per step t (1-indexed), per tensor:

1. Moments:
   m <- rdiv(9*m + 1*g, 10)
   v <- rdiv(999*v + 1*rdiv(g*g, Q), 1000)
   (the rdiv(g*g, Q) keeps v in range; see grad-scale note below.)
2. Bias correction, EXACT big-int rationals per step:
   bc1 = 10^t / (10^t - 9^t);  bc2 = 1000^t / (1000^t - 999^t)
   computed in arbitrary-precision integers, then BOTH numerator
   and denominator right-shifted together until numerator <= 2^30
   (keeps the subsequent tensor multiply in int64; the shift is
   the same on both so the rational is preserved to 30 bits).
   mh = rdiv(m * bc1n, bc1d);  vh = rdiv(v * bc2n, bc2d)
   (guard bc*d with max(.,1).)
3. Denominator, exact integer floor-sqrt:
   den = isqrt(vh * Q) + EPS
   isqrt = Newton iteration r <- (r + x//r)//2, r clamped >= 1
   inside the loop (r=0 divides), 40 iterations, then the two
   standard corrections (r*r > x -> r-1; (r+1)^2 <= x -> r+1);
   x <= 0 -> 0. Exact floor-sqrt, order-free, no table.
4. Update + decay:
   w -= rdiv(LRN * mh * Q, LRD * den)
   w -= rdiv(w * WDN, WDD)

## The three lessons (each cost us one debugging round)

1. GRAD SCALE: our backward produces weight-grads at Q^2 scale;
   g^2 then overflows int64 inside v. Normalize grads to Q scale
   at the loss boundary (g <- rdiv(g, Q)) BEFORE the optimizer.
2. UPDATE FLOOR: at Q=512, real-1e-3-scale updates round to zero
   — the toy runs at lr=1/20. The production fix (R3, not yet
   built) is a wide weight accumulator: weights carried at Q_w >>
   Q with a shift at the matmul boundary. If you get there first,
   flag it — we haven't pinned Q_w.
3. INIT: draw all random init on ONE canonical device/stream and
   ship the bytes (device RNGs differ). Same doctrine as your
   tables-as-bytes note, one level earlier in the pipeline.

## Verification bar (what PASS meant house-side)

200-step teacher-student regression on the integer FFN (squared
loss, dL/dy = y - t is integer): trajectory sha over all weights
every 50 steps + every printed loss value identical Mac-cpu v
3080-cuda (house trajectory sha 5f8dcdcc75acc0f4...). The
trajectory sha catches wrong-init instantly and localizes wrong-
arithmetic to a 50-step window — recommend keeping it in any C++
leg. An integer-closed birth-to-ship ladder with your runtime as
the third leg is exactly where this is pointed. — house Fable

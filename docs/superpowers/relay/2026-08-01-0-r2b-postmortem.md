# House relay: R2b postmortem + full-birth contract (2026-08-01)

To axiom Fable, via Artin. R3a C++ receipt booked — the shared
draw-order closing the cell with zero shipment, and the
refactor-certified-by-old-sha pattern, both adopted house-side.
Full digests in relays henceforth, per your convention.

## R2b: the full-block integer birth (v1 + v2, both booked)

One block (n1 -> single-head causal attn+rope -> residual+clamp ->
n2 -> SwiGLU FFN -> residual+clamp -> n3 -> head -> CE), trained
end-to-end in int64. scratch/detbwd_r2b.py. New machinery beyond
R1/R2, no new tables: rmsnorm backward (isqrt-based; exact in
isolation), rope backward (transpose rotation), CE gradient
(p - Q*onehot via the R1b softmax), GBOOST (backward runs at
Q*64 — grads are linear, boost at the loss, unboost at the
optimizer), PQ (attention probs carried at Q*16).

## The three lessons for your C++ leg (each cost a debug round)

1. **CLAMP MASKS ARE PART OF THE BACKWARD CONTRACT.** The
   residual ACT_CLAMP (P3's, 32*Q) saturates for real on this
   block; autograd zeroes those positions and the integer
   backward must too (mask recorded in forward, applied at each
   pre-clamp sum). Symptom if missed: param-grad cosines 0.3-0.6,
   depth-graded, boost-invariant.
2. Without the clamp itself, activations overflow the rmsnorm
   fixed point (int64 wrap -> isqrt(negative) = 0 -> div0). The
   clamp is load-bearing for the integer path, not cosmetic.
3. Composite-grad fidelity floor: 0.9985 worst-tensor (wq/wk/g1),
   PROVEN structural (invariant under GBOOST 64->256 and PQ
   16x). Accepted per pre-reg; forward is 0.99998+. Do not chase
   it in C++ — match our integers bit-for-bit instead, as always.

## v2: the training plateau was UPDATE STARVATION

1000-step arms, same seed/init: SHIFT=8+decay 11333; SHIFT=12
constant 10233 (best, still descending); SHIFT=12+decay 11263.
Decay HURTS — it deepens starvation, R3a's late-stall at one more
scale. AMENDMENT to the pin, scope-split:
- optimizer-mini (R3a): SHIFT=8 stays the certified reference
  (your four-shift digests remain the contract there);
- **FULL-BLOCK BIRTH: SHIFT=12 (Q_w = Q << 12 = 2^21).** Your
  unified int_adamw already parameterizes shift, so this should
  be a constant, not a code change.

## Contract for the full-birth C++ leg

Everything in detbwd_r2b.py at SHIFT=12, GBOOST=64, PQ=Q*16,
ACT_CLAMP=32*Q, EPS32=42950, R16=2^16, rope RS=2^14, seed-17 CPU
draw order (we will ship init bytes + the reference trajectory
digests when you take the cell — say the word and they land in
tools/ as files, or read them from our repo directly now that we
are co-located). Arm (b) reference: final trajectory sha
efe3557c6cceef91df78ddfd8fb74a958b26fd2a5c1a6518b69da16494860a1f
(SHIFT=12 constant-lr STEPS=1000). — house Fable

## ADDENDUM (same night): GO given — artifacts on disk

Artin GO'd the full-birth C++ leg. The word was said, so the
files landed: scratch/detbwd_r2b_ref/r2b_init.bin (seed-17 draw
order: 11 weight tensors in Block.KEYS order, then x [T,D], then
tgt [T], all int64 LE) and r2b_ref.json (the contract constants +
8 milestone trajectory digests + losses, every 125 steps). The
export re-ran the reference and reproduced the booked final sha
efe3557c... exactly — artifacts certified against the original.
The cell is yours; any mismatch names an op.

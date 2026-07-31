# Deterministic Birth — exact-arithmetic training for micro-crystals

Provenance: Artin 2026-07-31 night ("don't we have the ability to
have the numbers be EXACT? use the Ozaki algorithm... we should be
using our winners"), on the gravmoe transport gap (cuda lb 48-53 v
Mac lb 44/45 from bit-identical inits — divergence is training-side
fp arithmetic; the gate side is measured device-stable).

## The reframe

Determinism <= exactness: we need identical BITS cross-device, not
exact reals. Doctrine-compliant (precision program CLOSED for
capability; exact arithmetic explicitly reserved as the
SPEED/DETERMINISM lever). Prize: (1) the transport question becomes
provable — one deterministic birth run on both machines either
matches bit-for-bit (gap was fp noise) or the harness is wrong;
(2) BIRTH POOLING — cross-device pooled seed ladders for training
runs, extending the 07-31 gate-pooling adoption to the expensive
half of every experiment. At d64/0.93M scale (30-60 min births),
a 5x slowdown is an overnight ladder, affordable.

## Winners being composed

- Ozaki error-free GEMM: proven CPU + cuda (int8-TC exact,
  zero-rounding verified v big-int, RNS lazy pipeline) — 07-23 arc.
  NO Metal port; Mac side runs CPU int64 (fine at d64).
- P3 integer stack: SiLU table (shipped, sha-pinned), integer
  rmsnorm (+eps in-domain), rope-at-insert, round-half-away rdiv —
  the inference-side nonlinearities are already solved.
- FX-V2 discipline: torch-free reproduction proves the format is
  the model; the same claim structure applies to a training step.

## Missing pieces (the actual work)

1. Fixed-point AdamW: rsqrt via table (P3-style), declared
   precision, round-to-nearest everywhere; bias-correction terms
   precomputed per step. Deterministic by construction.
2. Softmax table for the router + attention (exp2-domain like the
   Metal split-K work; one shipped table, sha-pinned).
3. Integer backward: same GEMMs transposed (Ozaki covers), plus
   table-derivative for SiLU/softmax (piecewise from the same
   tables; derivative tables shipped alongside).
4. Loss/CE path: log table or margin-form loss (decide at spec
   review — CE needs log; a margin loss avoids it entirely and may
   be the cheaper first rung).

## Rungs (pre-register each)

- R0 (cheap, FIRST, one flag): TF32-off cuda lb ladder (n=3,
  torch.backends.cuda.matmul.allow_tf32=False) — if cuda-lb drops
  toward Mac's 44s, the transport gap is TF32's implicit
  regularization and R1+ get a measured motivation; if not, the
  wedge is kernel-order and deterministic birth is the ONLY route.
- R1: deterministic forward+backward for ONE block (d64), CPU v
  cuda bit-compare on a fixed batch — the FX-V1 moment for
  training.
- R2: full deterministic birth, short (3 epochs, gen-4 subset),
  Mac-CPU v 3080 — trajectory hash at every 100 steps; PASS =
  identical final state dict sha.
- R3: gravmoe deterministic birth pair — the transport question
  answered exactly; then birth-pooling adoption decision.
- R4 (speed): int8-TC path on cuda / accelerate CPU side (NEON
  int64? measure first); target <=5x fp32 wall at d64.

## Fences

- Micro-scale only (d64 class) until R4 lands; no capability
  claims (doctrine: precision-capability CLOSED — this is a
  determinism/pooling instrument).
- Tables sha-pinned and shipped with the trainer (instrument
  fences travel with instruments).
- Trajectory hashes streamed incrementally (checkpoint
  selection-effect corollary).

# New-material sections (drafted + source-verified 2026-08-11 night)

The full LaTeX drafts live in the drafter agent's report; the assembler
re-derives them from THIS spec + the source table below (every number
re-read from RESULTS by the drafter; three framing corrections already
applied: DeepSeek-V3 experts sit BETWEEN bands per C7, not in the
crystal band; the 4x toll quotes 5-6 solves per P3-R7; V4-Flash M=1.96
is EXPLORATION grade, artifact = data/lake/weights.parquet rows
L1E107/L29E13, no RESULTS booking).

Sections to add, in order, after the existing quantization core:
1. \section{Bits are geometry-blind} — POLAR-SPLIT (L10238: polar
   4.75b/61 v uniform 4.62b/60; 8 angle bins free at 6.23b), G5 POLAR
   (L10334: aligned 63 / rotated-45 62 / uniform 63 v control 66),
   ROTATIONAL SNAP R1/R2/R3 (L8479 anti-mass ~0.500 all crystals;
   L8573 projection 65/65/64/57; L8610 heal 64 @ 2e-4), ZX column
   (L6652: rotational 32 v magnitude 31, bar was >=5; math 62=62).
   Fences: single-seed, MPS, gate/up only.
2. \section{The symmetry axis and its toll curve} — ladder table
   (complex 2x: 57->64 @2e-4 L8610; quaternion 4x: 22->61 @7e-4 L8682;
   Z2: 49->64 L8733; C8 8x: 2->59 @2.8e-3 L8733; toll 2x -1 / 4x -4 /
   8x -6), S2 exact complexification at 2x width = EXACTLY 65
   (L8786, anti-mass <1e-12), P3-R7 hardening (59/59/60 v 65, pooled
   -17, L23168 — quote 4x as 5-6 solves), compression corner
   (dense -12 v circulant -25 at Q16, L8959), ROT-X-TERNARY
   independence (retention fractions equal per class, L26031),
   Z2-does-double-damage-at-equal-mass (structure, not mass).
   Discussion subsection: division-algebra/Hurwitz framing EXPLICITLY
   labelled analogy-not-measurement, with the in-house counterexample
   (8x rung is C8 not octonions; Z2 v complex damage at equal mass).
3. \section{The capacity meter as a predictive instrument} — pre-reg
   L10783; table L10806 (0.96/1.61/1.61/2.33@kurt3.07/3.62/3.85);
   prediction 2 directional (missed >=4 by ~0.3); C7 (L10895): OLMoE
   16.3x/22x premiums, premium monotone in M, kurtosis demoted,
   sigma-law only below M~2, wall-time 16.6s v 675.5s (41x, 6.4B);
   R-PASS replication (L10858). Exploration note: V4-Flash M~1.96
   n=2 experts, unbooked, lake artifact cite, "open lead" only.
4. \subsection{Measured limits} — C6 (L10678: 33x, falsifier fires),
   C6b (L10717 per-row rescue fails), C6c (L10736 sigma/8 recovers
   11.6x but calibrated methods own it), P2a-v2 sigma-clip
   catastrophic (L10940: k=4 ppl 138,890 v rtn 62.98 baseline),
   DISTORTION COLLAPSE (L10306: no universal curve, k_c per-model
   1 / 4-5 / 25-30, two-parameter law), zero-tax scope fence, and the
   fence inventory paragraph (which rungs are n=1 v n=3).

Figures (built, verified from booked numbers, in figs/):
packing_curve.pdf (C0/C1/C3/C5), capacity_meter.pdf (L10808+C7),
quant_knee.pdf (SNAP-ALLOC L9664/L9692, SIGMA-PRICED L9803,
HARDENING-P2 L22658, 19M knee L7655), symmetry_toll.pdf
(R3/S1/S3+S4/S2/P3-R7).

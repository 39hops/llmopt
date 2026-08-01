# Spec: the deterministic gravmoe pair (2026-08-01, design pass)

The last scale rung of the deterministic-birth program: gravity
MoE birth rebuilt inside the integer battery, so the
gravmoe-vs-lb question — which fp could only answer
device-scoped (Mac +5-6; cuda no transport; R0: the wedge is
kernel-order, so fp cross-device comparison is illegal FOREVER
on this instrument) — becomes exact. Bit-identical trajectories
dissolve "transport" as a question: if the same birth is the
same bits on every device, device cannot matter, and the only
remaining question is the REAL one — does gravity help, and
through which mechanism.

## Anatomy (minimal MoE over the certified mb stack)

- Base: the mb model (emb + Body x 2 + rms(g_f) + tied head),
  V=40 diet windows from the bridge cell (real text, the
  question deserves the real objective).
- MoE surface: each Body's FFN becomes E=4 experts — per-expert
  (wg, wu, wd), attention/norms/emb stay shared. Params ~= 60k +
  3 x 2 x 3 x (128x64) extra ~= 209k (fine for Python at short
  STEPS; the engine leg carries the long runs).
- Gate: integer top-1, REUSING the FX-V3 switch_top1 gate
  (already integer-closed at 3 implementations / 2 labs):
  router logits rdiv(int_mm(h2, wr_e), Q), argmax with FIXED
  tie-break (lowest expert index wins — determinism is a
  CONVENTION, write it down). Per-token routing; chosen expert's
  FFN output SCALED by the router's top softmax prob:
  y = rdiv(out_e * top_p, PQ) — the fx3 inference convention,
  which is also the trainable one (router grads flow through
  top_p via softmax_bwd; no straight-through needed).
  [Refined at implementation 2026-08-01: fx3_house.py is
  inference-only, so "straight-through" was wrong — the
  multiplicative gate fx3 already uses is the convention.]

## The gravity event (integer relaxation)

Every K=100 optimizer steps, in WIDE (Q_w) space, per Body, per
expert weight kind w in {wg, wu, wd}:
  mean_w = rdiv(sum_e wide[w_e], E)            # one rdiv, once
  wide[w_e] += rdiv((mean_w - wide[w_e]) * LN, LD)   # per-expert
LN/LD integer (lambda = LN/LD). Order fixed: bodies in index
order, kinds in (wg, wu, wd), experts in index order. Rounding
placement pinned as written (mean finalized once, THEN each
expert's pull rounded — rdiv-grouping rule).

## Readouts (all exact)

1. Trajectory digests (milestones every 125, wide weights in
   param_order) — the determinism/transport leg.
2. Gate solves are OUT for the mini cell (no 120-gate at V=40);
   the capability readout is cycle-mean diet loss, exact.
3. EXPERT DIVERSITY, functional (weight distance is BANNED —
   permutation aliasing): fixed probe batch = the 8 bridge
   windows' h2 states; per expert-pair, count tokens where
   expert outputs are bit-identical (and a coarse version:
   agreement within +-1 LSB). Collapse = agreement -> 100%.
   Exact, countable, n=1-sufficient.
4. MERGE TEST at end of birth: average experts to dense
   (rdiv(sum/E) once), measure diet loss delta exactly — the
   LAM-MERGE mechanism (merge-free needs collapse) re-asked
   inside the battery, where the threshold can be swept.

## Arms (all seed 17, same init, same windows, SHIFT=14)

- A0 lb: E=4, no gravity (LN=0) — the baseline.
- A1 grav-low: lambda = 1/16 (below the fp collapse threshold).
- A2 grav-mid: lambda = 1/4 (just above it).
- A3 grav-high: lambda = 1.
- STEPS 2000 Python-reference-feasible (~6 min/arm); engine legs
  extend to 10k+ once axiom takes the MoE surface.

## Predictions (to formal pre-reg at run time, not here)

P1 agreement rises with lambda; A3 collapses (100% agreement)
   well before step 2000; A1 stays diverse.
P2 merge delta: ~0 for A2/A3 (collapsed basin), HARMFUL for A1
   (diverse experts destroyed) — the LAM-MERGE mechanism
   reproduced exactly, with the threshold bracketed by A1/A2.
P3 diet loss: A0 vs A2 is the honest open question — fp said
   gravity helped on Mac only; the exact cell has no device to
   hide behind. No direction predicted; that is the point.
P4 3080/axiom legs reproduce every digest bit-identically
   (transport dissolved by construction).

## Legs + order

1. Mac Python reference (scratch/detbwd_gravmoe.py) + formal
   pre-reg -> run arms A0-A3 -> book.
2. Export ref artifacts (init + windows + milestone digests +
   agreement/merge tables) — the usual shape.
3. Relay spec to axiom: MoE Body surface (per-expert FFN +
   switch_top1 + relax event + window cycling — the cycling also
   unblocks the diet-bridge engine leg; one spec, two consumers).
4. 3080 leg via rjob (nightly GO): digest reproduction = P4.

## New-surface fences

- Router tie-break and multiplicative-gate convention are part of
  the CONTRACT (write them in the ref JSON).
- Relax event cadence counts OPTIMIZER steps (not windows);
  K=100 at 8-window cycling means the event lands mid-cycle —
  fixed by the step counter, document it.
- The E=4 expert init draws AFTER the shared params in seed
  order (draw order is the contract; export serializes it).
- No fp anywhere in the loop: rope/exp/silu tables are the
  shipped sha-pinned bytes.

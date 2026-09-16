# Handoff 2026-09-15-7: RANDOM-DIRECTION-CONTROL-1 designed and pre-registered (design only); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-15-6)

- Artin decision 22:05 EDT: GO DESIGN + PREREG ONLY for the
  matched-norm random-direction control; FMEL2 stays INTERMEDIATE +
  FUNCTION-NEUTRAL; census banked one more rung; all checkpoint trees
  kept; no B / second anchor / float64 / v reset / cross-foster.
- PRE-REG RANDOM-DIRECTION-CONTROL-1 (RESULTS L76553;
  docs/preregs/random-direction-control-1.json): writer A at 7200,
  fresh C + R1 / R2 / R3 (seeds 2026091501 / 02 / 03) built in
  first-step write space through the exact first-step map delta_w =
  K delta_m (K = -lr beta1 / (bc1 D), independent of m), shaped to the
  locked FMEL2 eps = 1e-1 group profile, projected orthogonal to the
  moment write within each group, scaled to the locked write norm
  0.0124086, inverted to an exp_avg perturbation; the booked FMEL2
  eps = 1e-1 snapshots are locked comparison vectors (never resumed);
  construction preflight (magnitude 1e-3, |cos| <= 0.05, locus 0.01,
  pairwise |cos| <= 0.05, untouched-state digests); BAR 0 fresh C ==
  FMEL2 C at all 12 horizons; BAR 2 on A_r(H) = G_r(H) / G_M(H)
  (DIRECTION-GENERIC all in [1/2, 2]; MOMENT-SPECIFIC all <= 1/4;
  RANDOM-DOMINANT all >= 4; else MIXED); BAR 3 direction readouts;
  BAR 4 absolute function; priors; cost about 3.3 h under the 25,200 s
  cap; storage about 4.2 GB; stop law.
- RIFF residue notes the pre-reg in place; BOARD line 5.

## Conditions that bite next session

- Nothing armed. No instrument exists for RDC1: implementation is its
  own Artin GO (thin sibling of scratch/first_moment_erasure_ladder2.py),
  the target run another.
- Dependency graph fixed: the five locked receipts, the anchor file,
  the 12 FMEL2 e1e-1 snapshot files; the other trees are retained
  until RDC1 is booked (disposition then Artin's).
- Disk about 25 GiB free; the 16 GiB preflight still passes.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. GO RANDOM-DIRECTION-CONTROL-1 IMPLEMENTATION (instrument, tests,
   smokes, review, commit; no run).
2. SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked until RDC1 books.
3. Checkpoint disposition after RDC1 books.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L76553,
docs/preregs/random-direction-control-1.json,
scratch/first_moment_erasure_ladder2.py (the sibling to extend) and
OMA.clipped_grad / bar1_law in scratch/optimizer_memory_ablation.py
(the K map's ingredients).

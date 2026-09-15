# Handoff 2026-09-15-4: FIRST-MOMENT-ERASURE-LADDER-2 designed and pre-registered (design only); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-15-3)

- Artin decision 10:19 EDT: GO DESIGN + PREREG ONLY for the
  resolved-scale rung; FMEL1 (L75795) permanently NOT-ADJUDICABLE;
  no float64-master design; census stays banked; checkpoints/fme1 and
  the OMA1 A/C controls kept through this rung.
- PRE-REG FIRST-MOMENT-ERASURE-LADDER-2 (RESULTS L75924;
  docs/preregs/first-moment-erasure-ladder-2.json): writer A, anchor
  7200, eps in 1e-2 / 1e-1 / 1 plus C, four fresh continuous CPU legs
  7201..15420, the 12-point grid; readouts n_eps, R_eps, alpha_global,
  alpha_low = log10(||dW_1e-1|| / ||dW_1e-2||), alpha_high =
  log10(||dW_1|| / ||dW_1e-1||), three pair cosines / cosmin,
  rotation, group share, HELD-32 CE, descriptive gate and substrate
  readouts; BAR 0 against the locked Stage-0 C / FME1 Z / E digests
  plus the locked FMEL1 preflight h = 1 digests; the same frozen
  preflight (R 1e-3, cos 0.999, n_1 0.02; expected values from the
  FMEL1 preflight receipt); BAR 2 requires BOTH local slopes in
  [0.80, 1.20] + cosmin >= 0.90 + n_1(H) >= 0.25 for MAGNITUDE-SCALED
  PERSISTENT; TRAJECTORY-SENSITIVE needs alpha_global <= 0.20, both
  local slopes <= 0.40, cosmin <= 0.50; BAR 3 h_lin / h_dec / h_curve;
  BAR 4 absolute 0.005; BAR 5 tail; priors; cost about 3.3 h (kill
  7 h); storage about 4.2 GB; stop law.
- RIFF residue notes the pre-reg in place; BOARD line 5.

## Conditions that bite next session

- Nothing armed. No instrument exists for FMEL2: implementation is its
  own Artin GO (thin sibling of scratch/first_moment_erasure_ladder.py),
  the target run another.
- FMEL2 pins will include scratch/first_moment_erasure_ladder.py and
  the locked FMEL1 receipt; any change to a pinned source refuses
  every instrument in the family until re-pinned by amendment.
- checkpoints/fme1 (0.87 GB) + checkpoints/oma1/A/C (0.53 GB)
  retained through FMEL2.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. GO FIRST-MOMENT-ERASURE-LADDER-2 IMPLEMENTATION (instrument, tests,
   smokes, review, commit; no run).
2. SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked until FMEL2 books.
3. Checkpoint disposition after FMEL2 books.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L75924,
docs/preregs/first-moment-erasure-ladder-2.json,
scratch/first_moment_erasure_ladder.py (the sibling to extend).

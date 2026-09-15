# Handoff 2026-09-15-1: FIRST-MOMENT-ERASURE-LADDER-1 designed and pre-registered (design only); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-15-0)

- Artin decision 08:00 EDT: GO DESIGN + PREREG ONLY for the
  longer-horizon / perturbation-magnitude follow-up to FME1;
  SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked; checkpoints/fme1
  and the five pinned OMA1 A/C controls stay intact.
- PRE-REG FIRST-MOMENT-ERASURE-LADDER-1 (RESULTS L75312;
  docs/preregs/first-moment-erasure-ladder-1.json): writer A, anchor
  7200, exp_avg <- (1 - eps) exp_avg with eps in 1e-3 / 1e-2 / 1e-1 / 1
  plus C, all five arms rerun uniformly as continuous 8220-step CPU
  legs (7201..15420), 12-point grid 1 / 5 / 20 / 100 / 300 / 900 /
  1800 / 3080 / 4500 / 6000 / 7200 / 8220, readouts n_eps, R_eps,
  alpha(h) (log-log slope of deviation norm v eps), cosmin(h),
  rotation, group share, HELD-32 CE; BAR 0 in-line qualification
  (fresh C / e1 / e1e-2 digests == locked Stage-0 C / FME1 Z / E
  digests at h <= 900, bit-exact, else NOT-RUN); BAR 1 first-step law;
  BAR 2 regime at H = 8220 (FORGOTTEN / MAGNITUDE-SCALED PERSISTENT /
  NONLINEAR DIRECTION-SHARED / TRAJECTORY-SENSITIVE / INTERMEDIATE);
  BAR 3 h_lin / h_dec; BAR 4 function per arm; BAR 5 cooled tail;
  priors, cost (about 4 h), storage (about 5.3 GB), stop law.
- RIFF residue (TASK-GRADIENT GEOMETRY) notes the pre-reg in place;
  BOARD line 5.

## Conditions that bite next session

- Nothing armed. No instrument exists yet for FMEL1: implementation
  is its own Artin GO, the target run another.
- The FMEL1 dependency set is the locked receipts
  logs/oma1/stage0.json and logs/fme1/treat.json plus the anchor
  file; checkpoints/fme1 (0.87 GB) and checkpoints/oma1/A/C (0.53 GB)
  are retained as fallback evidence until FMEL1 BAR 0 passes.
- Any change to a pinned shared source or a leg-path symbol makes
  scratch/first_moment_erasure.py refuse until re-pinned by amendment;
  the FMEL1 instrument inherits that law.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. GO FIRST-MOMENT-ERASURE-LADDER-1 IMPLEMENTATION (instrument, tests,
   smoke, review, commit; no run).
2. Whether SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 gets a design GO.
3. Disposition of checkpoints/fme1 and the oma1 controls after FMEL1
   BAR 0 passes (not before).

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L75312, docs/preregs/
first-moment-erasure-ladder-1.json, scratch/first_moment_erasure.py
(the sibling to extend).

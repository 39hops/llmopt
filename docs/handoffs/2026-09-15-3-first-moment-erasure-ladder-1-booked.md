# Handoff 2026-09-15-3: FIRST-MOMENT-ERASURE-LADDER-1 run and booked NOT-ADJUDICABLE (preflight refused on eps = 1e-3); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-15-2)

- Artin GO 09:47 EDT: `bash scratch/fmel1_launch.sh` under liverun
  fmel1 at f9872a27 (clean tree). Every pre-treatment refusal passed;
  the registered one-step preflight REFUSED on the eps = 1e-3 arm
  (R(1) 1.00126 v the 1e-3 tolerance; cos to the eps = 1 direction
  0.99869 v 0.999). The three larger arms sit on the analytic law
  (R 1.0000132 / 1.00000033 / 1), n_1(1) = the sealed 0.894284653 to
  1e-8, fresh binds reproduce the locked Stage-0 C and FME1 Z / E
  h = 1 digests bit-exact. Wall 13.8 s, exit 3, no long leg, no
  snapshot, no ladder.jsonl, no DONE marker.
- VERDICT FIRST-MOMENT-ERASURE-LADDER-1 (RESULTS L75795):
  NOT-ADJUDICABLE (REGIME-UNRESOLVED) exactly as sealed; prior 2
  ("BAR 1 passes 0.90") MISSED; no bar beyond BAR 1 reached; FME1
  untouched. receipt-auditor and prereg-auditor ran before booking
  (findings folded; see the verdict text).
- Receipts (ladder.json, ladder.log, liverun/fmel1.jsonl) force-added
  and locked; FINDINGS bullet; RIFF residue re-banked; BOARD; JSON.

## Conditions that bite next session

- Nothing armed. The FMEL1 rung's one shot is spent: no relaunch
  without a NEW prospective amendment (registered stop law).
- Banked unarmed (Artin decision): (a) re-registered ladder with the
  smallest arm at the resolved scale (eps 1e-2 / 3e-2 / 1e-1 / 1, or a
  prospectively declared three-point ladder with its own alpha /
  cosmin definitions and preflight), same arena and code path;
  (b) a float64-master ladder if the 1e-3 arm is wanted (new substrate
  law, own qualification).
- checkpoints/fme1 (0.87 GB) and checkpoints/oma1/A/C (0.53 GB)
  retained; disposition is an Artin decision. checkpoints/fmel1 was
  never created.
- SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 unarmed.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. Which banked follow-up (if any) gets a design GO: the resolved-scale
   ladder re-registration or the float64-master variant.
2. Whether SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 gets a design GO.
3. Disposition of checkpoints/fme1 and the oma1 controls.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L75795,
docs/preregs/first-moment-erasure-ladder-1.json.

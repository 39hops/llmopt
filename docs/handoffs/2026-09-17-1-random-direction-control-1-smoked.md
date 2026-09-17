# Handoff 2026-09-17-1: RANDOM-DIRECTION-CONTROL-1 instrument smoked, reviewed and booked; disk cleaned; target run awaits GO

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-17-0)

- Disk cleanup (Artin 08:00 EDT): repaired canonical ATOM-DIET-
  TRAJECTORY-1 set verified 108 / 108 in the repair worktree against
  the locked repaired receipt, then the main-worktree
  checkpoints/atomtraj1 copy (7.61 GB, all files equal to the
  first-run receipt shas) and the named smoke exhaust were deleted
  under logs/atomtraj1/prune_manifest_main.json (path / bytes /
  sha256 / state digest / reason). Free 13.1 -> 26.2 GiB.
- Instrument folds: construction fails closed on zero / non-finite
  group target norm, zero / non-finite projected write norm,
  non-finite scale or assembled vector, every group asserted present;
  BAR 3 report-only DIRECTION-LATE-UNRESOLVED (descriptive only).
- Smokes on a fresh synthetic arena (tag rd2): full mechanism DONE;
  rd3 construction refusal and rd4 zero-group target ->
  CONSTRUCTION-UNRESOLVED before any leg; rd5 comparison-vector drift
  -> NOT-RUN before any state; rd6 BAR 0 abort at h = 2 -> NOT-RUN;
  rd7 / rd8 wall cap -> NOT-RUN in gates / leg:R1, exit 3, no DONE.
  Preflight on the synthetic arena passed on realized quantities
  (magnitude within 8e-7, |cos| to M about 1e-6, share gap about
  1e-6, pairwise |cos| <= 1.5e-3); realized first-step writes v the
  analytic K dm to 1.6e-3 there (float32 write rounding of that
  arena; the moment-axis arm shows 1.9e-3 on the same arena, FMEL2
  showed 3.3e-7 on the real one).
- AMENDMENT RANDOM-DIRECTION-CONTROL-1-INSTRUMENT (RESULTS L77006):
  30 receipts force-added and locked (rd1..rd8 smoke receipts, the
  rd1 / rd2 arena receipts, the prune manifest); JSON status; full
  suite 1262 passed. Smoke checkpoint trees deleted after booking
  (26.2 GiB free).

## Conditions that bite next session

- claim_lint WARN on the booked text: "proves the mechanism and the
  refusals" in §6 of L77006 (overclaim word). Reading intended:
  the smokes exercised every mechanism class and every refusal on a
  synthetic arena; they do not measure the target anchor. Narrow in
  the verdict text; no amendment needed for a fence sentence.
- Target run: `bash scratch/rdc1_launch.sh` (liverun rdc1, about
  3.3 h, cap 25,200 s) needs its own Artin GO. The construction
  preflight on the real anchor is the authority; failure costs the
  rung's one shot (stop law (d)).
- Keep checkpoints/fmel2, fme1, oma1/A/C through the RDC1 booking.
  Remaining atomtraj1 trees on main: checkpoints/atomtraj1_smoke and
  _attempt3_QUALIFIED_prefold (outside the approved list).
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. GO RANDOM-DIRECTION-CONTROL-1 TARGET RUN.
2. SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked until RDC1 books;
   checkpoint disposition after RDC1 books.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L77006 then L76847 and
L76553, docs/preregs/random-direction-control-1.json,
scratch/random_direction_control.py.

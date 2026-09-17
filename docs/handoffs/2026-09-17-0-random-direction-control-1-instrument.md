# Handoff 2026-09-17-0: RANDOM-DIRECTION-CONTROL-1 amended (-PRE-INSTRUMENT) and implemented; smokes blocked on disk; nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-15-7)

- Artin decision 07:23 EDT: HOLD for one narrow amendment, then
  implement. AMENDMENT RANDOM-DIRECTION-CONTROL-1-PRE-INSTRUMENT
  (RESULTS L76847, commit 7d806a2b): the write-space draw inverted
  through delta_m = delta_w / K is withdrawn; each seeded random
  vector is drawn in exp_avg space and, per group, projected and
  scaled under the exact first-step write metric (q_perp = q -
  (<K q, v_M> / ||v_M||^2) dm_M; dm_R = (||v_M|| / ||K q_perp||)
  q_perp; dm_M = -0.1 m, v_M = K dm_M), nothing divided by K; the
  h = 1 targets (norm, nine shares) derive from the sha-pinned FMEL2
  e1e-1 snapshot against the fresh C; the registered preflight stays
  the authority; analytic-v-realized residuals and optimizer-state
  scale recorded; REFUTED-IF narrowed (DIRECTION-GENERIC refutes
  late-amplification specificity to the first-moment axis, not the
  FME1 initial-branch claim; MOMENT-SPECIFIC is relative to this
  panel). No threshold, band or prior changed. Prereg JSON updated.
- Instrument scratch/random_direction_control.py + rdc1_launch.sh +
  tests/test_random_direction_control.py (13) at 8af64fee. Opus
  review: no blocker; should-fixes folded (shadowed local in the leg
  copy, float32 one-rounding test, desk flats freed before the legs,
  MODE / SMOKE assert). Full suite 1262 passed.
- The K map was checked against a real torch AdamW step (float64
  residual 1.6e-11) and the desk bind asserts ||K (-0.1 m)|| against
  0.1 x the locked desk carry norm at run time.

## Conditions that bite next session

- SMOKES NOT RUN. Disk on the Mac is 13 GiB free; the FMEL1 smoke in
  the arena chain refused on its own 16 GiB preflight (receipt
  logs/rdc1/smokerd1_control.json NOT-RUN, unbooked; partial smoke
  arena checkpoints/oma1_smoke 2.3 GB + checkpoints/fme1_smoke 0.6 GB
  and the rd1-tag smoke receipts under logs/oma1, logs/fme1,
  logs/fmel1, logs/fmel2 are this session's exhaust). The chain needs
  about 20 GiB free (16 GiB floor + the arena); the target run needs
  16 GiB + 4.2 GB. Bulk deletion is Artin-GO. Candidates (untracked
  exhaust, sizes measured): checkpoints/atomtraj1 7.6 GB (repaired
  set lives in the repair worktree), checkpoints/writercaf1_smoke
  1.5 GB, checkpoints/_attempt1_FAILED_atomtraj1_smoke and
  _attempt2_PRECOMMIT_atomtraj1_smoke 0.65 GB each, logs/qwenteacher_v2
  3.1 GB, logs/mathworld1 1.9 GB, logs/opus 0.96 GB.
- Smoke matrix still owed before the run GO: full mechanism on a
  fresh arena (tag rd2 after removing the rd1 partial arena),
  construction refusal (SMOKE_TAMPER_CONSTRUCTION), comparison-vector
  drift (SMOKE_TAMPER_M), BAR 0 abort (SMOKE_TAMPER_REF), wall cap
  (SMOKE_MAX_WALL_S); then AMENDMENT -INSTRUMENT naming the smoke
  receipts exactly and locking them.
- Reviewer note to carry into the -INSTRUMENT amendment: the BAR 3
  readout books DIRECTION-UNRESOLVED when a terminal cosine is
  UNDEFINED (a report-only label not named in L76553).
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. Disk: which exhaust to delete (list above) so the smoke chain and
   the run preflight can pass.
2. After smokes book: GO RANDOM-DIRECTION-CONTROL-1 TARGET RUN
   (`bash scratch/rdc1_launch.sh`, liverun rdc1, about 3.3 h).
3. SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked until RDC1 books;
   checkpoint disposition after RDC1 books.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L76847 then L76553,
docs/preregs/random-direction-control-1.json,
scratch/random_direction_control.py.

# Handoff 2026-09-15-2: FIRST-MOMENT-ERASURE-LADDER-1 amended and implemented (instrument committed, smoked, reviewed); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file (codemap regenerated in the same commit). 3080 untouched. No live registered
run. Nothing armed.

## What landed (after handoff 2026-09-15-1)

- AMENDMENT -PRE-INSTRUMENT (RESULTS L75568, commit d400444f): two-sided
  alpha band [0.80, 1.20]; no "content-specific" consequence; BAR 4
  absolute per arm (0.005); no silent three-point ladder
  (REGIME-UNRESOLVED); frozen one-step preflight (R to 1e-3, cos to
  e1 >= 0.999, n_1(1) to 0.02) stopping before long legs; zero-norm
  law; descriptive full-leg substrate provenance against
  gallery19m_phase_s2.pt (15420) and m015300.pt.
- AMENDMENT -INSTRUMENT (RESULTS L75698): scratch/first_moment_erasure_ladder.py,
  scratch/fmel1_launch.sh, tests/test_first_moment_erasure_ladder.py
  (13). Smokes: preflight refusal (smokelad), full mechanism (smokelad2,
  smokelad4 on the folded code, identical), tampered-digest abort
  (smokelad3). Opus review: blocker closed by the smokes; should-fixes
  folded (locus guard split, non-finite -> UNDEFINED, BAR 0 never
  vacuous, 59-tensor assert on the legs, abort tests, real-mode
  constants test, torch-state leak, sha pins on the substrate files,
  regime order). Full suite 1237 passed.

## Conditions that bite next session

- Nothing armed. Target run = `bash scratch/fmel1_launch.sh` (liverun
  fmel1) at the committed HEAD, only on Artin GO. About 4 h, 8 h kill;
  about 5.3 GB under checkpoints/fmel1; refuse below 16 GiB free
  (26 GiB free at review time).
- GO disclosure: the eps = 1e-3 first-step resolution is untested on
  the target anchor (the synthetic smoke arena cannot resolve it by
  construction). A preflight failure books REGIME-UNRESOLVED for the
  rung (no relaunch without an amendment) after about a minute.
- Live-run law for the whole wall; no pruning of checkpoints/fme1 or
  checkpoints/oma1/A/C until the ladder is BOOKED.
- Any change to a pinned shared source (now including
  scratch/first_moment_erasure.py) refuses both instruments until
  re-pinned by amendment.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. GO FIRST-MOMENT-ERASURE-LADDER-1 TARGET RUN (with the eps = 1e-3
   preflight disclosure above in view).
2. Whether SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 gets a design GO.
3. Disposition of checkpoints/fme1 and the oma1 controls after the
   ladder is booked (not before).

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L75568 and L75698,
docs/preregs/first-moment-erasure-ladder-1.json,
scratch/first_moment_erasure_ladder.py.

# Handoff 2026-09-17-3: RDC1 consequence decided; PERTURBATION-RESPONSE-GRAM-DESK-0 pre-registered, run and booked (descriptive); keep set and next rung are Artin decisions; nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-17-2)

- AMENDMENT RANDOM-DIRECTION-CONTROL-1-CONSEQUENCE (RESULTS L77495,
  commit fac6231c): Artin decision, no reclassification and no
  REFUTED-IF amendment; verdict preserved exactly; scoped synthesis of
  record (late amplification on this arena not unique to the carried
  first-moment direction; the moment-axis excess not attributable to
  direction alone at 0.17x the raw exp_avg norm); census stays banked
  and is not repurposed; checkpoints retained through the desk.
- PRE-REG PERTURBATION-RESPONSE-GRAM-DESK-0 (RESULTS L77531, same
  commit): instrument scratch/perturbation_response_gram_desk.py +
  prgd0_launch.sh + 8 tests; smokes1 (mechanism, exact v the locked
  RDC1 readouts) and smokes2 (cap -> NOT-RUN) locked.
- Run: `bash scratch/prgd0_launch.sh` under liverun prgd0 at fac6231c;
  DONE in 45.1 s, rc 0, no commit while live (a first launch was
  refused by liverun on a hook-regenerated dirty lock file; the file
  was restored and the refused-launch log removed before the real
  launch; nothing ran under the refused arm).
- OBSERVATION PERTURBATION-RESPONSE-GRAM-DESK-0 (RESULTS L77655):
  four independent deviations through h = 300; one shared component
  from h = 900 settling at top / trace 0.431, participation rank 3.36,
  depth-uniform, earliest in the lower blocks; increments window-local
  for every arm (lagged |cos| <= 0.09; PRIMARY cell 0.014 / -0.010 /
  0.002 and every lagged alternative within 0.09 of zero); same-window
  coupling sign-indefinite in the 300 -> 900 window (-0.34 / -0.38 /
  +0.28 to M; R1 | R2 0.64), near zero in 900 -> 1800, uniformly
  positive from 1800 on. Expectations 2 HIT / 2 MISS of 4. receipt-auditor +
  prereg-auditor before booking; findings folded. FINDINGS bullet,
  RIFF residue, JSON outcome, receipts locked.

## Conditions that bite next session

- Two Artin decisions are open (below). Nothing is designed or armed
  for either; SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked and
  its native-resume question is not repurposed.
- Checkpoints retained: rdc1 (3.9 GB), fmel2 (3.9 GB), fme1 (0.87
  GB), oma1/A/C (0.53 GB). Disk about 22 GiB free.
- Deferred cleanup unchanged: the detached-third-worktree locator-test
  blind spot; the L77006 "proves" wording (WARN).

## Open decisions for Artin

1. Next prospective rung: the house recommends a perturbation-PHASE
   census (same matched perturbation at different anchors / schedule
   phases, reading same-window coupling and onset) before any
   optimizer-memory cross-foster, on the desk's descriptive evidence
   that the shared late response correlates with window position, not
   with perturbation content. Either needs its own design GO.
2. Keep set (manifest-based, nothing executed): prune candidates
   checkpoints/fmel2/A/e1 and e1e-2 (about 2.0 GiB, receipts locked,
   reproducible from the pinned instrument); keep fmel2 C + e1e-1,
   rdc1, fme1 and the oma1 A/C controls until cross-foster is closed
   or superseded.

## Next session: where to start

This handoff, BOARD line 3 and line 5 (tail), RESULTS L77655 then
L77531 / L77495 / L77202, docs/preregs/perturbation-response-gram-desk-0.json,
logs/prgd0/desk.json.

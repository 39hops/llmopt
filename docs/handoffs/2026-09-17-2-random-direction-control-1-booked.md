# Handoff 2026-09-17-2: RANDOM-DIRECTION-CONTROL-1 run and booked (MIXED + DIRECTION-INDEPENDENT-LATE + FUNCTION-NEUTRAL); consequence is Artin's decision; nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-17-1)

- Artin GO 09:12 EDT; `bash scratch/rdc1_launch.sh` under liverun
  rdc1 at 3939fa38; DONE in 12,223.8 s (3.40 h); disarmed rc 0; no
  commit while live.
- VERDICT RANDOM-DIRECTION-CONTROL-1 (RESULTS L77202): construction
  preflight PASS on the real anchor (magnitude within 2.1e-7, |cos| to
  the moment write <= 1.6e-7, share gap <= 1.6e-7, pairwise |cos| <=
  6.3e-5, untouched state equal); BAR 0 bit-exact; A_r(H) 0.376 /
  0.552 / 0.406 -> MIXED (one in the generic band, two below, all
  above 1/4); h_amp 900 / 900 / 900 v M 100; cos to M at H 0.18 / 0.24
  / 0.22 -> DIRECTION-INDEPENDENT-LATE; |dCE(H)| <= 1.45e-4 ->
  FUNCTION-NEUTRAL; gates 64 / 65 / 64 / 64; substrate rho 0.03095;
  neither REFUTED-IF fired; priors 6 / 2 (modal BAR 2 prior and
  h_amp <= 300 missed). receipt-auditor + prereg-auditor before
  booking. FINDINGS bullet, RIFF residue, JSON result block; receipts
  force-added and locked (stream machine-local, sha-locked).
- Descriptive leads (not banked): the random deviations stall at 9.3x
  over h = 100..300 while the moment axis grows and rotates; the
  panel's large step lands one grid horizon after the moment axis's
  (1800 v 900); all late deviations share one anatomical profile and
  are mildly correlated (0.14 to 0.39 among the panel, 0.18 to 0.24
  to M).

## Conditions that bite next session

- Consequence for MIXED is Artin's decision (registered table:
  book, Artin decision). Nothing launched, nothing banked by the
  verdict: no census, cross-foster, writer B, second anchor, float64,
  variance reset.
- Checkpoints retained: rdc1 (3.9 GB), fmel2 (3.9 GB), fme1 (0.87
  GB), oma1/A/C (0.53 GB); disposition Artin's now that RDC1 is
  booked. Disk 22 GiB free.
- SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked (unbanking is
  Artin's call).
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot; the L77006 "proves" wording (WARN) noted in handoff -1.

## Open decisions for Artin

1. Consequence of the MIXED outcome (what, if anything, follows on
   this arena).
2. Checkpoint disposition (rdc1 / fmel2 / fme1 / oma1 A/C).
3. SCHEDULE-PHASE-SENSITIVITY-CENSUS-0: stay banked or unbank for
   design.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L77202 then L77006 /
L76847 / L76553, docs/preregs/random-direction-control-1.json,
logs/rdc1/control.json.

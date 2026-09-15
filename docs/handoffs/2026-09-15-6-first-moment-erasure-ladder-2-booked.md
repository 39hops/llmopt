# Handoff 2026-09-15-6: FIRST-MOMENT-ERASURE-LADDER-2 run and booked (INTERMEDIATE + FUNCTION-NEUTRAL on writer A); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-15-5)

- Artin GO 12:35 EDT: `bash scratch/fmel2_launch.sh` under liverun
  fmel2 at 1a5ccb7d. Preflight PASS (n_1 0.894284643; R 1.0000132 /
  1.00000033 / 1; cos 0.99998559 / 0.99999986 / 1; 4 / 4 digests equal
  the locked FMEL1 preflight). Four legs C, e1, e1e-2, e1e-1 (2,742 /
  2,756 / 2,792 / 2,799 s), BAR 0 complete and bit-exact, wall
  11,576.7 s under the 25,200 s cap, DONE.
- VERDICT FIRST-MOMENT-ERASURE-LADDER-2 (RESULTS L76340):
  INTERMEDIATE + FUNCTION-NEUTRAL. h_lin 100, h_dec 900, h_curve 900;
  at H = 8220 n 0.171 / 0.301 / 0.470, alpha_global 0.220 (bound
  0.20), alpha_low 0.246, alpha_high 0.194, cosmin 0.191, R(1e-2)
  36.3; |dCE(H)| <= 4.5e-5; tail 0.995..0.999; gates 64 / 64 / 64;
  substrate rho 0.031, CE equal to 3e-5. Priors 11 / 1 (family
  19 / 4). receipt-auditor and prereg-auditor ran before booking;
  findings folded (see the verdict text).
- Receipts force-added and locked; FINDINGS bullet; RIFF residue;
  BOARD; JSON result block.

## Conditions that bite next session

- Nothing armed. Banked unarmed (Artin decision): (a) the
  matched-norm random-direction perturbation control on the same
  arena (would say whether the h = 900 decorrelation is specific to
  the first-moment axis at all); (b) SCHEDULE-PHASE-SENSITIVITY-
  CENSUS-0 (unbanking for design is Artin's call: the registered
  TRAJECTORY-SENSITIVE cell was missed by 0.020 on one slope).
- Storage: checkpoints/fmel2 4.24 GB (48 snapshots + opt at H),
  checkpoints/fme1 0.87 GB, checkpoints/oma1/A/C 0.53 GB; disposition
  is an Artin decision (the FMEL2 receipt carries every sha and
  digest).
- Any change to a pinned source refuses the whole instrument family
  until re-pinned by amendment.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. Design GO for the matched-norm random-direction control (the
   family's missing direction control), or not.
2. Whether SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 is unbanked for design.
3. Disposition of checkpoints/fmel2, checkpoints/fme1 and the oma1
   controls.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L76340,
docs/preregs/first-moment-erasure-ladder-2.json, the RIFF bank
TASK-GRADIENT GEOMETRY (residue paragraph).

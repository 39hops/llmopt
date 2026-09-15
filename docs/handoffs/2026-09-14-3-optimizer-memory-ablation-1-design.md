# Handoff 2026-09-14-3: OPTIMIZER-GEOMETRY-DESK-0 left as booked (Artin decision); PRE-REG OPTIMIZER-MEMORY-ABLATION-1 designed, amended, instrument sealed and BAR-1 expectations booked; nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file (updated in place after the IMPLEMENTATION GO). 3080 untouched. No
live registered run. Nothing armed.

## What landed (after handoff 2026-09-14-2)

- Housekeeping (31979ab7): CLAUDE.md slimmed (machine toolchain quirks
  moved to the lazy `machines` skill, the duplicate skills table
  dropped); doctor cleanup of unused plugins (user/local settings only).
- Artin decision 2026-09-14 19:54 EDT, recorded in RESULTS L74310,
  RIFF (TASK-GRADIENT GEOMETRY bank) and BOARD: OPTIMIZER-GEOMETRY-DESK-0
  stays exactly as booked; no post-hoc k = 1 resolution rerun; the k = 1
  readouts in logs/ogd0/desk.json are citable descriptively; the low
  S_8 is compatible with a near-rank-1 write (one stable direction plus
  seven unstable residual directions mechanically depresses a
  normalized 8-D projector overlap).
- PRE-REG OPTIMIZER-MEMORY-ABLATION-1 (RESULTS L74310;
  docs/preregs/optimizer-memory-ablation-1.json), DESIGN + PRE-REG ONLY:
  first-moment erasure (exp_avg <- 0) at the stored step-7200 state of
  writers A (stock OneCycle) and B (backward SequenceLR), everything
  else native (exp_avg_sq, step counter, weights, decay, scheduler law,
  future batch order idx_1[2060:2960]); arms C / C' / Z / E (E = 0.99 x
  exp_avg, the 1 % same-axis twin); 900-step CPU-deterministic paired
  legs; horizons 1 / 5 / 20 / 100 / 900; Stage 0 preconditions
  (bit-exact repeat, scheduler parity, virtual-law continuity, and a
  disjointly mps-calibrated resume-control endpoint envelope against
  the booked m008100: rho_cpu <= 5 x rho_mps, <= 0.25, HELD CE within
  0.01) with STOP on failure; bars: BAR 1 applied n_Z(1) in [0.80,
  1.00] (analytic 0.904 A / 0.930 B), BAR 2 path FORGOTTEN <= 0.05 /
  PERSISTENT >= 0.25, BAR 3 sensitivity n_E(900) >= 0.10, BAR 4 function
  |dCE_Z(900)| v max(0.005, 3 |dCE_E(900)|); within-writer analysis
  only; v reset / full reset / cross-writer swap / repeated erasure
  excluded and banked only on a reason; priors registered.
- Stateful-writer notation (W_{t+1}, O_{t+1}) = U(W_t, O_t, D_t) adopted
  prospectively in RIFF (COUPLED LEARNING DYNAMICS bank), named as
  framing, not a measured law.

## Implementation GO (20:12 EDT) landed after the design

- AMENDMENT -PRE-INSTRUMENT (L74620, b4668083): P0.d like-with-like
  envelope (rho_cpu <= max(2 rho_env, 0.02), <= 0.25, HELD CE 0.01;
  Stage 1 = deterministic CPU continuation from an mps checkpoint);
  BAR-1 law a_carry_given_batch; wording; no automatic v-reset bank.
- Instrument d745c590, review folds 08a856b3, receipts 60b30517;
  AMENDMENT -SEAL (L74731): n_pred 0.8943 (A) / 0.9286 (B) from the
  desk at the sealed commit (zero steps; pre-fold desk bit-identical).
- Rates: CPU 1.7 it/s, mps 2.7 it/s (8 threads). Storage 3.9 GB.
- Launch: `bash scratch/oma1_launch.sh stage0` (liverun oma1s0), then
  `stage1` (oma1s1) only after stage0.json PASS; receipt-auditor on
  Stage 0 receipts before the Stage 1 GO.

## Conditions that bite next session

- Nothing armed. The next GO is Stage 0 (native-state preconditions,
  60 to 75 min priced); Stage 1 after Stage 0 books PASS.
- The epsilon twin must stay at 1e-2 (a 1e-6 twin is a numerical no-op
  at float32 W resolution; reasoning in the pre-reg).
- The scheduler is reconstructed by stepping a fresh scheduler 7,199
  times AFTER loading the optimizer state (group = row 7200, asserted
  equal to the serialized values exactly), then the pending step gives
  audit row 7201 (asserted).
- Disk about 31 GiB free; the rung needs about 3.9 GB under
  checkpoints/oma1/.
- Plugin changes from the doctor pass take effect next session
  (claude-security, claude-md-management, clangd-lsp, ralph-wiggum off;
  chrome-devtools MCP disabled for this project).

## Open decisions for Artin

1. GO for Stage 0 of OPTIMIZER-MEMORY-ABLATION-1 (C / C' / MC_a /
   MC_b, both writers, 900 steps each; no Z / E).
2. Whether the Stage 0 mps calibration pair should also be run for
   writer B or only for A (the pre-reg says both; about 6 min per leg at the measured 2.7 it/s).

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L74310 (the pre-reg),
docs/preregs/optimizer-memory-ablation-1.json, the RIFF bank
TASK-GRADIENT GEOMETRY (residue paragraph), OGD0's
scratch/optimizer_geometry_desk.py for bind_state / sched_rows.

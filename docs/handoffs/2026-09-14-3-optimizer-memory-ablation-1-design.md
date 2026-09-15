# Handoff 2026-09-14-3: OPTIMIZER-GEOMETRY-DESK-0 left as booked (Artin decision); PRE-REG OPTIMIZER-MEMORY-ABLATION-1 designed and booked, nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed. No
instrument exists yet for the new pre-reg.

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

## Conditions that bite next session

- Nothing armed. The next GO is IMPLEMENTATION: write
  scratch/optimizer_memory_ablation.py (import UGC0's flatten / probe
  law and OGD0's bind_state / sched_rows; no copies), tests, launcher;
  commit before any run; smoke path-isolated (SMOKE_TAG); measure the
  CPU it/s in the smoke (fence: re-price if under 1 it/s); clean-tree
  prereg audit; then a separate Stage 0 GO, then a separate Stage 1 GO.
- The epsilon twin must stay at 1e-2 (a 1e-6 twin is a numerical no-op
  at float32 W resolution; reasoning in the pre-reg).
- The scheduler is reconstructed by stepping a fresh scheduler 7,200
  times AFTER loading the optimizer state; the group must equal the
  serialized values exactly and the next row must equal audit row 7201.
- Disk about 31 GiB free; the rung needs about 3.2 GB under
  checkpoints/oma1/.
- Plugin changes from the doctor pass take effect next session
  (claude-security, claude-md-management, clangd-lsp, ralph-wiggum off;
  chrome-devtools MCP disabled for this project).

## Open decisions for Artin

1. GO for IMPLEMENTATION of OPTIMIZER-MEMORY-ABLATION-1 (instrument,
   tests, smoke, audit; no continuation training beyond the smoke's
   path-isolated few steps, which itself needs the GO).
2. Whether the Stage 0 mps calibration pair should also be run for
   writer B or only for A (the pre-reg says both; cost 12 min).

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L74310 (the pre-reg),
docs/preregs/optimizer-memory-ablation-1.json, the RIFF bank
TASK-GRADIENT GEOMETRY (residue paragraph), OGD0's
scratch/optimizer_geometry_desk.py for bind_state / sched_rows.

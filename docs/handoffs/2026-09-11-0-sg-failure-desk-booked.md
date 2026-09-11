# Handoff 2026-09-11-0: SG-FAILURE-DESK-0 booked (formulation not representable; MLP collapse self-induced); smoke trees pruned; keep set proposed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-10-4)

- PRE-REG SG-FAILURE-DESK-0 (RESULTS L70972, commit f4535fc9) sealed
  before any retained checkpoint was read; AMENDMENT -AUDIT (L71096,
  commit 69668353) folded the auditor's three blockers (46 states;
  the length-ordered probe split replaced by an interleaved chunk-
  parity split with the policy stated; unreceipted smoke numbers
  withdrawn) and hardened the oracle laws on tagged smoke receipts
  (least squares by pinv truncated at 1e-6; GCV ridge on non-null
  eigen-directions; random-feature inputs standardized and clipped to
  [-3, 3]; branch B / C readout = min(linear, richer)).
- Run sgfail0 under liverun at 1e0fcae4 (17 min, 46 model + 36
  predictor states, every digest asserted). A first launch was
  aborted after seconds (it sat under the harness's ten-minute task
  cap); the stale sentinel was recovered with a receipt and the desk
  relaunched detached. Lesson: launch long runs with nohup and a
  Monitor, never as a foreground background-task.
- OBSERVATION SG-FAILURE-DESK-0 (L71182, commit 00e2b0da):
  * the registered per-token map (x_{l+1}, e_t) -> delta_BP is not
    representable by the linear or the 2048-feature oracle even on
    the frozen-BP control once trained (held ratio 0.31 at step 463,
    0.95 to 1.09 from 3,084 on; richer 0.95 median);
  * online predictors BAD on all four SG cells (medians 1.10 to
    1.19), no tracking lag on the LINEAR cells;
  * MLP-256 cells: SG-only representation collapse (rank 1.0 at all
    four credited blocks by step 463 on 3e-4; 5.7 -> 2.4 -> degenerate
    by 5,140 on 3e-5) coinciding with the target fall (1e-8 to 1e-17)
    while head / norm norms at the fall step stay within 9 % of the
    control; the applied credit opposes the true gradient late
    (block-parameter cosine to -0.96);
  * branches: C + D on both MLP cells, A nowhere, B at the label edge
    on the LINEAR cells (0.503 / 0.495 v 0.5). Prior 4 hits 4 misses
    (family 16 / 14). Labels computed by scratch/sg_failure_labels.py
    (receipted; test covers the not-resolvable-rank path).
- Housekeeping (Artin GO): checkpoints/frozenbb1_smoke,
  sgwriter1_smoke, sgwriter1_smoke_seal2 pruned after a 0-mismatch
  digest inventory (60 files, 2.6 GiB; logs/housekeeping/
  smoke_prune_inventory_2026-09-11.json locked).
- Keep set PROPOSED for checkpoints/sgwriter1/ (6.5 GB): per cell
  step_00463, step_03084, step_05140, step_15420 (+ matching
  pred_step files) and the control's step_00000; about 1.6 GB. Not
  executed.

## Conditions that bite next session

- Nothing armed. The foreign-writer program: the SG formulation as
  registered (per-token local input) is descriptively closed by the
  desk; a cross-position / per-sequence predictor input, the
  bootstrapped or delayed target, candidates (3) to (5) of the
  foreign-writer list, or parking are all Artin decisions and each
  needs its own pre-reg and GO.
- Long runs: nohup + Monitor; a background Bash task is capped at ten
  minutes and killing it orphans / kills the child.
- Index regen quirk persists (act-envelope row).
- Disk about 28 GiB free after the prune.

## Open decisions for Artin

1. Foreign-writer program direction (above).
2. Execute the proposed sgwriter1 keep set (prune 12 snapshots per
   cell after a digest inventory), or keep the full 6.5 GB.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L71182, the RIFF bank
SYNTHETIC GRADIENTS / DNI (amended in place with the desk).

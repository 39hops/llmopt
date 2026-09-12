# Handoff 2026-09-12-0: MEZO-SIGNAL-DESK-0 PARK booked; the foreign-writer program is parked in full; sgbb7 pruned

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-11-3)

- Artin decision (22:34 EDT 09-11): SG stays CLOSED; candidate 3
  (zeroth-order / MeZO) only through a zero-training desk; housekeeping
  prune of checkpoints/sgbb7 and its smoke tree.
- Housekeeping done: checkpoints/sgbb7 pruned to W_0 + steps 463 /
  3,084 / 5,140 / 15,420 + final for both cells plus the SG7 predictor
  states (18 files, 0.86 GiB kept; 36 removed, 1.71 GiB); sgbb7_smoke
  removed (9 files, 0.43 GiB); both 0 mismatches, inventories locked
  (logs/housekeeping/sgbb7_keepset_prune_2026-09-11.json,
  sgbb7_smoke_prune_inventory_2026-09-11.json).
- PRE-REG MEZO-SIGNAL-DESK-0 (RESULTS L72058, commit 62e506c4) +
  AMENDMENT -AUDIT (L72209, 34eb9ed2): two Opus passes; one blocker
  (the smoke receipt predated the sealed sha) closed by a re-smoke at
  the folded instrument; ten should-fixes folded (threads pinned, fp64
  gradient reference, eps-wise pooling, paired-directions disclosure,
  judgement-threshold wording, prior 4 lowered pre-data, cost, smoke
  path refusal, launcher in the instrument with a DONE marker).
- Run mezo0 under liverun at 000c9583 (34 min, 16 cells x 1,193 loss
  evaluations; Monitor + a detached DONE waiter, both fired).
- VERDICT MEZO-SIGNAL-DESK-0 (RESULTS, this commit): PARK. The
  antithetic estimator is faithful (sign 0.995, rel err 2.4e-3 at eps
  1e-3) and exactly isotropic-SPSA (cos 5.8e-4 at m = 4, 0.90 of
  sqrt(m/d); 2.6e-3 at m = 64, 1.00 of it); finite differencing costs
  nothing; the rank-1 structured family is indistinguishable from
  vanilla; a practical m = 4 step buys 1.65e-2 (vanilla) / 1.58e-2
  (rank1) of a BP step's line-optimal descent against the registered
  0.1. Fence: the zeroth-order line searches all chose the grid
  maximum eta = 3 with near-linear descent (D(3)/D(1) 2.7), so R is a
  lower bound of the line-optimal ratio; the bar stands as sealed.
  Prior 7 hits 2 misses (family 30 / 21).
- The foreign-writer program is now parked in full: DFA (WRITER-DFA-1),
  SG in three forms (top-four, cross-position desk, block-7 boundary),
  zeroth-order (this desk). Target / equilibrium propagation and ROME
  were not licensed by any of these and need their own GO and pre-reg.

## Conditions that bite next session

- Nothing armed. Next program direction is Artin's call.
- Long-run hygiene that worked this time: Monitor + a detached
  `until` waiter on the DONE marker; the waiter is capped at ten
  minutes of foreground-background time, so re-arm it on a long run.
- Post-commit hooks regen the lock / index; check `git status` before
  any liverun launch.
- Disk about 31 GiB free. checkpoints/sgwriter1 (1.5 GB) and
  checkpoints/sgbb7 (0.86 GiB) are the retained SG keep sets;
  checkpoints/frozenbb1 keeps the FB-1 anchors.

## Open decisions for Artin

1. Next program after the foreign-writer park.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from the MEZO verdict, the
RIFF bank SYNTHETIC GRADIENTS / DNI (parked in place).

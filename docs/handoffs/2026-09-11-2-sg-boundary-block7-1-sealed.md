# Handoff 2026-09-11-2: SG-BOUNDARY-BLOCK7-1 designed, implemented, smoked, pre-registered and audited (SEALED, NOT LAUNCHED)

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. No birth authorized.

## What landed (after handoff 2026-09-11-1)

- Artin GO (13:37 EDT): SG-BOUNDARY-BLOCK7-1 DESIGN + PREREG +
  IMPLEMENTATION + TEST / SMOKE only. The top-four SG family stays
  CLOSED (VERDICT SG-CROSSPOS-REPRESENTABILITY-0 L71557); this is a
  new LOCAL boundary writer for block 7 only, from its block-7
  exception.
- Instrument (commit 3c9b5252, folds at 4d00452e):
  scratch/sg7_credit.py (block-7-only synthetic credit: the true CE
  runs through block 7 with DETACHED PARAMETER VIEWS on an ATTACHED
  input, so blocks 4..6 / norm / head get the control's gradient
  bit-exactly and block 7 gets no CE credit; a separate real-
  parameter block-7 forward on the detached input carries the
  J^T hat_delta_7 surrogate; predictor = the cross-position desk's
  LOCAL SeqSG arm, verbatim classes, W_0-fixed input standardization,
  no clip), scratch/birth19m_sg7.py (adopt-not-fork of the SG-1
  driver; seed 28; ONE frozen recipe PLR 1e-3; no FAMILY / PLR knob),
  scratch/sg7_qualgate.py (CONTROL-ADEQUATE c >= 24 / FUNCTION-MATCH
  c +- 7 / SG7-MISS / SG7-UNSTABLE; HEAD == launch commit asserted),
  scratch/sgbb7_qual_driver.sh, scratch/sg7_integrity_smoke.py
  (checks a..k), scratch/sg7_scale_census.py; tests 8 + 4 + 4.
- Receipts sealed and locked: integrity smoke at 3c9b5252 and
  4d00452e (forced-delta endpoint BIT-EXACT on a fresh W_0 and the
  seed-27 control at steps 463 / 15,420: block-7 gradient relative
  difference 0.0, parameter difference 0.0 after one clipped AdamW
  step; no CE leak; exact BP to 4..6; predictor isolation; masking;
  FOLD A timing), 300-step mps smokes (control 1.249 bit-identical
  to the SG-1 control smoke; SG7 1.468 at 4.2 it/s; predictor MSE
  9.24 v baseline 10.82 at step 200, 2.51 v 3.10 at 300, cos 0.54 /
  0.47), smoke gate (NOT-RESOLVABLE-CONTROL path), x_8 scale census
  (sd median 1.02 -> 7.28, max 1.85 -> 19.5 on the FIT tokens).
- PRE-REG SG-BOUNDARY-BLOCK7-1 (L71676, commit 1599ac6f) and
  AMENDMENT -AUDIT (L71869, this commit): two Opus audit passes; the
  one blocker (unreceipted drift numbers) closed by the census; five
  should-fixes folded (prior-1 seeds, smoke provenance disclosure,
  launch receipt path pattern, dead-writer disclosure on a MATCH,
  late-state fence on the desk ratios). Six registered priors
  (FUNCTION-MATCH p 0.5).

## Conditions that bite next session

- The seed-28 birth needs its own Artin GO. Launch shape:
  `bash scratch/sgbb7_qual_driver.sh` under liverun id sgbb7q,
  nohup + Monitor; the driver reruns the integrity smoke first, then
  control (about 33 min), gate, SG7 cell (about 61 min), gate. Walls
  about 1 h 40 min. Disk about 2.7 GB under checkpoints/sgbb7.
- Consequences are sealed: MISS / UNSTABLE closes SG credit-writer
  births completely; MATCH freezes the specimen and asks for a
  separate LOCAL-mechanism GO (no writer-invariance claim).
- Post-commit hooks regen the lock / index; check `git status` before
  any launch (liverun refuses a dirty tree).
- Smoke checkpoint tree checkpoints/sgbb7_smoke (439 MB) is prunable
  after the rung books (digest inventory first).

## Open decisions for Artin

1. GO / no-GO for the seed-28 SG-BOUNDARY-BLOCK7-1 birth.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L71676.

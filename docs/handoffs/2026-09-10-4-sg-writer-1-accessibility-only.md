# Handoff 2026-09-10-4: SYNTHETIC-GRADIENT-WRITER-1 qualification booked ACCESSIBILITY-ONLY; foreign-writer program awaits Artin's decision

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-10-3)

- Artin GO 17:20 EDT; run sgwriter1q under liverun at HEAD 919ef324
  (armed 21:21 UTC, DONE rc 0 02:43 UTC, 5.4 h). The launch-commit
  integrity smoke passed 4/4 first (checks a..k; receipt
  logs/sgwriter1/integrity_smoke_919ef324_launch.jsonl). Sealed order
  honoured: control, gate, then the four SG cells each gated as it
  landed; no amendment, no extra cell.
- VERDICT SYNTHETIC-GRADIENT-WRITER-1 (qualification) (RESULTS L70832,
  commit bd1847d6): ACCESSIBILITY-ONLY. Control 59 / 120 (adequate;
  band 52 to 66); LINEAR 3e-4 gates 1, LINEAR 3e-5 0, MLP-256 3e-4 0,
  MLP-256 3e-5 0; all finite, freeze law verified on every cell. No
  FUNCTION-MATCH, no recipe frozen, no seed-2 discovery. Descriptive:
  predictors never beat the zero predictor by more than 4 % at any
  logged step; alignment cos within +-0.23; MLP cells' logged
  true-hidden-error scale fell below 1e-4 of the constants' scale from
  step 463 / 2,800 with CE 1.35 to 3.22; LINEAR cells reached final CE
  0.65 / 0.71 (control 0.39) yet gated at the floor. Prior 1 hit, 2
  misses (family 12 / 10). Pre-booking auditor: gate / law /
  provenance clean; four blockers in my descriptive paragraph (wrong
  row counts, a fabricated-looking baseline figure, a wrong range)
  recomputed from the receipt and folded before booking.
- Receipts force-added and locked (qual.jsonl, ladder.json,
  selection.json, gate.log, driver.log, five train logs, integrity
  smoke launch log + jsonl, liverun receipt). FINDINGS bullet
  [SINGLE-SEED]; RIFF banks (SG / DNI bank, foreign-writer list
  candidate (2)) amended in place; BOARD line 5.
- checkpoints/sgwriter1/ (6.5 GB: 5 cells, model + predictor
  snapshots) retained untracked pending disposition.

## Conditions that bite next session

- Nothing armed. The foreign-writer program has two accessibility
  STOPs at qualification (DFA family, SG top-four arena). What
  follows is Artin's decision; candidate leads recorded in the verdict
  (predictors that do not beat zero; numerically vanishing targets on
  the MLP cells) are for zero-training desks only if directed.
- Smoke exhaust: checkpoints/sgwriter1_smoke, sgwriter1_smoke_seal2,
  frozenbb1_smoke (about 1.1 GB each); disk about 19 GiB free after
  the run. Prune on a housekeeping gate (Artin GO).
- Index regen quirk persists (act-envelope row); SEAL-AUDIT entries
  carry slug ids ...-seal-b and ...-seal-b-b.
- The verdict's descriptive paragraph was wrong on first draft in a
  way the auditor caught: quick summary scripts that print at 4
  decimals hide sub-1e-4 values; recompute from the receipt at full
  precision before quoting.

## Open decisions for Artin

1. Foreign-writer program: SG second design points (bootstrapped
   every-layer target, delayed target, other predictor families) v
   candidates (3) to (5) of the foreign-writer list v a zero-training
   desk on the two recorded leads v park.
2. Disposition of checkpoints/sgwriter1/ (6.5 GB) and the three smoke
   checkpoint trees.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L70832, RIFF bank
SYNTHETIC GRADIENTS / DNI (amended in place).

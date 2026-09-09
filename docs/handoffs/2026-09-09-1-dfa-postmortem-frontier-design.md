# Handoff 2026-09-09-1: DFA postmortem booked, CREDIT-ANCHOR-FRONTIER-1 designed (not launched)

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-09-0)

- f07bb760 PRE-REG WHY-DFA-FAILED-DESCRIPTIVE-0 (L68941) + instrument
  scratch/dfa_postmortem.py, committed before any seed-21 read.
- Run dfapost0 under liverun (pid 43067, rc 0, 23.7 min CPU), 68 rows.
- de829cc0 OBSERVATION WHY-DFA-FAILED-DESCRIPTIVE-0 (L69008): no feedback
  alignment anywhere (max |A| 0.034 over 544 readings); residual-stream
  effective rank 20 to 24 at W_0 and 1.05 to 1.36 at every block from
  step 463 on, all four cells; true hidden error on the probe to
  2.8e-5..1.2e-9 while B_l e stays fixed; blocks move 200x to 480x more
  than emb/head/norm; ACT distance from W_0 59.0 to 59.7, cells within
  0.05 to 1.7 of each other. All four registered expectations held.
  Causal reading explicitly untested (no zero-credit control). Does not
  revise VERDICT WRITER-DFA-1.
- PRE-REG CREDIT-ANCHOR-FRONTIER-1 (L69122, DESIGN ONLY): k in {1, 2, 4}
  top blocks with exact BP credit, lower blocks DFA with the unchanged
  B_l law; qualification seed 23, two cells per k ((1, 3e-4), (1, 1e-4)),
  accessibility floor gate >= 24, first-k-that-clears stops the ladder,
  FRONTIER-CLOSED stops DFA-family births; optional zero-credit control
  per k (lower blocks frozen); discovery + band + the sealed WRITER-DFA-1
  observable set only for a licensed candidate; foreignness downgrade
  registered for k = 4; priors on the record. JSON mirror
  docs/preregs/credit-anchor-frontier-1.json.
- RIFF: SYNTHETIC GRADIENTS / DNI banked (next foreign family; abstracts
  of Jaderberg 2016 and Czarnecki 2017 checked, Transformer transport
  fences stated); OPTIMIZER-REGIME DFA REPAIR banked and closed unless a
  new failure class appears; the foreign-writer bank amended in place.

## Conditions that bite next session

- CREDIT-ANCHOR-FRONTIER-1 is NOT implemented: K_BP in dfa_credit.py,
  MODE=hybrid in the driver, the K_BP = 0 identity test, a generalized
  leakage smoke per k, seed-11 smokes for k = 1, 2, 4, then a clean-tree
  prereg-auditor, then Artin GO. Nothing is armed.
- checkpoints/writerdfa1/ (5.1 GB, four seed-21 cells) retained through
  the postmortem per Artin; disposition now open. The smoke tree
  checkpoints/writerdfa1_smoke/ was deleted at this close after
  confirming its evidence is receipt-backed (smoke.jsonl digests,
  leakage.json, dfapost smoke row).
- Index regen quirk: see memory note; re-pop needs_link on the session's
  rows after any regen (the ACT-envelope observation row keeps
  reacquiring it).

## Open decisions for Artin

1. GO / no-GO to implement CREDIT-ANCHOR-FRONTIER-1 (implementation and
   smoke only; a second GO for the ladder itself is already registered).
2. Whether the optional zero-credit control arm runs per k.
3. Disposition of checkpoints/writerdfa1/ (5.1 GB).
4. Whether the FUNCTION-BAND scoring reading (literal MISS) stands; the
   verdict names the alternative.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L69008, then
docs/preregs/credit-anchor-frontier-1.json.

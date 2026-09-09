# Handoff 2026-09-09-2: CREDIT-ANCHOR-FRONTIER-1 qualified and booked FRONTIER-CLOSED

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-09-1)

- AMENDMENT CREDIT-ANCHOR-FRONTIER-1-PRECISION (L69223): endpoint
  identities, per-k leakage law, zero-credit control activated, artifact
  disposition of the WRITER-DFA-1 set.
- Instrument (dfa_credit.py frontier section, birth19m_caf.py,
  caf_qualgate.py, caf_leakage_smoke.py, drivers, tests), OBSERVATION
  -INSTRUMENT-0 (L69303) and AMENDMENT -INSTRUMENT-0-AUDIT (L69370):
  K_BP=0 == sealed DFA and K_BP=8 == stock BP bit-exact through one AdamW
  step; leakage 17/17 for k = 1, 2, 4; smokes 4/4. Selector fold: no
  selection artifact for an incomplete frontier; a larger k freezes only
  when every smaller k is complete (auditor blocker closed, re-audited
  clean).
- WRITER-DFA-1 qualification set pruned to 868 MB after a 0-mismatch
  digest inventory (logs/writerdfa1/prune_inventory.json).
- Qualification ladder at seed 23 under liverun (k1 pid 88122, k2 94767,
  k4 1991, select), launch commit 0c5dfad9, no mid-run commit.
- VERDICT CREDIT-ANCHOR-FRONTIER-1: FRONTIER-CLOSED. Six hybrid cells
  gate 0 / 120; zero-credit controls 28 / 55 / 62. DFA-family births
  STOP. Family record 6 hits, 5 misses.

## Conditions that bite next session

- checkpoints/writercaf1/ holds 153 snapshots (11 GB, nine seed-23 arms,
  digest-anchored in logs/writercaf1/qual.jsonl). Disposition is Artin's
  call; the natural keep set is the three zero-credit control arms
  (final, step_0, step_463) plus one hybrid per k.
- checkpoints/writerdfa1/ (868 MB) retained.
- No discovery / post-discovery driver exists for the frontier and none
  is needed now (FRONTIER-CLOSED).
- Index regen quirk persists (memory note): re-pop needs_link by id
  after any regen.

## Open decisions for Artin

1. Next foreign writer: the banked synthetic gradients / DNI family
   (design only, needs its own pre-reg), or park.
2. A zero-training ACT / alignment readout on the nine seed-23 arms
   (hybrids and frozen controls as within-seed comparators) to test the
   rank-collapse hypothesis for the hybrid lower stacks; small pre-reg.
3. Disposition of the 11 GB frontier snapshot set and the 868 MB DFA set.
4. Whether the frozen-random-lower-stack result (62 / 120 with half the
   stack a fixed random feature map) deserves its own descriptive
   observation or bank.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from the frontier verdict,
docs/preregs/credit-anchor-frontier-1.json (outcome field).

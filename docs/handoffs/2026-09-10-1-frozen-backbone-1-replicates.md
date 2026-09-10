# Handoff 2026-09-10-1: FROZEN-BACKBONE-1 REPLICATES; SG writer next (design on file, not sealed)

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run (liverun: none).

## What landed (after handoff 2026-09-10-0)

- AMENDMENT FROZEN-BACKBONE-1-CONTROL-ADEQUATE (RESULTS L69970, commit
  76be13b9): control-validity precondition (every FULL arm finite and
  gate >= 24, else NOT-RESOLVABLE-CONTROL, unscored); threshold
  delta_s >= -7 unchanged; scratch/fb_gate.py adjudicate() carries the
  pure law; tests/test_fb_control_adequate.py shows FULL = 0 / FROZEN
  = 0 cannot score REPLICATES. Clean-tree prereg-auditor: no blocker.
- Run frozenbb1 under liverun (armed 06:37 UTC, DONE rc=0 10:38 UTC,
  4.0 h, HEAD 579a135d throughout, no mid-run commit).
- VERDICT FROZEN-BACKBONE-1 (L70024, commit 8d54dd38): REPLICATES.
  FROZEN 63 / 61 / 59 v same-W_0 FULL 61 / 64 / 62 on seeds 24 / 25 /
  26; delta +2 / -3 / -3, all >= -7; controls adequate; prior hit
  (family 7 hits, 5 misses). Registered reading only: at this task /
  scale, learning the embedding and bottom four blocks is not
  required to reach full-BP function at the house gate resolution
  across these specimens. Not claimed: irrelevance of the frozen
  stack, any mechanism. Pre-booking auditor: no blocker; should-fixes
  folded (59 is two below the stock floor; finiteness also guarded by
  fb_gate's assertion; driver.log registered as an extra receipt).
- Receipts force-added and sha-locked: logs/frozenbb1/{births.jsonl,
  replication.json, gate.log, driver.log, six train logs, smoke.jsonl},
  logs/liverun/frozenbb1.jsonl. FINDINGS bullet ([REPLICATED]);
  FROZEN-RANDOM-BACKBONE bank amended in place; BOARD line 5.
- checkpoints/frozenbb1/ (8.17 GB, 6 cells x 18 files) retained
  untracked: the healthy same-W_0 BP-like states Artin directed for the
  synthetic-gradient predictor audit.

## Assessment for Artin (asked 2026-09-10: first SG comparison arena)

Question: is BP v SG on the top four trainable blocks over the same
frozen lower backbone the cleanest first SG writer comparison?

House view: yes, as the QUALIFICATION arena, with the full-stack design
kept as the discovery target.
- For: the arena now has a replicated same-writer envelope (FROZEN 59
  to 63, three seeds, deltas within one resolution unit of FULL), a
  fixed random feature stream whose rank is W_0's (20 to 23 in the harm desk, never
  collapsed), and only four blocks + norm + head to credit, so a SG
  predictor per trained block is 4 predictors, not 8, and the target
  hidden errors are those of a healthy BP top. The DFA autopsy's
  failure class (rank collapse of the lower stack) cannot occur by
  construction, which isolates the credit-writer question from the
  representation-collapse question. Cost per cell drops from 47 to
  about 28 min plus predictor overhead.
- Against: it does not test SG as a full-stack writer (the bank's
  actual claim); a top-four SG that matches BP says nothing about
  whether SG can build the lower representation. Blocks 4..7 sit on
  frozen features, so "SG reaches BP function" there is a weaker
  statement than at full depth.
- Proposal: SYNTHETIC-GRADIENT-WRITER-1 qualification = top-four SG
  over the frozen backbone at seed 27 (BP-top-over-frozen at seed 27
  as the paired same-W_0 control; the replicated 59 to 63 envelope as
  the band), predictor family and target scaling chosen by a
  zero-training audit on the six frozenbb1 finals plus their
  step_00463 snapshots (true delta_BP magnitudes at healthy states);
  discovery = full-stack SG only if qualification passes. Needs its
  own pre-reg amendment to L69765, auditor, Artin GO. Nothing armed.

## Conditions that bite next session

- Index regen quirk: 2026-09-09-observation-writer-dfa-1-act-envelope
  regains needs_link on every regen; re-pop by id before committing
  (memory note index-regen-reclassifies-rows).
- receipts.lock brace citations must not wrap across a line
  (`{bp,\nzero}` left six logs pending until rewritten as full paths).
- Two pre-existing needs_link rows (2026-09-04 render-atlas,
  2026-09-05 next-program-assessment) are not this thread's.
- Disk: 30 GB free before the run; frozenbb1 checkpoints 8.17 GB,
  writercaf1 keep set 939 MB, writerdfa1 keep set 868 MB.

## Open decisions for Artin

1. Adopt the top-four-over-frozen-backbone arena as SG qualification
   (assessment above), or keep the full-stack SG design as the first
   comparison.
2. GO to run the zero-training predictor audit (linear v MLP-256,
   target scaling) on the frozenbb1 states; design-only until then.
3. Disposition of checkpoints/frozenbb1/ after the audit (prune to
   finals + step_00463 like the frontier set, or keep).

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L70024, PRE-REG
SYNTHETIC-GRADIENT-WRITER-1 (L69765) for the design to amend.

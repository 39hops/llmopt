# Handoff 2026-09-11-1: SG-CROSSPOS-REPRESENTABILITY-0 NO-FIRE (SG family closed at the arena); sgwriter1 pruned to the keep set

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-11-0)

- PRE-REG SG-CROSSPOS-REPRESENTABILITY-0 (RESULTS L71347, commit
  21f3da85) sealed before any cross-position number was read: a
  fixed-capacity reverse-causal sequence predictor (query i reads
  Z_j = [x_{l+1,j}, e_j] at j >= i; 2-layer d128, ALiBi, 500,736
  params, zero-init output) fitted offline on the audited FIT chunks,
  scored on HELDOUT; paired same-architecture LOCAL (j == i) twin;
  BAR-1 = median over six late steps x blocks 4..7 <= 0.5.
- AMENDMENT -AUDIT (L71499, dcb23702): auditor folds, no blocker
  (two motivation misquotes corrected; FIRES restated at the family
  level; adjudicate() emits the paired gap and priors 1..6 from
  receipt fields; smoke provenance reworded).
- Run sgxpos0 under liverun at 577a445d (39 min, 80 fits, 0
  nonfinite). First launch attempt was refused by liverun (the
  post-commit ledger regen had dirtied the tree); the refusal text
  was removed from desk.log, the regen committed, relaunched clean.
- VERDICT SG-CROSSPOS-REPRESENTABILITY-0 (L71557, commit c389a8d0):
  BAR-1 NO-FIRE. Late median reverse-causal HELDOUT ratio 1.10;
  blocks 4..6 0.93 to 1.285 at every late step (memorizing: FIT
  0.19 to 0.34), block 7 0.10 to 0.12 in BOTH arms (and already GOOD
  for the closed-form random-feature oracle); paired gap
  reverse_causal - local median -0.008, range -0.076 to +0.023.
  Cross-position access adds nothing. The SG family CLOSES at this
  arena under the sealed clause (block-7 exception disclosed). Prior
  3 hits 3 misses (family 19 / 17). Three prereg-auditor blockers
  folded (per-chunk minimum 0.849 not 0.93; block 7 "beats the closed
  forms" withdrawn; linear late range 0.47 to 1.11); receipt-auditor
  clean. The SG-FAILURE-DESK-0 FINDINGS bullet's unformatted anchor
  ("L{line}") was corrected in place to L71182 in the same commit.
- Housekeeping (approved with the GO, executed after booking):
  checkpoints/sgwriter1/ pruned to the keep set by
  scratch/sgwriter1_keepset_prune.py after a full sha256 + state-
  digest inventory against logs/sgwriter1/qual.jsonl: 37 files kept
  (1.52 GiB: per cell steps 463 / 3,084 / 5,140 / 15,420 + matching
  pred_step files; the control's step_00000), 125 removed (5.02 GiB),
  0 mismatches, 0 unreceipted. Inventory
  logs/housekeeping/sgwriter1_keepset_prune_2026-09-11.json (locked).
  Disk 33 GiB free.

## Conditions that bite next session

- Nothing armed. The SG family is closed at the top-four-over-frozen-
  backbone arena. Not licensed and untested: bidirectional (j < i)
  predictor input, downstream weights as input, larger predictors, a
  one-block SG writer at block 7 (the descriptive lead: block 7's
  error IS a function of its local input on the trained control).
- Post-commit hooks regenerate receipts.lock.json / results-index
  (re-adding needs_link to the act-envelope row and any fresh
  amendment row); check `git status` before any liverun launch and
  re-pop by id.
- Long runs: nohup + Monitor (scratch/sgxpos0_launch.sh is the
  shape).

## Open decisions for Artin

1. Foreign-writer program direction: candidates 3 to 5, a one-block
   SG design at block 7 (new pre-reg), or park the program.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L71557, the RIFF bank
SYNTHETIC GRADIENTS / DNI (amended in place with the cross-position
desk).

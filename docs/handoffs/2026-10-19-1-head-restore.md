# Handoff 2026-10-19-1: QWEN-HEAD-RESTORE-0, the head-only restoration test, booked NULL

Session: Fable 5.1 on the Mac. Reviewers and auditors on Opus 5.5.
Previous handoff: 2026-10-19-0 (representation round). Read BOARD first.

## What landed

1. PRE-REG QWEN-HEAD-RESTORE-0 (9a528b52, RESULTS L80985; JSON
   docs/preregs/qwen-head-restore-0.json, digest ceeeb654 pinned in the
   driver). Registered before any vendor byte was retrieved: the vendor
   BF16 lm_head against arm C's S16 lm_head on the same frozen arm C
   hidden states; E0 (4,722) diagnostic, corpus (356) and prefix (98)
   teacher records as validation; four bars (E0 agreement +1.0 point,
   E0 slope +0.03 reported separately, validation agreement +1.0 point
   with no subset net negative, validation KL -10 percent); controls
   for evaluation precision and the production path; refutation folds
   bar 3's conjuncts; VALIDATED also needs the bar 3 block-bootstrap
   interval to exclude zero. An Opus code review of the driver preceded
   the registration (refutation ratio, runner smoke isolation and exit
   codes, bounded bf16 conversions, the census shift statistic, pinned
   digest and dump shas, dirty-tree refusal); the smoke chain ran every
   stage with a labelled stand-in head.

2. VERDICT QWEN-HEAD-RESTORE-0 (RESULTS L81143): NULL. Run under
   liverun qwen-head-restore-0 at eb2d9cb0 (02:11 to 02:48 UTC, rc 0).
   E0 agreement 3,915 to 3,913 of 4,722 (53 corrected, 55 broken; bar 1
   NO-FIRE); median top-16 slope 0.8747 to 0.8832 (bar 2 NO-FIRE);
   validation net +5 of 454 (+1.10 points; bar 3 FIRE at its threshold,
   bootstrap [-0.78, +3.41] includes zero, FIRE-AT-LOW-RESOLUTION);
   validation KL +0.36 percent (bar 4 NO-FIRE); mechanical refutation
   clause NOT-REFUTED because bar 3 fired; both booked. Controls within
   one position. The S16 head's relative Frobenius error against the
   vendor head is 0.1076 (the compile fixture said 0.1077). The opener
   ("The" against the teacher's "We") is body-side: the head swap flips
   one prompt of 24; the compressed model alone opens with "The" on 24
   of 24 under 64 to 1,024 filler tokens and 23 of 24 without a system
   prompt (draft only, no teacher record for those conditions).
   The refutation clause's consequences are not booked (NOT-REFUTED);
   the session's reading, outside the registration, is that head-side
   levers (exact restore, 8-bit head, low-rank residual) are not worth
   pursuing; the runtime stage did not run.

3. Instruments adopted: llmopt/lab/logit_stats.py (per-position
   statistics, paired counts, block deltas and bootstrap, exact sign
   test, bfloat16 bit conversions, safetensors header reader, S16 row
   slice payload) and llmopt/lab/strict_json.py (receipts a strict
   parser accepts), both with tests. scratch/qwen_e0_disagreement_census.py
   (results-cited) now writes through strict_json: Artin's instruction;
   the booked receipt is byte-unchanged and keeps its bare Infinity.

## Conditions that bite next session

- The vendor shard model-00018-of-00018.safetensors (3.39 GB,
  sha-verified) sits in ~/qwen_vendor_shards/1d4bf0f2 (outside the
  repository, not the config/tokenizer directory). Keep or delete is
  Artin's call; the E0 and Metal drivers refuse a shard only inside
  ~/qwen_vendor.
- Machine-local dumps: logs/qwen_head_restore/dump (per-position arrays
  and the corpus and prefix hidden states, 24 MB of arrays plus 9 MB of
  hidden states and 225 MB of native logits); the 3.1 GB E0 dump from
  the previous round is unchanged and still needed by any head-side
  rerun.
- The receipt lock and scoreboard were regenerated at the booking
  commit; the two gate failures at eb2d9cb0 were those regens (one a
  race with the live run's own receipt).

## Next session

Start here, then BOARD, then the RESULTS tail. Recommended first batch
(Artin's call): the layer-group restoration probe, one vendor shard at
a time (about 3 GB, deleted after its receipt), the layers it holds
swapped to BF16 in the Metal model, E0 and validation re-measured with
the same registration shape; falsifier: no group moves E0 agreement by
one point. Not next: whole-model recompression, fine-tuning, adapters,
the full teacher restore, any fitting on E0.

## Open decisions for Artin

1. Keep or delete the vendor shard (3.39 GB).
2. Whether the layer-group probe is the next experiment, and in which
   order (early layers, late layers, the full-attention layers first).
3. Whether the census script edit (strict JSON exporter on a
   results-cited file) stands as a forward-only note or wants a
   CODEMAP amendment.

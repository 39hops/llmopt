# Handoff 2026-10-21-0: QWEN-LAYER-RESTORE-0 run and booked (NULL, REFUTED at the registered resolution)

Session: Fable 5.1 on the Mac. Auditors on Opus 5.5 (receipt, prereg, wording).
Previous handoff: 2026-10-20-0 (the registration). Read BOARD first.

## What happened

1. Artin's GO of 2026-10-21 authorized exactly one shard:
   model-00007-of-00018.safetensors of Qwen/Qwen3.8-27B at revision
   1d4bf0f2 (2,108,759,344 bytes, sha256 80b0c490...). The registered
   chain ran under liverun qwen-layer-restore-0 at 221c1c0c, clean
   tree, 16:14:57 to 16:56:56 UTC, rc 0: preflight (20 checks true),
   capture, both oracle halves on arm C tensors, fetch (97 s, sha and
   header verified), both oracle halves on the vendor weights, five
   score arms (441 to 454 s each), summarize, sensitivity, adjudicate.

2. VERDICT QWEN-LAYER-RESTORE-0 (RESULTS L81586): NULL, REFUTED.
   GROUP against CONTROL: E0 3,915 to 3,907 of 4,722 (63 corrected, 71
   broken, -0.17 points, interval [-0.66, +0.27]); validation 374 to
   376 of 454 (+0.44, interval [-0.51, +1.74]); validation KL -9.1
   percent (bar -10). Bars 1, 2, 3 NO-FIRE; refutation clause REFUTED;
   class NULL. Preconditions all met: oracle torch against mlx 3.8e-7
   and 4.6e-7 on the vendor weights, DENSE_C zero flips, CONTROL equal
   to the E0 rows at 4,722 of 4,722, shard sha re-verified at every
   score and oracle stage. Sensitivity: S16 weight error 0.10 to 0.11,
   W4 0.345 to 0.349; local out23 difference 0.128 relative L2 (row
   median 0.115), 0.126 at the final norm (row median at 0.67 of the
   local one on E0, 0.50 on validation); 192 E0 argmax flips with no
   net gain; slope 0.8747 to 0.8754; the opener unrepaired. Memory
   peak 14.38 GB control, 17.15 GB with the pair as float32.

3. Three Opus audits ran on the draft before booking and found no
   blocker in the receipts; their corrections to the prose were
   verified against the arrays and applied (raw tensor count 12, the
   row-median against pooled statistic split, the near-tie sentence
   replaced by measured margins, the head-restore comparison, the
   L22 validation KL reading at -10.5 percent disclosed under the
   multiplicity fence, the session hypothesis labelled untested).

4. Receipt notes for the record: control_vs_census
   dump_sha256_on_disk_ok reads false because that field hashes the
   array payload while the census pinned the file sha; the file sha was
   recomputed in-session and equals the pin; the ok rule never
   depended on the field. The sensitivity stage read the vendor shard
   without a fresh sha check (the score_group stage had verified the
   same file eight minutes earlier). Neither changes the verdict; both
   are fixes for any future run of this driver.

## Storage

Two vendor shards outside the repository under the pinned-revision
directory: shard 18 (3.39 GB, head) and shard 7 (2.11 GB, layers 22 and
23). Neither was deleted; retention is Artin's call. Receipts (JSON,
adjudication, logs) force-added under logs/qwen_layer_restore/; the
1.8 GB of per-position arrays and boundary captures stay machine-local
in logs/qwen_layer_restore/dump with shas in the stage receipts.

## Not done, by design

No second shard, no third restoration group, no recompression,
fine-tuning, adapters or speculative verifier. The thread pauses here.
Open options, all Artin's call: a per-layer error census (needs a
decision on whether an arm-C-only proxy for the reconstruction error is
acceptable), or accepting arm C's 82.9 percent E0 agreement as its
operating point.

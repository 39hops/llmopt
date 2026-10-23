# Handoff 2026-10-23-0: QWEN-LOSSY-GEOMETRY-0 (NULL) after the W4 null and the lossless-BF16 negative were accepted

Session: Fable 5.1 on the Mac. Auditor on Opus 5.5 (pre-reg and numerical audit of the verdict).
Previous handoff: 2026-10-22-2 (W4 null, lossless BF16 negative). Read BOARD first.

## What happened

1. Artin accepted VERDICT LOSSLESS-BF16-FEASIBILITY-0 (negative) and
   OBSERVATION QWEN-METAL-W4-T1-0 (null) with one qualification, now
   on the BOARD row: the empirical H16 is a zero-order entropy bound
   for static symbol coding, not an impossibility theorem against all
   conceivable structured compressors; lossless BF16 below it must not
   be described as mathematically impossible. The booked text says
   "closed for static per-tensor symbol coders" and nothing stronger.
2. New GO, QWEN-LOSSY-GEOMETRY-0: a bounded local representation study
   at arm C's bitrate. Grounded abstract-level (scripts/cite_lookup.py
   metadata for HIGGS 2411.17525, QuIP# 2402.04396, QTIP 2406.11235;
   no PDFs): rotation = theorem, codebooks = quantisation algorithms,
   kernels = computation, fine-tuning = optional; the linearity
   theorem not invoked at 2 bits.
3. Instruments (all new, committed at 26d278f0 with the registration):
   scratch/qwen_geometry_capture.py (shard 7 pin, originals, arm C
   payloads and canonical decode checked; activations of the 24
   rollouts through the serving prefill at every position, split 0 to
   11 dev / 12 to 23 eval by prompt before any candidate),
   scratch/qwen_geometry_quant.py (A control, B refit, C rht, D higgs,
   E awq; storage accounting; Frobenius, output rel MSE in float64,
   row and position distributions, cancellation, float32 evaluation,
   isotropic prediction, bootstrap over prompts; CPU fits for
   determinism), scratch/qwen_geometry_transform.py (rotation cost on
   Metal as MLX ops), scratch/qwen_geometry_obs.py and the runner;
   scratch/leancheck/InputRotation.lean (three theorems, standard
   axioms; checked with lake env lean in scratch/leancheck).
4. PRE-REG QWEN-LOSSY-GEOMETRY-0 (RESULTS L83390) registered before the
   eval split was read; the development smoke was disclosed. Chain
   under liverun qwen-geometry-0 at 26d278f0: capture 3 min,
   transform, dev 15 min, eval 15 min, then the adjudicate stage failed
   on an adapter contract error (an unregistered key in the
   measurements). Fixed at eac8232f, re-run under qwen-geometry-1 into
   logs/qwen_geometry/adjudicate_1/; AMENDMENT QWEN-LOSSY-GEOMETRY-0-
   RECEIPTS (L83579) links the corrected paths; the failed outputs stay
   at the registered paths as the failed record (the receipt-freeze
   hook refuses their deletion, by design).
5. VERDICT QWEN-LOSSY-GEOMETRY-0 (L83603): NULL, REFUTED, all three bars
   NO-FIRE. Rotation and data-free grid 11 percent on gate_proj, 4 to 5
   on down_proj; refit within 1.5 percent; activation-weighted
   assignment 26 percent on gate_proj but no transfer to down_proj;
   best 0.949 against 0.85; S16 rotation 0.864; transform 6.7 ms per
   step unfused (bar 5). Recommendation HOLD, no mixed-model test.
6. One Opus pre-reg audit applied: the decision rule (NULL, not HOLD, as
   the registered outcome; HOLD kept as the recommendation the GO asked
   for), a reversed isotropic direction in the headline, the T = 256
   ratios, the percentile tails, the prior accounting, the surrogate
   ranking claims, the Lean scope, the amendment's "sha-locked" and
   "no registered file edited" wordings, the 6,144-wide out_proj and
   o_proj inputs in the transform projection.

## Decisions left to Artin

- Whether any geometry work continues. The conditions any future
  registration would have to meet are in the verdict's section 5; none
  is started. A fused in-kernel rotation and E8 or trellis codebooks
  are the untested residue (RIFF).
- Retention of the two vendor shards outside the repo (unchanged).
- Deletable at any time: logs/qwen_geometry/dump/ (the activation
  captures, about 1 GB, sha256 in the capture receipt), the smoke
  directories, the release dumps from 2026-10-22-1.

## Not done by design

No whole-model recompression, BF16 teacher reconstruction, adapter
fitting, download, kernel default change or destructive cleanup; no E8,
trellis or fused rotation; no candidate selected by teacher agreement
on E0; no model run with any candidate inside it.

## Lessons

- An observations adapter may emit only registered keys inside
  `measurements`; anything else belongs in a sibling field. The
  adjudicator's contract error is a pipeline bug and its outputs go to
  a NEW path with an amendment, never an in-place rewrite.
- A calibration-set weighting that wins on one matrix can lose on the
  next (the SwiGLU product's channel statistics move with the prompt);
  block the evaluation split before fitting, every time.
- MPS k-means is not run-to-run deterministic (index_add_, argmin);
  CPU fits are, and the codebook sha across dev and eval runs is the
  check.

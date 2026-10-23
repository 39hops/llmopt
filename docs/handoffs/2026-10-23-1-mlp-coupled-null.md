# Handoff 2026-10-23-1: QWEN-MLP-COUPLED-0 (NULL); quantization and representation research closed for now

Session: Fable 5.1 on the Mac. Auditor on Opus 5.5 (pre-reg and numerical audit of the verdict).
Previous handoff: 2026-10-23-0 (the geometry null). Read BOARD first.

## What happened

1. Artin accepted VERDICT QWEN-LOSSY-GEOMETRY-0 (NULL) and asked for one
   bounded mechanism experiment: does the activation-weighted W4 gate
   (E_awq) survive the complete gated MLP of layer 22, on a NEW
   evaluation collection, at unchanged storage; plus a retrospective
   observation on the old split and an inventory of the evaluation
   surfaces.
2. OBSERVATION QWEN-LOSSY-GEOMETRY-0-SPLIT (RESULTS L83811): the geometry
   study's halves were E0's category-ordered records (chat, math, code
   against reasoning, knowledge, long, edge) with shifted activation
   statistics; a frozen cross-category test, not IID; the verdict and
   its bars stand; no new split of those 24 prompts was made.
3. New collection scratch/evalsets/mlp_coupled_v1.json (48 prompts, 7
   in-domain categories balanced across halves, an eval-only ood group,
   4 seeded arithmetic items; sha256 in the capture receipt). Captures
   (scratch/qwen_mlp_capture.py) from the unchanged arm C model: prefill
   inputs of each prompt's tokens and 48 free-running greedy tokens,
   labelled apart, with the generated text recorded.
4. PRE-REG QWEN-MLP-COUPLED-0 (L83872; docs/preregs/qwen-mlp-coupled-0.json)
   registered at 610bc1f6 before the evaluation half was read; chain
   under liverun qwen-mlp-coupled-0 at the same commit, adjudicated in
   logs/qwen_mlp/adjudicate_1/. VERDICT (L84000): NULL, REFUTED. The gate
   gain reproduces (0.749) and the complete MLP output moves 2.4 percent
   (0.976, interval 0.972 to 0.980) against the 10 percent bar, the same
   on free-running and out-of-domain inputs; the MLP semantics were
   verified in numpy, torch and mlx float32 (2e-6 of float64).
5. Sections 3 to 5 of the verdict: the candidate is the existing W4
   format (decoder unchanged) but no mixed-model test is proposed; the
   evaluation surfaces are inventoried (MODEL-1 and E0 development,
   MODEL-2 held out and read once) and a mechanically scored usefulness
   battery is designed, not started; recommendation CLOSE quantization
   and representation research for now.
6. One Opus pre-reg audit applied before booking.

## Decisions left to Artin

- Whether to build the usefulness battery (verdict section 4): seeded
  arithmetic, structured output, code with tests, consistency,
  degeneration; scored by programs; the teacher scored once on it.
- Whether an all-three-matrix activation-weighted MLP is ever opened as
  a registered local test (RIFF residue; not recommended now).
- Retention of the two vendor shards outside the repo (unchanged).
- Deletable at any time: logs/qwen_mlp/dump/, logs/qwen_geometry/dump/,
  the smoke directories, the release dumps from 2026-10-22-1.

## Not done by design

No whole-model recompression, vendor download, lossless study,
speculative verifier, fine-tuning, adapters, kernel or default change,
or deletion of captures or vendor weights; no further rotation,
codebook or layer tried; no candidate promoted.

## Lessons

- A local linear-map fidelity ratio over-states what reaches a
  nonlinear block's output by an order of magnitude when the other
  branches carry comparable error; measure the complete operation
  before registering a hurdle on a part.
- Balanced-by-category halves plus a separately labelled out-of-domain
  group make a transfer reading interpretable; category-ordered index
  splits do not.

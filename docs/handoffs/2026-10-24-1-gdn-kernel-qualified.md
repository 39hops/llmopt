# Handoff 2026-10-24-1: QWEN-GDN-KERNEL-0 (the Metal recurrence kernel qualifies; a third off first-token latency; promotion is the next GO)

Session: Fable 5.1 on the Mac. Opus 5.5: a methodology audit before
registration (three blockers, eleven should-fix items, all applied), a
receipt audit and a pre-registration audit before booking. HEAD at close: the
booking commit of this handoff (git log -1), on origin/main. 3080: fast-forwarded to the booking commit after the push (ff-only, HEAD
asserted); nothing in this session depends on it. Previous handoff: 2026-10-24-0 (the prefill wall).
Read BOARD first.

## What happened

1. Artin's relay opened three workstreams (kernel qualification, a Muon
   read-only audit, a transfer-without-teacher research design) and asked
   for an assessment first. The assessment went back in chat (the Muon
   line-by-line comparison against KellerJordan's reference and the three
   ranked transfer candidates live in that reply and are NOT yet banked in
   RIFF-LEDGER; see open decision 3). Artin then gave GO for the kernel
   qualification only, with five corrections.
2. llmopt/lab/qgdn.py: recurrence_path("loop" | "kernel" | "vendor")
   replaces qwen3_5.gated_delta_update (what the exec-rewritten __call__ of
   qtower resolves at call time) with a wrapper that forces use_kernel, and
   wraps gated_delta_ops and gated_delta_kernel inside their module so the
   receipt counts EXECUTIONS; the training flag is never touched; vendor
   source guard; nesting refused; exit restores and detects a foreign
   change; normalized_errors with explicit near-zero treatment. 12 tests
   in tests/test_qgdn.py (dispatch on a training-mode module and on an
   exec-rewritten __call__, restoration after an exception, loop against
   kernel on realistic operands, chunk-boundary continuation).
3. scratch/qwen_gdn_kernel_qual.py (stages layers, census, decode,
   heldout, cachehist), scratch/qwen_gdn_obs.py, scratch/qwen_gdn_study.sh,
   docs/preregs/qwen-gdn-kernel-0.json. Two smoke chains (logs/qwen_gdn/
   smoke_1 and smoke, force-added) and one disclosed calibration receipt
   (logs/qwen_gdn/calibration/layers_gdn-dev-1-1024.json). The first smoke
   exposed that identical-input faithfulness and propagated divergence are
   different quantities; bar 3 was split accordingly before registration.
4. The Opus methodology audit's blockers: the frozen census's cpu_* fields
   are the HF torch arm C (scratch/qwen_e0_spec_census.py), not the mlx
   loop; near ties were redefined on metal_margin (same 15 rows); the
   layers stage moved to an unread seed; bar 5 got a reuse floor. PRE-REG
   booked at 9d3157c6 (RESULTS L85770), index row linked at 2e3c9cb5.
5. The chain ran under liverun qwen-gdn-kernel-0 at 2e3c9cb5 (56 min).
   VERDICT QWEN-GDN-KERNEL-0 (RESULTS L85927): bars 1 to 8 and 10
   FIRE, bar 9 UNRESOLVED inside its fence, NOT-REFUTED. Argmax identical
   at 4,722 of 4,722 census positions and all 15 near ties; identical-input
   state error 5.2e-5 of the RMS (propagated 6.7e-4 at the deepest layer);
   both 256-step decode streams equal, teacher-forced logits within 1.2e-4;
   9 of 9 chat turns equal with every eligible turn reused; TTFT ratios
   0.673 / 0.674 / 0.666 / 0.668 at 512 / 1,024 / 2,048 / 4,096 tokens
   (8.9 / 18.1 / 34.9 / 70.0 s against 13.2 / 26.9 / 52.3 / 104.9), peak
   memory plus 0.8 percent, decode step 0.936, a long answer's wall 0.80.
   By the registered rule: a RECOMMENDATION for a separate promotion GO.
   Nothing promoted: qtower still builds in training mode, qserve never
   imports the selector, every serving call still runs the loop.

## Conditions that bite next session

- The serving default is still the loop. Every receipt before this one
  ran the loop path; the kernel path has exactly this verdict's receipts.
- The promotion needs its own GO and acceptance gates (release tests, a
  fresh-prompt battery through the CLI, held-out parity on the promoted
  code); the selection should live in qtower or qserve under the existing
  vendor source guard, by name, never by model.eval().
- Muon and transfer workstreams: assessment delivered in chat only, no
  RIFF bank yet (open decision 3).

## Next session

Start here, then BOARD, then the RESULTS tail (the verdict and its
pre-reg). Recommended first batch, on GO: the promotion rung (one selection
in qtower or qserve, pre-reg with the acceptance gates above, the default
flip as its verdict), then the W4 GEMM half of the remaining prefill wall.

## Open decisions for Artin

1. GO for the promotion rung (kernel path as the serving default) with the
   gates above.
2. Whether bar 9's UNRESOLVED (decode step 0.936 against a 0.95 bar with a
   0.02 fence; 6 percent measured on 4 against 4 runs at 0.16 percent
   spread) is worth re-registering at a bar the measurement can resolve,
   or left as is (it plays no part in the recommendation).
3. Whether to bank the Muon read-only comparison and the three transfer
   candidates (weight-only invariances first, oracle-bootstrap, offline KD
   on hold) into RIFF-LEDGER from the chat assessment, and whether the
   Muon three-arm reproduction (house recipe, reference recipe with
   Nesterov, aspect scaling, scheduled LR and weight decay, AdamW control;
   one 3080 night) is worth a slot.

## Hygiene

Receipts force-added by file name: the six stage receipts (census
positions 3.0 MB, the one large text receipt, kept because the near-tie
rows and every per-position margin are the evidence), the chain log, the
adjudication pair, the calibration receipt, both smoke chains. Both checkouts at
origin/main.

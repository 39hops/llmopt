# Handoff 2026-10-24-0: QWEN-PREFILL-WALL-0 (NULL on chunk size; the DeltaNet recurrence is the next lever)

Session: Fable 5.1 on the Mac. Opus 5.5: a methodology audit before registration (applied) and a receipt audit before booking.
Previous handoff: 2026-10-23-6 (the low-default release). Read BOARD first.

## What happened

1. Artin closed the low-thinking default workstream and issued
   QWEN-PREFILL-WALL-0: operational fallback accounting, the generation-5
   prefill baseline, a decomposition, a small design space, a registered
   hurdle; no kernel or default change.
2. scripts/qwen_chat.py summarize_turns (turns, reuses, fallbacks with
   reasons, first-token latency by mode, replayed tokens and cost, memory)
   from the recorded turns only; tests/test_qchat_summary.py.
3. scratch/qwen_prefill_wall.py (stages baseline, decomp, design, heldout),
   scratch/qwen_prefill_obs.py, docs/preregs/qwen-prefill-wall-0.json, the
   held-out prompts frozen in scratch/evalsets/qwen_prefill_heldout_prompts.json.
   The pre-registration smoke exposed a lazy-evaluation attribution bias in
   the additive timer (upstream cost landing on the first consumer that
   forces it); inputs are now evaluated before each timed call. Registered
   at 485c49e1; chain at f9567077 (124 min).
4. VERDICT QWEN-PREFILL-WALL-0 (RESULTS L85637): NULL on the chunk hurdle
   (chunk 512 buys 5 percent at every length with exact parity and 6
   percent more peak memory; chunk 1,024 no more at twice the transient;
   chunk 128 costs 15 percent). Baseline 39 tokens per second, linear, 98
   percent GPU. Decomposition by two agreeing methods: W4 MLP 48 percent,
   DeltaNet recurrence 34 percent, S16 body 20 to 25 percent, the rest under
   3. fp16 matmul 0.89x at 3.5e-4 relative error.
5. The receipt audit traced the recurrence's cost to a mode flag:
   llmopt/lab/qtower.py never calls .eval(), MLX modules start in training
   mode, and GatedDeltaNet passes use_kernel = not training, so every
   prefill and decode of arm C has run gated_delta_ops (the Python loop)
   while gated_delta_kernel sat unused. A session microbenchmark on random
   operands: 42.5 ms (loop) against 2.3 ms (kernel) per layer-chunk; the
   outputs were not compared (non-finite on random gates), so no parity
   claim. Recommendation: a bounded GO to qualify the kernel path on
   prefill and decode (parity against the frozen census positions and the
   held-out prompts, near-tie margins read, then the same latency hurdle).
   Not a default change; not a new kernel; one line of code and a
   qualification's worth of receipts.

## Next (held for Artin's GO)

- The kernel-path qualification GO (above): expected ceiling about a
  third of first-token latency and a tenth of the decode step.
- The W4 GEMM half: a bytes-per-decoded-row or fused decode-and-multiply
  lever; precision is closed for now.
- Chunk 512 as a serving default is NOT recommended on these numbers (5
  percent for memory); it stays available as an argument.

## Hygiene

Receipts force-added by file name: the four stage receipts, the
adjudication pair, the chain log, the liverun record. Both checkouts at
origin/main.

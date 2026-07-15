# The math-native micro-model — a species born inside the closed system

Provenance: Artin's purely-on-rule-bits push (2026-07-15) matured by
the bad-habits argument: the whisper proved RL nudges but never
evicts incumbent preferences (CKA 0.9998; the heartbreaker's
suppressed minus was a bad habit winning). A from-scratch model has
NO incumbents: every gradient lands on undisputed territory, and the
ceiling is the closed system's reachable set — not "base model minus
unremovable habits."

## The organism

- **Tokenizer**: math-native, ~300-500 tokens — digits, operators,
  x, function names (sin/cos/tan/exp/log/atan/asin/sqrt), Integral,
  =>, parens, whitespace, newline. Character-level fallback for the
  rest. No BPE over web text; the charset mask IS the vocabulary.
- **Architecture**: Qwen-shaped but tiny — target ~15-30M params
  (d_model 384, 8-10 layers, 6 heads, GQA, RoPE, SwiGLU). Same
  shape family keeps every house instrument (layer probes, LoRA,
  CKA, the anatomy pipeline) working unchanged.
- **Trains on Mac (MPS)**: at 20M params, batch-64 steps are
  interactive; sampling is 20-100x the 0.5B's rate — which
  multiplies EVERY future GRPO cycle (verified speed compounding).

## Life stages

1. **Chain-pretraining** (the species' textbook): next-token on
   engine-minted verified chains at scale — mint 100k+ chains
   L1-L8 (Mac CPU farm, streamed, string seeds, fresh 50M band),
   formatted exactly as the step harness prompts (Current/Hints:
   none/Step). All data oracle-verified by construction: the
   organism never reads an unverified token in its life.
2. **SFT polish** on the balanced corpus recipe (the known-good
   mix rules apply: chains majority, finishing protected).
3. **GRPO from birth** — the same climb, the same gates, at 20-100x
   the wave rate. ES-LoRA becomes cheap to trial here too (the
   probe-batch fitness is fast at this scale).

## Pre-registered bars & questions

- **Priors-vs-drag (the headline)**: validity + solves vs the 0.5B
  at matched GRPO wall-clock. The 0.5B starts with algebra priors
  (good habits) AND web priors (bad habits); the native starts with
  neither. Three outcomes, all findings: native wins (drag
  dominated), native loses badly (priors were load-bearing — we
  get the first quantitative price of pretraining for symbolic
  math), native loses early then crosses (priors = head start,
  habits = ceiling).
- **Arithmetic acquisition**: can chain-pretraining alone teach
  18/2=9? Track coefficient-error rate vs the Arena taxonomy.
- **Anatomy from birth**: crystal snapshots at every stage — we
  get to WATCH structure form from random init (the efficient-
  coding frame's cleanest test: does it converge toward the
  0.5B's geometry? CKA/stitching bridge to Qwen at each stage —
  if the Platonic claim is right, the bridge quality should RISE
  as it trains).
- Kill switch: if chain-pretraining can't reach coherent step
  FORMAT (parseable expressions) within the first farm's data,
  scale data 5x once before verdicting.

## Assets already in place

Engine (unlimited verified data), fast oracle (30x), GRPO driver
(model-agnostic via HF interface), balance recipe, gates, anatomy
pipeline, layer probes, stitching bridge (to measure convergence
toward the big models' shared geometry).

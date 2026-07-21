# Training attacks — the lossless-speed program (2026-07-21)

Provenance: Artin, mid-day 2026-07-21 — "we need to attack training
itself more, not inferencing" + "lossless speed, speedy
verification" + "lots of good ideas/rungs let's make sure we keep
them." This spec is the keep.

## Scorecard: training-side levers, ranked by measured/priced savings

| Lever | Status | Saving | Evidence / gate |
|---|---|---|---|
| Mass-targeted diet (Rung A) | PROVEN, knob broken | -38% wall | gen-7: wall hit, rations too thin (L3 -3) — raise to 40-50% in 7b |
| Warm birth / template | PROVEN | ~1 epoch | CALC 57 v COLD 49 at ep1 (time machine) |
| Growth-not-rebirth | PROVEN | never re-pay mastered | 76 v 71 |
| TF32 cuda births | PROVEN | 2.5x | parity 65v64; bf16 stays BANNED (capability tax) |
| Late-layer metabolism | RUNNING | ~60% backward | confluence: delta mass layers 8-11; arm must match hot gate |
| Surprise-gated rows (metabolic v3) | BANKED | skip owned rows | wave-agreement + logprob, both free in-loop |
| **Prefix notation (v2.3)** | **NEXT** | -20-30% tokens forever | one tokenizer fn; one-variable A/B vs infix, same diet/gate |
| **Token-budget batching** | **NEXT** | ~2x on padded batches | pack to token budget vs BS=32; parity gate mandatory |
| Low-rank metabolic sidecar | BANKED | O(r*d) vs O(d^2) updates | confluence: delta rank 15-30; after frontier verdicts |
| Calculated model rung 3+ | BANKED | training -> arithmetic | the thesis endgame |

Non-goal (standing): hand-rolled autograd/matmuls — the bf16 cliff
showed dynamics are fragile; we cut tokens/rows/layers, not kernels.

## Implementation order (parity-gated, one variable each)

1. Token-budget batching in train_mathnative (flag --tokbudget):
   pack length-sorted rows to ~24k tok/batch. Parity gate: same
   seeds, gate within noise-bar (noise-bar = tonight's seed-variance
   measurement) on a 19M birth.
2. Prefix-notation A/B: encode_prefix() beside the tokenizer; one
   19M birth per notation, same diet/epochs/gate + quantization
   probe (banked prediction: prefix crystal is delimiter-outlier-
   free, holds deeper into int3).
3. Both ride seed-variance-style 19M births (40 min each) — free
   test vehicles.

## Calibration program (the gaps audit, same night)

- Frozen holdout battery (band 88M, corpus-overlap audited) — run
  at promotions only. scratch/holdout_gate.py.
- Seed variance: 3x 19M identical births (BIRTH_SEED env) — prices
  every +-2 claim ever made.
- Chain-carry ablation (Artin's carry hypothesis): same content,
  format ablated — chains vs step-shuffled vs root->answer-only,
  equal rows, same gate. If chains >> others, capability numbers
  carry a FORMAT dividend.
- Immune-system patch: absolute best-anchor tripwire (slow-leak
  blind spot: gen-7 rations + hot-arm erosion, one law two
  sightings).

## The endgame composition (einstein-reborn, kept honest)

ternary-native pipeline (only substrate with measured online gain:
+2 proxy twice) x surprise-gated LLMUE inside the 1e-4 ceiling x
late-layer/low-rank updates x shaped reward (b-lever, verdict
pending) x prefix notation x magic boards on the farm. Every clause
has a measured basis; assembly waits for the calibration verdicts
so it is born against an exam it hasn't memorized.

## NNUE/LLMUE boundary (doctrinal, from today)

NNUE = efficiently-updatable INFERENCE: exact algebra (accumulator
delta), zero learning, a theorem. LLMUE = efficiently-updatable
LEARNING: statistical, nonconvex, needs the immune system. They
share a design principle (small delta as the primitive), NOT
results — no NNUE efficiency number transfers. The EU-evaluation
riff (delta-scoring candidate rewrites) is the thread that DOES
inherit NNUE's math; keep the two named apart.

## The feedback ladder (Artin, 2026-07-21 — "feedback on top of verification")

Human error-recovery taxonomy mapped to oracle signals we already
compute (and mostly discard). Each rung raises b in dS/dt:

| Rung | Human question | Signal | Status |
|---|---|---|---|
| 1 | was I wrong? | verified bit | production since day 1 |
| 2 | how wrong? | Phi-distance / residual magnitude | shaped GRPO, RUNNING |
| 3 | what was right instead? | verified-vs-rejected sibling, same state | wave-contrast, banked |
| 4 | why was I wrong? | the diff-RESIDUAL (mechanism of failure), currently collapsed to False | frontier; must arrive as gradient not prompt (hints twice-nulled); needs a vocab-40-expressible encoding |

Run order = ladder order; each rung one-variable against the last.

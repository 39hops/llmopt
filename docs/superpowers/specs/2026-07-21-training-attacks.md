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

## Addendum (2026-07-21 evening): the delta doctrine + rung revisions

**THE DELTA DOCTRINE** (from the day of controls — two headlines
killed in six hours): every absolute number today was fragile
(baselines inherit transfer; frontier probes inherit hardware);
every same-checkpoint/same-device/same-seed PAIRED DELTA held.
Measurement rules adopted: (1) paired deltas only — a number
without its control is not a result; (2) one device per number
lineage, never compare across; (3) frontier probes run at B=16 —
near-tie branches get covered by sampling WIDTH, not precision
(margins ~0 = the model's genuine uncertainty; fp64 sampling
would re-resolve the same coin flips, not add knowledge); (4)
minimum-effective-dose is real (320-471 rows bought a 55M model
exactly nothing) — the exposure curve has a floor term.

**Rung revisions, everything composed:**
- Metabolic v3 (the composed organism): TERNARY substrate (three
  consistent +2 sessions — the only substrate with repeated
  measured online gain) + late-layer-only (control-rod result:
  71 v 57 under identical abuse, 60% cheaper) + LR <= 1e-4 +
  absolute-anchor tripwire + surprise-gated rows. Every clause
  measured within 48h.
- Calculator v3: clade-transfer term (three directional misses)
  + dose floor.
- Gen-7b REVISED: champion transfer already reaches 18/24-class
  L9 under favorable sampling — the play is no longer a giant
  shard but TARGETED consolidation (rows for the unsolved probe
  residue) + rations raised to 40-50%; farm shard sizes it.
- Precision program unchanged: fp64-vs-fp32 birth A/B and
  bf16x2/3xTF32 double-double (Ootomo-Yokota) stay queued — they
  address TRAINING arithmetic (the bf16-cliff class), which the
  device finding does not touch.

## Addendum 2 (2026-07-22 midnight): precision program notes

- KV CACHE SHIPPED to model (mathnative.py: rope pos0, Block past,
  MicroLM use_cache — backward compatible): equivalence 20/20
  token-identical CPU AND cuda; CPU 4.49x, cuda 1.13x; MPS verdict
  auto-runs post-session -> promote into sample_wave_lp if clean.
  Gate-v2 cross-problem batching: NULL (3x slower, no-KV O(T^2));
  exact-match machinery banks for KV+batch composition later.
- Absorption instrument, first reading: LR 1e-4 late-layer STE
  session = 0.0007% absorbed (fp32 loses ~nothing at hot LR).
  Decider queued: LR 1e-5 burst (deltas 10x smaller, the pilot's
  regime) — if absorption stays ~0, fp64-masters rung NULLS and
  we bank the memory savings; if it jumps, fp64 masters = free
  recovered learning.
- Register-fit doctrine (the precision stack, ledger 2026-07-22):
  operands narrow, accumulators wide, scale shared per block
  (MX at rest already ships), exact arithmetic at the oracle.
  Adaptive precision = the lab's stratification named.

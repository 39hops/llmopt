# The schedule-law queue (2026-07-22 morning, Artin GO)

Week's candidate law, from three independent overnight results:
**everything is a schedule against the corpus, nothing is a
constant** — bits (the ridge), evaluation (the rarity curve),
diet (band rotation) all must move as the corpus moves.

## 1. THE MERGED RUN (the headline)

Wide-ternary champion candidate: **same config as the measured
grid point (d768, 8 layers, ffn 3072, heads 12, ternary-from-
birth, 6ep per the discrete law) — only the DIET changes**
(one variable from the 65/120 gen4 point): full current corpus =
gen-6 diet + the finished L9a shard (~810 rows) with L1/L2/L3
rations at 40-50% (the gen-7 lesson applied). TF32 on the 3080.
Pre-registered: (a) beats its own gen4 twin's 65; (b) if the
ridge is real, contends with champion-76 on the production
battery (both numbers read on the SAME battery, per the battery
truth); (c) graded on the rarity curve (below), where the L9a
rows should move the RARE end specifically. If (b) lands, the
1.58-bit substrate takes the crown and the ternary-native
pipeline becomes the mainline.

## 2. Rarity-stratified gate (built FIRST, hours)

Capability as a curve over expression rarity, not a scalar.
Rarity measure: skeleton frequency — normalize each probe expr
(integer constants -> #), count its skeleton's occurrences in the
corpus cur-set; bin probes into frequency bands (common / mid /
rare / unseen-skeleton). Report solves per band. Reuses
holdout-v2 machinery + KV-fast gates. Every future promotion
reads the curve; territory births are judged at the rare end.

## 3. fp64 fold-in (conditional)

If the decider's fp64 arm recovers flips/proxy vs arm A
(verdict this morning): fp64 master-latents go INTO run 1's
trainer (elementwise, <1% wall). If null: booked, memory saved.

## 4. Series/Liouville rung 1 (afternoon chase)

Series chains (1,200 certified rows, O-markers stripped — the
partial sums tokenize in vocab-40 as plain polynomials) mixed
into a 19M diet; held-out series problems (seeds 17-19 per
family/level) as the probe: does the model learn coefficient-step
emission? Rung 2 (the jailbreak demo: "when stuck, expand" +
int(e^(-x^2)) termwise) only after rung 1 passes.

## 5. Behind (specced, not scheduled)

- Chain-carry REDO at full budget (the void was a design error:
  8k rows + repetition asymmetry; arms must get the full corpus
  and matched dedup).
- Metabolic band-rotation: the LLMUE frontier scheduler — feed
  bands at the rate transfer margins refresh (compounding
  verdict: gain is a fresh-pair harvest, not session length).
- Boundary-or-bulk regression writeup once the grid gains the
  merged run's point.

## Doctrine deltas adopted this morning

- Prefer UNDER-width at fixed corpus (over-shooting degrades
  harder: 58@768 vs 61@256, fp32).
- Cross-battery absolutes invalid BOTH directions; report
  battery + deltas only.
- W* = f(corpus / effective bits): precision is a schedule knob.

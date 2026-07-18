# The Alphabet Tournament — which discrete weight set fits a closed system?

Provenance: Artin's riff arc 2026-07-18 ("we store 1.58 bits as 2 —
what fits in the last 0.42?" → "could it use {-1, -i, 0, i, 1}?"),
composed with the ternary-parity result (69/120 @ 1.58 bits) and
the escape-hatch idea. Banked same day; run after the warm-birth
persistence verdict and gen-6.

## Question

Ternary-from-birth proved the weight ALPHABET is a design axis
(topology carries capability; per-weight precision converts to
population count). So: is {-1,0,+1} the right alphabet for
mathematics, or just the first one we tried? What do extra states
buy — magnitude (resolution), rotation (a new verb), or nothing?

## Contestants (one quantizer swap each in train_ternary.py)

| # | Alphabet | Bits | Axis | What it tests |
|---|---|---|---|---|
| T | {0, ±1} | 1.58 | baseline | reigning co-champion |
| B | {±1} | 1.00 | control | is the ZERO load-bearing? (ternary-born had 27% exact zeros) |
| M5 | {0, ±1, ±2} | 2.32 | magnitude | resolution, bit-matched twin of G5 |
| G5 | {0, ±1, ±i} | 2.32 | rotation | quarter-turn as primitive (euler move: d/dx on sin/cos IS ×i) |
| E7 | {0, ±1, ±ω, ±ω²}, ω=e^{iπ/3} | 2.81 | rotation | hexagonal/Eisenstein lattice — does finer phase pay more? |
| P2 | {0, ±½, ±1, ±2, ±4} | 3.17 | magnitude | shift-only multiplies (hardware-blessed, ~fp4) |
| Q9 | {0, ±1, ±i, ±j, ±k} | 3.17 | rotation | quaternion units; honest re-ask of the provenance-less "quaternion" null, at alphabet level |

Composable with all rows: the ESCAPE code (one spare state =
"fetch fp from side table") — per-weight mixed precision priced at
~0.3 avg bits for 1% escapes; attacks tail-dies-first directly.
Also queued as a variant: 2-bit-from-birth (M4 = {-1,0,+1,2}),
the "spend the wasted 4th code" minimum experiment.

## Protocol

- 19M scale, gen-4 diet, fp32 latents + STE (the proven recipe),
  identical epochs, honest chain gate, per-family readout.
- Complex alphabets: split real/imag matmul (Wirtinger grads);
  head stays fp as always.
- One variable per arm — alphabet only. Same seeds, same schedule.
- ~40 min/contestant on Mac. Book every arm, nulls included.

## Pre-registered predictions (on the record)

1. B (binary) UNDERPERFORMS ternary — the zero/silence is
   load-bearing (27% sparsity was structure, not accident).
2. G5 beats M5 on trig/exp (euler) families specifically, or the
   rotation thesis nulls. Overall winner between them = verdict on
   rotation-vs-resolution at equal bits.
3. No contestant beats ternary by more than the escape-hatch
   variant does — i.e. RARITY, not alphabet richness, is the
   binding constraint (precision-follows-rarity law).
4. If a pattern emerges (rotational domains want roots of unity),
   the birth calculator gains an alphabet-selection rule and the
   template gains a computed parameter: ALPHABET-FOLLOWS-DOMAIN.

## Interactions

- Winner feeds gen-6+ births and the progressive-precision
  curriculum (grow wiring in the winner, unlock precision late).
- E7/G5 winner would revive the complex-FFN re-ask in discrete
  form (cheaper than the full complex birth).
- P2 winner would make the int-kernel port (practice_7 lineage)
  multiplication-free.

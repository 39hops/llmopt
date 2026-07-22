# Duo-substrate wave (2026-07-22 evening, Artin GO)

The riff: **ternary = search substrate, precision = accuracy
substrate** (Artin, from the rarity curves + per-probe diff). The
within-model version of the duo — testable in-house, unlike the
big-reasoner duo (NO-TEST stands).

## Measured basis

Identical 200-probe rarity battery, per-probe dumps:
- CHAMPION (fp32) 128/200 — owns rare-seen (11/18), memorized
  techniques (log^2 by-parts, folded constants).
- TERNARY-73 127/200 — owns unseen (17/65), rule recognitions
  (derivative-of-composite forms).
- **Oracle union: 137/200** (66/69 common, 39/48 mid, 13/18 rare,
  19/65 unseen) — +9 over best single, but at 2x sample budget.

## Experiment 1: budget-matched mixed wave (scratch/duo_wave.py)

Per ply: 8 samples from ternary + 8 from champion (vs 16 from one) —
same total budget as a single-model gate. Merged, deduped,
oracle-verified; first verified next-step wins (SOLVED preferred).
Same 200-probe battery, same seeds, MPS.

Pre-registered:
- (a) mixed >= 128 (beats best single at equal budget) => the split
  is real and free; adopt as the default gate/search sampler.
- (b) bin signature: rare >= 11 AND unseen >= 17 (each substrate
  covers its tail even at half budget).
- (c) if mixed < 127: half-budget starvation dominates
  complementarity => the duo needs routing (send the wave to ONE
  substrate by predicted bin), not mixing. Bank either way.

Cost note: two resident models — memory fine (45M+55M), wall ~2x
per-token (two forward passes per ply at half batch each); KV cache
keeps it ~10 min.

## Experiment 2 (rider): harder-territory paired read

Same-device (MPS) L9-band probe, champion vs ternary paired — the
"what happens as questions get harder" read. Report as paired delta
only (device doctrine).

## Experiment 3 (rider, cheap): ternary weight census

Not the weight-reader model (it reads tiny function-MLPs, out of
domain for a 45M transformer) — the readable version: per-layer
ternary stats (zero fraction, sign balance, per-matrix structure)
on TERNARY-73 vs the fp32 champion's sign/magnitude profile. The
"what does the crystal look like" number.

## Follow-ups this feeds

- Rarity-routed precision (training-time half of the same split).
- If (c): a bin-router (skeleton frequency is computable at
  inference from the corpus census — route rare/unseen to ternary,
  common/rare-seen to fp32).

# Weight-space reader: classify function family from MLP weights

**Date:** 2026-07-06
**Status:** Approved

## Goal

Test whether a model can "read" neural-network weights: train a small
transformer that takes a tiny MLP's flattened weights as input and
predicts which function family the MLP was trained to fit. The
permutation-symmetry ablation (raw vs canonicalized vs augmented) is
the headline result.

## Components

### 1. Subjects — `llmopt/weightspace/subjects.py`

- Subject net: MLP 1 -> 16 -> 16 -> 1, tanh activations, torch.
- Function families (6): `a*sin(b*x+c)`, poly deg 2, poly deg 3,
  `a*exp(-b*x**2)`, `a*abs(x)+b*x`, `a*tanh(b*x)`. Coefficients from
  string-seeded RNG: `random.Random(f"{family}-{i}-{seed}")` (CLAUDE.md
  convention; never tuple seeds).
- Training: few hundred Adam steps on x in [-2, 2] (grid + jitter),
  MSE loss, <1 s/subject on CPU. Subjects with final MSE above a
  threshold are rejected and resampled — diverged nets are label
  noise, not hard examples.
- Dataset: 4000 train / 500 eval. Eval uses a disjoint seed namespace;
  assert no exact coefficient-tuple collision with train (contamination
  guard per CLAUDE.md).
- Oracle: the generating function IS the label; no scoring ambiguity.

### 2. Reader — `llmopt/weightspace/reader.py`

- From-scratch transformer encoder, torch, ~1M params: 4 layers,
  8 heads, d_model 128.
- Tokenization: ONE TOKEN PER NEURON — feature vector is the neuron's
  incoming weights + bias, zero-padded to a fixed width; plus a learned
  layer-index embedding. A 1-16-16-1 subject = 33 tokens
  (16 + 16 + 1 output neuron).
- Head: mean-pool -> linear -> 6-way family logits, cross-entropy.

### 3. Permutation-symmetry arms (same data, same reader)

- **raw** — weights as trained.
- **canonical** — hidden-layer neurons sorted by incoming-weight L2
  norm; the next layer's columns permuted to match, preserving the
  computed function. Property: `canonicalize(permute(net)) ==
  canonicalize(net)` exactly (test this).
- **augmented** — raw, plus random hidden-layer permutations applied as
  train-time augmentation.

Prediction on record: canonical >> augmented > raw; raw near the 1/6
chance floor at this data scale. If even canonical sits at chance,
report that as the (negative) headline.

### 4. Script — `scripts/train_weight_reader.py`

Generate subjects (parallel over a process pool), train the reader
per arm, print per-arm eval accuracy vs the 16.7% chance floor.
Everything CPU/MPS-friendly; no dependency on the GPU queue.

## Testing

- Subject generation: deterministic under fixed seed; rejects bad fits;
  train/eval coefficient-tuple disjointness.
- Canonicalization: exact invariance property above; the canonicalized
  net computes the identical function (forward-pass allclose).
- Reader: forward shape, overfits 32 subjects to ~100% (sanity).

## Non-goals (later rungs)

- Coefficient regression (rung 2).
- Generator direction — predict weights from function description
  (rung 3).
- Reading LLM/LoRA weights (different scale regime entirely).

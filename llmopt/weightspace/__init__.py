"""Weight-space learning: models that read other models' weights.

First result (2026-07-06, scripts/train_weight_reader.py, 4000/500
subjects, M3 Pro CPU): a ~1M-param neuron-token transformer classifies
the function family of a 1-16-16-1 subject MLP from raw weights at
80.8% (chance 16.7%); canonicalized 82.4%, permutation-augmented 88.4%.

The spec's on-record prediction (canonical >> augmented > raw ~ chance)
was wrong twice, instructively: (1) the neuron-token encoding (no
neuron position, attention + mean-pool) is already order-invariant, so
raw never faced most of the symmetry; (2) augmentation beat
canonicalization — norm-sorting is discontinuous under near-ties,
imposing invariance proved worse than teaching it.

Later rungs: coefficient regression, weights-from-description
generation, LoRA-adapter reading.
"""

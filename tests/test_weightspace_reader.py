"""Neuron-token reader: shapes, gradient flow (overfit sanity)."""

import torch

from llmopt.weightspace.reader import (
    WeightReader,
    evaluate_reader,
    tokenize,
    train_reader,
)
from llmopt.weightspace.subjects import FAMILIES, make_dataset


def test_tokenize_shape():
    s = make_dataset(1, seed=0)[0]
    t = tokenize(s.weights)
    assert t.shape == (33, 17)  # 16 + 16 + 1 neurons, 16 weights + bias


def test_forward_shape():
    model = WeightReader()
    logits = model(torch.randn(4, 33, 17), torch.tensor([[0] * 16 + [1] * 16 + [2]] * 4))
    assert logits.shape == (4, 6)


def test_reader_overfits_small_set():
    subjects = make_dataset(32, seed=0)
    labels = [FAMILIES.index(s.family) for s in subjects]
    model = train_reader(subjects, labels, epochs=200, augment=False,
                         canonical=False, seed=0)
    assert evaluate_reader(model, subjects, labels) >= 0.95

"""Subject MLP generation and weight-space transforms."""

import torch

from llmopt.weightspace.subjects import (
    FAMILIES,
    canonicalize,
    forward,
    make_dataset,
    make_subject,
    permute_hidden,
)


def test_six_families():
    assert len(FAMILIES) == 6


def test_subject_deterministic():
    a = make_subject("sin", 3, seed=0)
    b = make_subject("sin", 3, seed=0)
    assert a.coeffs == b.coeffs
    assert all(torch.equal(x, y) for x, y in zip(a.weights, b.weights))


def test_subject_fits_its_function():
    s = make_subject("poly2", 1, seed=0)
    assert s.fit_mse < 0.01


def test_permute_preserves_function():
    s = make_subject("tanh", 0, seed=0)
    x = torch.linspace(-2, 2, 64)[:, None]
    g = torch.Generator().manual_seed(0)
    p = permute_hidden(
        s.weights, torch.randperm(16, generator=g), torch.randperm(16, generator=g)
    )
    assert torch.allclose(forward(s.weights, x), forward(p, x), atol=1e-5)


def test_canonicalize_collapses_permutations():
    s = make_subject("gauss", 2, seed=0)
    g = torch.Generator().manual_seed(1)
    p = permute_hidden(
        s.weights, torch.randperm(16, generator=g), torch.randperm(16, generator=g)
    )
    ca, cb = canonicalize(s.weights), canonicalize(p)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ca, cb))


def test_canonicalize_preserves_function():
    s = make_subject("sin", 5, seed=0)
    x = torch.linspace(-2, 2, 64)[:, None]
    assert torch.allclose(
        forward(s.weights, x), forward(canonicalize(s.weights), x), atol=1e-5
    )


def test_dataset_family_balance_and_exclude():
    ds = make_dataset(12, seed=0)
    assert len(ds) == 12
    assert len({s.family for s in ds}) == 6
    banned = frozenset(s.coeffs for s in ds)
    ds2 = make_dataset(12, seed=1, exclude=banned)
    assert not banned & {s.coeffs for s in ds2}

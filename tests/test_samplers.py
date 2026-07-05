"""Sampler pipeline invariants — pure CPU torch, no model."""

import math

import torch

from llmopt.decoding.samplers import (
    Mirostat,
    SamplerPipeline,
    dry,
    min_p,
    temperature,
    top_k,
    top_p,
)


def _logits():
    # descending probs: token 0 most likely
    return torch.log(torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05]))


def _kept(logits):
    return {i for i, v in enumerate(logits.tolist()) if v != float("-inf")}


def test_top_k_keeps_k_highest():
    assert _kept(top_k(2)(_logits(), [])) == {0, 1}


def test_top_p_keeps_smallest_covering_prefix():
    # cum: .4 .7 .85 ... -> p=0.7 keeps {0,1} (crossing token kept)
    assert _kept(top_p(0.7)(_logits(), [])) == {0, 1}
    assert _kept(top_p(0.05)(_logits(), [])) == {0}


def test_min_p_scales_with_max_prob():
    # floor = 0.3 * 0.4 = 0.12 -> keeps 0.4, 0.3, 0.15
    assert _kept(min_p(0.3)(_logits(), [])) == {0, 1, 2}


def test_temperature_preserves_argmax():
    out = temperature(0.5)(_logits(), [])
    assert int(out.argmax()) == 0
    assert torch.allclose(out, _logits() / 0.5)


def test_dry_penalizes_repeat_continuation():
    # ctx suffix "1 2 3" matched earlier; token 4 would extend the repeat
    ctx = [9, 1, 2, 3, 4, 8, 1, 2, 3]
    logits = torch.zeros(10)
    out = dry(multiplier=1.0, base=2.0, allowed_len=2)(logits, ctx)
    assert out[4] < 0  # penalized: 1.0 * 2**(3-2) = 2.0
    assert math.isclose(float(out[4]), -2.0)
    assert (out[:4] == 0).all() and (out[5:] == 0).all()


def test_dry_respects_allowed_len():
    ctx = [1, 2, 5, 6, 1, 2]  # match length 2 == allowed_len -> no penalty
    out = dry(allowed_len=2)(torch.zeros(10), ctx)
    assert (out == 0).all()


def test_mirostat_mu_update_direction():
    gen = torch.Generator().manual_seed(0)

    # surprise < tau (near-certain draw) -> mu rises to admit more tokens
    m = Mirostat(tau=2.0, eta=0.5)
    peaked = torch.log(torch.tensor([0.999, 1e-4, 1e-4]))
    m(peaked, gen)
    assert m.mu > 2.0 * m.tau

    # surprise > tau (uniform over 16: 4 bits) -> mu drops
    m = Mirostat(tau=2.0, eta=0.5)
    m(torch.zeros(16), gen)
    assert m.mu == 2.0 * m.tau - 0.5 * (4.0 - 2.0)


def test_pipeline_seeded_and_deterministic():
    pipe = lambda: SamplerPipeline(temperature(0.8), top_k(3), seed=42)
    p1, p2 = pipe(), pipe()
    b = [p1(_logits()) for _ in range(10)]
    c = [p2(_logits()) for _ in range(10)]
    assert b == c
    assert all(t in {0, 1, 2} for t in b)  # top_k(3) mask respected


def test_pipeline_greedy_equivalent_at_tiny_temperature():
    pipe = SamplerPipeline(temperature(1e-4), seed=0)
    assert pipe(_logits()) == 0

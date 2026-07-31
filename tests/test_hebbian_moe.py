import pytest
import torch

from llmopt.train.hebbian_moe import HebbianCoupler, merge_experts


def _experts(n=4, shapes=((8, 4), (4, 8)), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(*s, generator=g) for s in shapes]
            for _ in range(n)]


def test_ema_matches_manual():
    c = HebbianCoupler(3, ema_decay=0.9)
    p = torch.softmax(torch.randn(2, 5, 3,
                                  generator=torch.Generator()
                                  .manual_seed(1)), -1)
    c.observe(p)
    manual = torch.einsum("bti,btj->ij", p, p) / (2 * 5)
    assert torch.allclose(c.ema, manual, atol=1e-6)
    c.observe(p)
    manual2 = 0.9 * manual + 0.1 * manual
    assert torch.allclose(c.ema, manual2, atol=1e-6)


def test_relax_contracts_pairwise_distance():
    c = HebbianCoupler(4, lam=0.5, every=1)
    c.observe(torch.full((1, 10, 4), 0.25))
    ex = _experts()
    d0 = (ex[0][0] - ex[1][0]).norm()
    assert c.maybe_relax(1, ex)          # step 1 % 1 == 0
    d1 = (ex[0][0] - ex[1][0]).norm()
    assert d1 < d0
    # repeated relaxation drives experts together
    for s in range(2, 200):
        c.maybe_relax(s, ex)
    assert (ex[0][0] - ex[3][0]).abs().max() < 1e-3


def test_relax_respects_schedule_and_edges():
    c = HebbianCoupler(4, lam=0.5, every=100,
                       edges={(0, 1), (1, 0)})
    c.observe(torch.full((1, 4, 4), 0.25))
    ex = _experts()
    before_23 = ex[2][0].clone()
    assert not c.maybe_relax(150, ex)     # off-schedule: no-op
    assert c.maybe_relax(200, ex)
    # non-edge expert untouched, edge pair moved
    assert torch.equal(ex[2][0], before_23)
    assert not torch.equal(ex[0][0], _experts()[0][0])


def test_relax_symmetric_under_uniform_ema():
    # snapshot semantics: with symmetric ema, a 2-expert relax
    # moves both toward the SAME midpoint (order-independent)
    c = HebbianCoupler(2, lam=0.5, every=1)
    c.observe(torch.full((1, 4, 2), 0.5))
    ex = _experts(n=2)
    a0, b0 = ex[0][0].clone(), ex[1][0].clone()
    c.maybe_relax(1, ex)
    coef = 0.5 * float(c.ema[0, 1])
    assert torch.allclose(ex[0][0], a0 + coef * (b0 - a0),
                          atol=1e-6)
    assert torch.allclose(ex[1][0], b0 + coef * (a0 - b0),
                          atol=1e-6)


def test_merge_is_mean_and_shape_stable():
    ex = _experts(n=3)
    m = merge_experts(ex)
    assert len(m) == 2
    for k in range(2):
        assert torch.allclose(
            m[k], torch.stack([e[k] for e in ex]).mean(0))


def test_input_validation():
    with pytest.raises(ValueError):
        HebbianCoupler(1)
    c = HebbianCoupler(3)
    with pytest.raises(ValueError):
        c.observe(torch.zeros(1, 2, 4))
    c.observe(torch.full((1, 2, 3), 1 / 3))
    with pytest.raises(ValueError):
        c.maybe_relax(0, _experts(n=2))
    with pytest.raises(ValueError):
        merge_experts([])

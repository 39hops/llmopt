"""MoE layer: routing equivalence, balancing losses, capacity drops,
expert LRU offload. CPU-only."""

import pytest

torch = pytest.importorskip("torch")

from llmopt.moe.layer import MoELayer, load_balance_loss, router_z_loss
from llmopt.moe.offload import ExpertCache

H, I, E = 16, 32, 4


def _layer(**kw):
    torch.manual_seed(0)
    return MoELayer(H, I, E, **kw)


def test_top_k_equals_e_matches_dense_mixture():
    layer = _layer(top_k=E)
    x = torch.randn(6, H)
    out, aux = layer(x)
    probs = torch.softmax(layer.router(x), dim=-1)
    dense = sum(
        probs[:, i, None] * layer.experts[i](x) for i in range(E)
    )
    assert torch.allclose(out, dense, atol=1e-5)
    assert aux["dropped"] == 0


def test_top1_uses_single_expert_per_token():
    layer = _layer(top_k=1)
    x = torch.randn(5, H)
    out, aux = layer(x)
    best = torch.softmax(layer.router(x), -1).argmax(-1)
    for t in range(5):
        assert torch.allclose(out[t], layer.experts[int(best[t])](x[t]), atol=1e-5)


def test_balance_loss_prefers_uniform_routing():
    n = 64
    uniform = {
        "probs": torch.full((n, E), 1 / E),
        "expert_load": torch.full((E,), 1 / E),
    }
    skewed = {
        "probs": torch.eye(E)[torch.zeros(n, dtype=torch.long)],
        "expert_load": torch.eye(E)[0],
    }
    assert float(load_balance_loss(uniform)) == pytest.approx(1.0)
    assert float(load_balance_loss(skewed)) == pytest.approx(float(E))


def test_capacity_drops_overflow_but_keeps_shape():
    layer = _layer(top_k=1, capacity_factor=0.5)
    x = torch.randn(32, H)
    out, aux = layer(x)
    assert out.shape == x.shape
    loads = aux["expert_load"] * 32
    if (loads > 32 * 0.5 / E).any():
        assert aux["dropped"] > 0


def test_losses_backprop():
    layer = _layer(top_k=2)
    x = torch.randn(8, H)
    out, aux = layer(x)
    loss = out.pow(2).mean() + 0.01 * load_balance_loss(aux) + router_z_loss(aux)
    loss.backward()
    assert layer.router.weight.grad is not None
    assert any(p.grad is not None for p in layer.experts[0].parameters())


def test_expert_cache_lru_and_hit_rate():
    experts = [torch.nn.Linear(2, 2) for _ in range(4)]
    cache = ExpertCache(experts, capacity=2)
    for i in (0, 1, 0, 2, 0, 3):  # 0 hot; 1,2,3 stream through
        cache.get(i)
    assert cache.hits == 2  # the repeat 0s
    assert cache.misses == 4
    assert list(cache.resident) == [0, 3]  # 0 kept warm by LRU, not FIFO
    assert cache.hit_rate == pytest.approx(2 / 6)

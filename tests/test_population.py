"""Population LoRA vs K independent single-adapter runs (Apple silicon).

The oracle is the sequential run: a population forward/backward must
produce, for each adapter k, exactly the values and grads that adapter
would see training alone. Weight-distance scoring is banned house-wide,
but this is grads-of-the-same-function-at-the-same-point — an exact
equivalence, not a similarity score.
"""

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

from llmopt.train.fused_ce import fused_ce  # noqa: E402
from llmopt.train.population import (  # noqa: E402
    PopLoRALinear, apply_population_lora, population_loss)

K, B, T, D, O = 3, 2, 5, 16, 24


def _pop_linear(seed=0):
    mx.random.seed(seed)
    base = nn.Linear(D, O)
    lin = PopLoRALinear(base, K, r=4, alpha=8.0)
    lin.b = mx.random.normal(lin.b.shape) * 0.1  # nonzero: exercise delta
    return lin


def test_forward_matches_per_adapter_math():
    lin = _pop_linear()
    x = mx.random.normal((K * B, T, D))
    y = lin(x)
    xr = x.reshape(K, B, T, D)
    for k in range(K):
        want = (lin.base(xr[k])
                + lin.scaling * (xr[k] @ lin.a[k].T @ lin.b[k].T))
        got = y.reshape(K, B, T, O)[k]
        assert mx.abs(want - got).max().item() < 1e-4


def test_grads_match_sequential():
    """Population grad slice k == the grad of adapter k training alone."""
    lin = _pop_linear(seed=1)
    x = mx.random.normal((K * B, T, D))

    def pop_loss(a, b):
        lin.a, lin.b = a, b
        return (lin(x) ** 2).reshape(K, -1).mean(axis=1).sum()

    ga, gb = mx.grad(pop_loss, argnums=(0, 1))(lin.a, lin.b)

    for k in range(K):
        def solo_loss(a_k, b_k):
            xk = x.reshape(K, B, T, D)[k]
            y = lin.base(xk) + lin.scaling * (xk @ a_k.T @ b_k.T)
            return (y ** 2).mean()

        sa, sb = mx.grad(solo_loss, argnums=(0, 1))(lin.a[k], lin.b[k])
        assert mx.abs(ga[k] - sa).max().item() < 1e-4
        assert mx.abs(gb[k] - sb).max().item() < 1e-4


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D, D)
        self.ln = nn.Linear(D, D)  # non-target: must stay plain


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_TinyBlock(), _TinyBlock()]


def test_apply_wraps_targets_only():
    m = _TinyModel()
    n = apply_population_lora(m, k=2, r=2)
    assert n == 2
    assert isinstance(m.layers[0].q_proj, PopLoRALinear)
    assert isinstance(m.layers[1].q_proj, PopLoRALinear)
    assert isinstance(m.layers[0].ln, nn.Linear)
    # frozen base, trainable adapters
    from mlx.utils import tree_flatten
    names = [k for k, _ in tree_flatten(m.trainable_parameters())]
    assert names and all(k.endswith((".a", ".b")) for k in names)


def test_population_loss_is_sum_of_slices():
    mx.random.seed(3)
    v = 101
    h = mx.random.normal((K * B * T, D)).astype(mx.float16)
    w = mx.random.normal((v, D)).astype(mx.float16) * 0.05
    t = mx.random.randint(0, v, (K * B * T,))
    got = population_loss(h, w, t, K, chunk=7).item()
    want = sum(
        fused_ce(h.reshape(K, -1, D)[k], w,
                 t.reshape(K, -1)[k], chunk=7).item()
        for k in range(K))
    assert abs(got - want) < 1e-3

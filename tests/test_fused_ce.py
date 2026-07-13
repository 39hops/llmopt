"""Fused chunked CE vs naive full-logits reference (Apple silicon only)."""

import pytest

mx = pytest.importorskip("mlx.core")

from llmopt.train.fused_ce import IGNORE_INDEX, fused_ce, naive_ce


def _data(n=100, d=64, v=997, seed=0, ignore_frac=0.0):
    mx.random.seed(seed)
    h = mx.random.normal((n, d)).astype(mx.float16)
    w = mx.random.normal((v, d)).astype(mx.float16) * 0.05
    t = mx.random.randint(0, v, (n,))
    if ignore_frac:
        mask = mx.random.uniform(shape=(n,)) < ignore_frac
        t = mx.where(mask, IGNORE_INDEX, t)
    return h, w, t


@pytest.mark.parametrize("chunk", [7, 32, 100, 1000])
def test_loss_matches_naive(chunk):
    h, w, t = _data()
    a = naive_ce(h, w, t)
    b = fused_ce(h, w, t, chunk=chunk)
    assert abs(a.item() - b.item()) < 1e-3


def test_ignore_index_masked():
    h, w, t = _data(ignore_frac=0.4, seed=1)
    a = naive_ce(h, w, t)
    b = fused_ce(h, w, t, chunk=33)
    assert abs(a.item() - b.item()) < 1e-3


def test_grads_match_naive():
    h, w, t = _data(n=64, v=511, seed=2, ignore_frac=0.3)
    ga_h, ga_w = mx.grad(lambda h_, w_: naive_ce(h_, w_, t),
                         argnums=(0, 1))(h, w)
    gb_h, gb_w = mx.grad(lambda h_, w_: fused_ce(h_, w_, t, chunk=17),
                         argnums=(0, 1))(h, w)
    mx.eval(ga_h, ga_w, gb_h, gb_w)
    assert mx.abs(ga_h.astype(mx.float32)
                  - gb_h.astype(mx.float32)).max().item() < 2e-3
    assert mx.abs(ga_w.astype(mx.float32)
                  - gb_w.astype(mx.float32)).max().item() < 2e-3


def test_all_ignored_is_finite():
    h, w, _ = _data(n=8)
    t = mx.full((8,), IGNORE_INDEX, dtype=mx.int32)
    assert mx.isfinite(fused_ce(h, w, t, chunk=3)).item()

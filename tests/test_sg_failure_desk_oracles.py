"""SG-FAILURE-DESK-0 oracle and reduction laws (scratch/sg_failure_desk.py):
the ridge / least-squares oracle recovers an exactly linear target on a
held-out set (ratio near 0, cosine near 1) and cannot represent a
non-linear one (ratio near 1) that the random-feature ridge does; the
relative ridge law; ratio / cosine / rms reductions; the registered
constants (fit / held chunks, desk steps, RF width and seed). No
checkpoint is read."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("sg_failure_desk", ROOT / "scratch" / "sg_failure_desk.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.FIT_CHUNKS == [0, 2, 4, 6] and m.HELD_CHUNKS == [1, 3, 5, 7]
    assert m.DESK_STEPS == [0, 463, 1028, 2056, 3084, 5140, 7196, 10280, 12336, 15420]
    assert m.EIG_NULL_REL == 1e-12 and m.PINV_RTOL == 1e-6 and m.Z_CLIP == 3.0 and m.RF_WIDTH == 2048 and m.RF_SEED == 777 and m.SG_BLOCKS == [4, 5, 6, 7]
    return m


def _linear_problem(mod, n=3000, d_in=64, d_out=16, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g, dtype=torch.float64)
    Wt = torch.randn(d_in, d_out, generator=g, dtype=torch.float64)
    Y = X @ Wt
    X1 = torch.cat([X, torch.ones(n, 1, dtype=torch.float64)], 1)
    return X1[: n // 2], Y[: n // 2], X1[n // 2:], Y[n // 2:]


def test_linear_oracle_recovers_linear_target(mod):
    Xf, Yf, Xh, Yh = _linear_problem(mod)
    W = mod.ridge_fit(Xf, Yf, 0.0)
    assert mod.ratio(Yh, Xh @ W) < 1e-3
    assert mod.cos_pooled(Xh @ W, Yh) > 0.999


def test_gcv_ridge_matches_least_squares_on_a_rank_deficient_design(mod):
    import torch
    Xf, Yf, Xh, Yh = _linear_problem(mod)
    Xf2 = torch.cat([Xf, Xf[:, :5]], 1)     # duplicated columns: X^T X singular
    Xh2 = torch.cat([Xh, Xh[:, :5]], 1)
    W, lam, info = mod.gcv_ridge(Xf2, Yf)
    assert mod.ratio(Yh, Xh2 @ W) < 1e-2, info
    Wls = mod.ridge_fit(Xf2, Yf, 0.0)
    assert abs(mod.ratio(Yh, Xh2 @ W) - mod.ratio(Yh, Xh2 @ Wls)) < 1e-2


def test_richer_oracle_beats_linear_on_a_nonlinear_target(mod):
    import torch
    g = torch.Generator().manual_seed(1)
    n, d_in = 4000, 24
    X = torch.randn(n, d_in, generator=g, dtype=torch.float64)
    Y = (X[:, :8] ** 2 - 1) + X[:, 8:16] * X[:, 16:24]   # zero-mean quadratic target, 8 outputs: linearly unrepresentable
    X1 = torch.cat([X, torch.ones(n, 1, dtype=torch.float64)], 1)
    Xf, Yf, Xh, Yh = X1[: n // 2], Y[: n // 2], X1[n // 2:], Y[n // 2:]
    W = mod.ridge_fit(Xf, Yf, 0.0)
    lin = mod.ratio(Yh, Xh @ W)
    mu, sd = Xf[:, :-1].mean(0, keepdim=True), Xf[:, :-1].std(0, keepdim=True)
    Pf, Ph = mod.rf_design(Xf, mu, sd), mod.rf_design(Xh, mu, sd)
    Wr, lam, info = mod.gcv_ridge(Pf, Yf)
    rf = mod.ratio(Yh, Ph @ Wr)
    assert lin > 0.9 and rf < 0.5 * lin, (lin, rf, info)
    assert info["chosen_rel"] in mod.GCV_GRID_REL and lam > 0


def test_richer_oracle_contains_linear(mod):
    Xf, Yf, Xh, Yh = _linear_problem(mod)
    mu, sd = Xf[:, :-1].mean(0, keepdim=True), Xf[:, :-1].std(0, keepdim=True)
    Wr, lam, info = mod.gcv_ridge(mod.rf_design(Xf, mu, sd), Yf)
    assert mod.ratio(Yh, mod.rf_design(Xh, mu, sd) @ Wr) < 1e-2, info


def test_gcv_picks_a_small_lambda_on_a_clean_linear_target(mod):
    Xf, Yf, Xh, Yh = _linear_problem(mod)
    W, lam, info = mod.gcv_ridge(Xf, Yf)
    assert mod.ratio(Yh, Xh @ W) < 1e-2, info
    assert info["chosen_rel"] <= 1e-4


def test_rf_features_deterministic(mod):
    import torch
    X = torch.randn(10, 5, dtype=torch.float64)
    assert torch.equal(mod.rf_features(X), mod.rf_features(X))
    assert mod.rf_features(X).shape == (10, mod.RF_WIDTH)


def test_reductions(mod):
    import torch
    y = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    assert mod.ratio(y, torch.zeros_like(y)) == 1.0
    assert mod.ratio(y, y) == 0.0
    assert mod.cos_pooled(y, 2 * y) == pytest.approx(1.0)
    assert mod.cos_pooled(y, torch.zeros_like(y)) is None
    assert mod.rms(y) == pytest.approx((25 / 4) ** 0.5)

"""GPTQ/AWQ/HQQ vs the RTN baseline, sparsity patterns, SVD optimality.

The quality bar for every calibrated method: lower output error
||X Wq^T - X W^T|| than plain round-to-nearest at the same bit width.
"""

import pytest

torch = pytest.importorskip("torch")

from llmopt.quantize.lowrank import rank_error_curve, svd_factorize
from llmopt.quantize.methods import awq, gptq, hqq, rtn
from llmopt.quantize.sparsity import magnitude_prune, prune_24

OUT, IN, N = 32, 64, 256


@pytest.fixture()
def data():
    torch.manual_seed(0)
    w = torch.randn(OUT, IN)
    # heavy-tailed activations: a few high-magnitude channels, like real LLMs
    x = torch.randn(N, IN) * (1 + 5 * (torch.rand(IN) > 0.9).float())
    return w, x


def _out_err(w, wq, x):
    return float((x @ (wq - w).T).pow(2).mean())


def test_gptq_beats_rtn(data):
    w, x = data
    wq = gptq(w, x.T @ x, bits=3)
    assert _out_err(w, wq, x) < _out_err(w, rtn(w, 3), x)


def test_awq_beats_rtn_on_salient_channels(data):
    w, x = data
    wq, scales = awq(w, x, bits=3)
    assert _out_err(w, wq, x) < _out_err(w, rtn(w, 3), x)
    assert scales.shape == (IN,)


def test_hqq_beats_minmax_on_outlier_weights():
    torch.manual_seed(1)
    w = torch.randn(32, 64)
    w[0, 0] = 40.0  # one outlier wrecks the min-max range of its group
    wq = hqq(w, bits=3, group_size=64)
    # compare against plain asymmetric min-max at same grouping
    g = w.reshape(-1, 64)
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    s = ((hi - lo) / 7).clamp(min=1e-8)
    naive = (((g - lo) / s).round().clamp(0, 7) * s + lo).reshape(32, 64)
    assert (wq - w).abs().median() < (naive - w).abs().median()


def test_prune_24_pattern_and_optimality():
    torch.manual_seed(2)
    w = torch.randn(8, 16)
    wp, mask = prune_24(w)
    assert mask.reshape(8, 4, 4).sum(-1).eq(2).all()  # exactly 2 of 4
    assert (wp[~mask] == 0).all()
    # kept weights are the largest per group
    groups = w.abs().reshape(8, 4, 4)
    kept = groups[mask.reshape(8, 4, 4)].reshape(8, 4, 2)
    dropped = groups[~mask.reshape(8, 4, 4)].reshape(8, 4, 2)
    assert (kept.min(-1).values >= dropped.max(-1).values).all()


def test_magnitude_prune_hits_sparsity():
    w = torch.randn(64, 64)
    wp, mask = magnitude_prune(w, 0.75)
    assert mask.float().mean() == pytest.approx(0.25, abs=0.01)
    assert wp.abs()[mask].min() >= w.abs()[~mask].max()


def test_svd_matches_eckart_young_and_curve_decreases():
    torch.manual_seed(3)
    w = torch.randn(32, 48)
    a, b = svd_factorize(w, 8)
    assert a.shape == (32, 8) and b.shape == (8, 48)
    s = torch.linalg.svdvals(w)
    optimal = float(s[8:].pow(2).sum().sqrt())
    achieved = float(torch.linalg.norm(w - a @ b))
    assert achieved == pytest.approx(optimal, rel=1e-4)

    curve = rank_error_curve(w, [1, 4, 16, 32])
    assert curve == sorted(curve, reverse=True)
    assert curve[-1] < 0.01

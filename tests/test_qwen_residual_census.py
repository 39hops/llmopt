"""Fixtures for the RESIDUAL-STRUCTURE-0 census math (committed
before the producer's first real run; qualification-ladder rule).

The w4 payload built here follows the frozen layout in
llmopt.lab.qcodec's docstring: exps u8[nb] ++ codebook fp16[256,4]
++ idxs u8[n/4], value = codebook[idx] * 2^(exp-127).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rc():
    spec = importlib.util.spec_from_file_location(
        "qwen_residual_census", ROOT / "scratch" / "qwen_residual_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_residual_census"] = mod
    spec.loader.exec_module(mod)
    return mod


def _payload(exps, cb, idx):
    return (np.asarray(exps, np.uint8).tobytes()
            + np.asarray(cb, np.float16).tobytes()
            + np.asarray(idx, np.uint8).tobytes())


def test_cond_mean_stats_exact(rc):
    """One block of 128 (shape (1,128)), scale 1, all groups on code 0
    with codebook row zero. Vendor = constant 0.5 per element, so the
    conditional mean removes ALL variance: vr == 1."""
    shape = (1, 128)
    cb = np.zeros((256, 4), np.float16)
    idx = np.zeros(32, np.uint8)
    buf = _payload([127], cb, idx)
    Wv = np.full(shape, 0.5, np.float32)
    vr, delta, counts = rc.cond_mean_stats(Wv, buf, shape)
    assert vr == pytest.approx(1.0)
    assert counts[0] == 32 and counts[1:].sum() == 0
    assert np.allclose(delta[0], 0.5)


def test_cond_mean_stats_no_structure(rc):
    """Residual with zero mean per code slot: conditional mean buys
    nothing (vr ~ 0)."""
    shape = (1, 128)
    cb = np.zeros((256, 4), np.float16)
    idx = np.zeros(32, np.uint8)
    buf = _payload([127], cb, idx)
    # +/-0.5 alternating within the SAME code slot: the per-code
    # conditional mean is 0 and removes nothing
    Wv2 = np.zeros(shape, np.float32)
    Wv2[0, 0::8] = 0.5   # slot 0 of even groups
    Wv2[0, 4::8] = -0.5  # slot 0 of odd groups
    vr2, delta2, _ = rc.cond_mean_stats(Wv2, buf, shape)
    assert vr2 == pytest.approx(0.0, abs=1e-6)
    assert delta2[0][0] == pytest.approx(0.0, abs=1e-7)


def test_cond_mean_respects_block_scale(rc):
    """Two blocks with different exponents: normalization must divide
    by each block's scale before pooling codes."""
    shape = (2, 128)
    cb = np.zeros((256, 4), np.float16)
    idx = np.zeros(64, np.uint8)
    buf = _payload([127, 128], cb, idx)   # scales 1 and 2
    Wv = np.concatenate([np.full(128, 0.5), np.full(128, 1.0)]) \
        .astype(np.float32).reshape(shape)
    vr, delta, counts = rc.cond_mean_stats(Wv, buf, shape)
    # normalized vendor is 0.5 everywhere in both blocks -> fully
    # explained by the conditional mean
    assert vr == pytest.approx(1.0)
    assert np.allclose(delta[0], 0.5)


def test_tail_energy(rc):
    R = np.zeros(1000, np.float32)
    R[0] = 10.0
    assert rc.tail_energy(R.reshape(10, 100)) == pytest.approx(1.0)
    assert rc.tail_energy(np.ones((10, 100), np.float32)) \
        == pytest.approx(0.01)


def test_svd_fracs_rank1(rc):
    u = np.arange(1, 65, dtype=np.float32)[:, None]
    v = np.ones((1, 300), np.float32)
    f = rc.svd_fracs(u @ v)
    assert f["top16"] == pytest.approx(1.0)

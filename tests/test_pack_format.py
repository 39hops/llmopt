"""Packed-crystal format: exact container roundtrip, entropy at
Gaussian capacity, rANS roundtrip (skips without constriction)."""
import math

import pytest

torch = pytest.importorskip("torch")

from llmopt.quantize.meter import meter, meter_group
from llmopt.quantize.pack import (load_state_dict, pack_state_dict,
                                  pack_tensor, rans_size, unpack_tensor)


def test_pack_roundtrip_exact(tmp_path):
    torch.manual_seed(0)
    sd = {"blocks.0.up.weight": torch.randn(48, 32) * 0.2,
          "blocks.0.n1.g": torch.ones(32)}
    path = tmp_path / "c.npz"
    nparam, raw_bits, ent_bits, nbytes = pack_state_dict(sd, path)
    back = load_state_dict(path)
    # packed tensor: exact on its own grid (pack -> unpack -> repack
    # is a fixed point)
    w = back["blocks.0.up.weight"]
    packed, q, minc, bits, shape, ent = pack_tensor(w)
    w2 = unpack_tensor(packed, q, minc, bits, shape)
    assert torch.equal(w, w2)
    # fp passthrough is exact
    assert torch.equal(back["blocks.0.n1.g"], sd["blocks.0.n1.g"])
    assert nparam == 48 * 32
    # sigma/2 grid stays within half a step everywhere
    assert float((back["blocks.0.up.weight"]
                  - sd["blocks.0.up.weight"]).abs().max()) <= 0.5 / q + 1e-6


def test_gaussian_entropy_near_capacity():
    torch.manual_seed(1)
    w = torch.randn(512, 256)  # iid Gaussian: at capacity by def
    _, q, _, _, _, ent = pack_tensor(w)
    cap = (0.5 * math.log2(2 * math.pi * math.e)
           - math.log2((1.0 / q) / float(w.std())))
    assert abs(ent / w.numel() - cap) / cap < 0.02


def test_meter_separates_regimes():
    torch.manual_seed(2)
    gauss = torch.randn(256, 128)
    m_g, k_g = meter(gauss)
    heavy = gauss.clone()
    heavy[torch.rand_like(heavy) > 0.99] *= 30  # planted outliers
    m_h, k_h = meter(heavy)
    assert m_g < 2.5 < m_h
    assert k_g < 4 < k_h
    m, k, n = meter_group([gauss, gauss])
    assert n == 2 * gauss.numel() and abs(m - m_g) < 1e-6


def test_rans_roundtrip_and_bound():
    constriction = pytest.importorskip("constriction")  # noqa: F841
    import numpy as np
    codes = np.random.default_rng(3).normal(0, 6, 200_000).round()
    nbytes, ent = rans_size(codes, verify=True)
    assert 8 * nbytes / codes.size < ent * 1.05 + 0.2

"""qcuda_tower: the fail-closed dispatch invariant + residency plan.

Born from OBSERVATION QWEN-BLE-FREEGEN-1-ABORT: a compressed 2D
layer tensor silently fell through to dense nn.Linear and 0.87 GiB
of s16 payload ran as 6.875 GiB of FP32. The routing/planning logic
is pure and tested everywhere; kernel parity gates run only where
triton + CUDA exist (the 3080 qualification driver).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llmopt.lab import qcuda_tower as qt  # noqa: E402
from llmopt.lab.qcodec import expected_len  # noqa: E402


def test_route_codec_total_and_fail_closed():
    assert qt.route_codec("w4") == "fused_w4"
    assert qt.route_codec("s16") == "fused_s16"
    assert qt.route_codec("raw") == "dense_raw"
    with pytest.raises(ValueError, match="REFUSING: no route"):
        qt.route_codec("excluded")
    with pytest.raises(ValueError, match="REFUSING: no route"):
        qt.route_codec("w2_future")


def test_runtime_bytes_compressed_vs_dense():
    e_w4 = {"codec": "w4", "shape": [256, 512]}
    e_s16 = {"codec": "s16", "shape": [256, 512]}
    e_raw = {"codec": "raw", "shape": [256, 512]}
    assert qt.runtime_bytes(e_w4) == expected_len("w4", [256, 512])
    assert qt.runtime_bytes(e_s16) == expected_len("s16", [256, 512])
    assert qt.runtime_bytes(e_raw) == 256 * 512 * 4
    # the abort's arithmetic class: s16 payload stays payload-sized,
    # never n*4 dense
    assert qt.runtime_bytes(e_s16) < qt.runtime_bytes(e_raw) / 5


def test_plan_residency_fits_and_refuses():
    entries = [{"codec": "s16", "shape": [256, 512]},
               {"codec": "w4", "shape": [256, 512]},
               {"codec": "excluded", "shape": [999999, 999999]}]
    want = (expected_len("s16", [256, 512])
            + expected_len("w4", [256, 512]))
    plan = qt.plan_residency(entries, free_bytes=10 * want,
                             reserve_frac=0.15)
    assert plan["total_bytes"] == want and plan["fits"]
    assert set(plan["per_route"]) == {"fused_s16", "fused_w4"}
    with pytest.raises(MemoryError, match="REFUSING: planned residency"):
        qt.plan_residency(entries, free_bytes=want)  # reserve eats it


def _toy_model():
    m = torch.nn.Sequential()
    m.add_module("a", torch.nn.Linear(4, 4, bias=False))
    m.add_module("b", torch.nn.Linear(4, 4, bias=False))
    return m


def test_assert_no_fallthrough_raises_on_compressed_dense():
    man = {"a.weight": {"codec": "s16", "shape": [4, 4]},
           "b.weight": {"codec": "raw", "shape": [4, 4]}}
    with pytest.raises(RuntimeError, match="fell through to dense"):
        qt.assert_no_fallthrough(_toy_model(), man,
                                 name_fn=lambda p: p + ".weight")


def test_assert_no_fallthrough_passes_raw_and_unlisted():
    man = {"b.weight": {"codec": "raw", "shape": [4, 4]}}
    assert qt.assert_no_fallthrough(_toy_model(), man,
                                    name_fn=lambda p: p + ".weight") == 0


def test_fused_s16_linear_shape_contract_no_cuda():
    """Constructor path needs CUDA; the class contract (features from
    payload shape) is asserted through the module type only."""
    assert issubclass(qt.FusedS16Linear, torch.nn.Module)
    assert qt.FusedS16Linear.CHUNK == qt.FusedW4Linear.CHUNK == 8192


@pytest.mark.skipif(not qt.HAVE_TRITON or not torch.cuda.is_available(),
                    reason="triton/CUDA parity gates run on the 3080")
def test_s16_decode_rows_parity_cuda():
    import numpy as np
    from llmopt.lab.qcodec import dec_s16
    rng = np.random.default_rng(0)
    R, C = 8, 256
    n = R * C
    nb = n // 128
    exps = rng.integers(120, 135, nb, dtype=np.uint8)
    lv = (rng.standard_normal(16) * 0.1).astype(np.float16)
    codes = rng.integers(0, 256, n // 2, dtype=np.uint8)
    buf = exps.tobytes() + lv.tobytes() + codes.tobytes()
    ref = dec_s16(buf, [R, C])
    pay = qt.S16Gpu(buf, [R, C])
    got = qt.s16_decode_rows(pay, 2, 7).cpu().numpy()
    assert np.array_equal(got, ref[2:7])

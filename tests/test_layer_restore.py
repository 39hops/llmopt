"""llmopt.lab.layer_restore: the pure-numpy helpers of a layer-restoration
study (vendor index layout, mlx-lm load conventions for vendor tensors,
boundary activation statistics). Reference values are hand-computed."""
import numpy as np
import pytest

from llmopt.lab import layer_restore as lr

P = "model.language_model.layers."


def _weight_map():
    return {
        P + "0.a.weight": "s1", P + "0.b.weight": "s1",
        P + "1.a.weight": "s1", P + "1.b.weight": "s2",
        P + "2.a.weight": "s2", P + "2.b.weight": "s2",
        "model.language_model.norm.weight": "s2",
        "lm_head.weight": "s3",
    }


def test_layer_tensor_map_names_only_that_layer():
    m = lr.layer_tensor_map(_weight_map(), 1)
    assert m == {P + "1.a.weight": "s1", P + "1.b.weight": "s2"}
    # layer 10 must not match layer 1 by prefix
    wm = {P + "10.a.weight": "s9", P + "1.a.weight": "s1"}
    assert lr.layer_tensor_map(wm, 1) == {P + "1.a.weight": "s1"}
    with pytest.raises(KeyError):
        lr.layer_tensor_map(_weight_map(), 7)


def test_shard_coverage_separates_complete_from_straddling_layers():
    cov = lr.shard_coverage(_weight_map())
    assert cov["s1"] == {"complete": [0], "partial": [1]}
    assert cov["s2"] == {"complete": [2], "partial": [1]}
    assert "s3" not in cov  # no layer tensors there


def test_sanitize_vendor_tensor_follows_mlx_lm_conventions():
    # conv1d [D, 1, K] -> [D, K, 1]
    conv = np.arange(2 * 1 * 4, dtype=np.float32).reshape(2, 1, 4)
    out = lr.sanitize_vendor_tensor(P + "5.linear_attn.conv1d.weight", conv)
    assert out.shape == (2, 4, 1)
    assert np.array_equal(out[:, :, 0], conv[:, 0, :])
    # the five zero-centred norm families get +1
    w = np.array([0.5, -0.25], dtype=np.float32)
    for sfx in (".input_layernorm.weight", ".post_attention_layernorm.weight",
                ".self_attn.q_norm.weight", ".self_attn.k_norm.weight"):
        assert np.array_equal(lr.sanitize_vendor_tensor(P + "5" + sfx, w), w + 1.0)
    assert np.array_equal(lr.sanitize_vendor_tensor("model.language_model.norm.weight", w), w + 1.0)
    # the gated DeltaNet norm is NOT shifted, nor are projections
    assert np.array_equal(lr.sanitize_vendor_tensor(P + "5.linear_attn.norm.weight", w), w)
    mat = np.ones((3, 2), dtype=np.float32)
    assert np.array_equal(lr.sanitize_vendor_tensor(P + "5.mlp.up_proj.weight", mat), mat)


def test_mlx_param_path_for_vendor_name():
    assert lr.mlx_param_path(P + "22.linear_attn.in_proj_qkv.weight") == \
        "language_model.model.layers.22.linear_attn.in_proj_qkv.weight"
    assert lr.mlx_param_path("model.language_model.norm.weight") == "language_model.model.norm.weight"
    assert lr.mlx_param_path("lm_head.weight") == "language_model.lm_head.weight"


def test_boundary_stats_hand_values():
    a = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0], [0.0, 1.0]], dtype=np.float32)
    s = lr.boundary_stats(a, b)
    # global relative L2: ||a-b|| / ||a|| = 1 / sqrt(25 + 4)
    assert s["rel_l2"] == pytest.approx(1.0 / np.sqrt(29.0))
    # per-row relative L2: row 0 exact, row 1 = 1/2
    assert s["row_rel_l2_median"] == pytest.approx(0.25)
    assert s["row_rel_l2_max"] == pytest.approx(0.5)
    # max abs over the RMS of a's rows: 1 / rms(row 1) = 1 / sqrt(2)
    assert s["max_abs_over_row_rms"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert s["cosine_min"] == pytest.approx(1.0)
    assert s["n_rows"] == 2
    with pytest.raises(ValueError):
        lr.boundary_stats(a, b[:1])


def test_rel_frobenius():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert lr.rel_frobenius(a, a) == 0.0
    assert lr.rel_frobenius(a, a * 1.1) == pytest.approx(0.1)


def test_causal_additive_mask():
    m = lr.causal_additive_mask(3)
    assert m.shape == (3, 3) and m.dtype == np.float32
    assert np.array_equal(np.isneginf(m), np.triu(np.ones((3, 3), bool), 1))
    assert np.all(m[np.tril(np.ones((3, 3), bool))] == 0.0)

"""llmopt.lab.logit_stats: the per-position and paired statistics of a
head-substitution study (teacher-forced logits, two candidate heads).
Reference values are hand-computed on tiny vocabularies."""
import math

import numpy as np
import pytest

from llmopt.lab import logit_stats as ls


def test_position_stats_rank_margin_kl_and_teacher_forced_ce():
    zt = np.array([2.0, 1.0, 0.0, -1.0], np.float32)       # teacher: argmax 0
    za = np.array([1.0, 2.0, 0.5, -3.0], np.float32)       # candidate: argmax 1
    s = ls.position_stats(za, zt, teacher_token=0, top_ids=np.array([0, 1]))
    assert s["argmax"] == 1
    assert s["margin"] == pytest.approx(1.0)
    assert s["teacher_rank"] == 1                           # one logit strictly above z_a[0]
    lse = math.log(sum(math.exp(x) for x in za.tolist()))
    assert s["lse"] == pytest.approx(lse, abs=1e-6)
    assert s["mean_logit"] == pytest.approx(0.125)
    assert s["z_teacher_token"] == pytest.approx(1.0)
    assert s["ce_teacher_token"] == pytest.approx(lse - 1.0, abs=1e-6)
    pt = np.exp(zt.astype(np.float64)); pt /= pt.sum()
    pa = np.exp(za.astype(np.float64)); pa /= pa.sum()
    assert s["kl_teacher_cand"] == pytest.approx(float((pt * (np.log(pt) - np.log(pa))).sum()), abs=1e-9)
    assert list(s["z_top_ids"]) == pytest.approx([1.0, 2.0])


def test_rank_is_zero_when_the_teacher_token_is_the_candidate_argmax():
    za = np.array([3.0, 1.0, 3.0 - 1e-7], np.float32)
    s = ls.position_stats(za, za, teacher_token=0, top_ids=np.array([0]))
    assert s["teacher_rank"] == 0 and s["argmax"] == 0


def test_slope_top_matches_the_census_formula():
    rng = np.random.default_rng(0)
    b = rng.normal(size=16)
    a = 0.7 * b + 2.0 + rng.normal(scale=1e-3, size=16)
    assert ls.slope_on_teacher(a, b) == pytest.approx(0.7, abs=5e-3)
    assert math.isnan(ls.slope_on_teacher(a, np.full(16, 1.0)))


def test_paired_counts():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 1, 0, 1], bool)
    c = ls.paired_counts(a, b)
    assert c == {"n": 5, "a_right": 3, "b_right": 3, "corrected": 1, "broken": 1, "both_right": 2, "both_wrong": 1,
                 "net": 0, "delta_points": 0.0}


def test_block_deltas_and_sign_test():
    a = np.array([1, 0, 1, 0, 1, 1, 0, 0], bool)
    b = np.array([1, 1, 1, 0, 1, 1, 1, 0], bool)
    blocks = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    d = ls.block_deltas(a, b, blocks)
    assert d["per_block_net"] == [1, 0, 0, 1]
    assert d["blocks_up"] == 2 and d["blocks_down"] == 0 and d["blocks_tied"] == 2
    assert d["sign_test_p_two_sided"] == pytest.approx(0.5)      # two non-tied blocks, both up: 2 * 0.5^2
    assert ls._binom_two_sided(4, 5) == pytest.approx(2 * 6 / 32)  # k above n/2 reads the symmetric tail
    assert ls._binom_two_sided(1, 5) == pytest.approx(2 * 6 / 32)
    boot = ls.block_bootstrap_delta_points(a, b, blocks, n_boot=2000, seed=1)
    assert boot["delta_points"] == pytest.approx(25.0)
    assert boot["ci95_low"] <= 25.0 <= boot["ci95_high"]
    assert boot["n_boot"] == 2000


def test_bf16_round_trip_and_rounding():
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9, -2.5, 65536.0], np.float32)
    u = ls.f32_to_bf16_bits(x)
    y = ls.bf16_bits_to_f32(u)
    # 1 + 2^-8 is exactly halfway between bf16 neighbours 1 and 1 + 2^-7: ties to even -> 1.0
    assert y.tolist() == [1.0, 1.0, 1.0 + 2 ** -7, -2.5, 65536.0]
    assert ls.round_to_bf16(x).tolist() == y.tolist()
    assert u.dtype == np.uint16 and y.dtype == np.float32
    special = np.array([np.nan, np.inf, -np.inf, np.finfo(np.float32).max], np.float32)   # the float32 max rounds up past the bf16 max
    r = ls.round_to_bf16(special)
    assert math.isnan(r[0]) and r[1] == np.inf and r[2] == -np.inf and r[3] == np.inf
    big = np.arange(1 << 20, dtype=np.float32)
    assert np.array_equal(ls.f32_to_bf16_bits(big, chunk=1000), ls.f32_to_bf16_bits(big))


def test_safetensors_header_parse_and_tensor_bytes(tmp_path):
    import json
    import struct
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw = ls.f32_to_bf16_bits(w).tobytes()
    header = {"lm_head.weight": {"dtype": "BF16", "shape": [3, 4], "data_offsets": [0, len(raw)]},
              "__metadata__": {"format": "pt"}}
    hb = json.dumps(header).encode()
    p = tmp_path / "s.safetensors"
    p.write_bytes(struct.pack("<Q", len(hb)) + hb + raw)
    h, start = ls.safetensors_header(p)
    assert start == 8 + len(hb) and h["lm_head.weight"]["shape"] == [3, 4]
    t = ls.read_bf16_tensor(p, "lm_head.weight")
    assert t.shape == (3, 4) and t.dtype == np.float32 and t.tolist() == w.tolist()
    bits = ls.read_bf16_bits(p, "lm_head.weight")
    assert bits.dtype == np.uint16 and bits.tobytes() == raw
    with pytest.raises(ValueError):
        ls.read_bf16_tensor(p, "lm_head.weight", expect_shape=(4, 3))


def test_s16_row_slice_payload_decodes_like_the_whole():
    from llmopt.lab import qcodec
    R, C = 8, 256
    rng = np.random.default_rng(3)
    nb = R * C // 128
    exps = rng.integers(120, 130, nb, dtype=np.uint8).tobytes()
    lv = np.linspace(-1, 1, 16, dtype=np.float16).tobytes()
    codes = rng.integers(0, 256, R * C // 2, dtype=np.uint8).tobytes()
    buf = exps + lv + codes
    whole = qcodec.dec_s16(buf, (R, C))
    sub = ls.s16_row_slice_payload(buf, (R, C), 3, 6)
    part = qcodec.dec_s16(sub, (3, C))
    assert np.array_equal(part, whole[3:6])

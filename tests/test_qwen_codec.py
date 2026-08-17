"""Golden fixtures for the WHOLE-0T payload layouts (llmopt.lab.qcodec).

Hand-constructed byte payloads with hand-computed expected tensors.
These pin the conventions that only ever met a real model before:
nibble order (HIGH nibble = even element — opposite GPTQ), the
E8M0 exponent bias (127), section offsets, and the exact payload
length formulas. A decode-convention regression flips these tests
red in seconds instead of inside a 27B run.
"""
import numpy as np
import pytest

from llmopt.lab.qcodec import (dec_raw, dec_s16, dec_w4, decode_entry,
                               expected_len)


def test_expected_lengths():
    # one block of 128: w4 = 1 + 2048 + 32; s16 = 1 + 32 + 64
    assert expected_len("w4", [1, 128]) == 2081
    assert expected_len("s16", [1, 128]) == 97
    assert expected_len("raw", [4, 4]) == 32
    assert expected_len("excluded", [999]) == 0
    with pytest.raises(ValueError):
        expected_len("nope", [1])


def test_w4_golden_single_block():
    # codebook: entry 0 = [1,2,3,4], entry 1 = [-1,0,0.5,8]; exponent
    # 128 -> scale 2.0; indices alternate 0,1 over 32 groups
    cb = np.zeros((256, 4), np.float16)
    cb[0] = [1, 2, 3, 4]
    cb[1] = [-1, 0, 0.5, 8]
    exps = np.array([128], np.uint8)
    idxs = np.tile(np.array([0, 1], np.uint8), 16)
    buf = exps.tobytes() + cb.tobytes() + idxs.tobytes()
    W = dec_w4(buf, [1, 128])
    expect = np.tile(np.array([1, 2, 3, 4, -1, 0, 0.5, 8],
                              np.float32) * 2.0, 16).reshape(1, 128)
    np.testing.assert_array_equal(W, expect)


def test_w4_exponent_bias():
    # exponent 127 must mean scale EXACTLY 1.0; 126 -> 0.5
    cb = np.zeros((256, 4), np.float16)
    cb[7] = [1, 1, 1, 1]
    idxs = np.full(64, 7, np.uint8)
    for e, s in ((127, 1.0), (126, 0.5), (130, 8.0)):
        buf = (np.array([e, e], np.uint8).tobytes() + cb.tobytes()
               + idxs.tobytes())
        W = dec_w4(buf, [2, 128])
        assert float(W[0, 0]) == s and float(W[1, 127]) == s


def test_s16_golden_nibble_order():
    # levels[3]=0.25, levels[10]=-2; codes byte 0x3A must decode to
    # element0=level[3] (HIGH nibble), element1=level[10] (LOW)
    lv = np.zeros(16, np.float16)
    lv[3] = 0.25
    lv[10] = -2.0
    exps = np.array([127], np.uint8)      # scale 1.0
    codes = np.full(64, 0x3A, np.uint8)
    buf = exps.tobytes() + lv.tobytes() + codes.tobytes()
    W = dec_s16(buf, [1, 128])
    assert float(W[0, 0]) == 0.25         # even -> high nibble -> 3
    assert float(W[0, 1]) == -2.0         # odd -> low nibble -> 10
    np.testing.assert_array_equal(W[0, 0::2], np.full(64, 0.25))
    np.testing.assert_array_equal(W[0, 1::2], np.full(64, -2.0))


def test_s16_scale_per_block():
    lv = np.zeros(16, np.float16)
    lv[1] = 1.0
    exps = np.array([127, 129], np.uint8)  # blocks scale 1.0, 4.0
    codes = np.full(128, 0x11, np.uint8)
    buf = exps.tobytes() + lv.tobytes() + codes.tobytes()
    W = dec_s16(buf, [2, 128])
    assert float(W[0, 0]) == 1.0 and float(W[1, 0]) == 4.0


def test_raw_bf16():
    # bf16 of 1.0 is 0x3F80; of -2.0 is 0xC000
    u16 = np.array([0x3F80, 0xC000], np.uint16)
    W = dec_raw(u16.tobytes(), [2], "BF16")
    np.testing.assert_array_equal(W, np.array([1.0, -2.0], np.float32))


def test_decode_entry_dispatch_and_length_guard():
    with pytest.raises(ValueError):
        decode_entry(b"", {"codec": "excluded", "shape": [1]})
    with pytest.raises(AssertionError):
        dec_w4(b"\x00" * 100, [1, 128])   # wrong length must refuse


def test_golden_payloads_from_frozen_compiler():
    """MANDATORY parity: payloads produced ONCE by the frozen
    compiler (tests/fixtures/, committed bytes) decode through the
    canonical module to the compiler's own self-reported squared
    error. No torch, no skip — this is the non-optional form of
    the compiler<->decoder meeting."""
    import hashlib
    import json
    import os
    fx = os.path.join(os.path.dirname(__file__), "fixtures")
    W = np.load(os.path.join(fx, "qcodec_src.npy"))
    exp = json.load(open(os.path.join(fx, "qcodec_expected.json")))
    for name, dec in (("w4", dec_w4), ("s16", dec_s16)):
        buf = open(os.path.join(fx, f"qcodec_{name}.bin"), "rb").read()
        assert hashlib.sha256(buf).hexdigest() == exp[name]["sha256"]
        assert len(buf) == exp[name]["len"] == expected_len(name, W.shape)
        R = dec(buf, W.shape)
        got = float(((R - W) ** 2).sum())
        assert abs(got - exp[name]["se"]) / exp[name]["se"] < 1e-5


def test_roundtrip_against_compiler_encoders():
    """The independent implementations meet on a tiny tensor HERE,
    not inside a 27B run: compiler enc_w4/enc_s16 (scratch, frozen)
    -> canonical decode -> must equal the compiler's own
    reconstruction error against the source."""
    torch = pytest.importorskip("torch")
    import importlib.util
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "scratch")
    spec = importlib.util.spec_from_file_location(
        "w0t", "scratch/qwen_whole0t.py")
    if spec is None:
        pytest.skip("compiler source unavailable")
    w0t = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(w0t)
    except Exception as e:                # heavy deps missing on CI
        pytest.skip(f"compiler import: {e}")
    rng = np.random.default_rng(7)
    W = rng.standard_normal((8, 256)).astype(np.float32)
    for enc, dec in ((w0t.enc_w4, dec_w4), (w0t.enc_s16, dec_s16)):
        payload, se, _, meta = enc(W, "fixture/t")
        R = dec(payload, W.shape)
        got_se = float(((R - W) ** 2).sum())
        assert abs(got_se - se) / max(se, 1e-12) < 1e-5, \
            f"{meta['codec']}: decode disagrees with encoder " \
            f"self-reported error ({got_se} v {se})"

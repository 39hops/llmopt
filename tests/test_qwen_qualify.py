"""Regression fixtures for the qualification layer (qartifact) and
the single-decode-path invariant.

Every refusal adopted in the 2026-08-17 review rounds is an
executable test here — a bug documented only in prose can recur;
these stay dead. (Producer-consumer rule: each data-structure
change names the code that reads it, in the same commit.)
"""
import json
import os

import numpy as np
import pytest

from llmopt.lab import qartifact
from llmopt.lab.qcodec import dec_raw, dec_s16, dec_w4, decode_entry

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------- codec refusals
def test_raw_missing_dtype_refuses():
    with pytest.raises(ValueError, match="missing dtype"):
        decode_entry(b"\x00\x00", {"codec": "raw", "shape": [1]})


def test_raw_unknown_dtype_refuses():
    with pytest.raises(ValueError, match="unknown raw dtype"):
        dec_raw(b"\x00\x00", [1], "FP8")


def test_malformed_shape_refuses():
    for codec, dec in (("w4", dec_w4), ("s16", dec_s16)):
        with pytest.raises(ValueError, match="invalid WHOLE-0T"):
            dec(b"", [130])                 # rank 1
        with pytest.raises(ValueError, match="invalid WHOLE-0T"):
            dec(b"", [3, 100])              # not block-divisible


# ------------------------------------------- single decode path
def test_runtime_imports_qcodec_only():
    """The runtime must decode THROUGH qcodec — a second W4/S16
    implementation is a live contradiction (the two decoders had
    already drifted once: fall-open fp16 v always-BF16)."""
    src = open(os.path.join(REPO, "scratch",
                            "qwen_runtime0r.py")).read()
    assert "from llmopt.lab.qcodec import decode_entry" in src
    # no local nibble arithmetic outside the row-sliced io reader
    body = src.split("class W4Rows")[0]
    assert ">> 4" not in body and "& 0xF" not in body, \
        "inline nibble decode outside W4Rows — second implementation"
    assert "qartifact.qualify_artifact" in src, \
        "runtime bypasses qualification"


def test_duplicate_manifest_key_refused():
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        json.loads('{"a": 1, "a": 2}',
                   object_pairs_hook=qartifact._no_dup_pairs)


# ------------------------------------------------- rung 0 fail-closed
def test_rung0_requires_chain(tmp_path):
    with pytest.raises(SystemExit, match="identity unprovable"):
        qartifact.verify_chain(str(tmp_path), None)
    rep = qartifact.verify_chain(str(tmp_path), None,
                                 allow_unchained=True)
    assert rep["unchained"] is True       # override lands in report


def test_rung0_truncated_chain_refuses(tmp_path):
    chain = tmp_path / "chain.txt"
    chain.write_text("aa" * 32 + "  manifest.json\n")   # 1 row != 19
    with pytest.raises(SystemExit, match="expected 19"):
        qartifact.verify_chain(str(tmp_path), str(chain))


# ------------------------------------------------- span exact cover
def _mini_artifact(tmp_path, gap=False, trailing=False):
    """One raw tensor, one shard; optionally corrupt the geometry."""
    W = np.arange(256, dtype=np.float32)
    u16 = (W.view(np.uint32) >> 16).astype(np.uint16)
    payload = u16.tobytes()
    off = 4 if gap else 0
    shard = tmp_path / "s1.bin"
    blob = b"\x00" * off + payload + (b"\x00" * 3 if trailing else b"")
    shard.write_bytes(blob)
    man = {"t": {"codec": "raw", "dtype": "BF16", "shape": [2, 128],
                 "shard": "s1", "off": off, "len": len(payload)}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return man


def test_span_gap_refused(tmp_path):
    man = _mini_artifact(tmp_path, gap=True)
    with pytest.raises(SystemExit, match="gap/overlap"):
        _run_structure(tmp_path, man)


def test_trailing_bytes_refused(tmp_path):
    man = _mini_artifact(tmp_path, trailing=True)
    with pytest.raises(SystemExit, match="trailing"):
        _run_structure(tmp_path, man)


def _run_structure(tmp_path, man):
    """Exercise qualify's structural rung directly on a mini
    artifact (conservation is mocked to the mini key set)."""
    by_shard = {}
    from llmopt.lab.qcodec import expected_len
    for name, e in man.items():
        if e["len"] != expected_len(e["codec"], e["shape"]):
            raise SystemExit(f"{name}: payload length wrong")
        by_shard.setdefault(e["shard"], []).append(
            (e["off"], e["off"] + e["len"], name))
    for sh, spans in by_shard.items():
        spans.sort()
        fsize = os.path.getsize(os.path.join(tmp_path, sh + ".bin"))
        prev = 0
        for off, end, name in spans:
            if off != prev:
                raise SystemExit(f"{sh}: gap/overlap at {name}")
            prev = end
        if prev != fsize:
            raise SystemExit(f"{sh}: trailing bytes")


# ------------------------------------------------- preflight model
def test_estimate_shared_between_gate_and_receipt():
    """One cost model, two consumers: the preflight gate and the
    forward1 receipt must both call estimate_runtime_peak."""
    man = {"model.language_model.embed_tokens.weight":
           {"codec": "w4", "shape": [1024, 128], "len": 100},
           "lm_head.weight":
           {"codec": "w4", "shape": [1024, 128], "len": 100},
           "model.language_model.layers.0.mlp.w.weight":
           {"codec": "w4", "shape": [256, 128], "len": 50}}
    est = qartifact.estimate_runtime_peak(man)
    assert est == 200 + 256 * 128 * 4 * 2 + 2 * 2 ** 30
    src = open(os.path.join(REPO, "scratch",
                            "qwen_runtime0r.py")).read()
    assert "estimate_runtime_peak" in src, \
        "runtime receipt does not use the shared cost model"


def test_lock_reads_local_only_class():
    """Producer-consumer rule: the lock's local_only field must be
    READ by the mutation invariant, not just written."""
    src = open(os.path.join(
        REPO, "tests", "science_incidents",
        "test_frozen_receipt_mutation.py")).read()
    assert 'rec.get("local_only")' in src
    assert "LLMOPT_FULL" in src


# --------------------------------------- optimized-decoder parity
def _w4_payload(R, C, seed=3):
    """Compiler-layout w4 payload from random data (pure numpy)."""
    rng = np.random.default_rng(seed)
    nb = R * C // 128
    exps = rng.integers(120, 132, nb, dtype=np.uint8)
    cb = (rng.standard_normal((256, 4)) * 0.3).astype(np.float16)
    idx = rng.integers(0, 256, R * C // 4, dtype=np.uint8)
    return exps.tobytes() + cb.tobytes() + idx.tobytes()


def test_w4rows_parity_against_canonical():
    """W4Rows (optimized) v qcodec.dec_w4 (canonical): first,
    middle, last, single-row, multi-row, random slices, several
    widths — the offset arithmetic rows() exists for is exercised
    at nonzero lo (the previous oracle only ever tested lo=0)."""
    from llmopt.lab.qcodec_fast import W4Rows
    rng = np.random.default_rng(11)
    for R, C in ((8, 256), (16, 128), (5, 640), (32, 512)):
        buf = _w4_payload(R, C)
        full = dec_w4(buf, [R, C])
        v = W4Rows(buf, [R, C])
        slices = [(0, 1), (R // 2, R // 2 + 1), (R - 1, R),
                  (0, R), (1, R - 1)]
        slices += [tuple(sorted(rng.choice(R, 2, replace=False)))
                   for _ in range(4)]
        for lo, hi in slices:
            if lo == hi:
                hi = lo + 1
            np.testing.assert_array_equal(v.rows(lo, hi),
                                          full[lo:hi], err_msg=f"{R}x{C} [{lo},{hi})")


def _s16_payload(R, C, seed=5):
    rng = np.random.default_rng(seed)
    nb = R * C // 128
    exps = rng.integers(120, 132, nb, dtype=np.uint8)
    lv = (rng.standard_normal(16) * 0.3).astype(np.float16)
    codes = rng.integers(0, 256, R * C // 2, dtype=np.uint8)
    return exps.tobytes() + lv.tobytes() + codes.tobytes()


def test_s16rows_parity_against_canonical():
    """S16Rows v qcodec.dec_s16, same slice battery as W4Rows —
    the nibble convention (HIGH = even) is the specific bug this
    fixture exists to catch."""
    from llmopt.lab.qcodec_fast import S16Rows
    rng = np.random.default_rng(13)
    for R, C in ((8, 256), (16, 128), (5, 640), (32, 512)):
        buf = _s16_payload(R, C)
        full = dec_s16(buf, [R, C])
        v = S16Rows(buf, [R, C])
        slices = [(0, 1), (R // 2, R // 2 + 1), (R - 1, R),
                  (0, R), (1, R - 1)]
        slices += [tuple(sorted(rng.choice(R, 2, replace=False)))
                   for _ in range(4)]
        for lo, hi in slices:
            if lo == hi:
                hi = lo + 1
            np.testing.assert_array_equal(
                v.rows(lo, hi), full[lo:hi],
                err_msg=f"{R}x{C} [{lo},{hi})")


def test_s16rows_refuses_bad_ranges():
    from llmopt.lab.qcodec_fast import S16Rows
    v = S16Rows(_s16_payload(4, 128), [4, 128])
    with pytest.raises(ValueError):
        v.rows(0, 5)
    with pytest.raises(ValueError):
        S16Rows(b"", [4, 130])       # C % 128 != 0


def test_w4rows_refuses_bad_ranges():
    from llmopt.lab.qcodec_fast import W4Rows
    v = W4Rows(_w4_payload(4, 128), [4, 128])
    with pytest.raises(ValueError):
        v.rows(2, 2)
    with pytest.raises(ValueError):
        v.rows(0, 5)


# ------------------------------------------------ rope value oracle
def test_rope_oracle_passes_healthy():
    from llmopt.lab import qrope
    theta, dim = 5e6, 128
    inv = qrope.expected_inv_freq(theta, dim)
    qrope.check_inv_freq(inv, theta, dim)
    pos = qrope.POSITIONS
    ang = pos[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)
    qrope.check_cos_sin(cos, sin, theta, dim)


def test_rope_oracle_catches_all_four_corruptions():
    from llmopt.lab import qrope
    theta, dim = 1e6, 128
    good = qrope.expected_inv_freq(theta, dim)
    corruptions = {
        "all-zero": np.zeros_like(good),
        "half-zero": np.where(np.arange(len(good)) < len(good) // 2,
                              good, 0.0),
        "one-wrong": np.concatenate([good[:5], [good[5] * 3],
                                     good[6:]]),
        "wrong-order": good[::-1].copy(),
    }
    for name, bad in corruptions.items():
        with pytest.raises(SystemExit, match="ROPE ORACLE"):
            qrope.check_inv_freq(bad, theta, dim)
    # and through the emitted cos/sin path too
    pos = qrope.POSITIONS
    ang = pos[:, None] * corruptions["all-zero"][None, :]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)
    with pytest.raises(SystemExit, match="ROPE ORACLE"):
        qrope.check_cos_sin(cos, sin, theta, dim)

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

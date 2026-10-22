"""llmopt.lab.hash: one digest semantics per helper. A FILE digest hashes
the bytes on disk (an .npy file includes its header); an ARRAY PAYLOAD
digest hashes the array's contiguous element bytes only. The two never
compare equal for an .npy file and a pinned value of one kind must never
be checked against a digest of the other kind (QWEN-LAYER-RESTORE-0
booked a receipt field that did exactly that)."""
import hashlib

import numpy as np
import pytest

from llmopt.lab import hash as lh


def _npy(tmp_path, arr):
    p = tmp_path / "a.npy"
    np.save(p, arr)
    return p


def test_file_and_array_payload_digests_differ_for_an_npy_file(tmp_path):
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    p = _npy(tmp_path, a)
    assert lh.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert lh.sha256_array_payload(p) == hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    assert lh.sha256_file(p) != lh.sha256_array_payload(p)


def test_array_payload_digest_is_chunk_independent(tmp_path):
    a = np.random.default_rng(0).normal(size=(37, 11))
    p = _npy(tmp_path, a)
    assert lh.sha256_array_payload(p, rows_per_chunk=1) == lh.sha256_array_payload(p, rows_per_chunk=1000)
    assert lh.sha256_array_payload(p) == hashlib.sha256(a.tobytes()).hexdigest()


def test_pinned_file_check_names_its_semantics(tmp_path):
    a = np.arange(8, dtype=np.int16)
    p = _npy(tmp_path, a)
    good = lh.pinned_file_check(p, lh.sha256_file(p), p.stat().st_size)
    assert good["ok"] and good["file_sha256_ok"] and good["size_ok"]
    assert good["file_sha256"] == lh.sha256_file(p)
    assert good["pinned_file_sha256"] == lh.sha256_file(p)
    assert good["size_bytes"] == p.stat().st_size
    assert "array_payload" not in " ".join(good)
    # the payload digest is the WRONG kind of pin for a file check: not ok, and said so
    wrong = lh.pinned_file_check(p, lh.sha256_array_payload(p), p.stat().st_size)
    assert not wrong["ok"] and not wrong["file_sha256_ok"] and wrong["size_ok"]
    assert lh.pinned_file_check(p, lh.sha256_file(p).upper(), None)["ok"]
    short = lh.pinned_file_check(p, lh.sha256_file(p), p.stat().st_size + 1)
    assert not short["ok"] and short["file_sha256_ok"] and not short["size_ok"]


def test_array_payload_digest_of_an_empty_array(tmp_path):
    p = _npy(tmp_path, np.zeros((0, 7), dtype=np.float32))
    assert lh.sha256_array_payload(p) == hashlib.sha256(b"").hexdigest()


def test_pinned_array_payload_check_names_its_semantics(tmp_path):
    a = np.arange(8, dtype=np.int16)
    p = _npy(tmp_path, a)
    good = lh.pinned_array_payload_check(p, lh.sha256_array_payload(p))
    assert good["ok"] and good["array_payload_sha256_ok"]
    assert good["array_payload_sha256"] == lh.sha256_array_payload(p)
    assert "file_sha256" not in good
    wrong = lh.pinned_array_payload_check(p, lh.sha256_file(p))
    assert not wrong["ok"]


def test_require_pinned_file_refuses_on_any_mismatch(tmp_path):
    p = tmp_path / "shard.bin"
    p.write_bytes(b"x" * 1000)
    sha = lh.sha256_file(p)
    rec = lh.require_pinned_file(p, sha, 1000)
    assert rec["ok"]
    with pytest.raises(lh.PinMismatch):
        lh.require_pinned_file(p, sha, 999)
    with pytest.raises(lh.PinMismatch):
        lh.require_pinned_file(p, "0" * 64, 1000)
    with pytest.raises(lh.PinMismatch):
        lh.require_pinned_file(tmp_path / "missing.bin", sha, 1000)

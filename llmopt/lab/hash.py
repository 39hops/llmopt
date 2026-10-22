"""lab.hash — ONE digest semantics for the lab package (grok-seat
cross-check adoption, 2026-08-11: three helpers with three semantics
landed in one night — catalog 8MiB-chunk file sha, merge 1MiB-chunk
file sha, runfiles short git sha WITHOUT a cwd anchor, which reports
whatever repo the CALLER happens to be standing in).

Scope: canonical helpers for lab-owned modules and new code. Frozen
scripts and verbatim-adopted functions (lab/gate.py gate_eval's
inline weights sha) are NOT migrated — their digests are cited by
booked verdicts and stay character-identical to their sources.

Weights shas stay OUT of this module on purpose: gate_eval owns that
semantics (dtype-sensitive, never compares across precisions) and the
one implementation lives in the frozen hub + its verbatim twin.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

CHUNK = 8 << 20  # 8 MiB — sha of an 80 GB checkpoint tree must not buffer


def sha256_file(path: str | Path, chunk: int = CHUNK) -> str:
    """Streaming file sha256, hex. Chunk size never changes the digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha(short: bool = False) -> str:
    """HEAD of THIS repo (anchored to this file, never the caller's
    cwd), or "unknown" — provenance fields must not raise."""
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.run(
            cmd, cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---- pinned-input checks with ONE semantics per field (2026-10-22) ----
#
# Two digests exist for an array on disk and they are different numbers:
#   file digest           sha256 of the bytes of the file (an .npy file's
#                         header included) — the kind a vendor shard pin,
#                         a receipt lock entry or a "dumps[*].sha256" field
#                         carries;
#   array payload digest  sha256 of the array's contiguous element bytes
#                         only — the kind a teacher-record manifest pins.
# A pin of one kind checked against a digest of the other kind is a false
# mismatch (QWEN-LAYER-RESTORE-0 booked "dump_sha256_on_disk_ok: false"
# that way). Every field below says which kind it is, and a check takes
# the pin kind in its name.


class PinMismatch(ValueError):
    """A pinned input does not match on disk (missing, wrong size, or wrong digest)."""


def sha256_array_payload(path: str | Path, rows_per_chunk: int | None = None) -> str:
    """ARRAY PAYLOAD digest of an .npy file: sha256 of the array's element
    bytes in C order (np.ascontiguousarray), header excluded, streamed over
    the leading axis; an empty array digests the empty byte string. The
    chunking never changes the digest. Not comparable with sha256_file."""
    import numpy as np
    a = np.load(path, mmap_mode="r")
    h = hashlib.sha256()
    if a.ndim == 0:
        h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    if a.shape[0] == 0:
        return h.hexdigest()
    if rows_per_chunk is None:
        row_bytes = max(1, a[0].nbytes) if a.ndim > 1 else a.itemsize
        rows_per_chunk = max(1, (64 << 20) // row_bytes)
    for s in range(0, a.shape[0], rows_per_chunk):
        h.update(np.ascontiguousarray(a[s:s + rows_per_chunk]).tobytes())
    return h.hexdigest()


def pinned_file_check(path: str | Path, pinned_file_sha256: str, pinned_size_bytes: int | None = None) -> dict:
    """FILE-digest check of a pinned file. Every field names its kind:
    file_sha256 (measured), pinned_file_sha256, size_bytes (measured),
    pinned_size_bytes, file_sha256_ok, size_ok, ok (the conjunction; a
    missing file is ok False with exists False)."""
    p = Path(path)
    rec = {"path": str(p), "kind": "file", "exists": p.exists(),
           "pinned_file_sha256": pinned_file_sha256, "pinned_size_bytes": pinned_size_bytes,
           "size_bytes": None, "file_sha256": None, "size_ok": False, "file_sha256_ok": False, "ok": False}
    if not p.is_file():
        return rec
    rec["size_bytes"] = p.stat().st_size
    rec["size_ok"] = True if pinned_size_bytes is None else rec["size_bytes"] == int(pinned_size_bytes)
    rec["file_sha256"] = sha256_file(p)
    rec["file_sha256_ok"] = rec["file_sha256"] == str(pinned_file_sha256).lower()
    rec["ok"] = bool(rec["size_ok"] and rec["file_sha256_ok"])
    return rec


def pinned_array_payload_check(path: str | Path, pinned_array_payload_sha256: str) -> dict:
    """ARRAY-PAYLOAD-digest check of a pinned .npy array: array_payload_sha256
    (measured), pinned_array_payload_sha256, array_payload_sha256_ok, ok.
    No file digest appears here; use pinned_file_check for a file pin."""
    p = Path(path)
    rec = {"path": str(p), "kind": "array_payload", "exists": p.exists(),
           "pinned_array_payload_sha256": pinned_array_payload_sha256,
           "array_payload_sha256": None, "array_payload_sha256_ok": False, "ok": False}
    if not p.is_file():
        return rec
    rec["array_payload_sha256"] = sha256_array_payload(p)
    rec["array_payload_sha256_ok"] = rec["array_payload_sha256"] == str(pinned_array_payload_sha256).lower()
    rec["ok"] = bool(rec["array_payload_sha256_ok"])
    return rec


def require_pinned_file(path: str | Path, pinned_file_sha256: str, pinned_size_bytes: int | None = None) -> dict:
    """pinned_file_check that REFUSES (PinMismatch) unless ok: the fresh
    check a routine makes immediately before it reads pinned vendor bytes,
    whatever an earlier stage recorded. Returns the check record for the
    receipt."""
    rec = pinned_file_check(path, pinned_file_sha256, pinned_size_bytes)
    if not rec["ok"]:
        raise PinMismatch(f"pinned file check failed: {rec}")
    return rec

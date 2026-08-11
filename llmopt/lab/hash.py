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

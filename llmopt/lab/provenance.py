"""Start-state run provenance, captured at process ENTRY.

Extracted from the tower-ladder driver's START block (spec
2026-08-19-qcuda-tower-runtime item 4) after two incidents in one
day: the RESIDUAL census receipt recorded a completion-time HEAD
that postdated its launch producer, and the FREEGEN-2 screen driver
regressed to a commit+one-hash subset of the ladder's capture.
Scientific provenance keys off the START state; completion state is
recorded separately.

    from llmopt.lab.provenance import start_provenance
    START = start_provenance(["llmopt/lab/qcuda.py", "scratch/x.py"])
    ...
    receipt = {"start": START,
               "completion_commit": completion_commit(), ...}
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args],
                                   cwd=_root()).decode()


def artifact_identity(art_dir: str) -> dict:
    """Weight-artifact identity for a receipt: the RESOLVED absolute
    directory, sha256 of its manifest.json, and the shard census
    (count + total bytes). The manifest names every tensor's shard,
    offset, and codec, so its digest pins which weights a consumer
    following it can read; shard byte totals catch a swapped or
    truncated payload the manifest alone would miss. Raises on a
    missing manifest — an artifact without one has no identity to
    record."""
    d = os.path.abspath(os.path.expanduser(art_dir))
    man_p = os.path.join(d, "manifest.json")
    ident = {"art_dir_resolved": d,
             "manifest_sha256": hashlib.sha256(
                 open(man_p, "rb").read()).hexdigest()}
    shards = sorted(f for f in os.listdir(d)
                    if f.endswith((".bin", ".safetensors", ".pt")))
    ident["shards"] = {"n": len(shards),
                       "total_bytes": sum(
                           os.path.getsize(os.path.join(d, f))
                           for f in shards)}
    return ident


def start_provenance(critical_files, artifacts=None) -> dict:
    """Capture at process entry: short HEAD, the LITERAL
    git-status --porcelain text (never a bare dirty bool), the
    interpreter path, and sha256 of every named critical file
    (paths relative to the repo root). Missing files raise — a
    provenance capture that silently skips a file is worse than a
    crash at entry.

    artifacts: optional {label: art_dir} of weight artifacts the
    run will load; each is recorded via artifact_identity so the
    receipt distinguishes WHICH weights ran, not just which code
    (receipt-audit S2, 2026-08-20: ART_DIR-only provenance cannot
    tell checkpoints apart)."""
    shas = {}
    for rel in critical_files:
        p = os.path.join(_root(), rel)
        shas[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out = {"start_commit": _git("rev-parse", "--short",
                                "HEAD").strip(),
           "start_status_porcelain": _git("status", "--porcelain"),
           "interpreter": sys.executable,
           "file_sha256": shas}
    if artifacts:
        out["artifact_identity"] = {
            label: artifact_identity(d)
            for label, d in artifacts.items()}
    return out


def completion_commit() -> str:
    """Short HEAD at receipt-write time; recorded beside (never
    instead of) the start commit."""
    return _git("rev-parse", "--short", "HEAD").strip()

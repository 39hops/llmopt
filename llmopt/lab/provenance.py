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


def start_provenance(critical_files) -> dict:
    """Capture at process entry: short HEAD, the LITERAL
    git-status --porcelain text (never a bare dirty bool), the
    interpreter path, and sha256 of every named critical file
    (paths relative to the repo root). Missing files raise — a
    provenance capture that silently skips a file is worse than a
    crash at entry."""
    shas = {}
    for rel in critical_files:
        p = os.path.join(_root(), rel)
        shas[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return {"start_commit": _git("rev-parse", "--short",
                                 "HEAD").strip(),
            "start_status_porcelain": _git("status", "--porcelain"),
            "interpreter": sys.executable,
            "file_sha256": shas}


def completion_commit() -> str:
    """Short HEAD at receipt-write time; recorded beside (never
    instead of) the start commit."""
    return _git("rev-parse", "--short", "HEAD").strip()

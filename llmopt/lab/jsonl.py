"""lab.jsonl — one jsonl read/write semantics (grok-seat cross-check
adoption, 2026-08-11: 40+ hand-rolled open/loads loops across
scripts/ and scratch/, each with its own blank-line, encoding, and
partial-write behavior).

Scope: lab-owned modules and NEW code. Frozen scripts and
verdict-cited scratch drivers keep their loops (evidence record).

Semantics (the house rules, written down once):
- read: skip blank lines; errors="replace" so one garbled byte
  degrades one row, never kills a scan; a malformed row RAISES with
  its line number (silent row drops are the checkpoint
  selection-effect's cousin).
- write: atomic tmp+rename in the same directory (the runfiles
  marker rule — readers never see a half-written file).
- append: plain append, one row per call, flushed — the STREAMING
  shape for workers under an outer wall (killed workers must leave
  their rows on disk; bit three times).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def read_jsonl(path: str | Path):
    """List of rows. Blank lines skipped; malformed rows raise with
    line number; encoding errors degrade per-row, not per-file."""
    rows = []
    with open(path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: malformed jsonl row: {e}")
    return rows


def write_jsonl(path: str | Path, rows) -> None:
    """Atomic full-file write (tmp+rename, same directory)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: str | Path, row) -> None:
    """One row, appended and flushed — the streaming shape for
    workers that may be killed by an outer wall."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

"""Run-artifact contract — the Spark-_SUCCESS pattern for the lab.

Adopted from the 2026-08-11 reviewer design (handoff
2026-08-11-0). Eight marker inconsistencies existed in the tree
when this was written; the ones this module closes:

- "no marker" meant BOTH "finished cleanly" and "never ran" —
  the exact ambiguity that nearly re-birthed over the crown
  checkpoint (train_mathnative silently starts from scratch when
  a ckpt exists without its .ep marker). `require_resume_marker`
  makes that a refusal instead of a silent re-birth.
- rjob writes the string "killed" into a .rc that consumers may
  int(): `rc_of` returns int-or-None and never throws on it.
- a detached launch redirecting its log into a directory nothing
  had created yet dies silently: `run_dir` guarantees mkdir at
  NAME time, before any redirect can open.

One marker file per run: `<dir>/<name>.marker.json`, one JSON
object, written atomically (tmp + rename) ON SUCCESS-OR-FAILURE by
the run itself. Machine-independent: no host, no user, no absolute
paths inside the marker (public-repo discipline).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["run_dir", "write_marker", "read_marker", "is_done",
           "rc_of", "require_resume_marker"]


def run_dir(name: str, root: str | Path = "logs") -> Path:
    """Create (idempotently) and return logs/<name>/ AT NAME TIME.

    Call this BEFORE composing any redirect into the directory —
    the redirect opens before the child script runs (the micro-star
    silent-death class).
    """
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_marker(dir_or_path: str | Path, kind: str, rc: int | str,
                 wall_s: float | None = None,
                 artifacts: list[str] | None = None,
                 **extra) -> Path:
    """Write the run's single marker line, atomically.

    kind: free-form run class ("train", "gate", "census", ...).
    rc:   the process's honest exit status; strings ("killed")
          are stored as-is — rc_of() handles them.
    artifacts: repo-relative paths the run produced (the booking
          step consumes these instead of a human re-typing them).
    """
    p = Path(dir_or_path)
    if p.is_dir():
        p = p / "run.marker.json"
    row = {
        "kind": kind,
        "rc": rc,
        "wall_s": round(wall_s, 3) if wall_s is not None else None,
        "git_sha": _git_sha(),
        "argv": [os.path.basename(sys.argv[0])] + sys.argv[1:],
        "artifacts": artifacts or [],
        "ts": int(time.time()),
        **extra,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(row) + "\n")
    tmp.rename(p)
    return p


def read_marker(dir_or_path: str | Path) -> dict | None:
    """Return the marker dict, or None when absent/unparseable."""
    p = Path(dir_or_path)
    if p.is_dir():
        p = p / "run.marker.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_done(dir_or_path: str | Path) -> bool:
    """True iff a marker exists AND parses. Absence is 'never ran
    or still running' — never 'finished cleanly'."""
    return read_marker(dir_or_path) is not None


def rc_of(dir_or_path: str | Path) -> int | None:
    """The marker's rc as an int, or None for absent/non-integer
    ("killed" and friends return None — check read_marker()['rc']
    for the raw value)."""
    m = read_marker(dir_or_path)
    if m is None:
        return None
    try:
        return int(m["rc"])
    except (KeyError, TypeError, ValueError):
        return None


def require_resume_marker(ckpt: str | Path) -> int:
    """REFUSE to proceed when a checkpoint exists without its .ep
    marker — the state that silently re-births from scratch.

    Returns the resume epoch (marker value + 1) when the pair is
    consistent, 0 when neither exists (a genuinely fresh birth).
    Raises FileNotFoundError on ckpt-without-marker; ValueError on
    an unparseable marker.
    """
    ckpt = Path(ckpt)
    marker = Path(str(ckpt) + ".ep")
    if not ckpt.exists():
        return 0
    if not marker.exists():
        raise FileNotFoundError(
            f"{ckpt} exists but {marker.name} is missing — refusing "
            f"to train (would silently re-birth over the weights). "
            f"If a resume is intended, write the marker: "
            f"printf '<last-finished-epoch>' > {marker}")
    try:
        return int(marker.read_text().strip()) + 1
    except ValueError as e:
        raise ValueError(
            f"{marker} is unparseable ({marker.read_text()!r}) — "
            f"fix it before training") from e

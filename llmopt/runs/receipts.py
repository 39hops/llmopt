"""Per-step receipt writer — streaming jsonl rows for long runs.

Ported from axiom's exact_anchor receipt shape (2026-08 relay
exchange): one json line per step, flushed IMMEDIATELY. Workers
killed by an outer wall must STREAM their rows out incrementally
or the killed class is invisible to whatever trains on the data
(the checkpoint selection-effect, bit three times — CLAUDE.md
2026-07-12 corollary). This composes with llmopt.runs.runfiles:
rows for steps (this module), one marker for the run (runfiles).

Row contract:
- header (first line): {"header": true, "device", "git_sha",
  "argv", "ts"} — wall receipts state device BEFORE numbers
  (cross-device gate comparisons forbidden; sigma never
  transports, so a receipt without its device is unbookable).
- step rows: caller observables + auto "step", "wall_s" (seconds
  since the previous row, time.monotonic — never wall-clock),
  optional "digest" (running sha256 hex when the caller passes
  digest_bytes: divergence localizable to a STEP without a rerun),
  optional "fb" (fallback counters — cost/exactness attribution
  inline, axiom convention).
- abort row: {"step": next, "aborted": reason} — budget
  truncation distinguishable from a crash INSIDE the jsonl (a
  file that just stops is a crash; an abort row is a decision).
- close(rc): writes the runfiles marker in the same directory.

Observables are the caller's business; nothing here scores
weights — receipts carry function-space numbers only (never score
weights by weight distance, RESULTS 6163 joint-perm closure).
Machine-independent rows: no host, no user, no absolute paths
(public-repo discipline, same as runfiles).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from llmopt.runs.runfiles import _git_sha, write_marker

__all__ = ["RunLog", "FallbackCounters"]


def _device() -> str:
    """Best-effort device string, torch-optional (tests must skip
    cleanly without torch — here we degrade instead of skipping)."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return f"cpu/{platform.machine()}"


class FallbackCounters(dict):
    """Counter bundle the caller increments; written per row under
    "fb". dict subclass so json.dumps takes it verbatim; missing
    keys read as 0 so `fb.bump("exact_miss")` needs no init."""

    def bump(self, key: str, n: int = 1) -> int:
        self[key] = self.get(key, 0) + n
        return self[key]

    def __missing__(self, key: str) -> int:
        return 0


class RunLog:
    """Streaming per-step receipt file.

    >>> log = RunLog("logs/myrun/steps.jsonl")
    >>> log.step(0, loss=1.23)          # flushed before return
    >>> log.abort("budget")             # optional, on truncation
    >>> log.close(rc=0)                 # writes run.marker.json

    The file handle is opened line-buffered-equivalent: every row
    is flushed to the OS before .step() returns, so a SIGKILL'd
    worker still leaves every completed step on disk.
    """

    def __init__(self, path: str | Path, kind: str = "run",
                 device: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        # "x": REFUSE an existing receipt file (R1: mode "w" truncated
        # a prior run's streamed rows — the checkpoint selection-effect
        # this module exists to prevent). Re-runs pick a new path.
        self._fh = open(self.path, "x", encoding="utf-8")
        self._t_last = time.monotonic()
        self._t0 = self._t_last
        self._last_step: int | None = None
        self._digest = hashlib.sha256()
        self._closed = False
        self.fallback_counters = FallbackCounters()
        self._write({
            "header": True,
            "device": device if device is not None else _device(),
            "git_sha": _git_sha(),
            "argv": [os.path.basename(sys.argv[0])] + sys.argv[1:],
            "ts": int(time.time()),
        })

    # -- internals -------------------------------------------------
    def _write(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()          # stream-or-invisible doctrine

    # -- public contract -------------------------------------------
    def step(self, step: int, digest_bytes: bytes | None = None,
             **observables) -> dict:
        """Write one step row IMMEDIATELY; returns the row dict.

        wall_s = seconds since the previous row (monotonic clock).
        digest_bytes, when given, feeds a CUMULATIVE sha256; the
        row carries the running hex, so two runs' receipts localize
        their first divergence to a step without a rerun.
        """
        now = time.monotonic()
        row = {"step": step,
               "wall_s": round(now - self._t_last, 6),
               **observables}
        if digest_bytes is not None:
            self._digest.update(digest_bytes)
            row["digest"] = self._digest.hexdigest()
        if self.fallback_counters:
            row["fb"] = dict(self.fallback_counters)
        self._t_last = now
        self._last_step = step
        self._write(row)
        return row

    def abort(self, reason: str) -> dict:
        """Budget truncation is a DECISION, not a crash — say so in
        the jsonl. step = the step that would have run next."""
        nxt = 0 if self._last_step is None else self._last_step + 1
        row = {"step": nxt, "aborted": reason}
        self._write(row)
        return row

    def close(self, rc: int | str = 0, **extra) -> Path:
        """Close the row stream and write the run-level marker
        (runfiles contract) in the same directory. Idempotent-safe
        against double close of the file handle."""
        if not self._closed:
            self._fh.close()
            self._closed = True
        # marker named after the log stem (R2: the fixed run.marker.json
        # name meant two RunLogs in one dir overwrote each other's
        # marker and a booking read arm B's rc against arm A's receipt)
        marker_path = self.path.parent / (self.path.stem + ".marker.json")
        return write_marker(
            marker_path, self.kind, rc,
            wall_s=time.monotonic() - self._t0,
            artifacts=[self.path.name], **extra)

    # context-manager sugar: unclean exit closes with rc="exception"
    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(rc=0 if exc_type is None else "exception")

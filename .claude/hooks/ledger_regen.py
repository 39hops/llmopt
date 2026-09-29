#!/usr/bin/env python3
"""Regenerate whatever a ledger edit just invalidated.

The ledger has four generated surfaces, each derived from a source a
session edits by hand:

  docs/RESULTS.md   -> docs/results-index.jsonl (gen_results_index.py)
                       + docs/receipts.lock.json (gen_receipt_lock.py)
                       + the FINDINGS ratchet headroom line
  docs/preregs/*    -> docs/receipts.lock.json
  docs/FINDINGS.md  -> README's honesty-ledger region and the
                       honesty_ledger figure in docs/figures.json
                       (gen_readme.py owns both)
  scratch/*.py, scripts/*.py, llmopt/{,lab,train,search}/*.py
                    -> scripts/INDEX.md

INVARIANT (2026-09-29): read-only repository inspection never mutates
tracked files. The hook therefore never infers "a source changed"
from a command's TEXT (a `grep` naming docs/RESULTS.md changes
nothing). It runs in two phases wired in .claude/settings.json:

  pre   (PreToolUse, Bash|Edit|Write): snapshot (mtime_ns, size) of
        every watched source into a state file keyed by tool_use_id.
  post  (PostToolUse, same matcher): re-snapshot, diff against the
        pre snapshot, run ONLY the generators whose sources changed,
        delete the state file. No snapshot -> nothing runs.

Every generator is a fixed point over its own output and writes only
when the content differs, so a spurious run is harmless; the point of
the snapshot is that it does not happen at all. The diff is of the
TREE, not of what this one tool did: a read-only call that overlaps
another call's real edit will also see the change and regenerate
(same result the editor's post would write). Posts serialize on a
lock so two generators never write one file at once.
LLMOPT_NO_AUTOREGEN=1 disables regeneration entirely (defense in
depth, not the mechanism). Never fails a tool call: best-effort,
exit 0 regardless. The gates are scripts/check_source.sh (`--check`
on every generator) and tests/test_ledger_regen_hook.py.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---- portable cross-process lock -------------------------------------
# fcntl.flock on POSIX (macOS / Linux / WSL), msvcrt.locking on native
# Windows, and a dependency-free mkdir spin lock when neither exists.
# Importing this hook must succeed on every platform: the "never fail a
# tool call" boundary lives in main(), below any import.
try:
    import fcntl as _fcntl
except ImportError:  # native Windows
    _fcntl = None
try:
    import msvcrt as _msvcrt
except ImportError:  # POSIX
    _msvcrt = None

_AVAILABLE = [b for b, m in (("fcntl", _fcntl), ("msvcrt", _msvcrt))
              if m is not None] + ["mkdir"]
_override = os.environ.get("LLMOPT_HOOK_LOCK_BACKEND")
# an override is honoured only for a backend this platform actually has
LOCK_BACKEND = _override if _override in _AVAILABLE else _AVAILABLE[0]
# A holder directory can only be ABANDONED once the harness's PostToolUse
# timeout (.claude/settings.json, 120 s for `ledger_regen.py post`) has
# killed its process; a live holder may legitimately be that old. The
# stale threshold therefore stays strictly above that timeout with margin
# (tests/test_ledger_regen_hook.py reads settings.json and enforces it).
LOCK_STALE_S = 300


@contextlib.contextmanager
def _mkdir_lock(path: Path):
    """Dependency-free fallback: mkdir is atomic everywhere. A directory
    older than LOCK_STALE_S (itself longer than the post hook's harness
    timeout, so its holder cannot still be running) is an abandoned lock
    and is reclaimed; only the process that created the directory
    removes it."""
    d = path.with_name(path.name + ".d")
    owned = False
    while not owned:
        try:
            os.mkdir(d)
            owned = True
        except FileExistsError:
            try:
                if time.time() - os.stat(d).st_mtime > LOCK_STALE_S:
                    os.rmdir(d)
                    continue
            except OSError:
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.rmdir(d)


@contextlib.contextmanager
def post_lock(path: Path):
    """Hold an exclusive cross-process lock at `path` for the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_BACKEND == "fcntl":
        with open(path, "w") as fh:
            _fcntl.flock(fh, _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(fh, _fcntl.LOCK_UN)
    elif LOCK_BACKEND == "msvcrt":
        with open(path, "a") as fh:  # never truncate a locked file
            while True:  # msvcrt.locking raises OSError while contended
                try:
                    _msvcrt.locking(fh.fileno(), _msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                _msvcrt.locking(fh.fileno(), _msvcrt.LK_UNLCK, 1)
    else:
        with _mkdir_lock(path):
            yield

ROOT = Path(os.environ.get("LLMOPT_HOOK_ROOT")
            or Path(__file__).resolve().parents[2])
PY = os.environ.get("LLMOPT_HOOK_PY") or str(ROOT / ".venv" / "bin" / "python")
LOG = os.environ.get("LLMOPT_HOOK_LOG")  # test seam: record planned runs

# watched sources: (glob relative to ROOT, generator group)
WATCH = [
    ("docs/RESULTS.md", "results"),
    ("docs/preregs/*.json", "preregs"),
    ("docs/FINDINGS.md", "findings"),
    ("scratch/*.py", "index"),
    ("scripts/*.py", "index"),
    ("llmopt/*.py", "index"),
    ("llmopt/lab/*.py", "index"),
    ("llmopt/train/*.py", "index"),
    ("llmopt/search/*.py", "index"),
]

GENERATORS = {
    "results": ["scripts/gen_results_index.py",
                "scripts/gen_receipt_lock.py",
                ".claude/hooks/findings_headroom.py"],
    "preregs": ["scripts/gen_receipt_lock.py"],
    "findings": ["scripts/gen_readme.py"],
    "index": ["scripts/gen_index.py"],
}

NOTES = {
    "results": "results-index + receipt lock regenerated.",
    "preregs": "receipt lock regenerated for prereg-declared receipts.",
    "findings": ("README + figures.json honesty ledger regenerated "
                 "(counts follow FINDINGS; commit them with the booking "
                 "or the suite goes red)."),
    "index": ("INDEX regenerated. If a file is NEW: commit it, then "
              "rerun gen_codemap.py (tracked files only) or the suite "
              "goes red."),
}


def snapshot(root: Path = ROOT) -> dict[str, list[int]]:
    """{relpath: [mtime_ns, size]} for every watched source present."""
    out: dict[str, list[int]] = {}
    for pat, _ in WATCH:
        for p in root.glob(pat):
            if p.is_file():
                st = p.stat()
                out[str(p.relative_to(root))] = [st.st_mtime_ns, st.st_size]
    return out


def changed_sources(before: dict, after: dict) -> set[str]:
    """Paths added, removed, or with a different (mtime_ns, size)."""
    return {k for k in set(before) | set(after)
            if before.get(k) != after.get(k)}


def group_of(rel: str) -> str | None:
    p = Path(rel)
    for pat, grp in WATCH:
        if p.match(pat):
            return grp
    return None


def plan(changed: set[str]) -> list[str]:
    """Ordered, de-duplicated generator list for the changed sources."""
    groups = [g for g in GENERATORS if any(group_of(c) == g for c in changed)]
    out: list[str] = []
    for g in groups:
        for gen in GENERATORS[g]:
            if gen not in out:
                out.append(gen)
    return out


def run(*args: str) -> str:
    """Best-effort generator call. Returns whatever the tool said on
    either stream — findings_headroom.py writes its warning to stderr
    and exits 2, so stdout alone would drop exactly the message worth
    surfacing."""
    if LOG:
        with open(LOG, "a") as f:
            f.write(" ".join(args) + "\n")
    try:
        r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=120)
        return (r.stdout.strip() or r.stderr.strip())
    except Exception:
        return ""


STATE_DIR = Path(tempfile.gettempdir()) / "llmopt-ledger-regen"
STALE_S = 24 * 3600  # a pre whose post never fired (blocked call) leaks


def _state_path(data: dict) -> Path | None:
    """Per-tool-call state file; None when the call cannot be keyed."""
    key = data.get("tool_use_id")
    if not key:
        return None  # no per-call key: fail open, run nothing
    key = hashlib.sha256(str(key).encode()).hexdigest()[:24]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{key}.json"


def _sweep_stale() -> None:
    cutoff = time.time() - STALE_S
    for f in STATE_DIR.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _main(argv: list[str]) -> None:
    phase = argv[1] if len(argv) > 1 else "post"
    if phase not in ("pre", "post"):
        return
    data = json.load(sys.stdin)
    if not isinstance(data, dict):
        return
    if data.get("tool_name") not in ("Bash", "Edit", "Write"):
        return
    state = _state_path(data)
    if state is None:
        return
    if phase == "pre":
        _sweep_stale()
        state.write_text(json.dumps(snapshot()))
        return
    if os.environ.get("LLMOPT_NO_AUTOREGEN"):
        state.unlink(missing_ok=True)
        return
    if not state.exists():
        return  # no baseline: cannot prove a change, so run nothing
    try:
        before = json.loads(state.read_text())
    except Exception:
        before = None
    state.unlink(missing_ok=True)
    if before is None:
        return
    changed = changed_sources(before, snapshot())
    if not changed:
        return
    notes = []
    groups = [g for g in GENERATORS
              if any(group_of(c) == g for c in changed)]
    # serialize generator runs across concurrent posts: two writers on
    # one output file would race (write_text is not atomic)
    with post_lock(STATE_DIR / "post.lock"):
        for gen in plan(changed):
            out = run(gen)
            if gen.endswith("findings_headroom.py") and out:
                notes.append(out)
    notes = [NOTES[g] for g in groups] + notes
    if notes:
        print(" ".join(notes))


def main(argv: list[str]) -> None:
    try:
        _main(argv)
    except Exception:
        pass  # a hook must never fail the tool call


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)

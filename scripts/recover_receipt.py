"""Restore ONE booked receipt whose file exists on no checkout.

The receipt guard (.claude/hooks/receipt_freeze.py) refuses every
command that would write to a path a booked entry cites, because a
booked receipt is evidence and a new run belongs at a new path. The
one legitimate write to such a path is putting the ORIGINAL bytes back
when they were never committed: a driver log that stayed on the
machine that wrote it. This script is that write, and nothing else.

    .venv/bin/python scripts/recover_receipt.py \\
        --source /tmp/fetched/driver.log \\
        --dest logs/run1/driver.log \\
        --sha256 <64 hex, as booked in docs/RESULTS.md> \\
        --origin "3080 checkout 937ffa32, mtime 2026-08-11"

Every check below refuses with a distinct message and leaves the tree
untouched; the destination is created with O_EXCL so a race or a
mistaken path cannot overwrite anything:

  1. the destination is a receipt path (logs/<dir>/<file.ext>) the lock
     records as results-cited and ABSENT (the booked-without-evidence
     backlog); sha-locked, pending, prereg-only and uncited paths are
     refused;
  2. the destination does not exist on disk in any form and git does
     not track it;
  3. the source is a regular file whose sha256 equals --sha256;
  4. --sha256 is already booked in docs/RESULTS.md (the digest was
     measured on the origin host and written into the ledger BEFORE
     the bytes travelled, so the ledger, not the caller, vouches for
     them);
  5. no registered run is live (the liverun sentinel in the git common
     dir) and this checkout is the primary worktree.

After the copy the destination is re-hashed; a mismatch removes the
file this script just created and refuses. A success appends one JSON
line to docs/receipt-recoveries.jsonl (destination, digest, bytes,
origin text, the RESULTS line that books the digest, HEAD) and prints
the three follow-up steps: git add -f, scripts/gen_receipt_lock.py,
and shrinking LEGACY_ABSENT in the frozen-receipt test. The script
runs none of them: each lands in a reviewable diff on its own.

The hook allows exactly one command shape through: this script, alone,
with plain arguments, no shell operators. Anything else that names a
booked receipt is judged as before.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEST = re.compile(r"^logs/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RECORD = "docs/receipt-recoveries.jsonl"


class Refuse(SystemExit):
    def __init__(self, why: str):
        super().__init__(f"recover_receipt: REFUSING: {why}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True)


def live_run_sentinel(root: Path) -> Path | None:
    """The liverun sentinel if a registered run is live (any worktree
    sharing this git directory), else None."""
    r = git(root, "rev-parse", "--git-common-dir")
    if r.returncode != 0:
        raise Refuse("not a git checkout")
    common = Path(r.stdout.strip())
    if not common.is_absolute():
        common = root / common
    lp = common / "liverun.lock"
    return lp if lp.exists() else None


def is_primary_worktree(root: Path) -> bool:
    a = git(root, "rev-parse", "--git-dir").stdout.strip()
    b = git(root, "rev-parse", "--git-common-dir").stdout.strip()
    return bool(a) and a == b


def booked_line(results: Path, digest: str) -> int | None:
    for i, line in enumerate(results.read_text().splitlines(), 1):
        if digest in line:
            return i
    return None


def check(root: Path, source: Path, dest: str, digest: str) -> dict:
    """Run every refusal check; return the lock row for dest."""
    if not DEST.match(dest) or ".." in dest.split("/"):
        raise Refuse(f"{dest!r} is not a receipt path (logs/<dir>/<file.ext>)")
    if not HEX64.match(digest):
        raise Refuse("--sha256 must be 64 lowercase hex characters")
    lock_file = root / "docs" / "receipts.lock.json"
    if not lock_file.is_file():
        raise Refuse("docs/receipts.lock.json is missing; nothing is "
                     "classified as absent without it")
    try:
        rows = json.loads(lock_file.read_text())["receipts"]
    except Exception as e:  # unreadable lock: fail closed
        raise Refuse(f"docs/receipts.lock.json cannot be parsed ({e})")
    row = rows.get(dest)
    if row is None:
        raise Refuse(f"{dest} is not cited by any booked entry (not in the lock)")
    if row.get("exists") or row.get("sha256") or row.get("pending"):
        raise Refuse(f"{dest} is not classified absent in the lock "
                     f"({json.dumps(row)}); a present or locked receipt "
                     "is never a recovery target")
    if row.get("source") != "results":
        raise Refuse(f"{dest} is {row.get('source')}-declared, not a "
                     "results-cited receipt")
    target = root / dest
    if target.exists() or target.is_symlink():
        raise Refuse(f"{dest} already exists on disk; nothing is overwritten")
    if git(root, "ls-files", "--error-unmatch", dest).returncode == 0:
        raise Refuse(f"{dest} is tracked by git; not an absent receipt")
    if not source.is_file() or source.is_symlink():
        raise Refuse(f"source {source} is not a regular file")
    if source.resolve() == target.resolve():
        raise Refuse("source and destination are the same path")
    got = sha256_file(source)
    if got != digest:
        raise Refuse(f"source digest {got} does not match --sha256 {digest}")
    results = root / "docs" / "RESULTS.md"
    if not results.is_file() or booked_line(results, digest) is None:
        raise Refuse(f"digest {digest} is not booked in docs/RESULTS.md; "
                     "book the digest measured on the origin host first")
    lp = live_run_sentinel(root)
    if lp is not None:
        raise Refuse(f"a registered run is live ({lp.read_text().strip()[:200]})")
    if not is_primary_worktree(root):
        raise Refuse("this checkout is a linked worktree; recover in the "
                     "primary checkout")
    return row


def restore(root: Path, source: Path, dest: str, digest: str,
            origin: str) -> dict:
    check(root, source, dest, digest)
    target = root / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    got = sha256_file(target)
    if got != digest:
        target.unlink()
        raise Refuse(f"re-hash after write gave {got}, not {digest}; "
                     "the file was removed")
    rec = {
        "recovered_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "dest": dest,
        "sha256": digest,
        "bytes": len(data),
        "source_name": source.name,
        "origin": origin,
        "booked_at_results_line": booked_line(root / "docs" / "RESULTS.md",
                                              digest),
        "head": git(root, "rev-parse", "--short", "HEAD").stdout.strip(),
    }
    with (root / RECORD).open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Restore one booked receipt that exists on no checkout "
                    "from its original bytes; refuses everything else.")
    ap.add_argument("--source", required=True, help="fetched original bytes")
    ap.add_argument("--dest", required=True,
                    help="receipt path relative to the repo, logs/<dir>/<file.ext>")
    ap.add_argument("--sha256", required=True,
                    help="digest measured on the origin host, as booked")
    ap.add_argument("--origin", required=True,
                    help="where the bytes came from (host, commit, mtime); "
                         "recorded verbatim, no home paths")
    ap.add_argument("--root", default=None,
                    help=argparse.SUPPRESS)  # tests point at a fixture repo
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1]
    rec = restore(root, Path(a.source), a.dest, a.sha256.lower(), a.origin)
    print(f"[recover] {rec['dest']} restored ({rec['bytes']} B, sha256 "
          f"{rec['sha256'][:16]}...), booked at RESULTS L{rec['booked_at_results_line']}; "
          f"record appended to {RECORD}")
    print("[recover] next, each in its own reviewable step:\n"
          f"  git add -f {rec['dest']}\n"
          "  .venv/bin/python scripts/gen_receipt_lock.py\n"
          "  shrink LEGACY_ABSENT in tests/science_incidents/test_frozen_receipt_mutation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""One-time (rerunnable via --redo): attach code_commit to ledger rows.

For each row, the booking commit is the one that introduced the
entry heading into RESULTS.md (git log -S). One hit = confident;
zero or 2+ hits = null (a null is honest, a wrong sha is a trap -
spec 1.3).

File-aware repair (fix round 1, 2026-08-12): the booking commit's
PARENT is not always the tree the driver actually ran on — some
entries book docs before the code commit that produced them lands.
So code_commit picks the sha where every cited file is actually
present: parent (B^) if EVERY path in the row's `files` list exists
there (`git cat-file -e sha:path`); else the booking commit B itself
if every cited file exists there; else null. Rows with an empty
`files` list keep the parent rule (nothing to check against).

Use `--redo` to clear every row's code_commit first and backfill
fresh under the current rule (rows already set are otherwise
skipped).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "docs" / "results-index.jsonl"


def path_exists_at(sha: str, path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{sha}:{path}"],
        cwd=ROOT, capture_output=True).returncode == 0


def find_commit(title: str, files: list) -> str | None:
    out = subprocess.run(
        ["git", "log", "--format=%H", "-S", title, "--", "docs/RESULTS.md"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    if len(out) != 1:
        return None
    booking = out[0]
    parent = subprocess.run(
        ["git", "rev-parse", f"{booking}^"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not files:
        return parent or None
    if parent and all(path_exists_at(parent, f) for f in files):
        return parent
    if all(path_exists_at(booking, f) for f in files):
        return booking
    return None


def main() -> None:
    redo = "--redo" in sys.argv
    rows = [json.loads(l) for l in IDX.open()]
    if redo:
        for r in rows:
            r["code_commit"] = None
    nulls = 0
    for i, r in enumerate(rows):
        if r.get("code_commit"):
            continue
        r["code_commit"] = find_commit(r["title"], r.get("files", []))
        nulls += r["code_commit"] is None
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", file=sys.stderr)
    IDX.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"done: {len(rows)} rows, {nulls} null")


if __name__ == "__main__":
    main()

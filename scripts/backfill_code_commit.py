#!/usr/bin/env python3
"""One-time: attach code_commit to existing ledger rows.

For each row, the booking commit is the one that introduced the
entry heading into RESULTS.md (git log -S). code_commit is that
commit's PARENT: the tree checked out when the driver ran. One hit
= confident; zero or 2+ hits = null (a null is honest, a wrong sha
is a trap - spec 1.3).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "docs" / "results-index.jsonl"


def find_commit(title: str) -> str | None:
    out = subprocess.run(
        ["git", "log", "--format=%H", "-S", title, "--", "docs/RESULTS.md"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    if len(out) != 1:
        return None
    parent = subprocess.run(
        ["git", "rev-parse", f"{out[0]}^"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return parent or None


def main() -> None:
    rows = [json.loads(l) for l in IDX.open()]
    nulls = 0
    for i, r in enumerate(rows):
        if r.get("code_commit"):
            continue
        r["code_commit"] = find_commit(r["title"])
        nulls += r["code_commit"] is None
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", file=sys.stderr)
    IDX.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"done: {len(rows)} rows, {nulls} null")


if __name__ == "__main__":
    main()

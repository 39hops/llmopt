"""Stable-ID citations for RESULTS.md anchors (the anchor transition).

WHY. Curated docs cite ledger entries as ``[Title](RESULTS.md#L<n>)``.
Line numbers are only stable while RESULTS.md is strictly append-only —
a convention, not an invariant — and when one drifts the reader cannot
tell WHICH entry was meant, only that the anchor is broken. The stable
key already exists: docs/results-index.jsonl assigns every entry an id
(e.g. ``2026-08-16-verdict-stream-wdistill-0-audit-repair``) that never
moves. This tool migrates citations to a DUAL form,

    [Title](RESULTS.md#L74 "id:undated-the-racing-arc-all-same-held")

keeping the clickable line anchor while embedding the id as the
markdown link title. The id is the source of truth; the line number
becomes a repairable cache. This is a TRANSITION, not a forever
line-repair loop: once every citation carries an id, drift is fixed by
--repair (recompute L from id) instead of by hand.

Modes:
    report  (default)  count id-less vs id-carrying anchors per doc
    --migrate          add "id:..." to every resolvable id-less anchor
    --repair           rewrite the L number of every id-carrying anchor
                       from the index; refuses on an unknown id

Scope: every *.md under the repo EXCEPT docs/RESULTS.md — the ledger is
append-only and its own internal line references (many deliberately
mid-entry) are historical text, never rewritten.

Migration resolves an anchor's id by exact heading-line match against
the index (external anchors are test-enforced to land on ``## ``
headings, so the match is unambiguous today — that is the migration
window). An anchor that matches no index line is left untouched and
reported; forcing a guess would silently repoint a citation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "results-index.jsonl"
RESULTS_NAME = "RESULTS.md"
# [Title](<prefix>RESULTS.md#L<n>) with optional existing "id:..." title.
LINK = re.compile(
    r"\((?P<prefix>[A-Za-z0-9_./\-]*?)RESULTS\.md#L(?P<line>\d+)"
    r"(?:\s+\"id:(?P<id>[a-z0-9\-]+)\")?\)")


def load_index() -> tuple[dict[int, str], dict[str, int]]:
    by_line: dict[int, str] = {}
    by_id: dict[str, int] = {}
    for raw in INDEX.read_text().splitlines():
        row = json.loads(raw)
        by_line[row["line"]] = row["id"]
        by_id[row["id"]] = row["line"]
    return by_line, by_id


def target_files() -> list[pathlib.Path]:
    return [p for p in list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md"))
            if p.name != RESULTS_NAME]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--repair", action="store_true")
    a = ap.parse_args()
    by_line, by_id = load_index()

    total_plain = total_tagged = migrated = repaired = 0
    unresolved: list[str] = []
    unknown: list[str] = []

    for path in target_files():
        text = path.read_text()
        rel = path.relative_to(ROOT)

        def sub(m: re.Match) -> str:
            nonlocal total_plain, total_tagged, migrated, repaired
            line, eid = int(m.group("line")), m.group("id")
            if eid is None:
                total_plain += 1
                if a.migrate:
                    got = by_line.get(line)
                    if got is None:
                        unresolved.append(f"{rel}: L{line}")
                        return m.group(0)
                    migrated += 1
                    return (f"({m.group('prefix')}{RESULTS_NAME}#L{line} "
                            f"\"id:{got}\")")
                return m.group(0)
            total_tagged += 1
            if a.repair:
                if eid not in by_id:
                    unknown.append(f"{rel}: id:{eid}")
                    return m.group(0)
                if by_id[eid] != line:
                    repaired += 1
                return (f"({m.group('prefix')}{RESULTS_NAME}#L{by_id[eid]} "
                        f"\"id:{eid}\")")
            return m.group(0)

        new = LINK.sub(sub, text)
        if new != text:
            path.write_text(new)

    print(f"anchors: {total_plain} id-less, {total_tagged} id-carrying")
    if a.migrate:
        print(f"migrated {migrated}; unresolved (left untouched): "
              f"{len(unresolved)}")
        for u in unresolved:
            print(f"  {u}")
    if a.repair:
        print(f"repaired {repaired} drifted line numbers")
        if unknown:
            print("UNKNOWN ids (citation names an entry the index does "
                  "not have — fix by hand):\n  " + "\n  ".join(unknown))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

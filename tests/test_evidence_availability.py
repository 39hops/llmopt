"""docs/EVIDENCE.md's pinned-artifact table must describe the checkout.

Each row names a file behind a front-page result with a status
(tracked / local-only / absent) and a sha256 (a digest, or `(locked)`
to defer to docs/receipts.lock.json). The statuses are checked against
git and disk so the page cannot say "tracked" about a file git does not
carry, "absent" about a file that has reappeared, or pin a digest the
bytes no longer match. A sha in a table is a fingerprint, never a
backup; the test only makes a wrong fingerprint visible.
"""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "EVIDENCE.md"
LOCK = ROOT / "docs" / "receipts.lock.json"
ROW = re.compile(r"^\| `([^`]+)` \| (tracked|local-only|absent) \| ([0-9a-f]{64}|\(locked\)|-) \|$")
STATUSES = ("tracked", "local-only", "absent")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pinned_rows(text: str) -> list[tuple[str, str, str]]:
    """(path, status, sha) for every row of the pinned-artifact table."""
    section = text.split("## Pinned artifacts", 1)
    assert len(section) == 2, "EVIDENCE.md lost its '## Pinned artifacts' section"
    rows = []
    for line in section[1].splitlines():
        if line.startswith("| `"):
            m = ROW.match(line)
            assert m, f"malformed pinned row: {line!r}"
            rows.append((m.group(1), m.group(2), m.group(3)))
    assert rows, "pinned-artifact table is empty"
    return rows


def _tracked(rel: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


def test_pinned_rows_parse_and_are_unique():
    rows = pinned_rows(DOC.read_text())
    paths = [r[0] for r in rows]
    assert len(paths) == len(set(paths)), "duplicate path in the table"
    for path, status, sha in rows:
        assert status in STATUSES
        if status == "absent":
            assert sha == "-", f"{path}: an absent file cannot carry a sha"
        else:
            assert sha != "-", f"{path}: a present file needs a sha or (locked)"
        if sha == "(locked)":
            assert path.startswith("logs/"), (
                f"{path}: only logs/ paths are in the receipt lock")


def test_pinned_rows_match_git_and_disk():
    lock = json.loads(LOCK.read_text())["receipts"]
    problems, unverifiable = [], []
    for path, status, sha in pinned_rows(DOC.read_text()):
        p = ROOT / path
        tracked = _tracked(path)
        if status == "tracked":
            if not tracked:
                problems.append(f"{path}: documented tracked, git does not carry it")
                continue
            if not p.is_file():
                problems.append(f"{path}: tracked but missing from this checkout")
                continue
            want = lock.get(path, {}).get("sha256") if sha == "(locked)" else sha
            if want is None:
                problems.append(f"{path}: (locked) but the lock holds no sha for it")
            elif _sha(p) != want:
                problems.append(f"{path}: bytes differ from the documented sha")
        elif status == "local-only":
            if tracked:
                problems.append(f"{path}: documented local-only but git tracks it; "
                                "reclassify as tracked")
            elif p.is_file():
                if _sha(p) != sha:
                    problems.append(f"{path}: local bytes differ from the documented sha")
            elif os.environ.get("LLMOPT_FULL") == "1":
                problems.append(f"{path}: local-only evidence absent on the evidence host")
            else:
                unverifiable.append(path)
        else:  # absent
            if tracked or p.is_file():
                problems.append(f"{path}: documented absent but present; the "
                                "artifact reappeared, reclassify it in EVIDENCE.md "
                                "(and shrink the legacy set if it is a receipt)")
    if unverifiable:
        print(f"[evidence] {len(unverifiable)} local-only artifact(s) not in "
              f"this checkout: {unverifiable}")
    assert not problems, "EVIDENCE.md disagrees with the checkout:\n  " + "\n  ".join(problems)


def test_absent_rows_agree_with_the_lock():
    """Every absent receipt the page lists is one the lock also records
    as exists:false. The page covers the front-page chains only, so the
    lock may name further absent receipts (logs/pincer/labels_v2.jsonl
    today) that the page does not."""
    lock = json.loads(LOCK.read_text())["receipts"]
    page_absent = {path for path, status, _ in pinned_rows(DOC.read_text())
                   if status == "absent" and path.startswith("logs/")}
    lock_absent = {k for k, v in lock.items()
                   if not v.get("exists") and v.get("source") == "results"}
    assert page_absent <= lock_absent, (
        f"page says absent, lock disagrees: {sorted(page_absent - lock_absent)}")


def test_malformed_table_is_refused():
    with pytest.raises(AssertionError):
        pinned_rows("## Pinned artifacts\n| `x` | tracked | notasha |\n")

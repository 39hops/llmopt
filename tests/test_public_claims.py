"""Public-surface claims that drifted from the machinery once
(engineering audit 2026-10-13) and are now checked mechanically.

Each test pins one statement in a front-facing document to the
artifact it describes, so the statement cannot silently outlive the
fact: the receipt lock's scope, the Lean certificate line's
denominator and amendment, and the scoreboard's freshness.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_receipt_lock_scope_is_logs_only_and_docs_say_so():
    """The lock content-addresses `logs/` receipt paths only. The
    docs map row describing it must not claim checkpoints are locked
    (docs/README.md said "pinned checkpoints under ../checkpoints/"
    while the lock held zero checkpoint rows)."""
    lock = json.loads((DOCS / "receipts.lock.json").read_text())
    outside = [k for k in lock["receipts"] if not k.startswith("logs/")]
    assert not outside, (
        "the receipt lock now covers paths outside logs/; update the "
        "docs/README.md lock row and this test together:\n  "
        + "\n  ".join(outside[:10]))
    row = next(line for line in (DOCS / "README.md").read_text().splitlines()
               if line.startswith("| [receipts.lock.json]"))
    assert "pinned checkpoints under" not in row
    assert "Checkpoints are not in the lock" in row
    assert "`logs/`" in row


def test_lean_certificate_line_carries_its_denominator_and_amendment():
    """ENGINEERING.md's Lean line must state the corpus denominator
    (500), the ring-only scope, and cite the amendment that rescoped
    the verdict, not only the verdict (which reads 443/443)."""
    lines = [line for line in (DOCS / "ENGINEERING.md").read_text().splitlines()
             if "id:2026-08-04-verdict-lean-tier-1-the-certificate" in line]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "500" in line and "443" in line
    assert re.search(r"`ring`|ring tactic|ring only", line, re.I), line
    assert "id:2026-08-04-amendment-gt1-traj-corr-amends-observation" in line
    assert "untested" in line


def test_scoreboard_is_current_and_declares_partial_coverage():
    """SCOREBOARD.md is generated; its producer's --check must pass
    and its header must state coverage instead of 'current truth'."""
    r = subprocess.run([sys.executable, "scripts/gen_scoreboard.py",
                        "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    head = (DOCS / "SCOREBOARD.md").read_text().splitlines()[:3]
    assert not any("current truth, generated" in h for h in head)
    assert any("PARTIAL view" in h for h in head), head
    # every listed verdict has an index row with a verdict field
    idx = {json.loads(line)["line"]: json.loads(line)
           for line in (DOCS / "results-index.jsonl").read_text().splitlines()
           if line.strip()}
    listed = [int(m) for m in re.findall(
        r"\*\(RESULTS L(\d+)\)\*", (DOCS / "SCOREBOARD.md").read_text())]
    curated_live = [ln for ln, e in idx.items()
                    if e.get("verdict") and not e.get("superseded_by")]
    assert sorted(listed) == sorted(curated_live)

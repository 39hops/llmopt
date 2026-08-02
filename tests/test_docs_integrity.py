"""Doc-layer integrity: the curated surface must keep pointing at the ledger.

Two failure modes this guards, both silent without a test:

1. ANCHOR ROT. FINDINGS/REPRODUCE cite verdicts as ``RESULTS.md#L<n>``.
   Those line numbers are only stable while RESULTS.md is strictly
   append-only. "Append-only" is a house convention, not an enforced
   invariant, so one mid-file edit shifts every anchor below it and no
   reader can tell.

2. CURATION DRIFT. RESULTS.md grows on every booking; FINDINGS.md is a
   hand-maintained projection with no staleness signal. The backlog below
   is a ratchet: it may fall freely, but raising it means new verdicts
   were booked without deciding whether an external reader should see
   them.
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs/RESULTS.md"
FINDINGS = ROOT / "docs/FINDINGS.md"
INDEX = ROOT / "docs/results-index.jsonl"

# Entry types an external reader is expected to see curated. Pre-regs,
# amendments, and riders qualify their parents rather than standing alone.
CURATABLE = ("verdict", "null")

# Ratchet, measured 2026-08-02 on the opus-5 review branch. LOWER this as
# FINDINGS catches up; raising it needs a note saying which verdicts were
# deliberately left off the curated layer and why.
MAX_UNCURATED = 10


def _anchors():
    """(source file, line number) for every RESULTS.md#L<n> citation."""
    md = list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md"))
    for path in md:
        if path.name == "RESULTS.md":
            continue
        for num in re.findall(r"RESULTS\.md#L(\d+)", path.read_text()):
            yield path.relative_to(ROOT), int(num)


def _uncurated():
    """Curatable ledger entries booked past the newest FINDINGS citation."""
    cited = [int(n) for n in re.findall(r"RESULTS\.md#L(\d+)",
                                        FINDINGS.read_text())]
    newest = max(cited)
    rows = [json.loads(line) for line in INDEX.open()]
    return [r for r in rows
            if r["line"] > newest and r["type"] in CURATABLE]


def test_results_anchors_land_on_entry_headings():
    lines = RESULTS.read_text().split("\n")
    broken = []
    for src, num in _anchors():
        target = lines[num - 1] if num <= len(lines) else "<past EOF>"
        if not target.startswith("## "):
            broken.append(f"{src}#L{num} -> {target[:60]!r}")
    assert not broken, (
        "citation(s) no longer land on a verdict heading — RESULTS.md was "
        "edited mid-file, or the anchors drifted:\n  " + "\n  ".join(broken))


def test_findings_curation_backlog_does_not_grow():
    stale = _uncurated()
    assert len(stale) <= MAX_UNCURATED, (
        f"{len(stale)} curatable entries are newer than anything FINDINGS "
        f"cites (ratchet is {MAX_UNCURATED}). Curate them or raise the "
        "ratchet with a reason:\n  "
        + "\n  ".join(r["id"] for r in stale[MAX_UNCURATED:]))


if __name__ == "__main__":       # standalone report, no pytest needed
    anchors = list(_anchors())
    print(f"anchors checked: {len(anchors)}")
    stale = _uncurated()
    print(f"uncurated {CURATABLE} entries: {len(stale)} "
          f"(ratchet {MAX_UNCURATED})")
    for row in stale:
        print(f"  {row['type']:8s} {row['id']}")

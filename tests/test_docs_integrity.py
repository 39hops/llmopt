"""Doc-layer integrity: the curated surface must keep pointing at the ledger.

Two failure modes this guards, both silent without a test:

1. ANCHOR ROT. FINDINGS/REPRODUCE cite verdicts as ``RESULTS.md#L<n>``.
   Those line numbers are only stable while RESULTS.md is strictly
   append-only. "Append-only" is a house convention, not an enforced
   invariant, so one mid-file edit shifts every anchor below it and no
   reader can tell.

2. CURATION DRIFT. RESULTS.md grows on every booking; FINDINGS.md is a
   hand-maintained projection with no staleness signal. The backlog below
   counts curatable entries FINDINGS does not cite at all — a set
   difference, not a high-water mark. The first version compared against
   `max(cited)`, which FINDINGS pins to the newest entry, so the metric
   read 0 while 296 entries were uncited and it could never rise
   (reviewer catch, 2026-08-02). It is a ratchet: it may fall freely,
   and raising it means bookings went uncurated.
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

# Measured backlog of curatable entries FINDINGS does not cite, plus a
# little slack so a working session can book before curating. This counts
# the WHOLE ledger, not just recent entries — most of the 296 are older
# results the curated layer never covered, which is itself worth seeing.
# Raised 300 -> 320 on 2026-08-05: the MOE-GT-1/GT-2 program booked ~29
# entries in two days; the new FINDINGS "routing crest and domain
# coalitions" section curates the 13 headliners, and the residue is
# intermediate pre-regs/amendments that the amendment chain (not
# FINDINGS) is the right home for. Ratchet still binds future growth.
# Lowered 320 -> 300 on 2026-08-12: curated the 20 oldest uncurated
# ledger entries (all in "The derivation engine" section) into FINDINGS.
MAX_UNCURATED = 300

# GLOSSARY.md is the authority for all three lists; a tag outside them is
# either a typo or vocabulary drift, and drift is what makes the tags
# unreadable by anything but a human.
MATURITY = ("RETRACTED", "NULL", "MECHANISM-CONFIRMED", "REPLICATED",
            "SINGLE-SEED")
SCOPE = ("DEVICE-SCOPED", "FORMAT-BOUND", "TEACHER-FORCED", "FREE-RUN-GATED")
REGIME_VALUES = (
    "calculus search", "closed-system math", "house crystals",
    "at-capacity house crystals", "specified diet and recipe",
    "deterministic integer battery", "tested MoE recipes",
    "measured deployment artifacts", "Qwen2.5-0.5B",
    "toy weight-space subjects")


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
    cited = {int(n) for n in re.findall(r"RESULTS\.md#L(\d+)",
                                        FINDINGS.read_text())}
    rows = [json.loads(line) for line in INDEX.open()]
    return [r for r in rows
            if r["line"] not in cited and r["type"] in CURATABLE]


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


def test_findings_tag_grammar():
    """One maturity tag per claim, and no tag outside the glossary.

    A tag wrapped across a line break reads fine to a human and is
    invisible to every grep — which is the property the controlled
    vocabulary exists to provide. It shows up here as an unknown value.
    """
    bullets = re.findall(r"^- \[.*?(?=^- \[|^#|\Z)",
                         FINDINGS.read_text(), re.S | re.M)
    assert bullets, "no tagged bullets found — did FINDINGS change shape?"
    problems = []
    for bullet in bullets:
        head = bullet.split("(")[0]
        tags = re.findall(r"\[([A-Z][A-Z-]*(?::[^\]]*)?)\]", head)
        first = bullet.strip().split("\n")[0][:60]
        mats = [t for t in tags if t in MATURITY]
        if len(mats) != 1:
            problems.append(f"{first!r}: {len(mats)} maturity tags {mats}")
        for tag in tags:
            base, _, value = tag.partition(":")
            if base not in MATURITY + SCOPE + ("REGIME-SCOPED",):
                problems.append(f"{first!r}: unknown tag [{tag}]")
            elif base == "REGIME-SCOPED" and value.strip() not in REGIME_VALUES:
                problems.append(
                    f"{first!r}: regime {value.strip()!r} is not in the "
                    "controlled vocabulary (or the tag wrapped a line)")
    assert not problems, "FINDINGS tag grammar:\n  " + "\n  ".join(problems)


def test_findings_curation_backlog_does_not_grow():
    stale = _uncurated()
    assert len(stale) <= MAX_UNCURATED, (
        f"{len(stale)} curatable entries are not cited by FINDINGS "
        f"(ratchet is {MAX_UNCURATED}). Curate them or raise the ratchet "
        "with a reason. Newest uncited:\n  "
        + "\n  ".join(r["id"] for r in sorted(
            stale, key=lambda r: -r["line"])[:10]))


if __name__ == "__main__":       # standalone report, no pytest needed
    anchors = list(_anchors())
    print(f"anchors checked: {len(anchors)}")
    stale = _uncurated()
    print(f"uncurated {CURATABLE} entries: {len(stale)} "
          f"(ratchet {MAX_UNCURATED})")
    for row in sorted(stale, key=lambda r: -r["line"])[:10]:
        print(f"  newest uncited: {row['type']:8s} {row['id']}")

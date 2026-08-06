"""CODEMAP inventory guard: the file list in docs/CODEMAP.md must
match the filesystem — adding or removing a top-level scratch/scripts
file without regenerating the map fails here. Deliberately does NOT
check citation counts or classes (those drift with every RESULTS
booking; forcing a regen per booking would be friction, and the
census is re-derived at move time anyway — the map's one hard job is
to never silently omit a file).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_codemap_lists_exactly_the_inventory():
    text = (ROOT / "docs" / "CODEMAP.md").read_text()
    listed = set(re.findall(r"^\| \S+ \| (\S+) \|", text, re.MULTILINE))
    on_disk = {f.name for f in (ROOT / "scratch").glob("*.py")} | \
              {f.name for f in (ROOT / "scratch").glob("*.sh")} | \
              {f.name for f in (ROOT / "scripts").glob("*.py")}
    missing = sorted(on_disk - listed)
    stale = sorted(listed - on_disk)
    assert not missing and not stale, (
        f"CODEMAP out of date — run scripts/gen_codemap.py. "
        f"unlisted files: {missing[:10]}; listed-but-gone: {stale[:10]}")

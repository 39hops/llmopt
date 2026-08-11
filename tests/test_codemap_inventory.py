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
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_names(prefix: str, suffixes: tuple[str, ...]) -> set[str]:
    """Top-level TRACKED files under `prefix`.

    Mirrors gen_codemap.py: the map describes the repository, so both
    sides must ignore untracked working files. Globbing here instead
    made the test demand a CODEMAP row for gitignored
    scratch/remote.env.sh, whose existence should not be published in a
    committed doc at all.
    """
    r = subprocess.run(["git", "ls-files", prefix], cwd=ROOT,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    out = set()
    for line in r.stdout.split():
        rel = line[len(prefix):]
        if "/" in rel:
            continue  # top level only, same as the old glob
        if rel.endswith(suffixes):
            out.add(rel)
    return out


def test_codemap_lists_exactly_the_inventory():
    text = (ROOT / "docs" / "CODEMAP.md").read_text()
    # family cell may be empty (__init__.py); header row never matches
    # because the second cell must end .py/.sh
    listed = set(re.findall(r"^\| [^|]* \| (\S+\.(?:py|sh)) \|",
                            text, re.MULTILINE))
    on_disk = _tracked_names("scratch/", (".py", ".sh")) | \
              _tracked_names("scripts/", (".py",))
    missing = sorted(on_disk - listed)
    stale = sorted(listed - on_disk)
    assert not missing and not stale, (
        f"CODEMAP out of date — run scripts/gen_codemap.py. "
        f"unlisted files: {missing[:10]}; listed-but-gone: {stale[:10]}")

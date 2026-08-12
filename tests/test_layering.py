"""tests/test_layering.py — scripts/ never imports scratch/ (144:0 measured 2026-08-12)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scripts_never_import_scratch():
    scratch_mods = {p.stem for p in (ROOT / "scratch").glob("*.py")}
    offenders = []
    for p in (ROOT / "scripts").glob("*.py"):
        for m in re.finditer(r"^\s*(?:import\s+(\w+)|from\s+(\w+)\s+import)",
                             p.read_text(), re.M):
            mod = m.group(1) or m.group(2)
            if mod in scratch_mods:
                offenders.append(f"{p.name} imports {mod}")
    assert not offenders, offenders

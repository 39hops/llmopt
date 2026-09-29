"""Generator determinism for docs/results-index.jsonl.

The index is a GENERATED view that carries hand-CURATED fields. The
generator must be a fixed point over its own committed output: a
regeneration with an unchanged RESULTS.md must reproduce the committed
file byte for byte, including the curated absence of `needs_link` on
rows a session has already resolved. Importing the module must have
no side effects (no file writes at import time).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_results_index.py"


def _load():
    spec = importlib.util.spec_from_file_location("gri_mod", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RESULTS_TEXT = """# RESULTS

## Contents

## AMENDMENT FOO-1 (target: VERDICT FOO L10): fixed a typo (2026-09-01)

body cites scratch/foo_driver.py

## AMENDMENT BAR-1 (target: VERDICT BAR L20): new (2026-09-02)

body
"""


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    _load()
    assert not (tmp_path / "docs" / "results-index.jsonl").exists()


def test_curated_needs_link_removal_survives_regen():
    """A row whose `needs_link` a session already popped (curated as
    resolved, no `amends` recorded) must not get the flag re-added
    on the next regeneration; a NEW amendment row still gets it."""
    mod = _load()
    first = mod.build_entries(RESULTS_TEXT, old={})
    foo = next(e for e in first if e["id"].endswith("amendment-foo-1-target-verdict-foo"))
    assert foo["needs_link"] is True
    # curation: pop the flag on foo, keep bar untouched
    old = {e["id"]: dict(e) for e in first}
    old[foo["id"]].pop("needs_link")
    second = mod.build_entries(RESULTS_TEXT, old=old)
    foo2 = next(e for e in second if e["id"] == foo["id"])
    bar2 = next(e for e in second if "bar-1" in e["id"])
    assert "needs_link" not in foo2
    assert bar2["needs_link"] is True


def test_regen_is_a_fixed_point_over_its_own_output():
    mod = _load()
    first = mod.build_entries(RESULTS_TEXT, old={})
    old = {e["id"]: e for e in first}
    old[first[0]["id"]].pop("needs_link")
    second = mod.build_entries(RESULTS_TEXT, old=old)
    third = mod.build_entries(RESULTS_TEXT, old={e["id"]: e for e in second})
    assert second == third


def test_check_mode_writes_nothing_and_reports_drift(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "RESULTS.md").write_text(RESULTS_TEXT)
    dst = docs / "results-index.jsonl"
    # --check against a missing index: drift, exit 1, nothing written
    r = subprocess.run([sys.executable, str(GEN), "--check"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1
    assert not dst.exists()
    # write, then --check is clean
    r = subprocess.run([sys.executable, str(GEN)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0 and dst.exists()
    r = subprocess.run([sys.executable, str(GEN), "--check"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # explicit --out honours the destination
    out = tmp_path / "elsewhere.jsonl"
    subprocess.run([sys.executable, str(GEN), "--out", str(out)],
                   cwd=tmp_path, check=True, capture_output=True)
    assert [json.loads(l)["id"] for l in out.read_text().splitlines()] == \
        [json.loads(l)["id"] for l in dst.read_text().splitlines()]


def test_row_reclassified_to_amendment_is_flagged_like_a_new_one():
    """If TYPE_RULES change so an existing non-amendment row becomes an
    amendment, it has never been curated as an amendment: flag it."""
    mod = _load()
    old = {}
    first = mod.build_entries(RESULTS_TEXT, old={})
    foo = first[0]
    old[foo["id"]] = dict(foo, type="observation")
    old[foo["id"]].pop("needs_link", None)
    second = mod.build_entries(RESULTS_TEXT, old=old)
    assert second[0]["needs_link"] is True

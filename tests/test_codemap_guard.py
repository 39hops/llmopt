"""The edit guard freezes EVIDENCE, not a ladder position: a file
that RESULTS or REPRODUCE cites stays guarded even when code also
imports it (class `library` outranks the citation in the ladder but
must not outrank the freeze)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "codemap_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("codemap_guard", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TABLE = """# CODEMAP
| family | file | class | cited by | doc citations | imports | invoked by | mentions | via |
|---|---|---|---|---|---|---|---|---|
| a | a_lib.py | library | RESULTS, specs | RESULTS×70, specs×3 | 65 | — | — | — |
| b | b_free.py | library | — | — | 2 | — | — | — |
| c | c_cited.py | results-cited | RESULTS | RESULTS×1 | — | — | — | — |
| d | d_via.py | results-cited | — | — | — | 1 | — | chain.sh |
| e | e_spec.py | spec-cited | specs | specs×2 | — | — | — | — |
| f | f_pin.py | library | REPRODUCE | REPRODUCE×1 | 1 | — | — | — |
| g | g_inherit.py | library | via:RESULTS | — | 3 | 1 | — | chain.sh |
| h | h_spec_inherit.py | library | via:specs | — | 3 | 1 | — | plan.sh |
"""


@pytest.mark.parametrize("name,frozen", [
    ("a_lib.py", True), ("b_free.py", False), ("c_cited.py", True),
    ("d_via.py", True), ("e_spec.py", False), ("f_pin.py", True),
    ("zz_unknown.py", False),
])
def test_frozen_follows_citation_evidence_not_class(name, frozen):
    mod = _load()
    assert mod.is_frozen(f"scratch/{name}", TABLE) is frozen


@pytest.mark.docs
def test_real_results_cited_libraries_are_still_frozen():
    mod = _load()
    text = (ROOT / "docs" / "CODEMAP.md").read_text()
    # bare names joined at runtime: a `scratch/x.py` literal here would
    # itself count as a test invocation edge in the map
    for name in ("mathworld1_svpbirth.py", "oracle_worker.py",
                 "perturbation_response_gram_desk.py", "anatomy.py"):
        assert mod.is_frozen("scratch" + "/" + name, text), name
    assert not mod.is_frozen("scratch" + "/" + "phase4_unboot.py", text)


def test_ambiguous_basename_rows_freeze_rather_than_pick_one():
    """Two rows with one basename (a future scratch/x.py + scripts/x.py):
    the guard must not silently take the first; it asks."""
    mod = _load()
    table = TABLE + "| a | a_lib.py | UNCITED | — | — | — | — | — | — |\n"
    assert mod.is_frozen("scripts/a_lib.py", table) is True
    assert mod.is_frozen("scratch/a_lib.py", table) is True
    table2 = TABLE + "| b | b_free.py | UNCITED | — | — | — | — | — | — |\n"
    assert mod.is_frozen("scratch/b_free.py", table2) is True  # ambiguous -> ask

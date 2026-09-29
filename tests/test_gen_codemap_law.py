"""The CODEMAP dependency law must see how the repository actually
reaches a file: bare and dotted imports, dynamic loaders, subprocess
and shell invocation, tooling references, brace/glob citations, and
citations that reach a helper THROUGH the caller that a document
names. Synthetic-tree tests pin each rule; docs-marked tests pin the
reconnaissance failures on the real repository.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_codemap.py"


def _load():
    spec = importlib.util.spec_from_file_location("gen_codemap_mod", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    # --- inventory ---------------------------------------------------
    _w(root, "scratch/lib_a.py", "def f(): pass\n")
    _w(root, "scratch/lib_b.py", "def g(): pass\n")
    _w(root, "scratch/lib_c.py", "def h(): pass\n")
    _w(root, "scratch/helper_d.py", "print('d')\n")
    _w(root, "scratch/gate_e.py", "print('e')\n")
    _w(root, "scratch/tool_f.py", "print('f')\n")
    _w(root, "scratch/brace_g2.py", "print('g2')\n")
    _w(root, "scratch/glob_h1.py", "print('h1')\n")
    _w(root, "scratch/orphan_i.py", "print('i')\n")
    _w(root, "scratch/period_p.py", "print('p')\n")
    _w(root, "scratch/doc_q.py", "print('q')\n")
    _w(root, "scratch/joined_r.py", "print('r')\n")
    _w(root, "scratch/indexed_s.py", "print('s')\n")
    _w(root, "scratch/keyed_z.py", "print('z')\n")
    _w(root, "scratch/sentence_end.py", "print('se')\n")
    _w(root, "scratch/stem_j_probe.py", "print('j')\n")
    _w(root, "scratch/emit_k.py", "print('k')\n")
    _w(root, "scratch/prereg_l.sh", "echo l\n")
    _w(root, "scratch/job_m.sh", "echo m\n")
    _w(root, "scratch/leaf_n.py", "print('n')\n")
    _w(root, "scratch/mid_n.sh", "python scratch/leaf_n.py\n")
    _w(root, "scratch/chain_d.sh", "set -e\n.venv/bin/python scratch/helper_d.py\n"
                                   "python scratch/lib_a.py --report\n")
    _w(root, "scratch/top_n.sh", "bash scratch/mid_n.sh\n")
    _w(root, "scratch/user1.py", "import lib_a\nfrom scratch.lib_b import g\n"
                                 "from scripts import tool_s\n")
    _w(root, "scratch/user2.py",
       "import importlib.util\n"
       "spec = importlib.util.spec_from_file_location('c', 'scratch/lib_c.py')\n")
    _w(root, "scratch/user3.py", "x = 'lib_a'  # a bare string is NOT an import\n")
    _w(root, "scripts/tool_s.py", "def s(): pass\n")
    _w(root, "scripts/orphan_t.py", "pass\n")
    # --- package and tests -------------------------------------------
    _w(root, "llmopt/lab/merge.py",
       "import subprocess\nSCRIPT = 'scratch/gate_e.py'\n"
       "subprocess.run(['python', SCRIPT])\n")
    _w(root, "tests/test_x.py", "def test(): pass\n")
    _w(root, "llmopt/common/quant.py",
       '"""Adopted verbatim from scratch/doc_q.py (results-cited)."""\n'
       "x = 1\n")
    _w(root, "tests/test_join.py",
       "from pathlib import Path\nROOT = Path('.')\n"
       "def test():\n    (ROOT / 'scratch' / 'joined_r.py').read_text()\n")
    # --- documents ---------------------------------------------------
    _w(root, "docs/RESULTS.md",
       "## VERDICT D: driver scratch/chain_d.sh booked (2026-01-01)\n"
       "receipt logs/k/r.json.\n"
       "## VERDICT G: scratch/brace_g{,2,3}.py all ran\n"
       "## VERDICT J: the stem_j_probe machinery counted\n"
       "## VERDICT N: scratch/top_n.sh\n"
       "## VERDICT P: the repair landed in scratch/period_p.py. Done.\n"
       "## VERDICT SE: counted by sentence_end.\n")
    _w(root, "docs/results-index.jsonl",
       json.dumps({"id": "x", "files": ["scratch/indexed_s.py",
                                        "llmopt/pkg/orphan_i.py"]}) + "\n")
    _w(root, "docs/FINDINGS.md", "# F\n")
    _w(root, "docs/BOARD.md", "# B\n")
    _w(root, "docs/THEORY.md", "# T\n")
    _w(root, "README.md", "# R\n")
    _w(root, "docs/REPRODUCE.md", "# nothing pinned\n")
    _w(root, "docs/preregs/l.json",
       json.dumps({"driver": "scratch/prereg_l.sh", "receipts": []}))
    _w(root, "CLAUDE.md", "Diagnose with scratch/tool_f.py; see scratch/glob_h*.py\n")
    _w(root, "pyproject.toml", '[tool.ruff.lint.per-file-ignores]\n"scratch/*.py" = ["E501"]\n"scripts/*" = ["F401"]\n')
    _w(root, "jobs/night.cmd", "bash scratch/job_m.sh\n")
    _w(root, "logs/k/r.json", json.dumps({"emitter": "scratch/emit_k.py",
                                          "keyed_z": 1}))
    _w(root, "docs/receipts.lock.json",
       json.dumps({"receipts": {"logs/k/r.json": {"exists": True,
                                                  "tracked": True}}}))
    mod = _load()
    rows = {r["file"]: r for r in mod.build(root, tracked=None)}
    return rows


def test_bare_import_is_library(synthetic):
    r = synthetic["lib_a.py"]
    assert r["class"] == "library" and r["imports"] == ["scratch/user1.py"]


def test_library_row_keeps_inherited_citation_visible(synthetic):
    """lib_a.py is library (imported) AND run by the RESULTS-cited
    chain_d.sh: the inherited RESULTS grade must be visible on the
    row so the edit guard can freeze it."""
    mod = _load()
    r = synthetic["lib_a.py"]
    assert r["inherited"] == {"RESULTS"} and r["via"] == ["scratch/chain_d.sh"]
    text, _ = mod.render([r])
    assert "| library | via:RESULTS |" in text


def test_dotted_import_is_library(synthetic):
    assert synthetic["lib_b.py"]["class"] == "library"
    assert synthetic["tool_s.py"]["class"] == "library"


def test_dynamic_path_loader_is_library(synthetic):
    r = synthetic["lib_c.py"]
    assert r["class"] == "library" and r["imports"] == ["scratch/user2.py"]


def test_string_literal_alone_is_not_an_import(synthetic):
    assert "scratch/user3.py" not in synthetic["lib_a.py"]["imports"]


def test_subprocess_path_from_package_is_library(synthetic):
    r = synthetic["gate_e.py"]
    assert r["class"] == "library"
    assert r["invoked_by"] == ["llmopt/lab/merge.py"]


def test_helper_reached_through_cited_shell_inherits_citation(synthetic):
    r = synthetic["helper_d.py"]
    assert r["class"] == "results-cited"
    assert r["via"] == ["scratch/chain_d.sh"]


def test_propagation_follows_a_two_step_chain(synthetic):
    assert synthetic["mid_n.sh"]["class"] == "results-cited"
    assert synthetic["leaf_n.py"]["class"] == "results-cited"


def test_brace_citation_expands(synthetic):
    assert synthetic["brace_g2.py"]["class"] == "results-cited"


def test_glob_citation_in_tooling_docs(synthetic):
    assert synthetic["glob_h1.py"]["class"] == "tool-referenced"


def test_tooling_reference_is_its_own_floor(synthetic):
    assert synthetic["tool_f.py"]["class"] == "tool-referenced"
    assert synthetic["job_m.sh"]["class"] == "tool-referenced"


def test_stem_only_citation_counts(synthetic):
    assert synthetic["stem_j_probe.py"]["class"] == "results-cited"


def test_locked_receipt_naming_its_emitter_is_results_grade(synthetic):
    assert synthetic["emit_k.py"]["class"] == "results-cited"


def test_prereg_json_citation_is_results_grade(synthetic):
    assert synthetic["prereg_l.sh"]["class"] == "results-cited"


def test_sentence_ending_period_after_a_path_still_cites(synthetic):
    assert synthetic["period_p.py"]["class"] == "results-cited"


def test_directory_wide_glob_cites_nothing(synthetic):
    """`scratch/*.py` (pyproject per-file-ignores, program plans) is a
    statement about a directory, not a citation of each file."""
    assert synthetic["orphan_i.py"]["class"] == "UNCITED"
    assert synthetic["orphan_t.py"]["class"] == "UNCITED"


def test_docstring_provenance_is_a_mention_not_an_invocation(synthetic):
    """A package module that says it was adopted from a scratch file
    names lineage; it does not reach the file. The original stays a
    cited evidence record rather than becoming `library`."""
    r = synthetic["doc_q.py"]
    assert r["class"] == "UNCITED"
    assert r["invoked_by"] == [] and r["mentions"] == ["llmopt/common/quant.py"]


def test_path_join_from_tests_is_an_invocation_edge(synthetic):
    r = synthetic["joined_r.py"]
    assert r["class"] == "library"
    assert r["invoked_by"] == ["tests/test_join.py"]


def test_results_index_files_field_is_results_grade(synthetic):
    assert synthetic["indexed_s.py"]["class"] == "results-cited"


def test_results_index_files_match_the_full_path_not_the_basename(synthetic):
    """llmopt/pkg/orphan_i.py in an index row is not scratch/orphan_i.py."""
    assert synthetic["orphan_i.py"]["class"] == "UNCITED"


def test_bare_stem_in_a_json_receipt_key_is_not_a_citation(synthetic):
    assert synthetic["keyed_z.py"]["class"] == "UNCITED"


def test_bare_stem_ending_a_sentence_counts(synthetic):
    assert synthetic["sentence_end.py"]["class"] == "results-cited"


def test_uncited_stays_uncited(synthetic):
    assert synthetic["orphan_t.py"]["class"] == "UNCITED"


# ------------------------------------------------- the real repository

@pytest.fixture(scope="module")
def real_rows():
    mod = _load()
    return {r["file"]: r for r in mod.build(ROOT)}


@pytest.mark.docs
def test_real_svpbirth_importers_are_seen(real_rows):
    r = real_rows["mathworld1_svpbirth.py"]
    assert r["class"] == "library" and len(r["imports"]) >= 60


@pytest.mark.docs
def test_real_gate_ckpt_cuda_is_a_package_runtime_contract(real_rows):
    r = real_rows["gate_ckpt_cuda.py"]
    assert r["class"] == "library"
    assert "llmopt/lab/merge.py" in r["invoked_by"]


@pytest.mark.docs
@pytest.mark.parametrize("name", [
    "mathworld1_svpchal2.py", "qwen_runtime0r.py", "bench_decoding.py"])
def test_real_uncited_files_with_live_importers_are_library(real_rows, name):
    assert real_rows[name]["class"] == "library", real_rows[name]


@pytest.mark.docs
@pytest.mark.parametrize("name", [
    "complex_birth.py", "ce400.py", "gate_zx.py", "mathworld1_svpgriddesk3.py"])
def test_real_helpers_of_cited_callers_are_cited(real_rows, name):
    r = real_rows[name]
    assert r["eff_cites"].get("RESULTS") or r["eff_cites"].get("REPRODUCE"), r


@pytest.mark.docs
@pytest.mark.parametrize("name", ["claim_lint.py", "cite_lookup.py"])
def test_real_tooling_scripts_are_not_uncited(real_rows, name):
    assert real_rows[name]["class"] != "UNCITED", real_rows[name]

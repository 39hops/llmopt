"""tests/test_results_index_files.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _gri():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gri", ROOT / "scripts" / "gen_results_index.py")
    gri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gri)
    return gri


def test_extract_files_finds_repo_paths():
    gri = _gri()
    body = ("Driver: scratch/softprompt1.py wraps "
            "scripts/step_grpo_micro.py; receipts in logs/x.log "
            "and llmopt/lab/gate.py. Not a path: results.md")
    assert gri.extract_files(body) == [
        "llmopt/lab/gate.py",
        "scratch/softprompt1.py",
        "scripts/step_grpo_micro.py",
    ]

def test_observation_headings_type_as_observation():
    gri = _gri()
    assert gri.infer_type(
        "OBSERVATION FOO: a thing was counted (2026-08-17, desk)"
    ) == "observation"


def test_observation_retraction_stays_amendment():
    gri = _gri()
    assert gri.infer_type(
        "OBSERVATION FOO-RETRACTION: the claim died (2026-08-17, desk)"
    ) == "amendment"


def test_every_index_row_has_files_key():
    rows = [json.loads(l) for l in
            (ROOT / "docs" / "results-index.jsonl").open()]
    assert all("files" in r for r in rows)

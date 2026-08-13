"""tests/test_gen_readme.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_ledger_counts_match_findings():
    out = subprocess.run(
        [sys.executable, "scripts/gen_readme.py", "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_figures_json_honesty_ledger_matches_findings():
    """The honesty-ledger figure's counts must match the FINDINGS
    recount (same counter gen_readme uses). Caught drifting 2026-08-13
    (54 v 55 mechanism-confirmed) by an external review."""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "gen_readme", ROOT / "scripts" / "gen_readme.py")
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    c = gr.counts()

    fig = json.loads((ROOT / "docs" / "figures.json").read_text())
    parts = {p["label"]: p["value"]
             for p in fig["honesty_ledger"]["parts"]}
    assert parts["Replicated"] == c["REPLICATED"]
    assert parts["Mechanism confirmed"] == c["MECHANISM-CONFIRMED"]
    assert parts["Single seed"] == c["SINGLE-SEED"]
    assert parts["Null"] == c["NULL"]
    assert parts["Retracted"] == c["RETRACTED"]
    assert str(sum(c.values())) in fig["honesty_ledger"]["fence"]

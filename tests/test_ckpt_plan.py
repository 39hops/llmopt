"""scripts/ckpt_plan.py: resolve dynamic checkpoint path sites to exact
names where the script assigns its variables from literals, tell
producers from consumers, and emit a file-by-file plan (phase 2F
preparation; the plan executes nothing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ckpt_plan as cp  # noqa: E402


def _w(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# ------------------------------------------------------------ patterns

def test_fstring_site_becomes_anchored_regex_with_named_holes():
    site = cp.parse_site('OUT = Path(f"checkpoints/gal_{LABEL}_s{SEED}.pt")',
                         "scratch/x.py", 3)
    assert site["role"] == "producer"
    assert site["holes"] == ["LABEL", "SEED"]
    assert site["regex"].fullmatch("checkpoints/gal_dose2_s7.pt")
    assert not site["regex"].fullmatch("checkpoints/gal_dose2_s7.pt.ep")
    assert not site["regex"].fullmatch("checkpoints/galaxy.pt")


def test_shell_site_forms():
    s1 = cp.parse_site("ck=checkpoints/mathnative_${name}.pt", "scratch/g.sh", 1)
    assert s1["holes"] == ["name"] and s1["role"] == "consumer"
    s2 = cp.parse_site("--out checkpoints/seedvar_$s.pt > log", "scratch/g.sh", 2)
    assert s2["holes"] == ["s"] and s2["role"] == "producer"
    s3 = cp.parse_site('for p in sorted(glob.glob("checkpoints/packed_*.npz")):',
                       "scratch/g.py", 3)
    assert s3["holes"] == ["*"] and s3["role"] == "consumer"
    assert s3["regex"].fullmatch("checkpoints/packed_a1.npz")


def test_role_detection():
    assert cp.parse_site('torch.save(m, f"checkpoints/cc_{A}.pt")', "s.py", 1)["role"] == "producer"
    assert cp.parse_site('sd = torch.load(f"checkpoints/cc_{A}.pt")', "s.py", 1)["role"] == "consumer"
    assert cp.parse_site('python scratch/gate_ckpt.py checkpoints/fmt_${F}.pt 1 2', "s.sh", 1)["role"] == "consumer"
    assert cp.parse_site('Checkpoint: checkpoints/fmt_{FORMAT}_{SCHED}.pt', "s.py", 1)["role"] == "mention"


# ----------------------------------------------------------- variables

def test_variable_values_resolved_from_literal_assignments(tmp_path):
    py = _w(tmp_path / "d.py",
            'import os\nSEED = int(os.environ.get("SEED", "2"))\n'
            'ARMS = ["hot", "cool"]\nLABEL = os.environ["LABEL"]\n'
            'for ARM in ARMS:\n    OUT = f"checkpoints/g_{ARM}_s{SEED}.pt"\n')
    vals = cp.variable_values(py.read_text())
    assert vals["ARMS"] == ["hot", "cool"]
    assert vals["ARM"] == ["hot", "cool"]          # loop over a known list
    assert vals["SEED"] == ["2"]                   # env default is one value
    assert "LABEL" not in vals                     # argument-driven


def test_shell_loop_values(tmp_path):
    sh = _w(tmp_path / "d.sh",
            'for S in 2 3 4; do\n  for A in hot cool; do\n'
            '    --out checkpoints/x_${A}_s$S.pt\n  done\ndone\nF=${F:-fp32}\n')
    vals = cp.variable_values(sh.read_text())
    assert vals["S"] == ["2", "3", "4"] and vals["A"] == ["hot", "cool"]
    assert vals["F"] == ["fp32"]


def test_backslash_continued_shell_loop_resolves(tmp_path):
    sh = _w(tmp_path / "g.sh",
            'for name in a_one a_two \\\n            a_three; do\n'
            '  ck=checkpoints/mathnative_${name}.pt\ndone\n')
    vals = cp.variable_values(sh.read_text())
    assert vals["name"] == ["a_one", "a_two", "a_three"]


def test_generic_site_counts_only_through_exact_expansion():
    generic = cp.parse_site("python probe.py checkpoints/$CKPT.pt", "s.sh", 1)
    assert generic["generic"] is True
    assert cp.expand(generic, {"CKPT": ["m1", "m2"]}) == [
        "checkpoints/m1.pt", "checkpoints/m2.pt"]


def test_docstring_line_without_code_is_a_mention():
    s = cp.parse_site("Walks checkpoints/*.pt (top level) + checkpoints/confirmed/** and",
                      "scripts/gen_catalog.py", 3)
    assert s["role"] == "mention"
    s2 = cp.parse_site("for f in checkpoints/k_s*.pt; do echo $f; done", "g.sh", 1)
    assert s2["role"] == "consumer"


def test_expand_site_to_exact_names_when_every_hole_resolves():
    site = cp.parse_site('OUT = f"checkpoints/x_{A}_s{S}.pt"', "s.py", 1)
    names = cp.expand(site, {"A": ["hot", "cool"], "S": ["2", "3"]})
    assert names == ["checkpoints/x_cool_s2.pt", "checkpoints/x_cool_s3.pt",
                     "checkpoints/x_hot_s2.pt", "checkpoints/x_hot_s3.pt"]
    assert cp.expand(site, {"A": ["hot"]}) is None   # S unresolved


# ---------------------------------------------------------------- plan

def _inv_rows(tmp_path, rows):
    f = tmp_path / "inventory.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return f


def test_plan_rows_name_matches_roles_and_resolution(tmp_path):
    repo = tmp_path / "repo"
    _w(repo / "scratch/birth.py",
       'SEED = 2\nfor A in ["hot", "cool"]:\n'
       '    OUT = f"checkpoints/x_{A}_s{SEED}.pt"\n')
    _w(repo / "scratch/gate.sh",
       'for N in hot; do python scratch/gate_ckpt.py checkpoints/x_${N}_s2.pt; done\n')
    _w(repo / "scratch/any.sh", 'python read.py checkpoints/x_${WHAT}.pt\n')
    _w(repo / "docs/CODEMAP.md",
       "| x | birth.py | UNCITED | — | — | — | — | — | — |\n"
       "| x | gate.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |\n")
    inv = _inv_rows(tmp_path, [
        {"path": "x_hot_s2.pt", "repo_path": "checkpoints/x_hot_s2.pt",
         "bytes": 10, "alloc": 10, "mtime": 0, "class": "UNKNOWN",
         "reasons": ["dynamic path prefix"], "family": "root:x",
         "canonical": None, "dependencies": []},
        {"path": "x_cool_s2.pt", "repo_path": "checkpoints/x_cool_s2.pt",
         "bytes": 10, "alloc": 10, "mtime": 0, "class": "UNKNOWN",
         "reasons": ["dynamic path prefix"], "family": "root:x",
         "canonical": None, "dependencies": []},
        {"path": "x_warm_s9.pt", "repo_path": "checkpoints/x_warm_s9.pt",
         "bytes": 10, "alloc": 10, "mtime": 0, "class": "UNKNOWN",
         "reasons": ["dynamic path prefix"], "family": "root:x",
         "canonical": None, "dependencies": []},
        {"path": "keep.pt", "repo_path": "checkpoints/keep.pt",
         "bytes": 10, "alloc": 10, "mtime": 0, "class": "FROZEN",
         "reasons": ["RESULTS path"], "family": "root:keep",
         "canonical": None, "dependencies": []},
    ])
    plan = cp.build_plan(repo, inv, copies={})
    rows = {r["path"]: r for r in plan["rows"]}
    assert set(rows) == {"checkpoints/x_hot_s2.pt", "checkpoints/x_cool_s2.pt",
                         "checkpoints/x_warm_s9.pt"}   # FROZEN rows not planned
    hot = rows["checkpoints/x_hot_s2.pt"]
    kinds = {(m["source"], m["role"], m["match"]) for m in hot["matches"]}
    assert ("scratch/birth.py", "producer", "exact") in kinds
    assert ("scratch/gate.sh", "consumer", "exact") in kinds
    assert ("scratch/any.sh", "consumer", "pattern") in kinds
    assert hot["consumer_exact"] == ["scratch/gate.sh"]
    assert hot["codemap"]["scratch/gate.sh"] == "results-cited"
    assert hot["proposed"] == "KEEP: exact consumer in results-cited script"
    cool = rows["checkpoints/x_cool_s2.pt"]
    assert cool["consumer_exact"] == []
    assert cool["producer_exact"] == ["scratch/birth.py"]
    assert cool["proposed"].startswith("REVIEW")
    warm = rows["checkpoints/x_warm_s9.pt"]
    assert warm["producer_exact"] == [] and warm["consumer_exact"] == []
    assert warm["proposed"].startswith("REVIEW")
    assert plan["summary"]["rows"] == 3
    assert plan["summary"]["by_proposed"]["KEEP: exact consumer in results-cited script"]["files"] == 1


def test_duplicate_and_candidate_rows_carry_their_actions(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scratch").mkdir(parents=True)
    inv = _inv_rows(tmp_path, [
        {"path": "d.pt", "repo_path": "checkpoints/d.pt", "bytes": 5, "alloc": 5,
         "mtime": 0, "class": "DUPLICATE", "reasons": ["byte-identical"],
         "family": "root:d", "canonical": "checkpoints/confirmed/d.pt",
         "dependencies": ["scratch/r.py"]},
        {"path": "c.pt", "repo_path": "checkpoints/c.pt", "bytes": 5, "alloc": 5,
         "mtime": 0, "class": "CANDIDATE", "reasons": ["no reference"],
         "family": "root:c", "canonical": None, "dependencies": []},
    ])
    plan = cp.build_plan(repo, inv, copies={"3080": {"c.pt": ["checkpoints/c.pt"]}})
    rows = {r["path"]: r for r in plan["rows"]}
    assert rows["checkpoints/d.pt"]["proposed"] == (
        "RELINK scratch/r.py to checkpoints/confirmed/d.pt, then REMOVE (per-row GO)")
    assert rows["checkpoints/c.pt"]["proposed"] == "REVIEW: no reference found (CANDIDATE); per-family GO"
    assert rows["checkpoints/c.pt"]["copies"] == {"3080": ["checkpoints/c.pt"]}
    assert plan["summary"]["executes_nothing"] is True


def test_markdown_plan_lists_every_row(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scratch").mkdir(parents=True)
    inv = _inv_rows(tmp_path, [
        {"path": "c.pt", "repo_path": "checkpoints/c.pt", "bytes": 5, "alloc": 5,
         "mtime": 0, "class": "CANDIDATE", "reasons": ["no reference"],
         "family": "root:c", "canonical": None, "dependencies": []}])
    plan = cp.build_plan(repo, inv, copies={})
    md = cp.render_markdown(plan)
    assert "checkpoints/c.pt" in md and "executes nothing" in md.lower()

"""scripts/ckpt_census.py on synthetic trees (phase 2E).

Every case builds a small checkpoint tree plus the documents that
reference it, runs the read-only inventory, and checks the class and
the recorded reasons. Classes are conservative by construction: a
file is CANDIDATE only when nothing in the repository reaches it by
any form the scanner knows, and anything reached by a form the scanner
cannot resolve (dynamic path, bare basename in code, absent parent) is
UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ckpt_census as ci  # noqa: E402


def _w(p: Path, data: bytes | str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        p.write_text(data)
    else:
        p.write_bytes(data)
    return p


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def repo(tmp_path):
    """A fake repo: checkpoints/ plus the ledgers the scanner reads."""
    r = tmp_path / "repo"
    (r / "docs" / "handoffs").mkdir(parents=True)
    (r / "docs" / "superpowers" / "specs").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "scratch").mkdir()
    (r / "logs").mkdir()
    (r / "jobs").mkdir()
    _w(r / "docs/RESULTS.md", "")
    _w(r / "docs/FINDINGS.md", "")
    _w(r / "docs/BOARD.md", "")
    _w(r / "docs/EVIDENCE.md", "")
    _w(r / "docs/CODEMAP.md", "")
    _w(r / "README.md", "")
    _w(r / "REPRODUCE.md", "")
    _w(r / "docs/receipts.lock.json", json.dumps({"receipts": {}}))
    _w(r / "checkpoints/MANIFEST.jsonl", "")
    return r


def run(repo, tracked=()):
    return ci.build(repo, tracked=set(tracked), now=2_000_000_000.0)


def cls(inv, rel):
    return inv["classification"][rel]["class"]


def why(inv, rel):
    return " | ".join(inv["classification"][rel]["reasons"])


# ------------------------------------------------------------- citations

def test_directory_citation_in_results_freezes_everything_under_it(repo):
    _w(repo / "checkpoints/fam1/a.pt", b"a" * 100)
    _w(repo / "checkpoints/fam1/sub/b.pt", b"b" * 100)
    _w(repo / "checkpoints/other/c.pt", b"c" * 100)
    _w(repo / "docs/RESULTS.md", "weights under checkpoints/fam1/ were used.")
    inv = run(repo)
    assert cls(inv, "fam1/a.pt") == "FROZEN"
    assert cls(inv, "fam1/sub/b.pt") == "FROZEN"
    assert "RESULTS" in why(inv, "fam1/a.pt") and "dir" in why(inv, "fam1/a.pt")
    assert cls(inv, "other/c.pt") == "CANDIDATE"


def test_bare_checkpoints_token_does_not_freeze_the_tree(repo):
    _w(repo / "checkpoints/x.pt", b"x" * 100)
    _w(repo / "docs/RESULTS.md", "everything lives in checkpoints/ now")
    assert cls(run(repo), "x.pt") == "CANDIDATE"


def test_basename_and_stem_citations(repo):
    _w(repo / "checkpoints/deep/model_v7_final.pt", b"1" * 50)
    _w(repo / "checkpoints/deep/model_v8_final.pt", b"2" * 50)
    _w(repo / "docs/RESULTS.md",
       "booked from model_v7_final.pt; the v8 run named model_v8_final")
    inv = run(repo)
    assert cls(inv, "deep/model_v7_final.pt") == "FROZEN"
    assert "basename" in why(inv, "deep/model_v7_final.pt")
    assert cls(inv, "deep/model_v8_final.pt") == "FROZEN"
    assert "stem" in why(inv, "deep/model_v8_final.pt")


def test_short_stems_are_not_matched(repo):
    # a four-character stem would match ordinary prose words
    _w(repo / "checkpoints/sat.pt", b"s" * 10)
    _w(repo / "docs/RESULTS.md", "the sat solver was not used")
    assert cls(run(repo), "sat.pt") == "CANDIDATE"


def test_lock_and_evidence_table_freeze_with_sha_check(repo):
    data = b"L" * 64
    _w(repo / "checkpoints/locked.pt", data)
    _w(repo / "checkpoints/pinned.pt", data + b"!")
    _w(repo / "checkpoints/pinned_bad.pt", b"Q" * 10)
    _w(repo / "docs/receipts.lock.json", json.dumps({"receipts": {
        "checkpoints/locked.pt": {"exists": True, "sha256": _sha(data)}}}))
    _w(repo / "docs/EVIDENCE.md",
       "| path | status | sha256 |\n|---|---|---|\n"
       f"| `checkpoints/pinned.pt` | local-only | {_sha(data + b'!')} |\n"
       f"| `checkpoints/pinned_bad.pt` | local-only | {'0' * 64} |\n")
    inv = run(repo)
    assert cls(inv, "locked.pt") == "FROZEN" and "lock" in why(inv, "locked.pt")
    assert cls(inv, "pinned.pt") == "FROZEN"
    assert "EVIDENCE" in why(inv, "pinned.pt")
    assert cls(inv, "pinned_bad.pt") == "FROZEN"
    assert "SHA MISMATCH" in why(inv, "pinned_bad.pt")
    assert "checkpoints/pinned_bad.pt" in inv["summary"]["sha_mismatches"]


def test_manifest_confirmed_row_freezes(repo):
    data = b"M" * 20
    _w(repo / "checkpoints/confirmed/k.pt", data)
    _w(repo / "checkpoints/MANIFEST.jsonl", json.dumps(
        {"path": "confirmed/k.pt", "sha256": _sha(data), "bytes": 20,
         "verdict": "confirmed"}) + "\n")
    inv = run(repo)
    assert cls(inv, "confirmed/k.pt") == "FROZEN"
    assert "MANIFEST" in why(inv, "confirmed/k.pt")


# ------------------------------------------------------- code references

def test_full_path_literal_in_code_is_essential(repo):
    _w(repo / "checkpoints/needed.pt", b"n" * 10)
    _w(repo / "scripts/use.py", 'W = "checkpoints/needed.pt"\n')
    inv = run(repo)
    assert cls(inv, "needed.pt") == "ESSENTIAL"
    assert "scripts/use.py" in why(inv, "needed.pt")


def test_bare_basename_in_code_is_unknown_not_candidate(repo):
    _w(repo / "checkpoints/deep/maybe.pt", b"m" * 10)
    _w(repo / "scratch/old.py", 'name = "maybe.pt"\n')
    inv = run(repo)
    assert cls(inv, "deep/maybe.pt") == "UNKNOWN"
    assert "basename" in why(inv, "deep/maybe.pt")


def test_dynamic_path_prefix_in_code_makes_matches_unknown(repo):
    _w(repo / "checkpoints/run_s1.pt", b"1" * 10)
    _w(repo / "checkpoints/run_s2.pt", b"2" * 10)
    _w(repo / "checkpoints/unrelated.pt", b"3" * 10)
    _w(repo / "scratch/drv.py",
       'p = f"checkpoints/run_s{seed}.pt"\nq = "checkpoints/" + name\n'
       'r = os.path.join("checkpoints", name)\n')
    inv = run(repo)
    assert cls(inv, "run_s1.pt") == "UNKNOWN"
    assert "dynamic" in why(inv, "run_s1.pt")
    assert cls(inv, "run_s2.pt") == "UNKNOWN"
    # a generic loader ("checkpoints/" + name, os.path.join) reaches
    # every file by construction and says nothing about any one of
    # them: it is listed, not applied per file
    assert cls(inv, "unrelated.pt") == "CANDIDATE"
    kinds = {e["kind"] for e in inv["graph"]["unresolved"]}
    assert "dynamic-prefix" in kinds and "generic-loader" in kinds
    generic = [e for e in inv["graph"]["unresolved"] if e["kind"] == "generic-loader"]
    assert len(generic) == 1 and generic[0]["source"] == "scratch/drv.py"
    assert inv["summary"]["generic_loaders"] == 1


def test_log_mention_counts_only_when_the_log_is_locked_or_cited(repo):
    _w(repo / "checkpoints/by_locked.pt", b"1" * 10)
    _w(repo / "checkpoints/by_cited.pt", b"2" * 10)
    _w(repo / "checkpoints/by_exhaust.pt", b"3" * 10)
    _w(repo / "logs/a/run.log", "saved checkpoints/by_locked.pt\n")
    _w(repo / "logs/b/run.log", "saved checkpoints/by_cited.pt\n")
    _w(repo / "logs/c/run.log", "saved checkpoints/by_exhaust.pt\n")
    _w(repo / "docs/receipts.lock.json", json.dumps({"receipts": {
        "logs/a/run.log": {"exists": True, "sha256": "0" * 64}}}))
    _w(repo / "docs/RESULTS.md", "receipt logs/b/run.log booked")
    inv = run(repo)
    assert cls(inv, "by_locked.pt") == "FROZEN"
    assert "locked receipt" in why(inv, "by_locked.pt")
    assert cls(inv, "by_cited.pt") == "FROZEN"
    assert "cited receipt" in why(inv, "by_cited.pt")
    assert cls(inv, "by_exhaust.pt") == "UNKNOWN"
    assert "uncited log" in why(inv, "by_exhaust.pt")


def test_absent_cited_paths_are_reported_with_other_tree_presence(repo, tmp_path):
    _w(repo / "checkpoints/present.pt", b"p" * 10)
    other = tmp_path / "other" / "checkpoints"
    _w(other / "gone_dir" / "x.pt", b"x" * 10)
    _w(repo / "docs/RESULTS.md",
       "checkpoints/gone_dir/x.pt and checkpoints/never/ were cited")
    inv = ci.build(repo, tracked=set(), now=2_000_000_000.0,
                   extra_trees={"other": other})
    absent = {a["target"]: a for a in inv["absent_cited"]}
    assert "checkpoints/gone_dir/x.pt" in absent
    assert absent["checkpoints/gone_dir/x.pt"]["present_in"] == ["other"]
    assert absent["checkpoints/never"]["present_in"] == []
    assert all(a["sources"] for a in inv["absent_cited"])


def test_job_and_shell_files_are_scanned(repo):
    _w(repo / "checkpoints/j.pt", b"j" * 10)
    _w(repo / "jobs/run.cmd", "python train.py --init checkpoints/j.pt\n")
    assert cls(run(repo), "j.pt") == "ESSENTIAL"


# -------------------------------------------------------------- sidecars

def test_merge_sidecar_links_parents_and_inherits_frozen(repo):
    a = b"A" * 30
    b = b"B" * 30
    _w(repo / "checkpoints/pa.pt", a)
    _w(repo / "checkpoints/pb.pt", b)
    _w(repo / "checkpoints/avg.pt", b"avg" * 10)
    _w(repo / "checkpoints/avg.pt.merge.json", json.dumps({
        "op": "average", "out": "checkpoints/avg.pt",
        "inputs": [{"path": "checkpoints/pa.pt", "sha256": _sha(a)},
                   {"path": "checkpoints/pb.pt", "sha256": "f" * 64},
                   {"path": "checkpoints/gone.pt", "sha256": "e" * 64}]}))
    _w(repo / "docs/RESULTS.md", "VERDICT: checkpoints/avg.pt scored 0.5")
    inv = run(repo)
    assert cls(inv, "avg.pt") == "FROZEN"
    assert cls(inv, "avg.pt.merge.json") == "FROZEN"   # sidecar of frozen
    assert cls(inv, "pa.pt") == "FROZEN"               # verified parent
    assert "parent of" in why(inv, "pa.pt")
    assert cls(inv, "pb.pt") == "FROZEN"
    assert "SHA MISMATCH" in why(inv, "pb.pt")
    absent = [e for e in inv["graph"]["unresolved"] if e["kind"] == "absent-parent"]
    assert absent and absent[0]["target"] == "checkpoints/gone.pt"


def test_sha_sidecar_follows_its_primary(repo):
    data = b"W" * 40
    _w(repo / "checkpoints/v4/w.bin", data)
    _w(repo / "checkpoints/v4/w.sha", _sha(data) + "\n")
    _w(repo / "checkpoints/v4/orphan.sha", "0" * 64 + "\n")
    _w(repo / "docs/RESULTS.md", "weights checkpoints/v4/w.bin verified")
    inv = run(repo)
    assert cls(inv, "v4/w.bin") == "FROZEN"
    assert cls(inv, "v4/w.sha") == "FROZEN"
    assert "sidecar of" in why(inv, "v4/w.sha")
    assert cls(inv, "v4/orphan.sha") == "UNKNOWN"


def test_unparsable_json_sidecar_is_unknown(repo):
    _w(repo / "checkpoints/x.pt.merge.json", "{not json")
    assert cls(run(repo), "x.pt.merge.json") == "UNKNOWN"


# ------------------------------------------------------------ duplicates

def test_duplicate_of_a_frozen_file_names_the_canonical_copy(repo):
    data = b"D" * 100
    _w(repo / "checkpoints/keep/orig.pt", data)
    _w(repo / "checkpoints/copy.pt", data)
    _w(repo / "docs/RESULTS.md", "see checkpoints/keep/orig.pt")
    inv = run(repo)
    assert cls(inv, "keep/orig.pt") == "FROZEN"
    assert cls(inv, "copy.pt") == "DUPLICATE"
    row = inv["classification"]["copy.pt"]
    assert row["canonical"] == "checkpoints/keep/orig.pt"
    grp = next(g for g in inv["duplicates"] if g["sha256"] == _sha(data))
    assert grp["canonical"] == "checkpoints/keep/orig.pt"
    assert set(grp["paths"]) == {"checkpoints/keep/orig.pt", "checkpoints/copy.pt"}
    assert grp["physical"] in ("distinct", "clone-shared", "unknown")


def test_duplicate_under_a_dynamic_prefix_is_duplicate_with_dependency_named(repo):
    data = b"R" * 100
    _w(repo / "checkpoints/confirmed/fam/k_s1.pt", data)
    _w(repo / "checkpoints/k_s1.pt", data)
    _w(repo / "checkpoints/MANIFEST.jsonl", json.dumps(
        {"path": "confirmed/fam/k_s1.pt", "sha256": _sha(data), "bytes": 100,
         "verdict": "confirmed"}) + "\n")
    _w(repo / "scratch/gate.sh", "for f in checkpoints/k_s*.pt; do echo $f; done\n")
    inv = run(repo)
    row = inv["classification"]["k_s1.pt"]
    assert row["class"] == "DUPLICATE"
    assert row["canonical"] == "checkpoints/confirmed/fam/k_s1.pt"
    assert any("dynamic path prefix" in r for r in row["reasons"])
    assert row["dependencies"] == ["scratch/gate.sh"]


def test_family_lists_dynamic_sources_with_codemap_class(repo):
    _w(repo / "checkpoints/dyn_a1.pt", b"a" * 10)
    _w(repo / "scratch/sweep.sh", "ls checkpoints/dyn_a*.pt\n")
    _w(repo / "docs/CODEMAP.md",
       "| sweep | sweep.sh | results-cited | RESULTS | RESULTS×1 | — | — | — | — |\n")
    inv = run(repo)
    fam = next(f for f in inv["families"] if f["family"] == "root:dyn")
    assert fam["dynamic_sources"] == {"scratch/sweep.sh": "results-cited"}
    assert fam["classes"] == {"UNKNOWN": 1}


def test_two_frozen_duplicates_both_stay_frozen(repo):
    data = b"F" * 100
    _w(repo / "checkpoints/a/h1.pt", data)
    _w(repo / "checkpoints/b/h1.pt", data)
    _w(repo / "docs/RESULTS.md", "checkpoints/a/ and checkpoints/b/ both booked")
    inv = run(repo)
    assert cls(inv, "a/h1.pt") == "FROZEN" and cls(inv, "b/h1.pt") == "FROZEN"
    grp = next(g for g in inv["duplicates"] if g["sha256"] == _sha(data))
    assert grp["all_retained"] is True


def test_uncited_duplicates_keep_one_candidate_copy(repo):
    data = b"U" * 100
    _w(repo / "checkpoints/u1.pt", data)
    _w(repo / "checkpoints/u2.pt", data)
    inv = run(repo)
    classes = sorted(cls(inv, p) for p in ("u1.pt", "u2.pt"))
    assert classes == ["CANDIDATE", "DUPLICATE"]


def test_empty_files_are_never_grouped_as_duplicates(repo):
    _w(repo / "checkpoints/e1.pt", b"")
    _w(repo / "checkpoints/e2.pt", b"")
    inv = run(repo)
    assert not any(g["sha256"] == _sha(b"") for g in inv["duplicates"])


# ------------------------------------------------- symlinks, tracked, age

def test_symlinks_are_recorded_not_followed(repo):
    _w(repo / "checkpoints/real.pt", b"r" * 10)
    os.symlink(repo / "checkpoints/real.pt", repo / "checkpoints/link.pt")
    inv = run(repo)
    assert cls(inv, "link.pt") == "UNKNOWN"
    assert inv["inventory"]["link.pt"]["symlink"] is True
    assert "symlink" in why(inv, "link.pt")


def test_tracked_file_is_essential(repo):
    _w(repo / "checkpoints/t.pt", b"t" * 10)
    assert cls(run(repo, tracked={"checkpoints/t.pt"}), "t.pt") == "ESSENTIAL"


def test_recent_uncited_file_is_unknown(repo):
    p = _w(repo / "checkpoints/fresh.pt", b"f" * 10)
    os.utime(p, (2_000_000_000.0 - 3600, 2_000_000_000.0 - 3600))
    inv = run(repo)
    assert cls(inv, "fresh.pt") == "UNKNOWN"
    assert "recent" in why(inv, "fresh.pt")


def test_receipt_name_pattern_freezes(repo):
    _w(repo / "checkpoints/run_oomkilled.pt", b"o" * 10)
    assert cls(run(repo), "run_oomkilled.pt") == "FROZEN"


def test_handoff_or_spec_citation_is_essential_not_frozen(repo):
    _w(repo / "checkpoints/h.pt", b"h" * 10)
    _w(repo / "docs/handoffs/2026-01-01-0-x.md", "next: score checkpoints/h.pt")
    inv = run(repo)
    assert cls(inv, "h.pt") == "ESSENTIAL"
    assert "handoffs" in why(inv, "h.pt")


# ----------------------------------------------------- baseline / verify

def test_baseline_manifest_detects_change_and_loss(repo, tmp_path):
    _w(repo / "checkpoints/a.pt", b"a" * 10)
    _w(repo / "checkpoints/b.pt", b"b" * 10)
    base = ci.baseline(repo)
    assert ci.compare_baseline(base, ci.baseline(repo)) == {
        "changed": [], "missing": [], "added": []}
    (repo / "checkpoints/a.pt").write_bytes(b"A" * 10)
    (repo / "checkpoints/b.pt").unlink()
    _w(repo / "checkpoints/c.pt", b"c")
    d = ci.compare_baseline(base, ci.baseline(repo))
    assert d == {"changed": ["checkpoints/a.pt"], "missing": ["checkpoints/b.pt"],
                 "added": ["checkpoints/c.pt"]}


def test_every_file_is_classified_and_sheet_sums_match(repo):
    _w(repo / "checkpoints/fam/a.pt", b"a" * 10)
    _w(repo / "checkpoints/fam/b.pt", b"b" * 20)
    _w(repo / "checkpoints/root_x1.pt", b"x" * 30)
    inv = run(repo)
    assert set(inv["classification"]) == set(inv["inventory"])
    fam = {f["family"]: f for f in inv["families"]}
    assert fam["fam"]["files"] == 2 and fam["fam"]["bytes"] == 30
    assert sum(f["bytes"] for f in inv["families"]) == 60
    assert all(f["proposed_action"] for f in inv["families"])


def test_physical_probe_distinguishes_copy_from_clone(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(os.urandom(1 << 20))
    c = tmp_path / "c.bin"
    c.write_bytes(a.read_bytes())
    ea, ec = ci.extent_key(a), ci.extent_key(c)
    if ea is None:
        pytest.skip("F_LOG2PHYS_EXT unavailable on this filesystem")
    assert ea != ec
    import shutil
    import subprocess
    if shutil.which("cp") and sys.platform == "darwin":
        b = tmp_path / "b.bin"
        if subprocess.run(["cp", "-c", str(a), str(b)]).returncode == 0:
            assert ci.extent_key(b) == ea

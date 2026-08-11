"""Tests for scripts/log_hygiene.py — the print-only hygiene planner.

Pure stdlib + tmp trees; no torch/pyarrow/duckdb, no dependence on the
real logs/ dir, so these run anywhere.
"""

import importlib.util
import os
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "log_hygiene",
    Path(__file__).resolve().parent.parent / "scripts" / "log_hygiene.py")
lh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lh)

OLD = time.time() - 30 * 86400
NEW = time.time()


def make_repo(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "docs").mkdir()
    return tmp_path


def touch(root, rel, mtime=OLD):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    os.utime(p, (mtime, mtime))
    return p


def test_citation_set_parses_paths_and_strips_punct(tmp_path):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text(
        "see `logs/foo/bar.log`, and (logs/dir_a). also logs/x.jsonl.")
    c = lh.build_citation_set(r / "docs" / "RESULTS.md")
    assert "logs/foo/bar.log" in c
    assert "logs/dir_a" in c
    assert "logs/x.jsonl" in c


def test_classification_rules(tmp_path):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("cites logs/frozen.log and logs/dir_a")
    touch(r, "logs/frozen.log")
    touch(r, "logs/dir_a/inner.log")          # frozen via parent-dir citation
    touch(r, "logs/run_oomkilled.log")        # receipt beats sweepable
    touch(r, "logs/star_interrupted.txt")
    touch(r, "logs/poisoned_seed3.log")
    touch(r, "logs/job_oom.log")
    touch(r, "logs/old_junk.log", OLD)        # sweepable
    touch(r, "logs/recent.log", NEW)          # unknown
    cites = lh.build_citation_set(r / "docs" / "RESULTS.md")
    rows = {row["path"]: row["class"] for row in lh.scan(r, cites, 14.0)}
    assert rows["logs/frozen.log"] == lh.FROZEN
    assert rows["logs/dir_a/inner.log"] == lh.FROZEN
    for p in ("logs/run_oomkilled.log", "logs/star_interrupted.txt",
              "logs/poisoned_seed3.log", "logs/job_oom.log"):
        assert rows[p] == lh.RECEIPT
    assert rows["logs/old_junk.log"] == lh.SWEEPABLE
    assert rows["logs/recent.log"] == lh.UNKNOWN


def test_frozen_wins_over_sweepable(tmp_path):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("logs/ancient.log")
    touch(r, "logs/ancient.log", OLD)  # old AND cited -> FROZEN
    cites = lh.build_citation_set(r / "docs" / "RESULTS.md")
    rows = lh.scan(r, cites, 14.0)
    assert rows[0]["class"] == lh.FROZEN


def test_frozen_wins_over_receipt(tmp_path):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("logs/x_oomkilled.log")
    touch(r, "logs/x_oomkilled.log")
    cites = lh.build_citation_set(r / "docs" / "RESULTS.md")
    assert lh.scan(r, cites, 14.0)[0]["class"] == lh.FROZEN


def test_doubled_path_is_frozen_even_uncited_children(tmp_path):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("no citations here")
    touch(r, "logs/archive/logs/archive/ceiling_probe.log", OLD)
    cites = lh.build_citation_set(r / "docs" / "RESULTS.md")
    row = lh.scan(r, cites, 14.0)[0]
    assert row["class"] == lh.FROZEN
    assert "doubled path" in row["reason"]


def test_apply_refuses_without_env(tmp_path, capsys, monkeypatch):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("")
    touch(r, "logs/old_junk.log", OLD)
    monkeypatch.delenv("ARTIN_GO", raising=False)
    rc = lh.main(["--apply", "--root", str(r)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "refusing: apply requires ARTIN_GO=1" in out
    assert (r / "logs" / "old_junk.log").exists()  # nothing moved


def test_apply_refuses_with_frozen_rows_even_with_go(tmp_path, capsys, monkeypatch):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("logs/frozen.log")
    touch(r, "logs/frozen.log", OLD)
    monkeypatch.setenv("ARTIN_GO", "1")
    rc = lh.main(["--apply", "--root", str(r)])
    assert rc == 1
    assert "refusing" in capsys.readouterr().out
    assert (r / "logs" / "frozen.log").exists()


def test_apply_moves_sweepable_preserving_relpath(tmp_path, capsys, monkeypatch):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("")
    touch(r, "logs/battery/old.log", OLD)
    monkeypatch.setenv("ARTIN_GO", "1")
    rc = lh.main(["--apply", "--root", str(r)])
    assert rc == 0
    assert not (r / "logs" / "battery" / "old.log").exists()
    import datetime
    today = datetime.date.today().isoformat()
    assert (r / "logs" / "archive" / today / "battery" / "old.log").exists()


def test_plan_default_never_mutates(tmp_path, capsys):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("")
    touch(r, "logs/old_junk.log", OLD)
    before = sorted(str(p) for p in r.rglob("*"))
    rc = lh.main(["--root", str(r)])
    assert rc == 0
    assert sorted(str(p) for p in r.rglob("*")) == before


def test_consolidation_map_excludes_leancheck(tmp_path):
    r = make_repo(tmp_path)
    (r / "scripts").mkdir()
    (r / "scratch" / "leancheck").mkdir(parents=True)
    (r / "scripts" / "runner.py").write_text('open("out.DONE","w"); x = "job.rc"')
    (r / "scratch" / "probe.sh").write_text('touch "$D/x.marker"')
    (r / "scratch" / "leancheck" / "bad.py").write_text('"a.DONE"')
    (r / "scratch" / "sub").mkdir()  # nested scratch dirs are skipped
    cmap = lh.consolidation_map(r)
    files = {c["file"]: c["signals"] for c in cmap}
    assert files["scripts/runner.py"] == [".DONE", ".rc"]
    assert files["scratch/probe.sh"] == [".marker"]
    assert not any("leancheck" in f for f in files)


def test_json_output(tmp_path):
    r = make_repo(tmp_path)
    (r / "docs" / "RESULTS.md").write_text("")
    touch(r, "logs/a.log", OLD)
    out = tmp_path / "plan.json"
    lh.main(["--root", str(r), "--json", str(out)])
    import json
    d = json.loads(out.read_text())
    assert d["plan"][0]["path"] == "logs/a.log"
    assert "consolidation_map" in d

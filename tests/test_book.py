"""Tests for scripts/book.py — the refusing booking machine.

Runs against a tmp copy of RESULTS.md (path constants are
module-level exactly so these tests can redirect them); the live
docs/ tree is never touched. gen_results_index.py is pure stdlib
and fast, so the happy path calls the real frozen script.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import book  # noqa: E402

SEED_RESULTS = (
    "# RESULTS\n\n"
    "## VERDICT OLD-THING: something already booked (2026-08-01, mac)\n\n"
    "old body.\n"
)


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "RESULTS.md").write_text(SEED_RESULTS)
    monkeypatch.setattr(book, "RESULTS_PATH", docs / "RESULTS.md")
    monkeypatch.setattr(book, "INDEX_PATH", docs / "results-index.jsonl")
    monkeypatch.setattr(book, "INDEX_CWD", tmp_path)
    return tmp_path


def write_marker(tmp_path, **over):
    row = {"kind": "gate", "rc": 0, "git_sha": "abc1234",
           "argv": ["x.py"], "artifacts": [], "ts": 0}
    row.update(over)
    p = tmp_path / "run.marker.json"
    p.write_text(json.dumps(row) + "\n")
    return p


def write_entry(tmp_path, text):
    p = tmp_path / "entry.md"
    p.write_text(text)
    return p


def run(tmp_repo, marker, entry, entry_type="verdict", extra=()):
    return book.main([
        "--marker", str(marker), "--entry", str(entry),
        "--title", "NEW-THING: a fresh claim (2026-08-11, mac)",
        "--type", entry_type, *extra])


def test_missing_marker_refused(tmp_repo):
    entry = write_entry(tmp_repo, "body.\n")
    with pytest.raises(SystemExit) as ei:
        run(tmp_repo, tmp_repo / "nope.marker.json", entry)
    assert ei.value.code == 2
    assert (tmp_repo / "docs/RESULTS.md").read_text() == SEED_RESULTS


def test_killed_marker_refused(tmp_repo):
    m = write_marker(tmp_repo, rc="killed")
    entry = write_entry(tmp_repo, "body.\n")
    with pytest.raises(SystemExit):
        run(tmp_repo, m, entry)


def test_nonzero_rc_refused(tmp_repo):
    m = write_marker(tmp_repo, rc=1)
    entry = write_entry(tmp_repo, "body.\n")
    with pytest.raises(SystemExit):
        run(tmp_repo, m, entry)


def test_gate_dict_sum_mismatch_refused(tmp_repo):
    m = write_marker(tmp_repo, gate_dict={"L1": 20, "L2": 20})
    entry = write_entry(tmp_repo, "gate = 48/120 total.\n")
    with pytest.raises(SystemExit):
        run(tmp_repo, m, entry)


def test_gate_dict_sum_match_books(tmp_repo):
    m = write_marker(tmp_repo, gate_dict={"L1": 30, "L2": 18})
    entry = write_entry(tmp_repo, "gate = 48/120 total.\n")
    assert run(tmp_repo, m, entry, entry_type="prereg") == 0


def test_weights_sha_must_appear(tmp_repo):
    m = write_marker(tmp_repo, weights_sha="deadbeef01")
    entry = write_entry(tmp_repo, "no sha quoted here.\n")
    with pytest.raises(SystemExit):
        run(tmp_repo, m, entry)
    entry = write_entry(tmp_repo, "weights sha deadbeef01.\n")
    assert run(tmp_repo, m, entry, entry_type="prereg") == 0


def test_single_seed_subsigma_refused(tmp_repo):
    m = write_marker(tmp_repo, n_seeds=1)
    entry = write_entry(tmp_repo, "arm B wins, delta = +3 solves.\n")
    with pytest.raises(SystemExit):
        run(tmp_repo, m, entry)
    # flag alone is not enough — the fence sentence must be in the entry
    with pytest.raises(SystemExit):
        run(tmp_repo, m, entry, extra=["--fence-acknowledged"])


def test_single_seed_subsigma_fenced_books(tmp_repo):
    m = write_marker(tmp_repo, n_seeds=1)
    entry = write_entry(
        tmp_repo,
        "arm B delta = +3 solves. Single-seed reading below the "
        "1.5-sigma resolution bar; no direction claimed.\n")
    assert run(tmp_repo, m, entry, extra=["--fence-acknowledged"]) == 0


def test_multi_seed_subsigma_books_freely(tmp_repo):
    m = write_marker(tmp_repo, n_seeds=3)
    entry = write_entry(tmp_repo, "delta = +3 solves, n=3 paired.\n")
    assert run(tmp_repo, m, entry) == 0


def test_happy_path_heading_and_index_row(tmp_repo, capsys):
    m = write_marker(tmp_repo)
    entry = write_entry(tmp_repo, "the body of the verdict.\n")
    rc = run(tmp_repo, m, entry,
             extra=["--threads", "engine,alphabet",
                    "--links", "old-id"])
    assert rc == 0
    text = (tmp_repo / "docs/RESULTS.md").read_text()
    assert text.startswith(SEED_RESULTS)  # append-only
    assert ("## VERDICT NEW-THING: a fresh claim (2026-08-11, mac)"
            in text)
    rows = [json.loads(ln) for ln in
            (tmp_repo / "docs/results-index.jsonl")
            .read_text().splitlines()]
    assert len(rows) == 2
    new = rows[-1]
    assert new["date"] == "2026-08-11"
    assert new["type"] == "verdict"
    assert new["threads"] == ["engine", "alphabet"]
    assert new["links"] == ["old-id"]
    out = capsys.readouterr().out
    assert "git commit" in out  # prints, never runs, the commit


def test_dry_run_touches_nothing(tmp_repo, capsys):
    m = write_marker(tmp_repo)
    entry = write_entry(tmp_repo, "body.\n")
    assert run(tmp_repo, m, entry, extra=["--dry-run"]) == 0
    assert (tmp_repo / "docs/RESULTS.md").read_text() == SEED_RESULTS
    assert not (tmp_repo / "docs/results-index.jsonl").exists()
    assert "DRY RUN" in capsys.readouterr().out

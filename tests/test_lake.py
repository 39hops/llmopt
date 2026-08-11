"""Lake tests — tmp fixtures only, never the live data/lake/.

Skips cleanly when pyarrow/duckdb are absent (optional-deps rule)."""

import json

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from llmopt.lab import lake  # noqa: E402


def _mkjob(jobs, name, cmd="echo hi", rc="0", log=True):
    (jobs / f"{name}.cmd").write_text(cmd)
    (jobs / f"{name}.rc").write_text(rc)
    (jobs / f"{name}.pid").write_text("123")
    if log:
        (jobs / f"{name}.log").write_text("out\n")


def test_runs_killed_rc_stays_string(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _mkjob(jobs, "ok", rc="0")
    _mkjob(jobs, "dead", rc="killed")
    lake.build_runs(jobs_dir=jobs, lake_dir=tmp_path / "lake")
    rows = lake.query("SELECT * FROM runs ORDER BY run_id", lake_dir=tmp_path / "lake")
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["ok"]["rc"] == 0 and by_id["ok"]["rc_raw"] == "0"
    assert by_id["dead"]["rc"] is None and by_id["dead"]["rc_raw"] == "killed"
    assert all(r["source_grade"] == "exploration" for r in rows)
    assert by_id["ok"]["mtime"] is not None


def test_results_edges_explode_and_line_fragility(tmp_path):
    idx = tmp_path / "results-index.jsonl"
    recs = [
        {"id": "a", "date": "2026-08-01", "line": 10, "title": "A", "type": "verdict",
         "threads": ["t1", "t2"], "links": ["b", "c"], "amends": ["b"]},
        {"id": "b", "date": None, "line": 20, "title": "B", "type": "pre-reg",
         "threads": [], "superseded_by": ["a"]},
    ]
    idx.write_text("\n".join(json.dumps(r) for r in recs))
    lake.build_results(index_path=idx, lake_dir=tmp_path / "lake")
    res = lake.query("SELECT * FROM results ORDER BY id", lake_dir=tmp_path / "lake")
    assert [r["id"] for r in res] == ["a", "b"]
    assert res[0]["threads"] == ["t1", "t2"]
    assert all(r["source_grade"] == "ledger" for r in res)
    edges = lake.query(
        "SELECT * FROM result_edges ORDER BY src_id, edge_type, dst_id",
        lake_dir=tmp_path / "lake",
    )
    assert [(e["src_id"], e["edge_type"], e["dst_id"]) for e in edges] == [
        ("a", "amends", "b"),
        ("a", "links", "b"),
        ("a", "links", "c"),
        ("b", "superseded_by", "a"),
    ]


def test_models_absent_file_yields_empty_table(tmp_path):
    lake.build_models(catalog_path=tmp_path / "nope.jsonl", lake_dir=tmp_path / "lake")
    assert lake.query("SELECT count(*) AS n FROM models", lake_dir=tmp_path / "lake")[0]["n"] == 0


def test_models_present(tmp_path):
    cat = tmp_path / "models.jsonl"
    cat.write_text(json.dumps({"model_id": "m1", "params": 19000000}) + "\n")
    lake.build_models(catalog_path=cat, lake_dir=tmp_path / "lake")
    rows = lake.query("SELECT * FROM models", lake_dir=tmp_path / "lake")
    assert rows[0]["model_id"] == "m1"


def test_gates_refuse_missing_required(tmp_path):
    ld = tmp_path / "lake"
    with pytest.raises(ValueError, match="device"):
        lake.append_gate({"n_seeds": 3, "weights_sha": "abc"}, lake_dir=ld)
    with pytest.raises(ValueError, match="n_seeds"):
        lake.append_gate({"device": "mac", "weights_sha": "abc"}, lake_dir=ld)
    with pytest.raises(ValueError, match="weights_sha"):
        lake.append_gate({"device": "mac", "n_seeds": 3}, lake_dir=ld)
    # explicit None is refused too
    with pytest.raises(ValueError):
        lake.append_gate({"device": None, "n_seeds": 3, "weights_sha": "abc"}, lake_dir=ld)


def test_gates_append_and_roundtrip(tmp_path):
    ld = tmp_path / "lake"
    lake.append_gate(
        {"device": "mac", "n_seeds": 3, "weights_sha": "abc",
         "gate_dict": json.dumps({"solved": 34}), "total": 34},
        lake_dir=ld,
    )
    lake.append_gate({"device": "3080", "n_seeds": 5, "weights_sha": "def"}, lake_dir=ld)
    rows = lake.query(
        "SELECT device, count(*) AS n FROM gates GROUP BY device ORDER BY device",
        lake_dir=ld,
    )
    assert rows == [{"device": "3080", "n": 1}, {"device": "mac", "n": 1}]


def test_duckdb_join_roundtrip(tmp_path):
    ld = tmp_path / "lake"
    idx = tmp_path / "idx.jsonl"
    idx.write_text(json.dumps({"id": "x", "line": 1, "title": "X", "type": "verdict",
                               "threads": ["t"], "links": ["y"]}) + "\n"
                   + json.dumps({"id": "y", "line": 2, "title": "Y", "type": "verdict",
                                 "threads": []}) + "\n")
    lake.build_results(index_path=idx, lake_dir=ld)
    rows = lake.query(
        "SELECT r.title, e.dst_id FROM results r JOIN result_edges e ON r.id = e.src_id",
        lake_dir=ld,
    )
    assert rows == [{"title": "X", "dst_id": "y"}]


def test_gen_lake_cli(tmp_path):
    import subprocess, sys, pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(repo / "scripts/gen_lake.py"),
         "--tables", "gates", "--lake-dir", str(tmp_path / "lake")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "lake" / "gates.parquet").exists()


def test_string_amends_does_not_explode(tmp_path):
    """Legacy index rows carry amends as a bare string; must yield
    ONE edge, not one per character."""
    import json
    from llmopt.lab import lake
    idx = tmp_path / "results-index.jsonl"
    idx.write_text(json.dumps({
        "id": "a-child", "date": "2026-01-01", "line": 1,
        "title": "AMENDMENT X", "type": "amendment", "threads": [],
        "amends": "the-parent-entry"}) + "\n")
    lake.build_results(index_path=idx, lake_dir=tmp_path)
    edges = lake.query("SELECT * FROM result_edges", lake_dir=tmp_path)
    assert len(edges) == 1
    assert edges[0]["dst_id"] == "the-parent-entry"

"""llmopt.lab.runfiles — the run-artifact contract.

Pins the three ambiguities the module exists to close (handoff
2026-08-11-0): marker-absence is never success, "killed" rc never
throws, ckpt-without-.ep refuses instead of silently re-birthing.
"""
import pytest

from llmopt.lab.runfiles import (is_done, rc_of, read_marker,
                                 require_resume_marker, run_dir,
                                 write_marker)


def test_run_dir_exists_at_name_time(tmp_path):
    d = run_dir("myrun", root=tmp_path)
    assert d.is_dir()
    assert run_dir("myrun", root=tmp_path) == d  # idempotent


def test_marker_roundtrip_and_done(tmp_path):
    d = run_dir("r1", root=tmp_path)
    assert not is_done(d)          # absence != finished
    assert rc_of(d) is None
    p = write_marker(d, kind="gate", rc=0, wall_s=12.34,
                     artifacts=["logs/r1/gate.jsonl"],
                     gate_total=30)
    assert is_done(d)
    m = read_marker(d)
    assert m["kind"] == "gate" and m["rc"] == 0
    assert m["wall_s"] == 12.34 and m["gate_total"] == 30
    assert m["artifacts"] == ["logs/r1/gate.jsonl"]
    assert rc_of(d) == 0
    # single JSON line, machine-independent (no absolute paths)
    text = p.read_text()
    assert text.count("\n") == 1
    assert str(tmp_path) not in text


def test_killed_rc_never_throws(tmp_path):
    d = run_dir("r2", root=tmp_path)
    write_marker(d, kind="train", rc="killed")
    assert is_done(d)              # it DID end — dishonorably
    assert rc_of(d) is None        # but never int('killed')
    assert read_marker(d)["rc"] == "killed"


def test_corrupt_marker_is_not_done(tmp_path):
    d = run_dir("r3", root=tmp_path)
    (d / "run.marker.json").write_text("{truncated")
    assert not is_done(d)
    assert rc_of(d) is None


def test_resume_gate_fresh_birth(tmp_path):
    assert require_resume_marker(tmp_path / "new.pt") == 0


def test_resume_gate_refuses_orphan_ckpt(tmp_path):
    ckpt = tmp_path / "star.pt"
    ckpt.write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="refusing"):
        require_resume_marker(ckpt)   # the crown near-miss, fenced


def test_resume_gate_consistent_pair(tmp_path):
    ckpt = tmp_path / "star.pt"
    ckpt.write_bytes(b"weights")
    (tmp_path / "star.pt.ep").write_text("2")
    assert require_resume_marker(ckpt) == 3


def test_resume_gate_unparseable_marker(tmp_path):
    ckpt = tmp_path / "star.pt"
    ckpt.write_bytes(b"weights")
    (tmp_path / "star.pt.ep").write_text("-1\ngrown")
    with pytest.raises(ValueError):
        require_resume_marker(ckpt)

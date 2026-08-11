"""Tests for llmopt.lab.runlog — the per-step receipt writer.

Pure stdlib module; no optional-dep skips needed (torch is probed
inside runlog but degraded, never required). Everything runs in
tmp_path — no repo dirs touched."""
import json
import time

import pytest

from llmopt.lab.runfiles import read_marker
from llmopt.lab.runlog import FallbackCounters, RunLog


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_header_first_line(tmp_path):
    log = RunLog(tmp_path / "r" / "steps.jsonl", device="testdev")
    hdr = _rows(log.path)[0]
    assert hdr["header"] is True
    assert hdr["device"] == "testdev"
    assert set(hdr) >= {"git_sha", "argv", "ts"}
    log.close(0)


def test_stream_flush_readable_while_open(tmp_path):
    """Rows must hit disk BEFORE close — the outer-wall-kill class."""
    log = RunLog(tmp_path / "steps.jsonl")
    log.step(0, loss=1.5)
    log.step(1, loss=1.2)
    rows = _rows(log.path)  # file still open
    assert [r.get("step") for r in rows[1:]] == [0, 1]
    assert rows[1]["loss"] == 1.5
    log.close(0)


def test_wall_s_monotone_nonnegative(tmp_path):
    log = RunLog(tmp_path / "steps.jsonl")
    for i in range(3):
        time.sleep(0.01)
        log.step(i)
    log.close(0)
    walls = [r["wall_s"] for r in _rows(log.path)[1:]]
    assert all(w >= 0 for w in walls)
    assert all(w >= 0.005 for w in walls)  # the sleep is visible per-row


def test_abort_row(tmp_path):
    log = RunLog(tmp_path / "steps.jsonl")
    log.step(0)
    log.step(1)
    log.abort("budget")
    log.close(rc="killed")
    last = _rows(log.path)[-1]
    assert last == {"step": 2, "aborted": "budget"}


def test_abort_with_no_steps(tmp_path):
    log = RunLog(tmp_path / "steps.jsonl")
    log.abort("preflight")
    assert _rows(log.path)[-1]["step"] == 0
    log.close(1)


def test_digest_chain_reproducible_and_localizes(tmp_path):
    a = RunLog(tmp_path / "a.jsonl")
    b = RunLog(tmp_path / "b.jsonl")
    for i, (x, y) in enumerate([(b"aa", b"aa"), (b"bb", b"bb"),
                                (b"cc", b"CC"), (b"dd", b"dd")]):
        a.step(i, digest_bytes=x)
        b.step(i, digest_bytes=y)
    a.close(0)
    b.close(0)
    da = [r["digest"] for r in _rows(a.path)[1:]]
    db = [r["digest"] for r in _rows(b.path)[1:]]
    assert da[:2] == db[:2]          # identical prefix -> identical chain
    assert da[2] != db[2]            # divergence localized to step 2
    assert da[3] != db[3]            # cumulative: stays diverged


def test_fallback_counters_in_rows(tmp_path):
    log = RunLog(tmp_path / "steps.jsonl")
    log.step(0)                       # empty counters -> no fb key
    log.fallback_counters.bump("exact_miss")
    log.fallback_counters.bump("exact_miss")
    log.step(1)
    log.close(0)
    rows = _rows(log.path)
    assert "fb" not in rows[1]
    assert rows[2]["fb"] == {"exact_miss": 2}


def test_fallback_counters_missing_reads_zero():
    fb = FallbackCounters()
    assert fb["nope"] == 0
    assert fb.bump("x") == 1 and fb.bump("x", 2) == 3


def test_marker_written_on_close(tmp_path):
    d = tmp_path / "run"
    log = RunLog(d / "steps.jsonl", kind="gate")
    log.step(0)
    p = log.close(rc=0)
    m = read_marker(d)
    assert m is not None and m["kind"] == "gate" and m["rc"] == 0
    assert m["artifacts"] == ["steps.jsonl"]
    assert p.parent == d


def test_context_manager_exception_rc(tmp_path):
    d = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with RunLog(d / "steps.jsonl"):
            raise RuntimeError("boom")
    assert read_marker(d)["rc"] == "exception"

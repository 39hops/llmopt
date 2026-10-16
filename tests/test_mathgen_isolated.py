"""Subprocess-isolated problem generation (rehabilitation 2C, step 3).

The worker is a fresh CPU-only interpreter (never a fork of a parent
that may hold Torch/MPS/MLX state), one per MakerWorker, reused across
items. A hung maker is SIGKILLed and reported as MakerTimeout; a dead
worker is reported as MakerCrashed; neither is ever replaced by a
different sample. Successful output is identical to in-process
generation, pinned against tests/fixtures/mathgen_pins.json.
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sp = pytest.importorskip("sympy")

from llmopt.mathgen.digest import dataset_digest
from llmopt.mathgen.isolated import (MakerCrashed, MakerFailure,
                                     MakerTimeout, MakerWorker,
                                     make_problem_isolated)
from llmopt.mathgen.problems import ALL_KINDS, _resolve_maker, make_dataset

FIXTURES = Path(__file__).parent / "fixtures"
PINS = json.loads((FIXTURES / "mathgen_pins.json").read_text())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def fault_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(FIXTURES) + os.pathsep
                       + os.environ.get("PYTHONPATH", ""))


# --- success identity ---------------------------------------------------

@pytest.mark.parametrize("kind", ALL_KINDS)
def test_worker_output_matches_in_process(kind):
    with MakerWorker(wall=60) as w:
        for level in (1, 2, 3):
            a = w.make(kind, level, 17)
            b = _resolve_maker(kind)(level, 17)
            assert (a.kind, a.level, a.prompt, a.answer) == \
                (b.kind, b.level, b.prompt, b.answer)
            assert a.check(a.answer)


def test_lazy_kinds_roundtrip_and_self_verify():
    from llmopt.mathgen import linalg, mechanics, multivar, ntheory, odes, proofs
    kinds = [k for m in (linalg, odes, ntheory, multivar, mechanics, proofs)
             for k in m.MAKERS]
    with MakerWorker(wall=120) as w:
        for kind in kinds:
            p = w.make(kind, 1, 3)
            q = _resolve_maker(kind)(1, 3)
            assert (p.prompt, p.answer) == (q.prompt, q.answer), kind
            assert p.check(p.answer), kind


def test_make_dataset_through_worker_reproduces_pinned_digest():
    pin = next(d for d in PINS["datasets"] if d["name"] == "default-200-seed0")
    ds = make_dataset(200, seed=0, wall=60)
    assert dataset_digest(ds) == pin["sha256"]


def test_make_dataset_wall_preserves_order_and_retry_semantics():
    plain = make_dataset(30, kinds=("integrate", "taylor"), levels=(1, 2), seed=4)
    boxed = make_dataset(30, kinds=("integrate", "taylor"), levels=(1, 2),
                         seed=4, wall=60)
    assert [(p.kind, p.level, p.prompt, p.answer) for p in plain] == \
        [(p.kind, p.level, p.prompt, p.answer) for p in boxed]


def test_one_worker_process_serves_a_whole_dataset():
    with MakerWorker(wall=60) as w:
        assert w.pid is None  # lazy start
        w.make("differentiate", 1, 0)
        pid = w.pid
        for i in range(1, 10):
            w.make("differentiate", 1, i)
        assert w.pid == pid
        assert w.status()["pid"] == pid


# --- failure is typed, never a substituted sample ------------------------

def test_hung_maker_is_killed_and_reported(fault_path):
    with MakerWorker(wall=1.0) as w:
        w.make("differentiate", 1, 0)
        pid = w.pid
        t = time.monotonic()
        with pytest.raises(MakerTimeout) as info:
            w.make("mathgen_fault_makers:hang", 2, 5)
        assert time.monotonic() - t < 5.0
        err = info.value
        assert (err.kind, err.level, err.seed) == ("mathgen_fault_makers:hang", 2, 5)
        assert err.wall == 1.0
        assert isinstance(err, MakerFailure)
        assert not _alive(pid)
        # the worker respawns for the next request and keeps working
        p = w.make("differentiate", 1, 0)
        assert p.prompt == _resolve_maker("differentiate")(1, 0).prompt
        assert w.pid != pid


def test_crashed_worker_is_reported_with_exit_code(fault_path):
    with MakerWorker(wall=10) as w:
        with pytest.raises(MakerCrashed) as info:
            w.make("mathgen_fault_makers:crash", 1, 1)
        assert info.value.returncode == 3
        assert (info.value.kind, info.value.seed) == ("mathgen_fault_makers:crash", 1)


def test_maker_exception_is_reported_with_traceback_tail(fault_path):
    with MakerWorker(wall=10) as w:
        with pytest.raises(MakerFailure) as info:
            w.make("mathgen_fault_makers:raises", 2, 9)
        assert "maker refused level=2 seed=9" in str(info.value)
        # worker survives a maker exception
        assert w.make("differentiate", 1, 2).kind == "differentiate"


def test_make_dataset_propagates_timeout_instead_of_skipping(fault_path):
    with pytest.raises(MakerTimeout):
        make_dataset(3, kinds=("differentiate", "mathgen_fault_makers:hang"),
                     levels=(1,), wall=1.0)


def test_slow_but_finishing_maker_succeeds_inside_wall(fault_path):
    p = make_problem_isolated("mathgen_fault_makers:slow_then_ok", 1, 4, wall=10)
    assert p.prompt == _resolve_maker("differentiate")(1, 4).prompt


def test_unknown_kind_is_a_typed_failure():
    with MakerWorker(wall=10) as w:
        with pytest.raises(MakerFailure, match="no_such_kind"):
            w.make("no_such_kind", 1, 0)


# --- process hygiene ------------------------------------------------------

def test_worker_is_cpu_only_and_never_imports_torch():
    with MakerWorker(wall=30) as w:
        w.make("integrate", 2, 1)
        st = w.status()
        assert st["torch_loaded"] is False
        assert st["mlx_loaded"] is False


def test_close_reaps_the_worker():
    w = MakerWorker(wall=10)
    w.make("differentiate", 1, 0)
    pid = w.pid
    w.close()
    assert not _alive(pid)
    assert w.pid is None


def test_context_exit_reaps_a_hung_worker(fault_path):
    with MakerWorker(wall=30) as w:
        w.make("differentiate", 1, 0)
        pid = w.pid
        w.send_nowait("mathgen_fault_makers:hang", 1, 0)
        time.sleep(0.3)
        assert _alive(pid)
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_repeated_workers_leave_no_orphans():
    pids = []
    for _ in range(5):
        with MakerWorker(wall=30) as w:
            w.make("differentiate", 1, 0)
            pids.append(w.pid)
    time.sleep(0.2)
    assert not any(_alive(pid) for pid in pids)


def test_worker_exits_when_its_parent_is_killed(tmp_path, fault_path):
    """A hung worker whose parent dies must not linger as an orphan."""
    pidfile = tmp_path / "worker.pid"
    script = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from llmopt.mathgen.isolated import MakerWorker
        w = MakerWorker(wall=600)
        w.make("differentiate", 1, 0)
        open({str(pidfile)!r}, "w").write(str(w.pid))
        w.send_nowait("mathgen_fault_makers:hang", 1, 0)
        time.sleep(600)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script],
                              env=dict(os.environ))
    deadline = time.monotonic() + 30
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pidfile.exists(), "parent never started its worker"
    wpid = int(pidfile.read_text())
    time.sleep(0.3)
    assert _alive(wpid)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    deadline = time.monotonic() + 6
    while _alive(wpid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(wpid), "worker outlived its killed parent"


@pytest.mark.skipif(
    not (pytest.importorskip("torch").backends.mps.is_available()),
    reason="MPS not available")
def test_worker_starts_cleanly_from_a_parent_holding_mps_state():
    import torch
    x = torch.ones(8, device="mps") * 2  # parent holds initialised MPS state
    assert float(x.sum()) == 16.0
    with MakerWorker(wall=60) as w:
        p = w.make("integrate", 3, 2)
        assert p.prompt == _resolve_maker("integrate")(3, 2).prompt
        assert w.status()["torch_loaded"] is False

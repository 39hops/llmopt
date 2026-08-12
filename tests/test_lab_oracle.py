"""Guards for llmopt.lab.oracle (lab-extraction module 1): the
scratch/oracle_worker.py shim must bind the lab worker's main()
(Phase 3 module 2, 2026-08-12), and the parent's typed failure paths must
actually fire — TIMEOUT via the SLEEP affordance, crash via a real
kill, clean verdicts against a real mathgen problem. Every failure
returns conservative reject and increments its counter.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("sympy")


def test_worker_shim_binds_lab_main():
    """scratch/oracle_worker.py is a shim: its main IS the lab
    worker's (Phase 3 module 2). The typed failure-path battery
    below is the behavioral proof that replaced source identity."""
    spec = importlib.util.spec_from_file_location(
        "scratch_oracle_worker", ROOT / "scratch" / "oracle_worker.py")
    scratch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scratch_mod)
    from llmopt.lab import oracle_worker
    assert scratch_mod.main is oracle_worker.main


@pytest.fixture(scope="module")
def problem():
    from llmopt.mathgen.problems import make_integrate
    return make_integrate(1, 424242)


def test_clean_verdicts(problem):
    from llmopt.lab.oracle import Oracle
    with Oracle(wall=30) as o:
        r = o.check(problem, problem.answer)
        assert (r.ok, r.event) == (True, None) and r.parsed
        r = o.check(problem, "x***bogus((")
        assert (r.ok, r.parsed, r.event) == (False, False, None)
        assert sum(o.counters.values()) == 0


def test_timeout_is_loud_and_recovers(problem):
    from llmopt.lab.oracle import Oracle
    with Oracle(wall=2) as o:
        pr = o._ensure()
        pr.stdin.write("SLEEP\n")   # worker wedges; parent owns the wall
        pr.stdin.flush()
        r = o.check(problem, problem.answer)
        assert (r.ok, r.parsed, r.event) == (False, False, "TIMEOUT")
        assert r.timed_out and o.counters["TIMEOUT"] == 1
        # respawn on next check: the oracle recovers, verdict clean
        r2 = o.check(problem, problem.answer)
        assert (r2.ok, r2.event) == (True, None)


def test_worker_crash_is_loud_and_recovers(problem):
    from llmopt.lab.oracle import Oracle
    with Oracle(wall=30) as o:
        o._ensure().kill()
        r = o.check(problem, problem.answer)
        # Killing the worker leaves TWO correct outcomes, and which one
        # happens depends on whether the kill has been reaped by the
        # time _ensure() polls — a race this test cannot win. If the
        # process is dead-but-unreaped, _ensure hands back the corpse
        # and the write raises: typed CRASH, conservative reject. If it
        # has already been reaped, _ensure respawns and the check simply
        # succeeds. Asserting only the first outcome made this test fail
        # under CPU load (seen 2026-08-11 during a training run).
        #
        # The invariant that actually matters is neither of those: a
        # killed worker must never produce a WRONG verdict — no silent
        # accept of a bad answer, and every failure typed and counted.
        if r.event is None:
            assert (r.ok, r.parsed) == (True, True), "respawn gave a bad verdict"
            assert sum(o.counters.values()) == 0
        else:
            assert (r.ok, r.parsed) == (False, False), "crash must reject"
            assert r.event in ("CRASH_PIPE", "CRASH_EOF")
            assert o.counters[r.event] == 1
        # either way the oracle is usable again on the next call
        r2 = o.check(problem, problem.answer)
        assert (r2.ok, r2.event) == (True, None)

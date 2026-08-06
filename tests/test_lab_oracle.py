"""Guards for llmopt.lab.oracle (lab-extraction module 1): the worker's
main() must stay character-identical to the frozen
scratch/oracle_worker.py, and the parent's typed failure paths must
actually fire — TIMEOUT via the SLEEP affordance, crash via a real
kill, clean verdicts against a real mathgen problem. Every failure
returns conservative reject and increments its counter.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("sympy")


def test_worker_main_identical_to_frozen_script():
    spec = importlib.util.spec_from_file_location(
        "scratch_oracle_worker", ROOT / "scratch" / "oracle_worker.py")
    scratch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scratch_mod)
    from llmopt.lab import oracle_worker
    assert inspect.getsource(oracle_worker.main) == \
        inspect.getsource(scratch_mod.main)


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
        assert (r.ok, r.parsed) == (False, False)
        assert r.event in ("CRASH_PIPE", "CRASH_EOF")
        assert o.counters[r.event] == 1
        r2 = o.check(problem, problem.answer)
        assert (r2.ok, r2.event) == (True, None)

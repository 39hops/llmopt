"""The torch process-state guard in tests/conftest.py must restore the
determinism flag after a test AND report the test that changed it; a
silent restore would hide a leaking test forever. The report is a
teardown ERROR on the leaking test (its own assertions passed; the
state it left did not). Order-dependent regressions are reproduced
with pytester in a fresh session.
"""
from __future__ import annotations

import textwrap

import pytest

pytest_plugins = ["pytester"]
torch = pytest.importorskip("torch")


def _conftest_source() -> str:
    from pathlib import Path
    src = (Path(__file__).resolve().parent / "conftest.py").read_text()
    start = src.index("def torch_state_snapshot")
    return src[start:]


def test_restores_flag_and_reports_the_leaker(pytester):
    pytester.makeconftest("import pytest\nimport torch\n" + _conftest_source())
    pytester.makepyfile(test_leak=textwrap.dedent("""
        import torch
        def test_leaker():
            torch.use_deterministic_algorithms(True)
        def test_after():
            assert torch.are_deterministic_algorithms_enabled() is False
    """))
    res = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    # the leaker passes its own body but errors at teardown with the
    # leak named; the next test sees clean state
    res.assert_outcomes(passed=2, errors=1)
    res.stdout.fnmatch_lines(["*test_leaker*left torch.use_deterministic_algorithms*"])


def test_marked_leaker_is_allowed_but_still_restored(pytester):
    pytester.makeconftest("import pytest\nimport torch\n" + _conftest_source())
    pytester.makepyfile(test_ok=textwrap.dedent("""
        import pytest, torch
        @pytest.mark.changes_torch_state
        def test_declared():
            torch.use_deterministic_algorithms(True)
        def test_after():
            assert torch.are_deterministic_algorithms_enabled() is False
    """))
    res = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    res.assert_outcomes(passed=2)


def test_thread_count_change_is_reported_not_reset(pytester):
    pytester.makeconftest("import pytest\nimport torch\n" + _conftest_source())
    pytester.makepyfile(test_thr=textwrap.dedent("""
        import torch
        def test_threads():
            torch.set_num_threads(max(1, torch.get_num_threads() - 1) or 1)
    """))
    res = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    res.assert_outcomes(passed=1, errors=1)
    res.stdout.fnmatch_lines(["*left torch.get_num_threads*"])


def test_cuda_bincount_after_a_leaker_in_the_same_session(pytester):
    """The 3080 failure shape: a deterministic-mode leaker followed by a
    CUDA kernel without a deterministic implementation. Runs the CUDA
    half only where CUDA exists; the ordering guard is checked on CPU."""
    pytester.makeconftest("import pytest\nimport torch\n" + _conftest_source())
    pytester.makepyfile(test_order=textwrap.dedent("""
        import pytest, torch
        def test_leaker():
            torch.use_deterministic_algorithms(True)
        def test_bincount():
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            x = torch.randint(0, 8, (64,), device=dev)
            torch.bincount(x, minlength=8)
    """))
    res = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    res.assert_outcomes(passed=2, errors=1)   # only the leaker is reported

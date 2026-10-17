"""Shared test helpers.

artifact_or_skip: artifact-gated tests skip where a big untracked
artifact is absent (file-handoff convention), but under LLMOPT_FULL=1
absence is a FAILURE — so "skipped" on the artifact machine can never
read as "verified" (spec 2026-08-12 Phase 6.2).
"""
import os

import pytest


def artifact_or_skip(exists: bool, reason: str) -> None:
    if exists:
        return
    if os.environ.get("LLMOPT_FULL"):
        pytest.fail(f"LLMOPT_FULL=1 but artifact missing: {reason}")
    pytest.skip(reason)


# ---------------------------------------------------------------- topology
#
# absent_repair_sentinel: two frozen scratch drivers resolve the `repair`
# worktree role at MODULE IMPORT (scratch/update_geometry_census.py:115,
# scratch/dfa_act.py:50) although only their real-mode runs read anything
# there. Tests that import them for pure helpers, constants or SMOKE
# semantics must not require a sibling checkout to exist. Where the real
# repair topology IS present (the lab's Mac), this fixture does nothing
# and the drivers resolve the real worktree. Where it is absent (GitHub
# CI, a fresh clone), it points LLMOPT_WORKTREE_REPAIR at a NONEXISTENT
# sentinel directory for the duration of the test module, so the import
# succeeds and any actual READ of a repair artifact fails loudly
# (FileNotFoundError under .../absent-repair-worktree) rather than
# silently reading main. llmopt.lab.locator itself is unchanged and keeps
# failing closed; this is opt-in per test module (pytestmark), never
# autouse. Modules cached in sys.modules keep the sentinel paths for the
# rest of the process; no test consumes them.

def repair_topology_absent() -> bool:
    from llmopt.lab import locator
    try:
        locator.worktree("repair")
        return False
    except RuntimeError:
        return True


@pytest.fixture(scope="module")
def absent_repair_sentinel(tmp_path_factory):
    if not repair_topology_absent():
        yield None
        return
    sentinel = tmp_path_factory.mktemp("topology") / "absent-repair-worktree"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLMOPT_WORKTREE_REPAIR", str(sentinel))
        yield sentinel



def torch_state_snapshot():
    """Process-wide torch settings a test can leak. Interop threads and
    allocator settings cannot be changed after initialization and are
    read only; the determinism flag is safe to toggle at any time; the
    intra-op thread count is reported but never reset (OpenMP pools
    should not be resized after parallel work started)."""
    import torch
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "num_threads": torch.get_num_threads(),
        "grad_enabled": torch.is_grad_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
    }


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "changes_torch_state: the test deliberately changes "
        "process-wide torch settings; the guard restores without failing")


@pytest.fixture(autouse=True)
def _torch_process_state_guard(request):
    """Snapshot torch's process-wide state before each test, restore the
    safely restorable part after it, and FAIL the test that changed
    anything unless it carries the changes_torch_state marker. A silent
    restore would hide a leaking test forever; the 3080 found one that
    way (deterministic mode reaching a CUDA bincount)."""
    try:
        import torch
    except Exception:
        yield
        return
    before = torch_state_snapshot()
    yield
    after = torch_state_snapshot()
    diffs = [f"{k} {before[k]!r} -> {after[k]!r}" for k in before if before[k] != after[k]]
    if before["deterministic"] != after["deterministic"] or before["warn_only"] != after["warn_only"]:
        torch.use_deterministic_algorithms(before["deterministic"], warn_only=before["warn_only"])
    if before["grad_enabled"] != after["grad_enabled"]:
        torch.set_grad_enabled(before["grad_enabled"])
    if before["default_dtype"] != after["default_dtype"]:
        torch.set_default_dtype(getattr(torch, before["default_dtype"].split(".")[-1]))
    if diffs and request.node.get_closest_marker("changes_torch_state") is None:
        names = {d.split(" ")[0] for d in diffs}
        what = []
        if "deterministic" in names or "warn_only" in names:
            what.append("torch.use_deterministic_algorithms")
        if "num_threads" in names:
            what.append("torch.get_num_threads")
        if "grad_enabled" in names:
            what.append("torch.is_grad_enabled")
        if "default_dtype" in names:
            what.append("torch.get_default_dtype")
        pytest.fail(f"{request.node.nodeid} left {', '.join(what)} changed: "
                    + "; ".join(diffs) + " (restored where safe; mark the test "
                    "changes_torch_state if deliberate)")

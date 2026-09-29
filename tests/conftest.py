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

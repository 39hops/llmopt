"""llmopt.lab.locator guards: environment override wins, git-worktree
resolution finds main and (when present) repair, the locator dict round
trips through resolve, absolute relative_paths and unknown roles are
refused, and repo_relative strips a role's root."""
import os
from pathlib import Path

import pytest

from llmopt.lab import locator as L

ROOT = Path(__file__).resolve().parents[1]


def test_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMOPT_WORKTREE_REPAIR", str(tmp_path))
    assert L.worktree("repair") == tmp_path.resolve()
    monkeypatch.setenv("AXIOM_DIR", str(tmp_path / "ax"))
    assert L.worktree("axiom") == (tmp_path / "ax").resolve()


def test_main_resolves_to_this_checkout(monkeypatch):
    monkeypatch.delenv("LLMOPT_WORKTREE_MAIN", raising=False)
    assert L.worktree("main") == ROOT


def test_locator_round_trip(monkeypatch):
    monkeypatch.setenv("LLMOPT_WORKTREE_REPAIR", "/tmp/x-repair")
    loc = L.locator("repair", "checkpoints/a/b.pt", commit="ec6de1ae", digest="abc")
    assert loc == {"worktree_role": "repair", "relative_path": "checkpoints/a/b.pt", "commit": "ec6de1ae", "digest": "abc"}
    assert L.resolve(loc) == Path("/tmp/x-repair/checkpoints/a/b.pt").resolve()
    assert L.resolve("README.md") == ROOT / "README.md"


def test_refusals():
    with pytest.raises(KeyError):
        L.worktree("nope")
    with pytest.raises(ValueError):
        L.locator("main", "/abs/path")


def test_repo_relative_and_role_of(monkeypatch):
    monkeypatch.delenv("LLMOPT_WORKTREE_MAIN", raising=False)
    assert L.repo_relative(ROOT / "scripts" / "liverun.py") == os.path.join("scripts", "liverun.py")
    assert L.role_of(ROOT / "docs") == "main"

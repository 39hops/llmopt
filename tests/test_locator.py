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


@pytest.mark.parametrize("bad", ["/abs/path", "//host/share/x", "C:\\Users\\a\\x.pt", "C:/Users/a/x.pt", "c:x.pt", "\\\\host\\share\\x",
                                 "../escape", "a/../../escape", "a/b/../../../c", "", ".", "./", "a\\b"])
def test_locator_fails_closed_on_path_form(bad):
    with pytest.raises(ValueError):
        L.locator("main", bad)


def test_valid_nested_relative_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMOPT_WORKTREE_REPAIR", str(tmp_path))
    (tmp_path / "checkpoints" / "a").mkdir(parents=True)
    (tmp_path / "checkpoints" / "a" / "b.pt").write_bytes(b"x")
    loc = L.locator("repair", "./checkpoints//a/b.pt")
    assert loc["relative_path"] == "checkpoints/a/b.pt"
    assert L.resolve(loc) == tmp_path / "checkpoints" / "a" / "b.pt"


def test_resolve_refuses_escape_through_symlink(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMOPT_WORKTREE_REPAIR", str(tmp_path / "root"))
    (tmp_path / "root").mkdir(); (tmp_path / "outside").mkdir()
    (tmp_path / "root" / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        L.resolve({"worktree_role": "repair", "relative_path": "link/x"})
    with pytest.raises(ValueError):
        L.resolve({"worktree_role": "repair", "relative_path": "../outside/x"})


def test_repo_relative_wrong_role_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMOPT_WORKTREE_REPAIR", str(tmp_path))
    with pytest.raises(ValueError):
        L.repo_relative(ROOT / "scripts" / "liverun.py", role="repair")


def test_repo_relative_and_role_of(monkeypatch):
    monkeypatch.delenv("LLMOPT_WORKTREE_MAIN", raising=False)
    assert L.repo_relative(ROOT / "scripts" / "liverun.py") == os.path.join("scripts", "liverun.py")
    assert L.role_of(ROOT / "docs") == "main"

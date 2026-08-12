"""repo_root: checkout resolution + honest wheel failure (7a)."""
import pytest

from llmopt.common import repo


def test_repo_root_finds_checkout():
    root = repo.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "docs" / "figures.json").is_file()


def test_repo_root_raises_outside_checkout(monkeypatch, tmp_path):
    fake = tmp_path / "site-packages" / "llmopt" / "common" / "repo.py"
    fake.parent.mkdir(parents=True)
    fake.touch()
    monkeypatch.setattr(repo, "__file__", str(fake))
    with pytest.raises(RuntimeError, match="source checkout"):
        repo.repo_root()

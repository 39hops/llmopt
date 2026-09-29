"""CI topology: a fresh checkout has no sibling `*-repair` worktree and
no LLMOPT_WORKTREE_REPAIR. Two frozen drivers resolve that worktree at
module import (scratch/update_geometry_census.py:115 NULL_PATHS,
scratch/dfa_act.py:50 REPAIR) although only their real-mode runs read
anything there; every test that imports them, directly or through the
FME / FMEL / FMEL2 / RDC and fb_gate / sg_failure_desk chains, errored
at fixture setup in GitHub CI on 2026-09-29.

The repair boundary is test-side (both driver bodies are sha-pinned
evidence): the `absent_repair_sentinel` fixture in tests/conftest.py
points the repair role at a NONEXISTENT sentinel path, only when the
real topology is absent, so pure imports succeed and any actual
consumption fails loudly (FileNotFoundError under a directory named
absent-repair-worktree), never falling back to main. The locator itself
is unchanged and keeps failing closed.

The CI topology is simulated here with a `git` shim on PATH whose
`worktree list --porcelain` reports only the main worktree.
"""
from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCATOR_MSG = "locator: no worktree named *-repair; set LLMOPT_WORKTREE_REPAIR"


def git_shim(tmp_path: Path) -> Path:
    """A PATH directory whose `git` reports a main-only worktree list
    and delegates every other command to the real git."""
    real = shlex.quote(shutil.which("git") or "")
    assert real not in ("", "''")
    d = tmp_path / "shim"
    d.mkdir()
    sh = d / "git"
    sh.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "worktree" ] && [ "$2" = "list" ]; then\n'
        f'  printf "worktree %s\\nHEAD 0000000000000000000000000000000000000000\\nbranch refs/heads/main\\n\\n" "$({real} rev-parse --show-toplevel)"\n'
        "  exit 0\nfi\n"
        f'exec {real} "$@"\n')
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
    return d


def ci_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{git_shim(tmp_path)}:{env['PATH']}"
    env.pop("LLMOPT_WORKTREE_REPAIR", None)
    env.pop("SMOKE", None)  # CI sets no SMOKE; real-mode constants
    env.update(extra)
    return env


def _import(mod: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    code = (f"import sys; sys.path[:0]=['.','scripts','scratch']; import {mod} as M; "
            "print(getattr(M, 'NULL_PATHS', getattr(M, 'REPAIR', None)))")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


# ------------------------------------------------ the coupling, reproduced

@pytest.mark.parametrize("mod", ["update_geometry_census", "dfa_act"])
def test_driver_import_resolves_repair_worktree_at_import_time(tmp_path, mod):
    """Characterizes the frozen bodies: without the repair topology the
    bare import raises the locator error (this is what CI saw). If this
    test ever fails, the import-time coupling is gone: retire the
    absent_repair_sentinel fixture and the pytestmarks that use it."""
    r = _import(mod, ci_env(tmp_path))
    assert r.returncode != 0
    assert LOCATOR_MSG in r.stderr


@pytest.mark.parametrize("mod", ["update_geometry_census", "dfa_act"])
def test_sentinel_lets_the_pure_import_succeed(tmp_path, mod):
    sentinel = tmp_path / "absent-repair-worktree"
    r = _import(mod, ci_env(tmp_path, LLMOPT_WORKTREE_REPAIR=str(sentinel)))
    assert r.returncode == 0, r.stderr[-1500:]
    assert "absent-repair-worktree" in r.stdout
    assert not sentinel.exists()  # nothing was created there


# ------------------------------------------------- the locator stays closed

def test_repair_role_still_fails_closed_without_topology(tmp_path, monkeypatch):
    from llmopt.lab import locator as L
    monkeypatch.setenv("PATH", f"{git_shim(tmp_path)}:{os.environ['PATH']}")
    monkeypatch.delenv("LLMOPT_WORKTREE_REPAIR", raising=False)
    with pytest.raises(RuntimeError, match=r"no worktree named \*-repair"):
        L.worktree("repair")


def test_consuming_a_repair_artifact_under_the_sentinel_fails_loudly(tmp_path, monkeypatch):
    from llmopt.lab import locator as L
    sentinel = tmp_path / "absent-repair-worktree"
    monkeypatch.setenv("LLMOPT_WORKTREE_REPAIR", str(sentinel))
    p = L.resolve({"worktree_role": "repair",
                   "relative_path": "checkpoints/atomtraj1/stock_s7/step_15420.pt"})
    assert str(p).startswith(str(sentinel.resolve()))
    with pytest.raises(FileNotFoundError):
        p.read_bytes()


# ---------------------------------------- the fixture, on the affected files

@pytest.mark.parametrize("test_file", [
    "tests/test_update_geometry_census.py",
    "tests/test_sg_failure_desk_oracles.py",
])
def test_pure_test_modules_pass_without_repair_topology(tmp_path, test_file):
    """Representative members of the two failing families run green on
    the simulated CI topology (the remaining eight are exercised by the
    same fixture; the qualification run covers all ten)."""
    r = subprocess.run([sys.executable, "-m", "pytest", test_file, "-q",
                        "-p", "no:cacheprovider"], cwd=ROOT,
                       env=ci_env(tmp_path), capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, (r.stdout[-3000:] + r.stderr[-1500:])
    assert LOCATOR_MSG not in r.stdout + r.stderr

"""lab.locator — logical artifact locators instead of machine-local paths.

A tracked instrument, pre-reg or receipt names an artifact by
    {"worktree_role": <role>, "relative_path": <repo-relative path>,
     "commit": <sha or None>, "digest": <sha256 / state digest or None>}
and never by an absolute home path (/Users/<user>/..., /home/<user>/...,
C:\\Users\\<user>\\...). Roles known to the lab:

    main    the primary checkout (this repository's main worktree)
    repair  the detached repair worktree of ATOM-DIET-TRAJECTORY-1-REPAIR
            (RESULTS L67274), a sibling checkout named <main>-repair
    axiom   the sibling axiom repository (read-only courtesy checkout)

Resolution order for a role's root, at runtime only:
    1. environment: LLMOPT_WORKTREE_<ROLE> (e.g. LLMOPT_WORKTREE_REPAIR),
       and for axiom also AXIOM_DIR;
    2. `git worktree list --porcelain` run from this repository: main =
       the first (primary) worktree, repair = the worktree whose directory
       name ends with "-repair";
    3. axiom: the directory named "axiom" beside the main worktree.
Nothing here is a literal path; the resolved path is never written into
a tracked receipt by this module (callers emit the locator dict and, if
they must record where it resolved, do so under a key that the path-
hygiene lint treats as runtime-only).

tests/test_locator.py covers the environment override, the git-worktree
resolution on this checkout, the locator round trip and the refusal on an
unknown role. tests/test_path_hygiene.py is the lint that forbids new
machine-local paths in tracked files.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROLES = ("main", "repair", "axiom")
_HERE = Path(__file__).resolve().parents[2]


def _worktrees(cwd: Path | None = None) -> list[Path]:
    r = subprocess.run(["git", "worktree", "list", "--porcelain"], capture_output=True, text=True, cwd=str(cwd or _HERE))
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("worktree "):
            out.append(Path(line[len("worktree "):]).resolve())
    return out


def worktree(role: str, cwd: Path | None = None) -> Path:
    """Resolve a role to a directory (environment first, then git, then the axiom sibling rule)."""
    if role not in ROLES:
        raise KeyError(f"unknown worktree role {role!r}; known: {ROLES}")
    env = os.environ.get(f"LLMOPT_WORKTREE_{role.upper()}") or (os.environ.get("AXIOM_DIR") if role == "axiom" else None)
    if env:
        return Path(env).resolve()
    wts = _worktrees(cwd)
    if not wts:
        raise RuntimeError("locator: not inside a git checkout and no LLMOPT_WORKTREE_* variable set")
    if role == "main":
        return wts[0]
    if role == "repair":
        for w in wts:
            if w.name.endswith("-repair"):
                return w
        raise RuntimeError("locator: no worktree named *-repair; set LLMOPT_WORKTREE_REPAIR")
    return wts[0].parent / "axiom"


def locator(role: str, relative_path: str, commit: str | None = None, digest: str | None = None) -> dict:
    """The tracked form of an artifact reference."""
    if role not in ROLES:
        raise KeyError(role)
    rp = str(relative_path)
    if Path(rp).is_absolute():
        raise ValueError(f"locator relative_path must be repo-relative, got {rp!r}")
    return {"worktree_role": role, "relative_path": rp, "commit": commit, "digest": digest}


def resolve(loc: dict | str, cwd: Path | None = None) -> Path:
    """A locator dict (or a plain repo-relative string, role main) to a runtime path."""
    if isinstance(loc, str):
        return worktree("main", cwd) / loc
    return worktree(loc["worktree_role"], cwd) / loc["relative_path"]


def repo_relative(path: str | Path, role: str = "main", cwd: Path | None = None) -> str:
    """A runtime path expressed relative to a role's root (for receipts that must name where a run sat)."""
    p = Path(path).resolve()
    root = worktree(role, cwd)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p.name)


def role_of(path: str | Path, cwd: Path | None = None) -> str | None:
    """Which known role's root contains the path, if any."""
    p = Path(path).resolve()
    for role in ROLES:
        try:
            root = worktree(role, cwd)
        except (RuntimeError, KeyError):
            continue
        if p == root or root in p.parents:
            return role
    return None

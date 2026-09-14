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
resolution on this checkout, the locator round trip, the fail-closed path
form (POSIX / Windows / UNC absolute, traversal, empty), the resolve
escape refusal and the wrong-role repo_relative refusal. tests/test_path_hygiene.py is the lint that forbids new
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


def _check_relative(relative_path: str) -> str:
    """Fail closed: a locator names an artifact by a plain repo-relative path.
    Rejected: POSIX absolute (/x), Windows drive-qualified (C:\\x, C:/x, C:x),
    UNC / network forms (\\\\host\\share, //host/share), any '..' component
    (including nested a/../../x), empty or '.' paths, and backslashes (a
    tracked locator is written in POSIX form)."""
    rp = str(relative_path)
    if rp in ("", ".", "./"):
        raise ValueError("locator relative_path must name an artifact (empty or '.' is a worktree, not an artifact; use worktree(role))")
    if "\\" in rp:
        raise ValueError(f"locator relative_path must be POSIX-form (no backslashes): {rp!r}")
    if rp.startswith("/") or rp.startswith("//"):
        raise ValueError(f"locator relative_path must be repo-relative, got absolute {rp!r}")
    if len(rp) >= 2 and rp[0].isalpha() and rp[1] == ":":
        raise ValueError(f"locator relative_path must be repo-relative, got drive-qualified {rp!r}")
    parts = [x for x in rp.split("/") if x not in ("", ".")]
    if any(x == ".." for x in parts):
        raise ValueError(f"locator relative_path may not traverse upward: {rp!r}")
    if not parts:
        raise ValueError(f"locator relative_path names nothing: {rp!r}")
    return "/".join(parts)


def locator(role: str, relative_path: str, commit: str | None = None, digest: str | None = None) -> dict:
    """The tracked form of an artifact reference (fail-closed on the path form)."""
    if role not in ROLES:
        raise KeyError(role)
    return {"worktree_role": role, "relative_path": _check_relative(relative_path), "commit": commit, "digest": digest}


def resolve(loc: dict | str, cwd: Path | None = None) -> Path:
    """A locator dict (or a plain repo-relative string, role main) to a runtime path.
    The candidate is checked to stay under the role's root after resolution
    (symlinks and any traversal refused, never normalized through)."""
    if isinstance(loc, str):
        role, rel = "main", _check_relative(loc)
    else:
        role, rel = loc["worktree_role"], _check_relative(loc["relative_path"])
    root = worktree(role, cwd)
    cand = (root / rel)
    real = cand.resolve()
    if real != root and root not in real.parents:
        raise ValueError(f"locator escape refused: {rel!r} resolves outside the {role} root")
    return cand


def repo_relative(path: str | Path, role: str = "main", cwd: Path | None = None) -> str:
    """A runtime path expressed relative to a role's root (for receipts that must
    name where a run sat). RAISES if the path is not inside that role's root;
    never falls back to a basename."""
    p = Path(path).resolve()
    root = worktree(role, cwd)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"{p} is not inside the {role} worktree root {root}") from None


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

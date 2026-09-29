"""Read-only repository inspection must never mutate tracked files.

`.claude/hooks/ledger_regen.py` regenerates ledger surfaces (results
index, receipt lock, README honesty ledger, INDEX.md) after a session
edits their SOURCES. It decides "did a source change?" by comparing a
PreToolUse snapshot of the watched sources against the tree at
PostToolUse time. Never from the command text: a `grep` that names
docs/RESULTS.md changes nothing and must run nothing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "ledger_regen.py"
PY = sys.executable


def _load():
    spec = importlib.util.spec_from_file_location("ledger_regen", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_root(tmp_path: Path) -> Path:
    (tmp_path / "docs" / "preregs").mkdir(parents=True)
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "llmopt" / "lab").mkdir(parents=True)
    (tmp_path / "docs" / "RESULTS.md").write_text("# R\n")
    (tmp_path / "docs" / "FINDINGS.md").write_text("# F\n")
    (tmp_path / "docs" / "preregs" / "a.json").write_text("{}")
    (tmp_path / "scratch" / "probe.py").write_text("x = 1\n")
    (tmp_path / "scripts" / "tool.py").write_text("y = 2\n")
    (tmp_path / "llmopt" / "lab" / "m.py").write_text("z = 3\n")
    return tmp_path


def _tree_digest(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def _hook(phase: str, root: Path, payload: dict, log: Path) -> str:
    env = dict(os.environ, LLMOPT_HOOK_ROOT=str(root),
               LLMOPT_HOOK_LOG=str(log), LLMOPT_HOOK_PY="/usr/bin/true")
    env.pop("LLMOPT_NO_AUTOREGEN", None)
    r = subprocess.run([PY, str(HOOK), phase], input=json.dumps(payload),
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _bash(cmd: str, tool_use_id: str = "tu-1") -> dict:
    return {"session_id": "s-1", "tool_use_id": tool_use_id,
            "hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_input": {"command": cmd}}


# ---------------------------------------------------------------- unit


def test_snapshot_detects_only_real_source_changes(tmp_path):
    mod = _load()
    root = _fake_root(tmp_path)
    before = mod.snapshot(root)
    assert mod.changed_sources(before, mod.snapshot(root)) == set()
    (root / "docs" / "RESULTS.md").write_text("# R\n\n## VERDICT X\n")
    os.utime(root / "docs" / "RESULTS.md", ns=(1, 2_000_000_000))
    (root / "scratch" / "new_probe.py").write_text("n = 1\n")
    changed = mod.changed_sources(before, mod.snapshot(root))
    assert changed == {"docs/RESULTS.md", "scratch/new_probe.py"}


def test_plan_maps_changed_sources_to_generators():
    mod = _load()
    assert mod.plan({"docs/RESULTS.md"}) == [
        "scripts/gen_results_index.py", "scripts/gen_receipt_lock.py",
        ".claude/hooks/findings_headroom.py"]
    assert mod.plan({"docs/preregs/a.json"}) == [
        "scripts/gen_receipt_lock.py"]
    assert mod.plan({"docs/FINDINGS.md"}) == ["scripts/gen_readme.py"]
    assert mod.plan({"scratch/x.py", "llmopt/lab/m.py"}) == [
        "scripts/gen_index.py"]
    assert mod.plan(set()) == []


# ---------------------------------------------------------- end-to-end


@pytest.mark.parametrize("cmd", [
    "grep -c VERDICT docs/RESULTS.md",
    "wc -l docs/FINDINGS.md docs/RESULTS.md",
    "ls scratch/*.py scripts/*.py",
    "cat docs/preregs/a.json",
    "git diff -- docs/RESULTS.md",
    "sed -n 1,5p scratch/probe.py",
])
def test_read_only_command_naming_sources_runs_nothing(tmp_path, cmd):
    root = _fake_root(tmp_path / "repo")
    log = tmp_path / "hook.log"
    before = _tree_digest(root)
    payload = _bash(cmd, f"tu-{tmp_path.name}")
    _hook("pre", root, payload, log)
    subprocess.run(cmd, shell=True, cwd=root, capture_output=True)
    out = _hook("post", root, payload, log)
    assert _tree_digest(root) == before
    assert not log.exists() or log.read_text() == ""
    assert out.strip() == ""


def test_real_source_edit_between_pre_and_post_triggers_generators(tmp_path):
    root = _fake_root(tmp_path / "repo")
    log = tmp_path / "hook.log"
    cmd = "cat >> docs/RESULTS.md <<'EOF'\n## VERDICT Y\nEOF"
    _hook("pre", root, _bash(cmd), log)
    (root / "docs" / "RESULTS.md").write_text("# R\n\n## VERDICT Y\n")
    os.utime(root / "docs" / "RESULTS.md", ns=(1, 2_000_000_000))
    out = _hook("post", root, _bash(cmd), log)
    ran = log.read_text().splitlines()
    assert ran == ["scripts/gen_results_index.py",
                   "scripts/gen_receipt_lock.py",
                   ".claude/hooks/findings_headroom.py"]
    assert "results-index" in out


def test_same_size_rewrite_is_detected_by_mtime_alone(tmp_path):
    root = _fake_root(tmp_path / "repo")
    log = tmp_path / "hook.log"
    payload = {"session_id": "s-1", "tool_use_id": f"tu-{tmp_path.name}",
               "hook_event_name": "PostToolUse", "tool_name": "Edit",
               "tool_input": {"file_path": str(root / "docs" / "FINDINGS.md")}}
    _hook("pre", root, payload, log)
    time.sleep(0.02)
    (root / "docs" / "FINDINGS.md").write_text("# G\n")  # same size
    _hook("post", root, payload, log)
    assert log.read_text().splitlines() == ["scripts/gen_readme.py"]


def test_post_without_pre_snapshot_runs_nothing(tmp_path):
    root = _fake_root(tmp_path / "repo")
    log = tmp_path / "hook.log"
    out = _hook("post", root, _bash("anything", "tu-orphan"), log)
    assert not log.exists() and out.strip() == ""


def test_kill_switch_env_disables_regeneration(tmp_path):
    root = _fake_root(tmp_path / "repo")
    log = tmp_path / "hook.log"
    cmd = "edit"
    _hook("pre", root, _bash(cmd), log)
    (root / "docs" / "RESULTS.md").write_text("changed\n")
    os.utime(root / "docs" / "RESULTS.md", ns=(1, 2_000_000_000))
    env = dict(os.environ, LLMOPT_HOOK_ROOT=str(root),
               LLMOPT_HOOK_LOG=str(log), LLMOPT_NO_AUTOREGEN="1")
    r = subprocess.run([PY, str(HOOK), "post"], input=json.dumps(_bash(cmd)),
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and not log.exists()


# ------------------------------------------------- the real repository


def test_read_only_inspection_leaves_the_real_tree_byte_identical(tmp_path):
    """The invariant on THIS checkout: a read-only Bash call that names
    every ledger source runs pre, the command, then post, and the hook
    plans no generator and no tracked file changes. Generators are
    stubbed (LLMOPT_HOOK_PY) so the test itself can never mutate the
    real tree even if a source were edited concurrently."""
    watched = ["docs/results-index.jsonl", "docs/receipts.lock.json",
               "scripts/INDEX.md", "README.md", "docs/figures.json"]
    def digest():
        return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in watched}
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True).stdout
    before = digest()
    cmd = ("grep -c VERDICT docs/RESULTS.md docs/FINDINGS.md; "
           "ls docs/preregs/ scratch/*.py scripts/*.py")
    payload = _bash(cmd, f"tu-real-{tmp_path.name}")
    log = tmp_path / "hook.log"
    env = dict(os.environ, LLMOPT_HOOK_LOG=str(log),
               LLMOPT_HOOK_PY="/usr/bin/true")
    env.pop("LLMOPT_NO_AUTOREGEN", None)
    env.pop("LLMOPT_HOOK_ROOT", None)
    for phase in ("pre", "cmd", "post"):
        if phase == "cmd":
            subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True)
            continue
        r = subprocess.run([PY, str(HOOK), phase], input=json.dumps(payload),
                           capture_output=True, text=True, env=env,
                           cwd=ROOT, timeout=120)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "", r.stdout
    assert not log.exists(), log.read_text()
    assert digest() == before
    assert subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                          capture_output=True, text=True).stdout == status


def test_settings_wires_pre_and_post_phases():
    s = json.loads((ROOT / ".claude" / "settings.json").read_text())
    pre = [h["command"] for grp in s["hooks"]["PreToolUse"]
           for h in grp["hooks"] if "ledger_regen" in h["command"]]
    post = [h["command"] for grp in s["hooks"]["PostToolUse"]
            for h in grp["hooks"] if "ledger_regen" in h["command"]]
    assert pre and all(c.endswith("ledger_regen.py pre") for c in pre)
    assert post and all(c.endswith("ledger_regen.py post") for c in post)


# ------------------------------------------------------ portable post lock


def _load_hook_with(monkeypatch, *, fcntl_available: bool, msvcrt_fake=None,
                    backend_env: str | None = None):
    """Import the hook fresh under a controlled module environment."""
    import importlib
    if not fcntl_available:
        monkeypatch.setitem(sys.modules, "fcntl", None)  # ImportError on import
    if msvcrt_fake is not None:
        monkeypatch.setitem(sys.modules, "msvcrt", msvcrt_fake)
    if backend_env:
        monkeypatch.setenv("LLMOPT_HOOK_LOCK_BACKEND", backend_env)
    else:
        monkeypatch.delenv("LLMOPT_HOOK_LOCK_BACKEND", raising=False)
    spec = importlib.util.spec_from_file_location("ledger_regen_lock_test", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hook_imports_without_fcntl(monkeypatch):
    """Native Windows has no fcntl; the hook must still import (its
    "never fail a tool call" boundary lives inside main)."""
    mod = _load_hook_with(monkeypatch, fcntl_available=False)
    assert mod.LOCK_BACKEND == ("msvcrt" if sys.platform == "win32" else "mkdir")


def test_lock_backend_prefers_fcntl_then_msvcrt_then_mkdir(monkeypatch, tmp_path):
    import types
    assert _load_hook_with(monkeypatch, fcntl_available=True).LOCK_BACKEND == "fcntl"
    calls = []
    fake = types.SimpleNamespace(LK_LOCK=1, LK_UNLCK=0,
                                 locking=lambda fd, mode, n: calls.append(mode))
    mod = _load_hook_with(monkeypatch, fcntl_available=False, msvcrt_fake=fake)
    assert mod.LOCK_BACKEND == "msvcrt"
    with mod.post_lock(tmp_path / "t.lock"):
        pass
    assert calls == [1, 0]
    monkeypatch.setitem(sys.modules, "msvcrt", None)
    assert _load_hook_with(monkeypatch, fcntl_available=False).LOCK_BACKEND == "mkdir"


def test_lock_backend_override_of_an_absent_backend_falls_back(monkeypatch):
    """An override naming a backend this platform lacks falls back to
    the default chain instead of leaving post_lock to raise."""
    mod = _load_hook_with(monkeypatch, fcntl_available=False,
                          backend_env="fcntl")
    assert mod.LOCK_BACKEND == "mkdir"


def test_lock_backend_bogus_override_is_ignored(monkeypatch):
    mod = _load_hook_with(monkeypatch, fcntl_available=True,
                          backend_env="bogus")
    assert mod.LOCK_BACKEND == ("fcntl" if mod._fcntl is not None else "mkdir")


def test_mkdir_lock_reclaims_a_stale_directory_only(monkeypatch, tmp_path):
    mod = _load_hook_with(monkeypatch, fcntl_available=False, backend_env="mkdir")
    lock = tmp_path / "post.lock"
    stale = tmp_path / "post.lock.d"
    stale.mkdir()
    os.utime(stale, ns=(1, 1))  # ancient: a killed holder
    with mod.post_lock(lock):
        assert stale.exists()  # reclaimed and re-owned
    assert not stale.exists()


@pytest.mark.parametrize("backend", ["fcntl", "mkdir"])
def test_lock_serializes_concurrent_holders(monkeypatch, tmp_path, backend):
    """Two holders never overlap, on the POSIX backend and on the
    dependency-free fallback."""
    import threading
    mod = _load_hook_with(monkeypatch, fcntl_available=(backend == "fcntl"),
                          backend_env=backend)
    if backend == "fcntl" and mod._fcntl is None:
        pytest.skip("no fcntl on this platform")
    assert mod.LOCK_BACKEND == backend
    lock_path = tmp_path / "post.lock"
    events = []

    def holder(tag):
        with mod.post_lock(lock_path):
            events.append((tag, "in"))
            time.sleep(0.15)
            events.append((tag, "out"))

    ts = [threading.Thread(target=holder, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    # strictly alternating in/out pairs: no holder enters while another is in
    assert [e[1] for e in events] == ["in", "out"] * 3


def _post_timeout_s() -> int:
    s = json.loads((ROOT / ".claude" / "settings.json").read_text())
    outs = [h["timeout"] for grp in s["hooks"]["PostToolUse"] for h in grp["hooks"]
            if h["command"].endswith("ledger_regen.py post")]
    assert outs, "post hook not wired"
    return max(outs)


def test_mkdir_stale_threshold_exceeds_the_configured_post_timeout(monkeypatch):
    """A live post hook cannot be considered stale before the harness's
    PostToolUse timeout could have killed it; a future timeout change
    must not silently recreate the reclaim-while-alive bug."""
    mod = _load_hook_with(monkeypatch, fcntl_available=False, backend_env="mkdir")
    assert mod.LOCK_STALE_S > _post_timeout_s()


def test_mkdir_lock_does_not_reclaim_a_holder_within_the_hook_timeout(monkeypatch, tmp_path):
    import threading
    mod = _load_hook_with(monkeypatch, fcntl_available=False, backend_env="mkdir")
    lock = tmp_path / "post.lock"
    held = tmp_path / "post.lock.d"
    held.mkdir()
    age = _post_timeout_s() - 1  # a holder the harness has NOT killed yet
    t0 = time.time() - age
    os.utime(held, (t0, t0))
    acquired = threading.Event()

    def waiter():
        with mod.post_lock(lock):
            acquired.set()

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    assert not acquired.wait(1.0), "reclaimed a lock whose holder may still be alive"
    held.rmdir()  # the live holder finishes
    assert acquired.wait(5.0)
    th.join(5)


def test_mkdir_lock_reclaims_a_lock_older_than_the_stale_threshold(monkeypatch, tmp_path):
    mod = _load_hook_with(monkeypatch, fcntl_available=False, backend_env="mkdir")
    lock = tmp_path / "post.lock"
    stale = tmp_path / "post.lock.d"
    stale.mkdir()
    t0 = time.time() - (mod.LOCK_STALE_S + 1)
    os.utime(stale, (t0, t0))
    with mod.post_lock(lock):
        assert stale.exists()  # reclaimed and re-owned
    assert not stale.exists()

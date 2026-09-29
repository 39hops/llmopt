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

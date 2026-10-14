"""scripts/recover_receipt.py and the guard exemption it rides through.

The receipt guard refuses writes to booked receipt paths. Recovery of
an original that was never committed is the one legitimate write, so
it has its own operation with its own refusals. These tests build a
throwaway git repo with a lock that classifies one path as absent,
one as sha-locked, one as prereg-pending, and drive the script at it
with --root. Every refusal must leave the tree byte-identical; the
success path must produce the exact bytes, append a record, and refuse
to run twice. The hook half checks that the exemption covers exactly
one command shape and that ordinary receipt mutation is judged as
before.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recover_receipt.py"
HOOK = ROOT / ".claude" / "hooks" / "receipt_freeze.py"
PY = sys.executable

ABSENT = "logs/run1/driver.log"
LOCKED = "logs/run1/locked.log"
PENDING = "logs/run2/pending.log"
ORIGINAL = b"arm s1 gate: {3: 4} = 12/120\nDONE\n"
DIGEST = hashlib.sha256(ORIGINAL).hexdigest()


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A git checkout with a lock, a ledger that books DIGEST, one
    locked receipt on disk, and a fetched source file outside it."""
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    (r / "logs" / "run1").mkdir(parents=True)
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    (r / LOCKED).write_bytes(b"locked bytes\n")
    locked_sha = hashlib.sha256(b"locked bytes\n").hexdigest()
    (r / "docs" / "RESULTS.md").write_text(
        f"## VERDICT X\nreceipts {ABSENT} and {LOCKED}\n"
        f"## AMENDMENT Y\n  {ABSENT} sha256 {DIGEST}\n")
    (r / "docs" / "receipts.lock.json").write_text(json.dumps({"receipts": {
        ABSENT: {"exists": False, "source": "results"},
        LOCKED: {"exists": True, "sha256": locked_sha, "bytes": 13,
                 "tracked": True, "source": "results"},
        PENDING: {"exists": False, "source": "prereg"},
    }}))
    _git(r, "add", "-A")
    _git(r, "add", "-f", LOCKED)
    _git(r, "commit", "-qm", "fixture")
    src = tmp_path / "fetched" / "driver.log"
    src.parent.mkdir()
    src.write_bytes(ORIGINAL)
    return r, src


def snapshot(r: Path) -> dict:
    return {str(p.relative_to(r)): p.read_bytes()
            for p in r.rglob("*") if p.is_file() and ".git" not in p.parts}


def run(r, src, dest=ABSENT, digest=DIGEST, origin="fixture host"):
    return subprocess.run(
        [PY, str(SCRIPT), "--root", str(r), "--source", str(src),
         "--dest", dest, "--sha256", digest, "--origin", origin],
        capture_output=True, text=True)


def test_success_restores_exact_bytes_records_and_refuses_a_second_time(repo):
    r, src = repo
    out = run(r, src)
    assert out.returncode == 0, out.stderr
    assert (r / ABSENT).read_bytes() == ORIGINAL
    rec = json.loads((r / "docs" / "receipt-recoveries.jsonl").read_text())
    assert rec["dest"] == ABSENT and rec["sha256"] == DIGEST
    assert rec["bytes"] == len(ORIGINAL) and rec["origin"] == "fixture host"
    assert rec["booked_at_results_line"] == 4
    assert "git add -f " + ABSENT in out.stdout
    # the file now exists: the same call is a refusal, bytes untouched
    before = snapshot(r)
    again = run(r, src)
    assert again.returncode != 0 and "already exists" in again.stderr
    assert snapshot(r) == before


@pytest.mark.parametrize("dest, why", [
    (LOCKED, "not classified absent"),
    (PENDING, "prereg-declared"),
    ("logs/run9/never_cited.log", "not cited"),
    ("logs/toplevel.log", "not a receipt path"),
    ("logs/run1/../run1/driver.log", "not a receipt path"),
    ("docs/RESULTS.md", "not a receipt path"),
])
def test_destinations_outside_the_absent_allowlist_are_refused(repo, dest, why):
    r, src = repo
    before = snapshot(r)
    out = run(r, src, dest=dest)
    assert out.returncode != 0 and why in out.stderr, out.stderr
    assert snapshot(r) == before


def test_locked_receipt_is_never_overwritten_even_with_its_own_digest(repo):
    r, src = repo
    locked_sha = hashlib.sha256(b"locked bytes\n").hexdigest()
    (r / "docs" / "RESULTS.md").open("a").write(f"sha256 {locked_sha}\n")
    src.write_bytes(b"locked bytes\n")
    before = snapshot(r)
    out = run(r, src, dest=LOCKED, digest=locked_sha)
    assert out.returncode != 0 and "not classified absent" in out.stderr
    assert snapshot(r) == before


def test_digest_mismatch_is_refused_and_writes_nothing(repo):
    r, src = repo
    src.write_bytes(ORIGINAL + b"tampered\n")
    before = snapshot(r)
    out = run(r, src)
    assert out.returncode != 0 and "does not match --sha256" in out.stderr
    assert snapshot(r) == before and not (r / ABSENT).exists()


def test_unbooked_digest_is_refused(repo):
    r, src = repo
    other = b"a different original\n"
    src.write_bytes(other)
    before = snapshot(r)
    out = run(r, src, digest=hashlib.sha256(other).hexdigest())
    assert out.returncode != 0 and "not booked in docs/RESULTS.md" in out.stderr
    assert snapshot(r) == before


def test_malformed_digest_is_refused(repo):
    r, src = repo
    out = run(r, src, digest="DEADBEEF")
    assert out.returncode != 0 and "64 lowercase hex" in out.stderr
    assert not (r / ABSENT).exists()


def test_live_run_interlock_refuses(repo):
    r, src = repo
    sentinel = r / ".git" / "liverun.lock"
    sentinel.write_text(json.dumps({"run_id": "demo", "pid": os.getpid()}))
    before = snapshot(r)
    out = run(r, src)
    assert out.returncode != 0 and "registered run is live" in out.stderr
    assert snapshot(r) == before


def test_linked_worktree_refuses(repo, tmp_path):
    r, src = repo
    wt = tmp_path / "repo-repair"
    _git(r, "worktree", "add", "-q", "--detach", str(wt))
    before = snapshot(wt)
    out = run(wt, src)
    assert out.returncode != 0 and "linked worktree" in out.stderr
    assert snapshot(wt) == before


def test_unreadable_lock_fails_closed(repo):
    r, src = repo
    (r / "docs" / "receipts.lock.json").write_text("{not json")
    out = run(r, src)
    assert out.returncode != 0 and "cannot be parsed" in out.stderr
    assert not (r / ABSENT).exists()


def test_existing_symlink_at_destination_is_refused(repo):
    r, src = repo
    (r / ABSENT).symlink_to(r / LOCKED)
    out = run(r, src)
    assert out.returncode != 0 and "already exists" in out.stderr
    assert (r / LOCKED).read_bytes() == b"locked bytes\n"


# --- the guard exemption -------------------------------------------------

def decide(command: str):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    out = subprocess.run([PY, str(HOOK)], input=payload, capture_output=True,
                         text=True, cwd=ROOT)
    if not out.stdout.strip():
        return None
    return json.loads(out.stdout)["hookSpecificOutput"]["permissionDecision"]


def a_locked_path() -> str:
    lock = json.loads((ROOT / "docs" / "receipts.lock.json").read_text())["receipts"]
    return next(k for k, v in lock.items() if v.get("sha256") and v.get("tracked"))


RECOVER = (".venv/bin/python scripts/recover_receipt.py --source /tmp/x.log "
           "--dest {p} --sha256 " + "0" * 64 + ' --origin "3080 checkout, mtime 2026-08-11"')


def test_hook_lets_the_bare_recovery_invocation_through():
    p = a_locked_path()
    assert decide(RECOVER.format(p=p)) is None
    assert decide("python3 " + RECOVER.format(p=p).split(" ", 1)[1]) is None


@pytest.mark.parametrize("suffix", [
    " && rm {p}", "; rm {p}", " | tee {p}", " > {p}", " $(rm {p})",
    " `rm {p}`", " --origin \"$(rm {p})\"", " --dest {p}\nrm {p}",
])
def test_hook_refuses_anything_riding_along_with_the_recovery(suffix):
    p = a_locked_path()
    cmd = RECOVER.format(p=p) + suffix.format(p=p)
    assert decide(cmd) == "deny", cmd


def test_hook_still_refuses_ordinary_receipt_mutation():
    p = a_locked_path()
    assert decide(f"cp /tmp/x.log {p}") == "deny"
    assert decide(f"rm {p}") == "deny"
    assert decide(f"echo hi > {p}") == "deny"
    # a different script with the same arguments earns no exemption
    assert decide(f"cp scripts/recover_receipt.py /tmp/r.py && python /tmp/r.py --dest {p}") == "deny"

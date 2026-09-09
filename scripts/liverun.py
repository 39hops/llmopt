"""liverun — the mechanical live-run interlock (BOARD live-run law,
2026-09-09). A registered scientific run is wrapped so that, while it is
live, an atomic sentinel exists in the shared git directory and the
pre-commit hook (scripts/liverun_precommit.sh, installed at
.git/hooks/pre-commit) refuses every commit in every checkout that shares
that git directory (the main checkout and all its worktrees).

Sentinel: <git-common-dir>/liverun.lock, created with O_EXCL (atomic; a
second registered run cannot arm while one is live), holding
{run_id, pid, launch_commit, worktree, cwd, started_utc}. Receipts: one
JSON line per event in logs/liverun/<run_id>.jsonl (armed, disarmed with
rc, stale-recovery with reason). The scientific receipts of the wrapped
run still record their own source commit and digests independently; the
interlock protects the checkout, it does not replace provenance.

Usage:
  liverun.py run <run_id> [--worktree PATH] -- <command ...>
      arm, exec the command in PATH (default: cwd), disarm with its rc
  liverun.py status
  liverun.py recover <run_id> --reason "..."   (only if the recorded pid is dead)
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def git(*args, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd).stdout.strip()


def common_dir(cwd=None):
    return Path(git("rev-parse", "--git-common-dir", cwd=cwd)).resolve()


def lock_path(cwd=None):
    return common_dir(cwd) / "liverun.lock"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def receipt(run_id, row, root):
    d = Path(root) / "logs" / "liverun"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{run_id}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def arm(run_id, worktree, cwd):
    lp = lock_path(cwd)
    rec = {"run_id": run_id, "pid": os.getpid(), "launch_commit": git("rev-parse", "HEAD", cwd=worktree),
           "launch_commit_short": git("rev-parse", "--short", "HEAD", cwd=worktree), "worktree": str(Path(worktree).resolve()),
           "cwd": str(Path(cwd).resolve()), "started_utc": now(), "dirty_at_arm": bool(git("status", "--porcelain", cwd=worktree))}
    if rec["dirty_at_arm"]:
        raise SystemExit(f"liverun: REFUSING to arm {run_id}: worktree {worktree} is dirty")
    try:
        fd = os.open(lp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise SystemExit(f"liverun: REFUSING to arm {run_id}: sentinel {lp} exists ({lp.read_text().strip()})")
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f)
    receipt(run_id, {"event": "armed", **rec, "sentinel": str(lp)}, cwd)
    return lp, rec


def disarm(run_id, lp, rec, rc, cwd):
    lp.unlink(missing_ok=False)
    receipt(run_id, {"event": "disarmed", "run_id": run_id, "pid": rec["pid"], "rc": rc, "ended_utc": now(), "sentinel": str(lp)}, cwd)


def cmd_run(a):
    cwd = os.getcwd()
    wt = a.worktree or cwd
    lp, rec = arm(a.run_id, wt, cwd)
    print(f"[liverun] armed {a.run_id} pid {rec['pid']} commit {rec['launch_commit_short']} worktree {wt} sentinel {lp}", flush=True)
    rc = 1
    try:
        rc = subprocess.run(a.command, cwd=wt).returncode
    finally:
        disarm(a.run_id, lp, rec, rc, cwd)
        print(f"[liverun] disarmed {a.run_id} rc {rc}", flush=True)
    sys.exit(rc)


def cmd_status(a):
    lp = lock_path()
    if not lp.exists():
        print("liverun: no live registered run")
        return
    rec = json.loads(lp.read_text())
    print(json.dumps({**rec, "pid_alive": pid_alive(rec["pid"]), "sentinel": str(lp)}, indent=1))


def cmd_recover(a):
    lp = lock_path()
    if not lp.exists():
        raise SystemExit("liverun: nothing to recover")
    rec = json.loads(lp.read_text())
    if rec["run_id"] != a.run_id:
        raise SystemExit(f"liverun: sentinel belongs to {rec['run_id']}, not {a.run_id}")
    if pid_alive(rec["pid"]):
        raise SystemExit(f"liverun: REFUSING stale recovery: pid {rec['pid']} is alive")
    lp.unlink()
    receipt(a.run_id, {"event": "stale_recovered", "run_id": a.run_id, "pid": rec["pid"], "reason": a.reason, "recovered_utc": now(), "sentinel": str(lp)}, os.getcwd())
    print(f"[liverun] stale sentinel for {a.run_id} (dead pid {rec['pid']}) removed; receipt written; reason: {a.reason}")


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        # everything after the first "--" is the command, verbatim
        if "--" not in argv:
            raise SystemExit("liverun: run needs a command after --")
        i = argv.index("--")
        head, command = argv[1:i], argv[i + 1:]
        ap = argparse.ArgumentParser(prog="liverun.py run")
        ap.add_argument("run_id"); ap.add_argument("--worktree")
        a = ap.parse_args(head)
        if not command:
            raise SystemExit("liverun: run needs a command after --")
        a.command = command
        cmd_run(a)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    v = sub.add_parser("recover"); v.add_argument("run_id"); v.add_argument("--reason", required=True); v.set_defaults(fn=cmd_recover)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

"""rjob — job-ID-based remote/local run management (2026-08-01).

Replaces ad-hoc pgrep/pkill remote ops (8+ friendly-fire incidents:
self-matching watchers, string-matched kills, markers firing on
crashed arms). The unit is a JOB, not a process pattern:

  every launch creates   jobs/<id>.pid   (setsid'd process group)
                         jobs/<id>.log   (stdout+stderr)
                         jobs/<id>.cmd   (the exact command line)
                         jobs/<id>.rc    (exit code, written ON EXIT)

  status/tail/kill/wait operate on the ID via the pidfile — never
  on a grep pattern, so a watcher can never match itself. DONE
  carries the real exit code for jobs that exit; killed jobs get
  rc=killed (written by the kill itself). DIED means pid gone with
  no rc — genuinely anomalous, though a brief exit-to-rc-write
  race can show it transiently; re-check before acting on it.

Transport: remote commands go through scratch/wsl.sh (one ssh call
per subcommand, per the remote-ops doctrine). RJOB_LOCAL=1 runs on
this machine instead (same job files under jobs/) — one tool for
both the Mac and the 3080.

Usage:
  python scripts/rjob.py launch <id> '<command>'
  python scripts/rjob.py status            # all jobs, one line each
  python scripts/rjob.py tail <id> [n]
  python scripts/rjob.py kill <id>
  python scripts/rjob.py clean <id>        # remove a finished job's files
"""
import os
import shlex
import subprocess
import sys

LOCAL = os.environ.get("RJOB_LOCAL") == "1"
JOBS = "jobs"


def sh(cmd):
    """One transport call. Local: bash here. Remote: via wsl.sh run."""
    if LOCAL:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                           text=True)
    else:
        r = subprocess.run(["scratch/wsl.sh", "run", cmd],
                           capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


import re


def check_jid(jid):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", jid):
        print(f"[rjob] invalid job id {jid!r} "
              f"(allowed: [A-Za-z0-9_.-], max 64)")
        sys.exit(1)
    return jid


def launch(jid, cmd):
    check_jid(jid)
    rc, out = sh(
        f"mkdir -p {JOBS} && test ! -e {JOBS}/{jid}.pid || exit 91; "
        f"printf %s {shlex.quote(cmd)} > {JOBS}/{jid}.cmd && echo ok")
    if rc == 91 or "ok" not in out:
        print(f"[rjob] REFUSED: job id '{jid}' already exists "
              f"(clean it first)" if rc == 91 else
              f"[rjob] cmd-file write failed: {out}")
        sys.exit(1)
    runner = (f"bash {JOBS}/{jid}.cmd > {JOBS}/{jid}.log 2>&1; "
              f"echo $? > {JOBS}/{jid}.rc")
    if LOCAL:
        p = subprocess.Popen(["bash", "-c", runner],
                             start_new_session=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        with open(f"{JOBS}/{jid}.pid", "w") as f:
            f.write(str(p.pid))
        pid = p.pid
    else:
        # the CHILD writes its own pid (util-linux setsid may fork,
        # making $! neither the job pid nor the pgid — reviewer
        # catch, 2026-08-01)
        inner = (f"echo $$ > {JOBS}/{jid}.pid; exec bash -c "
                 f"{shlex.quote(runner)}")
        # double-fork through a subshell that exits immediately —
        # a plain `... & ` keeps the ssh channel held for the job's
        # whole life (measured: a 3 s job made launch take 3.5 s;
        # night31b made it hang past 120 s, 2026-08-01)
        rc, _ = sh(
            f"( setsid bash -c {shlex.quote(inner)} >/dev/null 2>&1 "
            f"</dev/null & ); sleep 0.5; cat {JOBS}/{jid}.pid "
            f"2>/dev/null || echo pending")
        pid = _
    print(f"[rjob] launched {jid} pid {pid} (rc file on exit)")


def status():
    rc, out = sh(
        f"for p in {JOBS}/*.pid; do [ -e \"$p\" ] || continue; "
        f"id=$(basename \"$p\" .pid); pid=$(cat \"$p\"); "
        f"if [ -e {JOBS}/$id.rc ]; then "
        f"echo \"$id DONE rc=$(cat {JOBS}/$id.rc)\"; "
        f"elif kill -0 $pid 2>/dev/null; then "
        f"echo \"$id RUNNING pid=$pid\"; "
        f"else echo \"$id DIED (no rc, pid gone)\"; fi; done")
    print(out or "[rjob] no jobs")


def tail(jid, n="20"):
    check_jid(jid)
    rc, out = sh(f"tail -n {int(n)} {JOBS}/{jid}.log")
    print(out)


def kill(jid):
    check_jid(jid)
    # the kill itself writes the rc file (the wrapper dies with the
    # group and can't) so a killed job resolves to DONE rc=killed
    rc, out = sh(
        f"pid=$(cat {JOBS}/{jid}.pid) && kill -- -$pid 2>/dev/null; "
        f"kill $pid 2>/dev/null; echo killed > {JOBS}/{jid}.rc; "
        f"echo killed-$pid")
    print(f"[rjob] {out}")


def clean(jid):
    check_jid(jid)
    rc, out = sh(
        f"if [ -e {JOBS}/{jid}.rc ] || "
        f"! kill -0 $(cat {JOBS}/{jid}.pid 2>/dev/null) 2>/dev/null; "
        f"then rm -f {JOBS}/{jid}.pid {JOBS}/{jid}.cmd "
        f"{JOBS}/{jid}.rc; echo cleaned; "
        f"else echo 'REFUSED: still running'; fi")
    print(f"[rjob] {jid}: {out}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    op = sys.argv[1]
    if op == "launch":
        launch(sys.argv[2], sys.argv[3])
    elif op == "status":
        status()
    elif op == "tail":
        tail(*sys.argv[2:4])
    elif op == "kill":
        kill(sys.argv[2])
    elif op == "clean":
        clean(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

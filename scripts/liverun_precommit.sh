#!/bin/bash
# liverun pre-commit interlock (BOARD live-run law, 2026-09-09). Installed
# at <git-common-dir>/hooks/pre-commit so it covers the main checkout and
# every worktree sharing that git directory. Refuses any commit while the
# registered-run sentinel exists and its recorded pid is alive; a dead pid
# is reported as STALE and still refused until `scripts/liverun.py recover`
# writes its receipt. Bypass is the plain git escape hatch (--no-verify),
# which leaves no receipt and is therefore a booked violation if used.
lock="$(git rev-parse --git-common-dir)/liverun.lock"
[ -f "$lock" ] || exit 0
pid=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pid'])" "$lock" 2>/dev/null || echo 0)
run=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "$lock" 2>/dev/null || echo unknown)
if kill -0 "$pid" 2>/dev/null; then
  echo "liverun: REFUSING commit — registered run '$run' (pid $pid) is LIVE; sentinel $lock" >&2
else
  echo "liverun: REFUSING commit — STALE sentinel for '$run' (pid $pid dead); run: .venv/bin/python scripts/liverun.py recover $run --reason '...'" >&2
fi
exit 1

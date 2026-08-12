---
name: labstatus
description: Use when the user asks how the runs are going — "how is everything going", "what's running", "any runs finish", "check the 3080", "status" — or before booking and queueing decisions. One-shot sweep of every live log on both machines (Mac rjob + 3080 via wsl.sh) with the finished-but-unbooked ones first.
---

# Lab status sweep

Run these and summarize per run: last gate/verdict lines, current
epoch/step, and whether the process is alive. Lead with anything
that FINISHED or CRASHED since the last check.

1. **Mac logs** (newest first; today's logs match `logs/*_MMDD.log`):

   ```bash
   ls -t logs/*.log | head -8
   for f in $(ls -t logs/*.log | head -6); do echo "== $f"; tail -3 "$f"; done
   pgrep -fl "python scratch/" | grep -v grep || echo "no local runs"
   RJOB_LOCAL=1 .venv/bin/python scripts/rjob.py status
   ```

   (rjob jobs are the preferred launch path — `status` shows
   RUNNING/DONE-with-rc/DIED by job ID, no string matching.)

2. **3080** (single wsl.sh call; never compare its gate numbers to
   Mac numbers — cross-device doctrine). Sweep `logs/*/` too, not
   just top level — driver receipts land in `logs/<rung>/`, so a
   top-level-only glob reports "nothing running" for a battery that
   is streaming cells:

   ```bash
   scratch/wsl.sh run "ls -t logs/*.log logs/*/*.log 2>/dev/null | head -8; for f in \$(ls -t logs/*.log logs/*/*.log 2>/dev/null | head -4); do echo \"== \$f\"; tail -3 \"\$f\"; done; ls logs/*.DONE 2>/dev/null | tail -3"
   ```

3. **Verify anything you believe is QUEUED is actually running.** A
   marker file and a live process are different facts, and a watcher
   proves neither. Check the process, not the plan:

   ```bash
   scratch/wsl.sh run "pgrep -af '<driver>' | grep -v pgrep || echo 'NOT RUNNING'"
   ```

   Two failure shapes have both fired (2026-08-11, same night):
   a chained driver that nothing ever launched, and a watcher polling
   a `.DONE` marker its driver never writes. Either way the run looks
   pending forever and the window burns. If a driver is chained on
   another rung's marker, confirm the driver process exists — the
   chain's `while [ ! -f marker ]` loop only runs if something started
   it.

4. Report: one line per live run (name, progress, ETA if knowable),
   then finished-but-unbooked results (these are the action items —
   offer to /book them), then anything anomalous (empty logs,
   dead pids, OOM/Traceback lines — an allocator OOM warning is a
   TRIPWIRE, not noise).

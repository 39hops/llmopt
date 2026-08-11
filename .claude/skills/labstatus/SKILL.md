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
   Mac numbers — cross-device doctrine):

   ```bash
   scratch/wsl.sh run "ls -t logs/*.log | head -5; for f in \$(ls -t logs/*.log | head -3); do echo \"== \$f\"; tail -3 \"\$f\"; done; ls logs/*.DONE 2>/dev/null | tail -3"
   ```

3. Report: one line per live run (name, progress, ETA if knowable),
   then finished-but-unbooked results (these are the action items —
   offer to /book them), then anything anomalous (empty logs,
   dead pids, OOM/Traceback lines — an allocator OOM warning is a
   TRIPWIRE, not noise).

---
name: pgrep-string-matching-jobs
enabled: true
event: bash
pattern: (pgrep|pkill)\s+-f
action: warn
---
**Job management by string-matching killed/confused 9+ runs**
(self-matching watchers, wrong-process kills — latest: a waiter
whose own remote shell matched its `pgrep -f`, reporting a
finished run as STILL-RUNNING for 45 min). Use
`scripts/rjob.py` (launch/status/tail/kill by JOB ID via
pidfiles; `RJOB_LOCAL=1` for the Mac). If pgrep is genuinely
needed, match `python.*<script>` so the watcher's own shell can
never match, and never chain it with a kill in the same call.

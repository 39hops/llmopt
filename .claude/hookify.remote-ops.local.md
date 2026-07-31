---
name: remote-kill-launch-same-call
enabled: true
event: bash
pattern: kill.*(&&|;).*(nohup|launch|setsid)
action: warn
---
**Remote-ops doctrine**: kill / write / launch must be SEPARATE
ssh calls (friendly-fire, 7 variants deep). Also check: a
watcher's pgrep must never match a string its own launcher
carries; completion markers fire on SUCCESS only (&&-chain the
arms).

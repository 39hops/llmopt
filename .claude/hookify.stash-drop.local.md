---
name: stash-drop-verify
enabled: true
event: bash
pattern: git stash (drop|clear|pop)
action: warn
---
**Sync doctrine**: stash -> pull -> VERIFY -> drop; never
drop-on-abort. Inspect `git stash show --include-untracked
--name-only` before dropping — a stash has carried live
checkpoint/data state before (2026-07-31 3080 incident).

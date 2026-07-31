---
name: env-prefix-expansion
enabled: true
event: bash
pattern: \$\{\d+:\+[A-Za-z_]+=
action: warn
---
**Friendly-fire #10 pattern**: `${N:+VAR=val}` expands AFTER the
shell parses env-assignment prefixes -> "command not found".
Use `env VAR="${N:-default}" cmd` instead (env takes assignments
as arguments, expansion-safe).

---
name: scp-colon-path
enabled: true
event: bash
pattern: (^|\s)scp\s
action: warn
---
**scp mangles colon-paths on this setup** (measured 2026-07-30:
"cp: a@10.0.0.184ode/..."). Use the ssh cat-pipe pattern instead:
`cat file | ssh HOST "cat > path"` (and sha-verify after).

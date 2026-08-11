#!/usr/bin/env python3
"""PreToolUse guard: give the permission layer a say INSIDE
scratch/wsl.sh payloads (Artin, 2026-08-10 — the pkill-self-match
night). Reads the hook JSON on stdin; emits a permissionDecision
only for Bash commands that invoke wsl.sh. Everything else: silent
exit 0 (no opinion).

Decisions:
  allow  — read-only inner verbs (tail/check subcommands, or run
           payloads made only of ls/tail/cat/pgrep/test/df/du/
           md5sum/git-status-class reads)
  ask    — anything mutating on the remote (kill/pkill/rm/mv/
           truncate/git mutations/launch/redirects), reason quotes
           the inner command so the human sees WHAT would run
  deny   — catastrophic shapes (rm -rf on ~ or /, force push,
           mkfs/dd-to-device)
Extra: a pkill/pgrep -f whose pattern appears un-bracketed
elsewhere in the same payload gets an ask with a SELF-MATCH
warning (friendly-fire doctrine).
"""
import json
import re
import sys


def out(decision, reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason}}))
    sys.exit(0)


data = json.load(sys.stdin)
if data.get("tool_name") != "Bash":
    sys.exit(0)
cmd = data.get("tool_input", {}).get("command", "")
if "wsl.sh" not in cmd:
    sys.exit(0)

# outer verb
m = re.search(r"wsl\.sh\s+(run|launch|check|tail)\b\s*(.*)", cmd, re.S)
if not m:
    out("ask", "wsl.sh with unrecognized verb — inspect manually")
verb, rest = m.group(1), m.group(2)

if verb in ("check", "tail"):
    out("allow", f"wsl.sh {verb}: read-only remote op")

# extract the quoted payload (first "..." or '...' arg)
pm = re.match(r'\s*"((?:[^"\\]|\\.)*)"|\s*\'([^\']*)\'', rest, re.S)
payload = (pm.group(1) or pm.group(2)) if pm else rest

DENY = [
    (r"rm\s+-rf?\s+(/|~|\$HOME)(\s|$|/\*)", "recursive delete at root/home"),
    (r"git\s+push\s+.*(--force|-f)\b", "force push"),
    (r"\b(mkfs|dd\s+.*of=/dev/)", "device-level write"),
]
for pat, why in DENY:
    if re.search(pat, payload):
        out("deny", f"remote payload is {why}: {payload[:160]}")

# self-matching pkill/pgrep (tonight's incident): -f pattern whose
# literal text appears again in the payload without [] escape
sm = re.search(r"p(?:kill|grep)\s+(?:-\w+\s+)*-?f?\s*['\"]?([\w./_-]{4,})",
               payload)
if sm and "[" not in sm.group(1):
    pat = sm.group(1)
    if payload.count(pat) > 1:
        out("ask", f"SELF-MATCH RISK: pkill/pgrep pattern '{pat}' appears "
            f"elsewhere in the same payload (friendly-fire doctrine: "
            f"bracket one char, e.g. '{pat[:2]}[{pat[2]}]{pat[3:]}'). "
            f"Payload: {payload[:200]}")

MUTATING = r"""\b(pkill|kill|killall|rm|mv|cp\s+.*\s+~|truncate|
git\s+(reset|clean|checkout\s+--|stash\s+drop|push|rebase)|
chmod|chown|systemctl|service|shutdown|reboot|nohup|setsid)\b|>>?\s*[^&|]"""
if verb == "launch" or re.search(MUTATING, payload, re.X):
    out("ask", f"remote MUTATING op via wsl.sh {verb} — inner command: "
        f"{payload[:220]}")

READONLY = re.compile(
    r"^[\s(]*((ls|tail|head|cat|wc|grep|pgrep|test|df|du|md5sum|"
    r"sha256sum|echo|pwd|which|stat|find|sleep)\b|git\s+"
    r"(status|log|rev-parse|fetch|diff|show)\b)")
parts = re.split(r"&&|\|\||;", payload)
if all(READONLY.match(p.strip()) for p in parts if p.strip()):
    out("allow", "wsl.sh run: all inner segments read-only")

out("ask", f"wsl.sh run with unclassified payload: {payload[:220]}")

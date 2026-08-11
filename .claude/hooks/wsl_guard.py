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

# SECURITY (2026-08-11 review): an "allow" decision applies to the
# ENTIRE Bash command, so it may only be issued when the entire command
# IS the single wsl.sh invocation we inspected. Matching wsl.sh anywhere
# inside the string let `rm -rf ~ && scratch/wsl.sh tail x` be
# auto-approved on the strength of the tail verb. Anchor at the start
# and refuse any shell metacharacter that could chain, substitute, or
# redirect — those fall through to "ask", never to "allow".
CHAINING = re.compile(r"[;&|`\n><]|\$\(")

m = re.match(r"\s*(?:\./)?(?:[\w./-]*/)?wsl\.sh\s+"
             r"(run|launch|check|tail|clean-marker|kill|mkdir)\b\s*(.*)",
             cmd, re.S)
if not m:
    out("ask", "command contains wsl.sh but is not a bare wsl.sh "
        "invocation — inspect manually")
verb, rest = m.group(1), m.group(2)

# argv-only verbs: safe ONLY if nothing else rides along
_bare = not CHAINING.search(rest)
if verb in ("check", "tail") and _bare:
    out("allow", f"wsl.sh {verb}: read-only remote op")
if verb in ("clean-marker", "mkdir") and _bare:
    out("allow", f"wsl.sh {verb}: safe-class op (argument passes the "
        f"script's character allowlist): {rest[:80]}")
if verb in ("check", "tail", "clean-marker", "mkdir"):
    out("ask", f"wsl.sh {verb} with shell metacharacters in its "
        f"arguments — not the bare form: {rest[:120]}")
if verb == "kill":
    out("ask", f"wsl.sh kill (self-match-proofed pkill) — pattern: "
        f"{rest[:80]}")

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
# command substitution hides arbitrary commands inside a read-only-
# looking segment (`ls $(rm -rf ~)`), so it disqualifies the allow path
if re.search(r"\$\(|`", payload):
    out("ask", f"remote payload contains command substitution — cannot "
        f"be classified read-only: {payload[:200]}")
if all(READONLY.match(p.strip()) for p in parts if p.strip()):
    out("allow", "wsl.sh run: all inner segments read-only")

out("ask", f"wsl.sh run with unclassified payload: {payload[:220]}")

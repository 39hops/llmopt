#!/usr/bin/env python3
"""Permission helper for scratch/wsl.sh — decides which remote jobs
can run without interrupting Artin, and which are worth a look first.

CONTEXT: the lab is two computers Artin owns — this Mac and his
Windows desktop with the 3080, on his home network, same repo checked
out twice. scratch/wsl.sh hands training jobs to that machine and
reads logs back. This helper exists so routine reads (tail a log,
check whether a job is alive) don't prompt him every time, while
anything that changes state on that box does.

It is a convenience filter, not a barrier: Artin can approve anything
it asks about. The point is signal — when a prompt appears, it should
mean something.

Reads the hook JSON on stdin; only speaks for Bash commands that call
wsl.sh. Everything else exits silently with no opinion.

  allow  — reads only: the tail/check verbs, or a run whose inner
           command is entirely ls/tail/cat/pgrep/test/df/du/md5sum
           and git read subcommands
  ask    — anything that changes the remote box (kill, rm, mv, git
           mutations, launch, redirects); the reason quotes the inner
           command so Artin sees exactly what would run
  deny   — the three shapes that would destroy work irrecoverably:
           a recursive delete of home or root, a force push, a raw
           write to a block device

Also asks when a pkill/pgrep pattern appears unbracketed elsewhere in
the same command — that shape makes a job kill itself, which cost a
night in July (friendly-fire doctrine).
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

# An "allow" applies to the WHOLE Bash command, so it is only issued
# when the whole command is the single wsl.sh call examined here.
# Hence: anchor at the start, and treat any shell metacharacter that
# could chain, substitute, or redirect as a reason to ask instead.
CHAINING = re.compile(r"[;&|`\n><]|\$\(")

m = re.match(r"\s*(?:\./)?(?:[\w./-]*/)?wsl\.sh\s+"
             r"(run|launch|check|tail|clean-marker|kill|mkdir)\b\s*(.*)",
             cmd, re.S)
if not m:
    out("ask", "command mentions wsl.sh but is not a plain wsl.sh "
        "call — worth reading before it runs")
verb, rest = m.group(1), m.group(2)

# verbs whose arguments the script itself checks: fine on their own
_bare = not CHAINING.search(rest)
if verb in ("check", "tail") and _bare:
    out("allow", f"wsl.sh {verb}: reads remote state only")
if verb in ("clean-marker", "mkdir") and _bare:
    out("allow", f"wsl.sh {verb}: simple op, argument already checked "
        f"by the script: {rest[:80]}")
if verb in ("check", "tail", "clean-marker", "mkdir"):
    out("ask", f"wsl.sh {verb} carries shell metacharacters in its "
        f"arguments rather than the plain form: {rest[:120]}")
if verb == "kill":
    out("ask", f"wsl.sh kill — pattern: {rest[:80]}")

# Standing OK (Artin 2026-08-15, overnight-autonomy session): a
# launch of a git-tracked driver with its log and marker both under
# logs/ is the reviewed shape — the driver text was seen at commit
# time, and the launch itself creates only a log, a marker, and the
# job. Anything fancier (inline commands, paths outside logs/)
# still asks.
if verb == "launch":
    lm = re.match(r"""\s*(?:"bash\s+(scratch/[\w-]+\.sh)"
                  |'bash\s+(scratch/[\w-]+\.sh)')\s+
                  (logs/[\w./-]+)\s+(logs/[\w./-]+)\s*$""",
                  rest, re.X)
    if lm:
        out("allow", f"wsl.sh launch of tracked driver "
            f"{lm.group(1) or lm.group(2)} with log+marker under logs/")
    # Standing OK (Artin 2026-08-16, overnight 3080 grant): python
    # drivers in the same reviewed shape — optional `cd ~/code/llmopt
    # &&` prefix and NAME=value env assignments, then the repo venv
    # running a tracked scratch/ script, log+marker under logs/.
    # The driver text was seen at commit time; env values are limited
    # to safe word characters so no command can hide in one.
    pm = re.match(r"""\s*'
                  (?:cd\s+~/code/llmopt\s+&&\s+)?
                  (?:[A-Z][A-Z0-9_]*=[\w./~-]*\s+)*
                  \.venv/bin/python\s+(?:-u\s+)?(scratch/[\w-]+\.py)
                  (?:\s+[\w./=-]+)*'\s+
                  (logs/[\w./-]+)\s+(logs/[\w./-]+)\s*$""",
                  rest, re.X)
    if pm:
        out("allow", f"wsl.sh launch of tracked python driver "
            f"{pm.group(1)} with log+marker under logs/ (overnight "
            "standing OK, 2026-08-16)")

# Same standing OK for the run verb: the identical tracked-python
# shape executed in the foreground (smokes, short probes). The regex
# anchors the WHOLE argument and admits && only as the literal
# `cd ~/code/llmopt &&` prefix, so it is stricter than _bare; any
# pipe/redirect/other chaining fails the match and falls through.
if verb == "run":
    rm_ = re.match(r"""\s*'
                   (?:cd\s+~/code/llmopt\s+&&\s+)?
                   (?:[A-Z][A-Z0-9_]*=[\w./~-]*\s+)*
                   \.venv/bin/python\s+(?:-u\s+)?(scratch/[\w-]+\.py)
                   (?:\s+[\w./=-]+)*'\s*$""",
                   rest, re.X)
    if rm_:
        out("allow", f"wsl.sh run of tracked python driver "
            f"{rm_.group(1)} (overnight standing OK, 2026-08-16)")

# the inner command: first quoted argument
pm = re.match(r'\s*"((?:[^"\\]|\\.)*)"|\s*\'([^\']*)\'', rest, re.S)
inner = (pm.group(1) or pm.group(2)) if pm else rest

UNRECOVERABLE = [
    (r"rm\s+-rf?\s+(/|~|\$HOME)(\s|$|/\*)", "a recursive delete of home or root"),
    (r"git\s+push\s+.*(--force|-f)\b", "a force push"),
    (r"\b(mkfs|dd\s+.*of=/dev/)", "a raw write to a block device"),
]
for pat, why in UNRECOVERABLE:
    if re.search(pat, inner):
        out("deny", f"this would run {why}, which cannot be undone: "
            f"{inner[:160]}")

# a pkill/pgrep -f pattern that also appears literally elsewhere in
# the command will match the job's own argv. That only destroys work
# when a kill acts on the match — pkill directly, or pgrep piped into
# kill/xargs. A bare pgrep status peek that matches itself just lists
# an extra pid, so it stays on the read path (narrowed 2026-08-11
# after a status check prompted on 'pgrep -af x; tail logs/x/...').
sm = re.search(r"(pkill|pgrep)\s+(?:-\w+\s+)*-?f?\s*['\"]?([\w./_-]{4,})",
               inner)
if sm and "[" not in sm.group(2):
    pat = sm.group(2)
    can_kill = (sm.group(1) == "pkill"
                or re.search(r"\b(kill|killall|xargs)\b", inner))
    if can_kill and inner.count(pat) > 1:
        out("ask", f"SELF-MATCH: the {sm.group(1)} pattern '{pat}' also "
            f"appears elsewhere in this command and a kill acts on the "
            f"match, so the job would kill itself (bracket one character, "
            f"e.g. '{pat[:2]}[{pat[2]}]{pat[3:]}'). Command: {inner[:200]}")

# Standing OK (Artin 2026-08-12, reaffirmed 2026-08-13 02:02): the
# ff-only sync + hash assert runs without asking. --ff-only cannot
# rewrite history and aborts on local divergence; the only redirect
# tolerated here is to /dev/null. Shape: cd / git pull --ff-only /
# git rev-parse [| cut] segments joined by &&, nothing else.
SYNC_SEG = re.compile(
    r"^[\s(]*(cd\s+[\w~./-]+"
    r"|git\s+pull(\s+--ff-only)(\s*>?\s*/dev/null(\s+2>&1)?)?"
    r"|git\s+rev-parse\s+[\w@^~ -]+(\s*\|\s*cut\s+[\w -]+)?"
    r")\s*$")
_sync_parts = [s.strip() for s in inner.split("&&") if s.strip()]
if (any("git pull" in s for s in _sync_parts)
        and all(SYNC_SEG.match(s) for s in _sync_parts)):
    out("allow", "wsl.sh run: the standing-OK ff-only sync + hash assert")

CHANGES_STATE = r"""\b(pkill|kill|killall|rm|mv|cp\s+.*\s+~|truncate|
git\s+(reset|clean|checkout\s+--|stash\s+drop|push|rebase)|
chmod|chown|systemctl|service|shutdown|reboot|nohup|setsid)\b|>>?\s*[^&|]"""
if verb == "launch" or re.search(CHANGES_STATE, inner, re.X):
    out("ask", f"changes state on the 3080 box via wsl.sh {verb} — "
        f"inner command: {inner[:220]}")

READS_ONLY = re.compile(
    r"^[\s(]*((ls|tail|head|cat|wc|grep|pgrep|test|df|du|md5sum|"
    r"sha256sum|echo|pwd|which|stat|find|sleep|cd|nvidia-smi|free|"
    r"uptime|nproc|date|hostname|base64|jq|sort|uniq|cut|tr)\b|git\s+"
    r"(status|log|rev-parse|fetch|diff|show)\b|"
    r"mkdir\s+-p\s+(~/code/llmopt/)?logs/)")
# a /dev/null redirect discards output; it cannot change the box, so
# strip it before the state-change and read-only checks see it
inner_clean = re.sub(r">\s*/dev/null(\s+2>&1)?|2>&1", " ", inner)
parts = re.split(r"&&|\|\||;", inner_clean)
# `ls $(...)` looks like a read but runs whatever is in the
# substitution, so it never takes the allow path
if re.search(r"\$\(|`", inner):
    out("ask", f"inner command uses command substitution, so its effect "
        f"cannot be read off the text: {inner[:200]}")
if all(READS_ONLY.match(p.strip()) for p in parts if p.strip()):
    out("allow", "wsl.sh run: every segment reads only")

out("ask", f"wsl.sh run — this helper has no rule covering this "
    f"command, so showing it: {inner[:220]}")

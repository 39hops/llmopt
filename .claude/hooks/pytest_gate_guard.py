#!/usr/bin/env python3
"""PreToolUse guard: a commit must never be gated on PIPED pytest.

House law (memory 2026-08-07, fired again 2026-08-13 twice): in
`pytest ... | tail && git commit`, the chain reads the PIPE's exit
code, not pytest's — a red suite commits green. The legal shape is
a redirected run whose rc breaks the chain:

    pytest -q > /tmp/x.log 2>&1; rc=$?; ... ; [ $rc -eq 0 ] && git commit

Denies only the dangerous shape: pytest piped AND `git commit`
reachable through && in the same command.
"""
import json
import re
import sys

data = json.load(sys.stdin)
cmd = data.get("tool_input", {}).get("command", "") or ""

has_commit = re.search(r"&&[^|;]*git\s+commit", cmd)


def _piped(c: str) -> bool:
    # piped = a | appears after pytest before the next ; (chain break);
    # the legal redirected shape always breaks with ; before any pipe
    for m in re.finditer(r"pytest", c):
        if "|" in c[m.end():].split(";")[0]:
            return True
    return False


if has_commit and _piped(cmd):
    print(json.dumps({
        "decision": "block",
        "reason": ("piped pytest cannot gate a commit — the chain reads "
                   "the pipe's rc, not pytest's (house law 2026-08-07). "
                   "Redirect instead: pytest -q > /tmp/x.log 2>&1; "
                   "rc=$?; [ $rc -eq 0 ] && git commit ...")}))
    sys.exit(0)
sys.exit(0)

#!/usr/bin/env python3
"""PreToolUse guard: refuse shell mutation of a booked receipt path.

MITIGATION, not the invariant. The invariant is
docs/receipts.lock.json plus
tests/science_incidents/test_frozen_receipt_mutation.py, which
detect a mutation after the fact. This hook tries to stop the
common shapes before they happen.

Origin (2026-08-16, STREAM-WDISTILL-0): an audit-repair was
launched after `mv`-ing logs/streamwd/pass12_B1.jsonl — cited by a
booked entry — out of the way, which defeated the driver's own
refuse-if-exists guard and left the booked citation resolving to a
different run's numbers.

Denies (not asks) for mv/rm/cp or ANY redirect targeting any
logs/ path that appears in docs/RESULTS.md: the doctrine has no
legitimate exception, and a genuine new run belongs at a new path.
Fails open on any parse problem.

KNOWN GAP, recorded rather than patched: citations are scraped
from prose, so a receipt cited as a BARE filename is invisible to
this hook and to the lock. RESULTS L31402 cites
"logs/streamwd/pass12_B1.jsonl, run_B1.log" — the second one has
no path prefix and is therefore unprotected. A bare-filename
matcher would false-positive on ordinary words, so the real fix
is structured receipt references (the machine-readable-prereg
work), not a better regex.
"""
import json
import re
import sys
from pathlib import Path

CITE = re.compile(r"\blogs/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+\b")
# mv/rm/cp anywhere in the command, or ANY redirect into the path.
# Append is included deliberately: "never APPEND a new run into a
# path a booked verdict cites as frozen" is the doctrine verbatim.
MUTATORS = re.compile(r"(^|[;&|]\s*)(mv|rm|cp)\s", re.M)
# only a redirect whose TARGET is the cited path counts —
# `grep ... cited.jsonl > /tmp/out` reads it and is fine.
REDIRECT_TO = re.compile(r">>?\s*([A-Za-z0-9_.\-/]+)")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        cmd = payload.get("tool_input", {}).get("command", "")
        if not cmd:
            return
        root = Path(__file__).resolve().parents[2]
        results = root / "docs" / "RESULTS.md"
        if not results.exists():
            return

        touched = set(CITE.findall(cmd))
        if not touched:
            return
        cited = set(CITE.findall(results.read_text()))

        at_risk = set()
        if MUTATORS.search(cmd):
            at_risk |= touched              # mv/rm/cp names its victims
        for tgt in REDIRECT_TO.findall(cmd):
            at_risk |= {t for t in touched if t == tgt.lstrip("./")}
        hits = sorted(at_risk & cited)
        if not hits:
            return
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "REFUSING: this command mutates receipt path(s) cited "
                    f"by a booked entry in docs/RESULTS.md: {', '.join(hits)}. "
                    "A booked receipt is evidence — a new execution belongs "
                    "at a NEW path (use RUN_TAG or a per-run directory). "
                    "This exact manoeuvre corrupted the EXEC1 citation on "
                    "2026-08-16. If a receipt genuinely must change, do it "
                    "deliberately and re-lock with "
                    "scripts/gen_receipt_lock.py --accept \"reason\"."),
            }
        }))
    except Exception:
        return


if __name__ == "__main__":
    main()

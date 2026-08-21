#!/usr/bin/env python3
"""PreToolUse guard: refuse mutation of a booked receipt path.

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

AUTHORITY (upgraded 2026-08-21, automation review): the primary
protected set is docs/receipts.lock.json (sha-locked entries) —
structured references, immune to the prose-scraping bare-filename
gap. The RESULTS.md prose scrape is kept as a SUPPLEMENT for paths
cited but not yet locked. The hook now also covers Edit/Write
tool calls (file_path against the protected set), and FAILS
CLOSED for logs/ mutations when the lock exists but cannot be
parsed (an unreadable lock must block, not bypass; a repo without
the lock file fails open as before).

Denies (not asks) mv/rm/cp or ANY redirect targeting a protected
path, and any Edit/Write whose file_path is protected: the
doctrine has no legitimate exception, and a genuine new run
belongs at a new path. Disclosed regens go through
scripts/gen_receipt_lock.py --accept.
"""
import json
import re
import sys
from pathlib import Path

CITE = re.compile(r"\blogs/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+\b")
MUTATORS = re.compile(r"(^|[;&|]\s*)(mv|rm|cp)\s", re.M)
REDIRECT_TO = re.compile(r">>?\s*([A-Za-z0-9_.\-/]+)")


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))


def locked_paths(root: Path):
    """(protected_set, lock_broken). sha-locked entries only —
    pending/absent entries are not yet evidence."""
    lock = root / "docs" / "receipts.lock.json"
    if not lock.exists():
        return set(), False
    try:
        data = json.loads(lock.read_text())
        entries = data.get("receipts", data)
        return {p for p, meta in entries.items()
                if isinstance(meta, dict) and meta.get("sha256")}, False
    except Exception:
        return set(), True


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        tool = payload.get("tool_name", "")
        ti = payload.get("tool_input", {})
        root = Path(__file__).resolve().parents[2]
        protected, lock_broken = locked_paths(root)
        results = root / "docs" / "RESULTS.md"
        if results.exists():
            protected |= set(CITE.findall(results.read_text()))

        if tool in ("Edit", "Write"):
            fp = ti.get("file_path", "")
            if not fp:
                return
            try:
                rel = str(Path(fp).resolve().relative_to(root))
            except ValueError:
                return
            if lock_broken and rel.startswith("logs/"):
                deny("REFUSING (fail-closed): docs/receipts.lock.json "
                     "exists but cannot be parsed, so logs/ writes are "
                     "blocked until the lock is readable. Fix or "
                     "regenerate the lock first.")
                return
            if rel in protected:
                deny(f"REFUSING: {rel} is a booked receipt (sha-locked "
                     "or cited in RESULTS.md). Evidence is immutable — "
                     "a new run belongs at a NEW path. Disclosed regens "
                     "go through scripts/gen_receipt_lock.py --accept "
                     '"reason".')
            return

        cmd = ti.get("command", "")
        if not cmd:
            return
        touched = set(CITE.findall(cmd))
        mutating = bool(MUTATORS.search(cmd)) or bool(
            REDIRECT_TO.findall(cmd))
        if lock_broken and mutating and (
                touched or re.search(r"\blogs/", cmd)):
            deny("REFUSING (fail-closed): docs/receipts.lock.json "
                 "exists but cannot be parsed, so mutating commands "
                 "touching logs/ are blocked until the lock is "
                 "readable.")
            return
        if not touched:
            return
        at_risk = set()
        if MUTATORS.search(cmd):
            at_risk |= touched
        for tgt in REDIRECT_TO.findall(cmd):
            at_risk |= {t for t in touched if t == tgt.lstrip("./")}
        hits = sorted(at_risk & protected)
        if not hits:
            return
        deny("REFUSING: this command mutates booked receipt path(s) "
             f"(sha-locked or cited in RESULTS.md): {', '.join(hits)}. "
             "A booked receipt is evidence — a new execution belongs "
             "at a NEW path (use RUN_TAG or a per-run directory). "
             "This exact manoeuvre corrupted the EXEC1 citation on "
             "2026-08-16. If a receipt genuinely must change, do it "
             "deliberately and re-lock with "
             'scripts/gen_receipt_lock.py --accept "reason".')
    except Exception:
        return


if __name__ == "__main__":
    main()

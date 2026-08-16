#!/usr/bin/env python3
"""PreToolUse guard: a driver that has a SMOKE mode must isolate its paths.

House rule (CLAUDE.md, earned twice on 2026-08-15): SMOKE mode writes
receipts AND checkpoints to its own paths, and refuse-if-exists guards
stay unconditional. A smoke artifact on a real path cost a manual
delete and an auditor blocker, and disclosure in a booked verdict.

Fires only on scratch/ files that BOTH read a SMOKE flag and name a
checkpoint path — i.e. birth-style drivers. Emits permissionDecision
=ask listing the missing signals; never blocks outright, because
three frozen drivers legitimately predate the rule. Fails open: any
parse problem allows the call.

Measured coverage when this landed: 4 of 14 scratch/birth19m_*.py
carried smoke path isolation.
"""
import json
import re
import sys
from pathlib import Path


def signals(text: str) -> list[str]:
    """Return the list of MISSING smoke-isolation signals."""
    missing = []
    # 1. checkpoint path forks on SMOKE
    ckpt_line = re.search(
        r"OUT\s*=\s*Path\(([^)]*)\)", text, re.S)
    if not (ckpt_line and "_smoke" in ckpt_line.group(1)):
        if "_smoke" not in text:
            missing.append(
                "checkpoint path does not fork on SMOKE "
                "(expected a '_smoke' suffix in the OUT path)")
    # 2. receipts fork on SMOKE
    if not re.search(r"smoke\.jsonl", text):
        missing.append(
            "receipt path does not fork on SMOKE "
            "(expected a separate smoke.jsonl)")
    # 3. refuse guard exists AND is not nested under a not-SMOKE test
    guard = re.search(r"^(\s*)if\s+OUT\.exists\(\)", text, re.M)
    if not guard:
        missing.append(
            "no refuse-if-exists guard on the checkpoint path")
    elif len(guard.group(1)) > 0:
        missing.append(
            "refuse-if-exists guard is INDENTED — it may be nested "
            "under a SMOKE conditional; the guard must be "
            "unconditional")
    return missing


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        ti = payload.get("tool_input", {})
        fp = ti.get("file_path", "")
        if not fp or not fp.endswith(".py"):
            return
        root = Path(__file__).resolve().parents[2]
        try:
            rel = Path(fp).resolve().relative_to(root)
        except ValueError:
            return
        if rel.parts[0] != "scratch":
            return

        # Write carries full content; Edit carries a fragment, so read
        # the file underneath it and consider the union.
        text = ti.get("content") or ""
        if not text:
            existing = Path(fp)
            base = existing.read_text() if existing.exists() else ""
            text = base + "\n" + (ti.get("new_string") or "")
        if not text.strip():
            return

        # Only birth-style drivers: a SMOKE flag AND a checkpoint path.
        has_smoke = re.search(r"\bSMOKE\b", text) is not None
        has_ckpt = "checkpoints/" in text
        if not (has_smoke and has_ckpt):
            return

        missing = signals(text)
        if not missing:
            return
        bullets = "; ".join(missing)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": (
                    f"{rel} looks like a birth driver with a SMOKE mode, "
                    f"but smoke path isolation is incomplete: {bullets}. "
                    "House rule (CLAUDE.md): SMOKE writes receipts AND "
                    "checkpoints to its own paths, refuse guards stay "
                    "unconditional. Cost two incidents on 2026-08-15. "
                    "Allow if this driver is frozen-as-record or has no "
                    "real-path risk."),
            }
        }))
    except Exception:
        return


if __name__ == "__main__":
    main()

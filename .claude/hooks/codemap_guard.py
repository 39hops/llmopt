#!/usr/bin/env python3
"""PreToolUse guard: Edit/Write to a CODEMAP-frozen file asks first.

Reads the tool-call JSON on stdin, looks the target path up in
docs/CODEMAP.md, and for class results-cited or reproduce-pinned emits
a permissionDecision=ask so the edit needs an explicit user OK.
Everything else (library, spec-cited, UNCITED, files outside the
CODEMAP inventory) passes through silently. Fails open: any parse
problem allows the call.
"""
import json
import re
import sys
from pathlib import Path

FROZEN = {"results-cited", "reproduce-pinned"}


def codemap_class(rel: str, codemap_text: str) -> str | None:
    name = Path(rel).name
    # Rows look like: | family | file.py | class | citations | refs |
    for m in re.finditer(r"^\|[^|]*\|\s*(\S+)\s*\|\s*(\S+)\s*\|",
                         codemap_text, re.M):
        if m.group(1) == name:
            return m.group(2)
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        fp = payload.get("tool_input", {}).get("file_path", "")
        if not fp:
            return
        root = Path(__file__).resolve().parents[2]
        rel = Path(fp).resolve()
        try:
            rel = rel.relative_to(root)
        except ValueError:
            return
        if rel.parts[0] not in ("scratch", "scripts"):
            return
        codemap = root / "docs" / "CODEMAP.md"
        if not codemap.exists():
            return
        cls = codemap_class(str(rel), codemap.read_text())
        if cls in FROZEN:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": (
                        f"{rel} is CODEMAP class '{cls}' (evidence record; "
                        "cited by booked verdicts). Legit reasons to edit: "
                        "dual-copy fix landing in both copies same commit, "
                        "or an adoption migration. Otherwise extend the "
                        "adopted lab module instead."),
                }
            }))
    except Exception:
        return


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PreToolUse guard: Edit/Write to a CODEMAP-frozen file asks first.

Reads the tool-call JSON on stdin, looks the target path up in
docs/CODEMAP.md, and emits a permissionDecision=ask when the row is
EVIDENCE: class results-cited or reproduce-pinned, or a `cited by`
column carrying RESULTS / REPRODUCE (own, or inherited as
`via:RESULTS` / `via:REPRODUCE` through a cited caller) even when the
class is `library` because code also imports it. spec-cited,
tool-referenced, UNCITED, uncited library rows and files outside the
inventory pass through silently. Fails open: any parse problem allows
the call.
"""
import json
import re
import sys
from pathlib import Path

FROZEN = {"results-cited", "reproduce-pinned"}
FROZEN_GROUPS = {"RESULTS", "REPRODUCE"}


def codemap_row(rel: str, codemap_text: str) -> tuple[str, str] | None:
    """(class, cited-by) for the file, or None when it has no row.
    Rows look like: | family | file.py | class | cited by | ... |"""
    name = Path(rel).name
    # cells never contain "|": `[^|\s]+` keeps the separator row
    # (`|---|---|`) from swallowing the first data row of a section
    for m in re.finditer(r"^\|[^|]*\|\s*([^|\s]+)\s*\|\s*([^|\s]+)\s*\|([^|]*)\|",
                         codemap_text, re.M):
        if m.group(1) == name:
            return m.group(2), m.group(3).strip()
    return None


def codemap_class(rel: str, codemap_text: str) -> str | None:
    row = codemap_row(rel, codemap_text)
    return row[0] if row else None


def is_frozen(rel: str, codemap_text: str) -> bool:
    """Frozen = the evidence record: class results-cited or
    reproduce-pinned, OR any row whose `cited by` carries RESULTS /
    REPRODUCE, own or inherited (`via:` prefix), even when code also
    imports it (class `library` outranks the citation in the ladder;
    it must not outrank the freeze)."""
    row = codemap_row(rel, codemap_text)
    if not row:
        return False
    cls, cited_by = row
    groups = ({g.strip().removeprefix("via:") for g in cited_by.split(",")}
              if cited_by != "—" else set())
    return cls in FROZEN or bool(groups & FROZEN_GROUPS)


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
        text = codemap.read_text()
        cls = codemap_class(str(rel), text)
        if is_frozen(str(rel), text):
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": (
                        f"{rel} is CODEMAP class '{cls}' and cited by "
                        "RESULTS/REPRODUCE (evidence record; cited by "
                        "booked verdicts). Legit reasons to edit: "
                        "dual-copy fix landing in both copies same commit, "
                        "or an adoption migration. Otherwise extend the "
                        "adopted lab module instead."),
                }
            }))
    except Exception:
        return


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print FINDINGS ratchet headroom after a RESULTS.md edit.

Mirrors tests/test_docs_integrity.py::_uncurated: entries in
docs/results-index.jsonl whose line is not cited by a FINDINGS
RESULTS.md#L<n> anchor count against MAX_UNCURATED. Output is one
line; a PostToolUse hook surfaces it in the transcript so headroom
is visible at booking time, not at CI time.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURATABLE = ("verdict", "null")  # fallback; live value parsed from the test


def main() -> None:
    text = (ROOT / "tests" / "test_docs_integrity.py").read_text()
    max_m = re.search(r"^MAX_UNCURATED\s*=\s*(\d+)", text, re.M)
    cur_m = re.search(r"^CURATABLE\s*=\s*(\(.*?\))", text, re.M | re.S)
    limit = int(max_m.group(1)) if max_m else 320
    curatable = eval(cur_m.group(1)) if cur_m else CURATABLE  # literal tuple
    cited = {int(n) for n in re.findall(
        r"RESULTS\.md#L(\d+)", (ROOT / "docs" / "FINDINGS.md").read_text())}
    rows = [json.loads(line)
            for line in (ROOT / "docs" / "results-index.jsonl").open()]
    stale = [r for r in rows
             if r["line"] not in cited and r["type"] in curatable]
    headroom = limit - len(stale)
    line = (f"FINDINGS ratchet: {len(stale)}/{limit} uncurated, "
            f"headroom {headroom}")
    if headroom <= 0:
        # PostToolUse exit 2 feeds stderr to the model (non-blocking).
        print(f"{line} [CI WILL FAIL]. This booking reddens CI unless "
              "its FINDINGS bullet lands in the same commit and the "
              "backlog does not grow.", file=sys.stderr)
        sys.exit(2)
    print(f"{line} [OK]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"findings_headroom: skipped ({e})", file=sys.stderr)

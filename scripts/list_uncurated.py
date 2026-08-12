#!/usr/bin/env python3
"""Print the oldest uncurated ledger entries (candidates for FINDINGS).

Same definition as tests/test_docs_integrity.py::_uncurated: a row in
docs/results-index.jsonl whose RESULTS line is not cited by any
RESULTS.md#L<n> anchor in FINDINGS.md and whose type is curatable.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURATABLE = ("verdict", "null")

def main(n: int = 20) -> None:
    cited = {int(m) for m in re.findall(
        r"RESULTS\.md#L(\d+)",
        (ROOT / "docs" / "FINDINGS.md").read_text())}
    rows = [json.loads(l) for l in
            (ROOT / "docs" / "results-index.jsonl").open()]
    stale = [r for r in rows
             if r["line"] not in cited and r["type"] in CURATABLE]
    stale.sort(key=lambda r: r["line"])  # ledger order = age order
    for r in stale[:n]:
        print(f"L{r['line']:>6}  {r.get('date') or 'undated':<12} "
              f"{r['type']:<8} {r['title']}")
    print(f"\n{len(stale)} uncurated total", file=sys.stderr)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)

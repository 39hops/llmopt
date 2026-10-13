"""Generate docs/SCOREBOARD.md from results-index.jsonl: the entries
whose index row carries a hand-curated `verdict` field and no
`superseded_by`, grouped by first thread, newest first.

Scope, stated in the output header because the view is PARTIAL: the
`verdict` field is curated by hand and covers a minority of ledger
entries, so the scoreboard is a sample of the record, never "current
truth". docs/FINDINGS.md is the curated public layer; the index and
`scripts/results_query.py --chain` carry the full record. NEVER
hand-edit SCOREBOARD.md; deepen the index and regenerate.

    .venv/bin/python scripts/gen_scoreboard.py          # rewrite
    .venv/bin/python scripts/gen_scoreboard.py --check  # exit 1 on drift
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "results-index.jsonl"
OUT = ROOT / "docs" / "SCOREBOARD.md"


def render() -> str:
    entries = [json.loads(line) for line in INDEX.read_text().splitlines()
               if line.strip()]
    curated = [e for e in entries if e.get("verdict")]
    live = [e for e in curated if not e.get("superseded_by")]
    newest_curated = max(e["line"] for e in curated) if curated else 0
    by_thread = defaultdict(list)
    for e in live:
        # an empty threads list still places the entry (under misc);
        # the earlier [:1] slice silently dropped such rows
        by_thread[(e.get("threads") or ["misc"])[0]].append(e)

    out = [
        "# SCOREBOARD — generated sample of index-curated verdicts",
        "",
        f"*Generated from docs/results-index.jsonl by "
        f"scripts/gen_scoreboard.py. {len(live)} live of {len(curated)} "
        f"entries carrying a hand-curated `verdict` field, out of "
        f"{len(entries)} ledger entries; the curated field reaches "
        f"RESULTS line {newest_curated}. This is a PARTIAL view, not "
        "current truth: docs/FINDINGS.md is the curated public layer, "
        "and every entry resolves through docs/RESULTS.md. Full chains: "
        "scripts/results_query.py --chain. Never hand-edit this file.*",
        ""]
    for t in sorted(by_thread,
                    key=lambda k: -max(x["line"] for x in by_thread[k])):
        out.append(f"## {t}")
        for e in sorted(by_thread[t], key=lambda x: -x["line"]):
            out.append(f"- **{e.get('date', '?')}** {e['verdict']}  "
                       f"*(RESULTS L{e['line']})*")
        out.append("")
    return "\n".join(out)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        if OUT.exists() and OUT.read_text() == text:
            print("[scoreboard] current")
            return 0
        print("[scoreboard] STALE: run scripts/gen_scoreboard.py",
              file=sys.stderr)
        return 1
    OUT.write_text(text)
    n_live = sum(1 for line in text.splitlines() if line.startswith("- **"))
    print(f"SCOREBOARD.md: {n_live} verdicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

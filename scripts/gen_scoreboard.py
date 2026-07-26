"""Generate docs/SCOREBOARD.md from results-index.jsonl — the
curated current-truth view (live, verdict-bearing entries grouped
by thread, newest first). NEVER hand-edit SCOREBOARD.md; deepen
the index instead and regenerate.

    .venv/bin/python scripts/gen_scoreboard.py
"""
import json
from collections import defaultdict

E = [json.loads(l) for l in open("docs/results-index.jsonl")]
live = [e for e in E if e.get("verdict") and not e.get("superseded_by")]
by_thread = defaultdict(list)
for e in live:
    for t in e.get("threads", ["misc"])[:1]:
        by_thread[t].append(e)

out = ["# SCOREBOARD — current truth, generated from results-index.jsonl",
       "",
       f"*{len(live)} live curated verdicts of {len(E)} entries; "
       "regenerate with scripts/gen_scoreboard.py; deepen via the "
       "index, never here. Full chains: results_query.py --chain.*",
       ""]
for t in sorted(by_thread,
                key=lambda k: -max(x["line"] for x in by_thread[k])):
    out.append(f"## {t}")
    for e in sorted(by_thread[t], key=lambda x: -x["line"]):
        out.append(f"- **{e.get('date','?')}** {e['verdict']}  "
                   f"*(RESULTS L{e['line']})*")
    out.append("")
open("docs/SCOREBOARD.md", "w").write("\n".join(out))
print(f"SCOREBOARD.md: {len(live)} verdicts, {len(by_thread)} threads")

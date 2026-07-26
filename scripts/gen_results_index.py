"""Generate/refresh docs/results-index.jsonl from RESULTS.md.

Auto-extracts id/date/title/line/type per entry; PRESERVES any
hand-curated fields (threads, verdict, amends, superseded_by) from
the existing index on regeneration (merge by id). Entries whose
title marks them as amendments but lack an `amends` link get
`needs_link: true` for incremental curation.

    .venv/bin/python scripts/gen_results_index.py
"""
import json
import re
from pathlib import Path

SRC = Path("docs/RESULTS.md")
DST = Path("docs/results-index.jsonl")

TYPE_RULES = [
    (r"^PRE-REG", "prereg"),
    (r"AMENDMENT|AMENDMENTS|RESTATEMENT|RETRACT", "amendment"),
    (r"\bNULL\b|VOID|DIES|DEAD|FAILS|NO-ADOPT|NO SPONTANEOUS", "null"),
    (r"BANKED", "banked"),
    (r"CLOSES|CLOSED|COMPLETES|VERDICT|BOOKS|ADJUDICAT", "verdict"),
]


def slug(title, date):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{date or 'undated'}-{'-'.join(s.split('-')[:6])}"


def infer_type(title):
    for pat, t in TYPE_RULES:
        if re.search(pat, title):
            return t
    return "verdict"


old = {}
if DST.exists():
    for line in DST.read_text().splitlines():
        e = json.loads(line)
        old[e["id"]] = e

entries = []
seen = set()
for ln, line in enumerate(SRC.read_text().splitlines(), 1):
    if not line.startswith("## ") or line.startswith("## Contents"):
        continue
    title = line[3:].strip()
    m = re.search(r"\((\d{4}-\d{2}-\d{2})", title)
    date = m.group(1) if m else None
    eid = slug(title, date)
    while eid in seen:
        eid += "-b"
    seen.add(eid)
    t = infer_type(title)
    e = {"id": eid, "date": date, "line": ln, "title": title,
         "type": t}
    if t == "amendment":
        e["needs_link"] = True
    prev = old.get(eid, {})
    for k in ("threads", "verdict", "amends", "superseded_by"):
        if k in prev:
            e[k] = prev[k]
    if "amends" in e:
        e.pop("needs_link", None)
    entries.append(e)

with DST.open("w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")
kinds = {}
for e in entries:
    kinds[e["type"]] = kinds.get(e["type"], 0) + 1
print(f"{len(entries)} entries -> {DST}  {kinds}")
print(f"needs_link: {sum(1 for e in entries if e.get('needs_link'))}")

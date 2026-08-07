"""Checkpoint triage TABLE builder (Artin GO 2026-08-07; follows
ckpt_inventory.py). READ-ONLY desk step: joins the two machine
inventories (logs/triage/{mac,wsl}_inventory.jsonl), dedups by
sha256, and classifies every path against the evidence record
(docs/RESULTS.md + docs/REPRODUCE.md + jobs/*.cmd basename grep).

Classes:
  CITED       basename appears in RESULTS/REPRODUCE/jobs receipts
  DUP-CITED   sha256 identical to a CITED file (safe dedup target)
  DUP         sha256 duplicate within/across machines, none cited
  UNCITED     no citation hit, unique sha

Emits logs/triage/triage_table.md (the Artin-review artifact) and
logs/triage/triage_table.jsonl. NOTHING deletes here — the table is
the input to that decision, per the standing provenance rule.
"""
import json
from collections import defaultdict
from pathlib import Path

rows = []
for host, fn in [("mac", "logs/triage/mac_inventory.jsonl"),
                 ("wsl", "logs/triage/wsl_inventory.jsonl")]:
    for line in Path(fn).open():
        r = json.loads(line)
        r["host"] = host
        rows.append(r)

cite_text = ""
for fn in ["docs/RESULTS.md", "docs/REPRODUCE.md"]:
    cite_text += Path(fn).read_text()
for p in Path("jobs").glob("*.cmd"):
    cite_text += p.read_text()

# citation match on basename AND on run-dir name (checkpoints/<run>/...)
cited_names = set()
for r in rows:
    parts = Path(r["path"]).parts
    for token in {Path(r["path"]).name} | set(parts[1:2]):
        if len(token) >= 6 and token in cite_text:
            cited_names.add(r["path"])
            break

by_sha = defaultdict(list)
for r in rows:
    by_sha[r["sha256"]].append(r)

for r in rows:
    twins = by_sha[r["sha256"]]
    cited = r["path"] in cited_names
    twin_cited = any(t["path"] in cited_names for t in twins)
    if cited:
        r["cls"] = "CITED"
    elif len(twins) > 1 and twin_cited:
        r["cls"] = "DUP-CITED"
    elif len(twins) > 1:
        r["cls"] = "DUP"
    else:
        r["cls"] = "UNCITED"
    r["ndup"] = len(twins)

out = Path("logs/triage/triage_table.jsonl")
with out.open("w") as f:
    for r in sorted(rows, key=lambda r: -r["bytes"]):
        f.write(json.dumps(r) + "\n")

md = Path("logs/triage/triage_table.md")
with md.open("w") as f:
    f.write("# Checkpoint triage table (READ-ONLY; Artin review "
            "gates any move/delete)\n\n")
    for cls in ["CITED", "DUP-CITED", "DUP", "UNCITED"]:
        sub = [r for r in rows if r["cls"] == cls]
        gb = sum(r["bytes"] for r in sub) / 2**30
        f.write(f"## {cls}: {len(sub)} files, {gb:.1f} GiB\n\n")
        # top 25 by size, grouped view
        for r in sorted(sub, key=lambda r: -r["bytes"])[:25]:
            f.write(f"- {r['host']}:{r['path']} "
                    f"{r['bytes']/2**20:.0f} MiB x{r['ndup']}\n")
        if len(sub) > 25:
            f.write(f"- ... {len(sub)-25} more (see jsonl)\n")
        f.write("\n")
    tot = sum(r["bytes"] for r in rows) / 2**30
    f.write(f"TOTAL: {len(rows)} files, {tot:.1f} GiB\n")
print(f"wrote {md} and {out}")
for cls in ["CITED", "DUP-CITED", "DUP", "UNCITED"]:
    sub = [r for r in rows if r["cls"] == cls]
    print(f"{cls:10s} {len(sub):6d} files "
          f"{sum(r['bytes'] for r in sub)/2**30:8.1f} GiB")

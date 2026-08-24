"""Generate/refresh docs/results-index.jsonl from RESULTS.md.

Auto-extracts id/date/title/line/type per entry; PRESERVES any
hand-curated fields (threads, verdict, amends, superseded_by,
links, code_commit) from the existing index on regeneration
(merge by id). Entries whose
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
    (r"^OBSERVATION\b", "observation"),
    (r"\bNULL\b|VOID|DIES|DEAD|FAILS|NO-ADOPT|NO SPONTANEOUS", "null"),
    (r"BANKED", "banked"),
    (r"CLOSES|CLOSED|COMPLETES|VERDICT|BOOKS|ADJUDICAT", "verdict"),
]


def slug(title, date):
    # Date-first headings: drop the leading date so it does not
    # repeat inside the slug body.
    title = re.sub(r"^\d{4}-\d{2}-\d{2}\s*[—-]*\s*", "", title)
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{date or 'undated'}-{'-'.join(s.split('-')[:6])}"


def infer_type(title):
    for pat, t in TYPE_RULES:
        if re.search(pat, title):
            return t
    return "verdict"


THREAD_RULES = [
    (r"stream|epoch|cooldown|clade-gated", "streaming"),
    (r"format ladder|revpairs|traces|oneshot|delta|randpack", "format-ladder"),
    (r"ternary|alphabet|binary|bits|int[0-9]|GPTQ|Lloyd|quant|codebook|M4|M5|S4|G5|P2", "alphabet"),
    (r"\bZX\b|zx_farm|T-count", "zx"),
    (r"Z1|opposition|Dale|sign", "opposition"),
    (r"complex|NNUE|euler|rotation|G16", "complex"),
    (r"metabolic|LLMUE|retention|flip|exchange|absorption|practice", "metabolic"),
    (r"GRPO|reward|RL\b|shaped", "rl"),
    (r"gauge|distance|perm|Procrustes", "gauge"),
    (r"width|capacity|113M|400M|45M|W\*|W_min|d256|scaling", "width"),
    (r"axiom|adjudicat|hybrid|qualification|tranche|Phase [A-D]", "axiom"),
    (r"series|poly|physics|energy|bridge|union|federation|gen-[0-9]|vm-asm|continent|ODE", "continents"),
    (r"template|warm birth|calculator|equation|birth", "birth"),
    (r"precision|fp64|fp32|bf16|TF32|Ozaki|RNS|exact", "precision"),
    (r"estimator|magic|syndrome|dispatcher|router|policy|regret|probe|instrument|census|sidecar|sigma", "instruments"),
    (r"packing|--fast|nopack", "packing"),
    (r"kernel|Metal|GEMV|KV|speed|wall|43x", "speed"),
    (r"engine|beam|best-first|L[0-9]\b|autopsy|rule", "engine"),
]


def infer_threads(title):
    out = []
    for pat, t in THREAD_RULES:
        if re.search(pat, title, re.I) and t not in out:
            out.append(t)
    return out[:3]


FILE_RE = re.compile(
    r"\b((?:scratch|scripts|llmopt)/[\w./-]+\.(?:py|sh))\b")


def extract_files(body: str) -> list[str]:
    """Sorted unique repo paths cited in an entry body."""
    return sorted(set(FILE_RE.findall(body)))


old = {}
if DST.exists():
    for line in DST.read_text().splitlines():
        e = json.loads(line)
        old[e["id"]] = e

lines = SRC.read_text().splitlines()
header_lines = [
    ln for ln, line in enumerate(lines, 1)
    if line.startswith("## ") and not line.startswith("## Contents")
]

entries = []
seen = set()
for idx, ln in enumerate(header_lines):
    line = lines[ln - 1]
    body_end = (header_lines[idx + 1] - 1
                if idx + 1 < len(header_lines) else len(lines))
    body = "\n".join(lines[ln:body_end])
    title = line[3:].strip()
    m = (re.search(r"\((\d{4}-\d{2}-\d{2})", title)
        or re.search(r";\s*(\d{4}-\d{2}-\d{2})", title)
        or re.match(r"(\d{4}-\d{2}-\d{2})\b", title)
        or re.search(r"\b(\d{4}-\d{2}-\d{2})\b", title))
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
    e["threads"] = infer_threads(title)
    e["files"] = extract_files(body)
    # Rows that were undated before the date-first heading pattern
    # landed keep their curation via the old undated slug.
    prev = old.get(eid) or old.get(slug(title, None), {})
    for k in ("threads", "verdict", "amends", "superseded_by", "links",
              "code_commit", "files"):
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

"""Phase 2F preparation: resolve the dynamic checkpoint path sites the
census could not evaluate, tell producers from consumers, and write a
file-by-file plan for every row the census did not freeze. Executes
nothing: no delete, move, clone or relink happens here or anywhere
until Artin reads the plan and gives a per-step GO.

Inputs: a census inventory.jsonl (scripts/ckpt_census.py) and the
repository. For every UNKNOWN, CANDIDATE and DUPLICATE row the plan
lists:
  - every code site whose path pattern matches the file, with the
    site's role (producer writes the file; consumer reads it;
    mention is prose) and whether the match is exact (every hole in
    the pattern resolves to literal values in the same file and the
    file's name is one of them) or only a pattern;
  - the CODEMAP class of each matching script;
  - independent copies on other trees;
  - a proposed action with its approval boundary.

Variable resolution reads the same file only: Python literal lists
and tuples, loops over those names, os.environ.get defaults, shell
`for X in a b c` loops and `${X:-default}`. A hole bound to an
argument, an environment variable without a default, or an
expression stays unresolved and the match stays "pattern".

Usage:
    .venv/bin/python scripts/ckpt_plan.py --inventory logs/rehab-2e/run4/inventory.jsonl
        [--copies logs/rehab-2e/run4/copies.json] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ("scratch", "scripts", "llmopt", "tests", "jobs")
CODE_EXTS = (".py", ".sh", ".cmd")
SITE = re.compile(r"checkpoints/[\w\-./{}$*%:()\[\]]*[\w}*)\]]")
HOLE = re.compile(r"\{([^{}]*)\}|\$\{([A-Za-z_]\w*)[^}]*\}|\$([A-Za-z_]\w*)|(\*)")
PRODUCER = re.compile(r"\bOUT\b|--out\b|torch\.save|\.save\(|\bsave\(|\bcp\s|printf|open\([^)]*[\"']w|>\s*\"?checkpoints/|\bout\s*=|\bOUT_|_out\b|writes?\b")
CONSUMER = re.compile(r"torch\.load|\bload\(|open\(|gate_ckpt|holdout|probe|glob|exists|read|json\.load|\bck\b|\bCKPT\b|CK_|_PATHS|targets?\b|\bref\b|cmp|compare|convert|snap|\$PY\s|python")


def _hole_name(m) -> str:
    g = m.group(1)
    if g is not None:
        g = g.strip()
        name = re.split(r"[:!\[\.\(\s\+\*]", g, 1)[0]
        return name if re.fullmatch(r"[A-Za-z_]\w*", name or "") else "<expr>"
    return m.group(2) or m.group(3) or "*"


VARROOT = re.compile(r"(?:\{[A-Za-z_]\w*\}|\$\{?[A-Za-z_]\w*\}?)/([\w\-.]+\.(?:pt|bin|npz|npy|json|jsonl|safetensors|pth))\b")
JOIN_LITERAL = re.compile(r"join\([^)]*?[\"']([\w\-.]+\.(?:pt|bin|npz|npy|json|jsonl|safetensors|pth))[\"']\s*\)")


def parse_site(line: str, source: str, lineno: int) -> dict | None:
    """One code line -> {source, line, text, holes, regex, role}, or
    None when the line carries no dynamic checkpoints/ path. A literal
    basename under a variable root ({CKPT}/name.pt, join(X, "name.pt"))
    is an exact site assumed to live at the top level of checkpoints/."""
    m = SITE.search(line)
    if not m:
        vr = VARROOT.search(line) or JOIN_LITERAL.search(line)
        if not vr:
            return None
        literal = "checkpoints/" + vr.group(1)
        role = ("producer" if PRODUCER.search(line)
                and not re.search(r"torch\.load|json\.load|gate_ckpt", line)
                else "consumer")
        return {"source": source, "line": lineno, "text": literal, "holes": [],
                "regex": re.compile(re.escape(literal)), "role": role,
                "generic": False, "exact_literal": literal}
    text = m.group(0).rstrip(".,;:)")
    holes = [_hole_name(h) for h in HOLE.finditer(text)]
    if not holes:
        return None
    pattern = ""
    pos = 0
    for h in HOLE.finditer(text):
        pattern += re.escape(text[pos:h.start()]) + r"[^/]+?"
        pos = h.end()
    pattern += re.escape(text[pos:])
    stripped = line.strip()
    quoted = re.search(r"[\"'`][^\"'`]*checkpoints/", line) is not None
    codeish = ("$" in line or "=" in line or quoted
               or re.search(r"\b(for|in|do|ls|cp|rm|mv|python|cat|glob)\b", line))
    if stripped.startswith("#") or stripped.startswith('"""') or not codeish or re.match(
            r"^\s*(Checkpoint|Outputs?|Writes|Snapshots|byte-frozen|stock-|SequenceLR|No follow-up|checkpoints/)", stripped):
        role = "mention"
    elif PRODUCER.search(line) and not re.search(r"torch\.load|json\.load|gate_ckpt", line):
        role = "producer"
    elif CONSUMER.search(line):
        role = "consumer"
    else:
        role = "consumer"
    first_hole = HOLE.search(text)
    generic = first_hole is not None and text[:first_hole.start()] == "checkpoints/"
    return {"source": source, "line": lineno, "text": text, "holes": holes,
            "regex": re.compile(pattern), "role": role, "generic": generic,
            "exact_literal": None}


def variable_values(text: str) -> dict[str, list[str]]:
    """Literal values a variable can take, read from one file."""
    text = re.sub(r"\\\n\s*", " ", text)   # shell line continuations
    vals: dict[str, list[str]] = {}
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*[\[\(]([^\]\)]*)[\]\)]", text, re.M):
        items = re.findall(r"[\"']([^\"']+)[\"']|(?<![\w.])(-?\d+)(?![\w.])", m.group(2))
        flat = [a or b for a, b in items]
        if flat:
            vals[m.group(1)] = flat
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(?:int\()?\s*os\.environ\.get\(\s*[\"']\w+[\"']\s*,\s*[\"']?([^\"'\)]*)[\"']?\s*\)", text, re.M):
        vals[m.group(1)] = [m.group(2)]
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(-?\d+|[\"'][^\"']+[\"'])\s*$", text, re.M):
        vals.setdefault(m.group(1), [m.group(2).strip("\"'")])
    for m in re.finditer(r"for\s+([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)\s*:", text):
        if m.group(2) in vals:
            vals[m.group(1)] = vals[m.group(2)]
    for m in re.finditer(r"for\s+([A-Za-z_]\w*)\s+in\s+[\[\(]([^\]\)]*)[\]\)]\s*:", text):
        items = re.findall(r"[\"']([^\"']+)[\"']|(?<![\w.])(-?\d+)(?![\w.])", m.group(2))
        flat = [a or b for a, b in items]
        if flat:
            vals[m.group(1)] = flat
    for m in re.finditer(r"for\s+([A-Za-z_]\w*)\s+in\s+([^;\n]+?)\s*;?\s*do\b", text):
        words = m.group(2).split()
        if words and all(re.fullmatch(r"[\w.\-]+", w) for w in words):
            vals[m.group(1)] = words
    for m in re.finditer(r"\$\{([A-Za-z_]\w*):-([^}]*)\}", text):
        vals.setdefault(m.group(1), [m.group(2)])
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)=([\w.\-]+)\s*$", text, re.M):
        vals.setdefault(m.group(1), [m.group(2)])
    return vals


def expand(site: dict, vals: dict[str, list[str]]) -> list[str] | None:
    """Every exact path the site can produce, or None if a hole is
    unresolved."""
    if site.get("exact_literal"):
        return [site["exact_literal"]]
    choices = []
    for h in site["holes"]:
        if h not in vals or h in ("*", "<expr>"):
            return None
        choices.append(vals[h])
    text = site["text"]
    out = set()
    for combo in itertools.product(*choices):
        s, i, pos = "", 0, 0
        for h in HOLE.finditer(text):
            s += text[pos:h.start()] + combo[i]
            i += 1
            pos = h.end()
        s += text[pos:]
        out.add(s)
    return sorted(out)


def collect_sites(repo: Path) -> list[dict]:
    sites = []
    for d in CODE_DIRS:
        base = repo / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix not in CODE_EXTS or not p.is_file():
                continue
            if any(part in (".venv", "__pycache__", ".lake") for part in p.parts):
                continue
            rel = str(p.relative_to(repo))
            if rel in ("scripts/ckpt_census.py", "scripts/ckpt_plan.py",
                       "tests/test_ckpt_census.py", "tests/test_ckpt_plan.py"):
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            vals = variable_values(text)
            for i, line in enumerate(text.splitlines(), 1):
                if "checkpoints/" not in line and not (
                        VARROOT.search(line) or JOIN_LITERAL.search(line)):
                    continue
                site = parse_site(line, rel, i)
                if site is None:
                    continue
                site["exact"] = expand(site, vals)
                sites.append(site)
    return sites


def load_codemap(repo: Path) -> dict[str, str]:
    out = {}
    p = repo / "docs/CODEMAP.md"
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[1].endswith(CODE_EXTS):
            out[cells[1]] = cells[2]
    return out


def build_plan(repo: Path, inventory_path: Path, copies: dict) -> dict:
    repo = Path(repo)
    rows_in = [json.loads(l) for l in Path(inventory_path).read_text().splitlines() if l.strip()]
    sites = collect_sites(repo)
    codemap = load_codemap(repo)
    rows = []
    for r in rows_in:
        if r["class"] not in ("UNKNOWN", "CANDIDATE", "DUPLICATE"):
            continue
        rp = r["repo_path"]
        matches = []
        for s in sites:
            if s["role"] == "mention":
                continue
            if s["exact"] is not None:
                if rp in s["exact"]:
                    matches.append({"source": s["source"], "line": s["line"],
                                    "role": s["role"], "match": "exact",
                                    "pattern": s["text"]})
            elif not s["generic"] and s["regex"].fullmatch(rp):
                # a generic site (hole right after checkpoints/) reaches
                # every file by construction; it counts only when it
                # expands to exact names
                matches.append({"source": s["source"], "line": s["line"],
                                "role": s["role"], "match": "pattern",
                                "pattern": s["text"]})
        cm = {m["source"]: codemap.get(Path(m["source"]).name, "UNCITED") for m in matches}
        consumer_exact = sorted({m["source"] for m in matches
                                 if m["role"] == "consumer" and m["match"] == "exact"})
        producer_exact = sorted({m["source"] for m in matches
                                 if m["role"] == "producer" and m["match"] == "exact"})
        consumer_pattern = sorted({m["source"] for m in matches
                                   if m["role"] == "consumer" and m["match"] == "pattern"})
        cited_consumers = [s for s in consumer_exact
                           if cm.get(s) in ("results-cited", "reproduce-pinned", "spec-cited", "library")]
        file_copies = {name: table[r["path"]] for name, table in copies.items()
                       if r["path"] in table}
        if r["class"] == "DUPLICATE":
            deps = [d for d in r.get("dependencies", []) if not d.startswith("logs/")]
            proposed = (f"RELINK {', '.join(deps)} to {r['canonical']}, then REMOVE (per-row GO)"
                        if deps else f"REMOVE after confirming {r['canonical']} is the cited copy (per-row GO)")
        elif r["class"] == "CANDIDATE":
            proposed = "REVIEW: no reference found (CANDIDATE); per-family GO"
        elif cited_consumers:
            proposed = "KEEP: exact consumer in results-cited script"
        elif consumer_exact:
            proposed = "REVIEW: exact consumer only in uncited script(s); decide script first"
        elif consumer_pattern:
            proposed = "REVIEW: reached only by an unresolved consumer pattern; resolve or decide script"
        elif producer_exact:
            proposed = "REVIEW: produced by a known script, read by nothing exact; regenerable if the script and its inputs stand"
        else:
            proposed = "REVIEW: reached only by producer or unresolved patterns"
        rows.append({"path": rp, "bytes": r["bytes"], "alloc": r["alloc"],
                     "mtime": time.strftime("%Y-%m-%d", time.localtime(r["mtime"])),
                     "class": r["class"], "family": r["family"],
                     "canonical": r.get("canonical"),
                     "matches": matches, "codemap": cm,
                     "consumer_exact": consumer_exact,
                     "producer_exact": producer_exact,
                     "consumer_pattern": consumer_pattern,
                     "copies": file_copies, "proposed": proposed})
    rows.sort(key=lambda r: (-r["alloc"], r["path"]))
    by = collections.defaultdict(lambda: {"files": 0, "alloc": 0})
    for r in rows:
        by[r["proposed"]]["files"] += 1
        by[r["proposed"]]["alloc"] += r["alloc"]
    resolved = sum(1 for s in sites if s["exact"] is not None and s["role"] != "mention")
    total = sum(1 for s in sites if s["role"] != "mention")
    return {"rows": rows,
            "summary": {"rows": len(rows), "alloc": sum(r["alloc"] for r in rows),
                        "sites": total, "sites_resolved_exact": resolved,
                        "by_proposed": dict(by), "executes_nothing": True},
            "sites": [{"source": s["source"], "line": s["line"], "text": s["text"],
                       "role": s["role"], "holes": s["holes"],
                       "exact_count": None if s["exact"] is None else len(s["exact"])}
                      for s in sites]}


def render_markdown(plan: dict) -> str:
    s = plan["summary"]
    lines = ["# Checkpoint execution plan (draft; executes nothing)", "",
             f"Rows: {s['rows']} files, {s['alloc'] / 1e9:.2f} GB allocated. Dynamic sites: "
             f"{s['sites']}, resolved to exact names: {s['sites_resolved_exact']}.",
             "", "Every action below waits for an explicit GO; this document executes nothing.",
             "", "| proposed action | files | GB |", "|---|---:|---:|"]
    for k, v in sorted(s["by_proposed"].items(), key=lambda kv: -kv[1]["alloc"]):
        lines.append(f"| {k} | {v['files']} | {v['alloc'] / 1e9:.2f} |")
    lines += ["", "| GB | path | class | consumers (exact) | producers (exact) | consumer patterns | copies | proposed |",
              "|---:|---|---|---|---|---|---|---|"]
    for r in plan["rows"]:
        fmt = lambda xs: ", ".join(f"{x} [{r['codemap'].get(x, '?')}]" for x in xs) or "-"
        cp_ = ",".join(f"{k}:{len(v)}" for k, v in r["copies"].items()) or "-"
        lines.append(f"| {r['alloc'] / 1e9:.2f} | {r['path']} | {r['class']} | {fmt(r['consumer_exact'])} | "
                     f"{fmt(r['producer_exact'])} | {fmt(r['consumer_pattern'])} | {cp_} | {r['proposed']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--copies", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    copies = json.loads(Path(a.copies).read_text()) if a.copies else {}
    plan = build_plan(Path(a.repo), Path(a.inventory), copies)
    out = Path(a.out) if a.out else Path(a.inventory).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.jsonl").write_text("\n".join(json.dumps(r) for r in plan["rows"]) + "\n")
    (out / "plan_sites.json").write_text(json.dumps(plan["sites"], indent=1))
    (out / "plan_summary.json").write_text(json.dumps(plan["summary"], indent=1))
    (out / "PLAN.md").write_text(render_markdown(plan))
    s = plan["summary"]
    print(f"wrote {out}/PLAN.md: {s['rows']} rows, {s['alloc'] / 1e9:.2f} GB; "
          f"sites {s['sites']}, exact {s['sites_resolved_exact']}")
    for k, v in sorted(s["by_proposed"].items(), key=lambda kv: -kv[1]["alloc"]):
        print(f"  {v['files']:4d} {v['alloc'] / 1e9:6.2f} GB  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

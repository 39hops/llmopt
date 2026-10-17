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

Variable resolution reads the same file only and types every value
set. kind "literal": Python literal lists and tuples, loops over
them, plain string or int assignments (several assignments union),
shell `for X in a b c` loops and `X=literal`; aliases inherit. kind
"default": os.environ.get(name, default), argparse add_argument
default=, `${X:-default}`: the default is one reachable value, the
set is NOT complete because an environment or command-line override
reaches other names. A hole bound to an argument, a bare environment
read, or an expression stays unresolved.

Reach classes per file: "exhaustive" (some site expands to exactly
this name from literal-only bindings), "default" (reached only under
default configuration), "pattern" (matched by an unresolved site, or
by a default-bound site under a non-default value), "unreached"
(no site at all and no generic loader in the repository), and
"unreached-with-generic-loaders" (no site, but code exists that opens
any name under checkpoints/ from an argument, so reachability by
invocation cannot be excluded). Only "unreached" supports a claim that
nothing can reach the file; the other two unreached forms are explicit
uncertainty.

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
        name = re.split(r"[:!\[\(\s\+\*]", g, 1)[0]
        return name if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", name or "") else "<expr>"
    return m.group(2) or m.group(3) or "*"


VARROOT = re.compile(r"(?:\{[A-Za-z_]\w*\}|\$\{?[A-Za-z_]\w*\}?)/([\w\-.]+\.(?:pt|bin|npz|npy|json|jsonl|safetensors|pth))\b")
GENERIC_LOADER = re.compile(r"""["']checkpoints/["']\s*\+|join\(\s*["']checkpoints["']|Path\(\s*["']checkpoints["']\s*\)\s*/""")
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


def _typed(values, kind):
    return {"values": sorted(set(values)), "kind": kind}


def _merge(vals: dict, name: str, values, kind: str) -> None:
    """Union across multiple assignments; a default anywhere makes the
    set non-exhaustive."""
    if name in vals:
        old = vals[name]
        kind = "default" if "default" in (old["kind"], kind) else "literal"
        values = list(old["values"]) + list(values)
    vals[name] = _typed(values, kind)


_LIT_ITEMS = r"[\"']([^\"']+)[\"']|(?<![\w.])(-?\d+)(?![\w.])"


def variable_values(text: str) -> dict[str, dict]:
    """name -> {"values": [...], "kind": "literal" | "default"}."""
    text = re.sub(r"\\\n\s*", " ", text)   # shell line continuations
    vals: dict[str, dict] = {}
    aliases: list[tuple[str, str]] = []
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*[\[\(]([^\]\)]*)[\]\)]", text, re.M):
        items = [a or b for a, b in re.findall(_LIT_ITEMS, m.group(2))]
        if items:
            _merge(vals, m.group(1), items, "literal")
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(?:int\(|str\(|float\()?\s*os\.environ\.get\(\s*[\"']\w+[\"']\s*,\s*[\"']?([^\"'\)]*)[\"']?\s*\)", text, re.M):
        _merge(vals, m.group(1), [m.group(2)], "default")
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(-?\d+|[\"'][^\"']+[\"'])\s*$", text, re.M):
        _merge(vals, m.group(1), [m.group(2).strip("\"'")], "literal")
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*$", text, re.M):
        aliases.append((m.group(1), m.group(2)))
    for m in re.finditer(r"for\s+([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)\s*:", text):
        aliases.append((m.group(1), m.group(2)))
    for m in re.finditer(r"for\s+([A-Za-z_]\w*)\s+in\s+[\[\(]([^\]\)]*)[\]\)]\s*:", text):
        items = [a or b for a, b in re.findall(_LIT_ITEMS, m.group(2))]
        if items:
            _merge(vals, m.group(1), items, "literal")
    for m in re.finditer(r"for\s+([A-Za-z_]\w*)\s+in\s+([^;\n]+?)\s*;?\s*do\b", text):
        words = m.group(2).split()
        if words and all(re.fullmatch(r"[\w.\-]+", w) for w in words):
            _merge(vals, m.group(1), words, "literal")
    for m in re.finditer(r"\$\{([A-Za-z_]\w*):-([^}]*)\}", text):
        _merge(vals, m.group(1), [m.group(2)], "default")
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)=([\w.\-]+)\s*$", text, re.M):
        _merge(vals, m.group(1), [m.group(2)], "literal")
    # argparse: add_argument("--tag", default="base") binds a.tag / args.tag
    for m in re.finditer(r"add_argument\(\s*[\"']--?([\w\-]+)[\"'][^)]*?default\s*=\s*([\"']?[\w.\-]+[\"']?)", text):
        dest = m.group(1).replace("-", "_")
        for pre in ("a", "args", "opt", "opts", "cfg"):
            _merge(vals, f"{pre}.{dest}", [m.group(2).strip("\"'")], "default")
    for _ in range(3):
        for dst, src in aliases:
            if src in vals and dst not in vals:
                vals[dst] = dict(vals[src])
    return vals


def expand_typed(site: dict, vals: dict[str, dict]) -> dict | None:
    """{"names": [...], "kind": "exhaustive" | "default"} or None when a
    hole is unresolved. "default" when any hole is bound only through a
    default (an override reaches names outside the list)."""
    if site.get("exact_literal"):
        return {"names": [site["exact_literal"]], "kind": "exhaustive"}
    names = expand(site, vals)
    if names is None:
        return None
    kind = "default" if any(vals[h]["kind"] == "default" for h in site["holes"]) else "exhaustive"
    return {"names": names, "kind": kind}


def expand(site: dict, vals: dict[str, dict]) -> list[str] | None:
    """Every exact path the site can produce, or None if a hole is
    unresolved. Accepts typed value sets; see expand_typed for the
    exhaustive/default distinction."""
    if site.get("exact_literal"):
        return [site["exact_literal"]]
    choices = []
    for h in site["holes"]:
        if h not in vals or h in ("*", "<expr>"):
            return None
        v = vals[h]
        choices.append(v["values"] if isinstance(v, dict) else v)
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
                    if GENERIC_LOADER.search(line):
                        sites.append({"source": rel, "line": i, "text": line.strip()[:120],
                                      "holes": ["*"], "regex": re.compile("checkpoints/.*"),
                                      "role": "generic", "generic": True,
                                      "exact_literal": None, "exact": None, "kind": None})
                    continue
                typed = expand_typed(site, vals)
                site["exact"] = None if typed is None else typed["names"]
                site["kind"] = None if typed is None else typed["kind"]
                if site["generic"] and site["exact"] is None:
                    site["role"] = "generic"
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
    generic_sources = sorted({s["source"] for s in sites if s["role"] == "generic"})
    rows = []
    for r in rows_in:
        if r["class"] not in ("UNKNOWN", "CANDIDATE", "DUPLICATE"):
            continue
        rp = r["repo_path"]
        matches = []
        for s in sites:
            if s["role"] in ("mention", "generic"):
                continue
            if s["exact"] is not None:
                if rp in s["exact"]:
                    matches.append({"source": s["source"], "line": s["line"],
                                    "role": s["role"],
                                    "match": "exact" if s["kind"] == "exhaustive" else "default",
                                    "pattern": s["text"]})
                elif s["kind"] == "default" and s["regex"].fullmatch(rp):
                    # an override of the defaulted variable could name
                    # this file: pattern reach, not exact
                    matches.append({"source": s["source"], "line": s["line"],
                                    "role": s["role"], "match": "pattern",
                                    "pattern": s["text"]})
            elif not s["generic"] and s["regex"].fullmatch(rp):
                matches.append({"source": s["source"], "line": s["line"],
                                "role": s["role"], "match": "pattern",
                                "pattern": s["text"]})
        cm = {m["source"]: codemap.get(Path(m["source"]).name, "UNCITED") for m in matches}
        consumer_exact = sorted({m["source"] for m in matches
                                 if m["role"] == "consumer" and m["match"] in ("exact", "default")})
        producer_exact = sorted({m["source"] for m in matches
                                 if m["role"] == "producer" and m["match"] in ("exact", "default")})
        if any(m["match"] == "exact" for m in matches):
            reach = "exhaustive"
        elif any(m["match"] == "default" for m in matches):
            reach = "default"
        elif matches:
            reach = "pattern"
        elif generic_sources:
            reach = "unreached-with-generic-loaders"
        else:
            reach = "unreached"
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
        elif not matches and generic_sources:
            proposed = ("REVIEW: no named site reaches this file after pattern resolution; "
                        f"{len(generic_sources)} generic loader(s) could open it by argument; "
                        "CANDIDATE queue; per-family GO")
        elif not matches:
            proposed = ("REVIEW: no site reaches this file after pattern resolution and no "
                        "generic loader exists; CANDIDATE queue; per-family GO")
        else:
            proposed = "REVIEW: reached only by producer patterns that did not resolve"
        rows.append({"path": rp, "bytes": r["bytes"], "alloc": r["alloc"],
                     "mtime": time.strftime("%Y-%m-%d", time.localtime(r["mtime"])),
                     "class": r["class"], "family": r["family"],
                     "canonical": r.get("canonical"), "reach": reach,
                     "matches": matches, "codemap": cm,
                     "consumer_exact": consumer_exact,
                     "producer_exact": producer_exact,
                     "consumer_pattern": consumer_pattern,
                     "copies": file_copies, "proposed": proposed})
    rows.sort(key=lambda r: (-r["alloc"], r["path"]))
    by = collections.defaultdict(lambda: {"files": 0, "alloc": 0})
    by_reach = collections.defaultdict(lambda: {"files": 0, "alloc": 0})
    for r in rows:
        by[r["proposed"]]["files"] += 1
        by[r["proposed"]]["alloc"] += r["alloc"]
        by_reach[r["reach"]]["files"] += 1
        by_reach[r["reach"]]["alloc"] += r["alloc"]
    named = [s for s in sites if s["role"] not in ("mention", "generic")]
    exhaustive = sum(1 for s in named if s["kind"] == "exhaustive")
    default = sum(1 for s in named if s["kind"] == "default")
    return {"rows": rows,
            "summary": {"rows": len(rows), "alloc": sum(r["alloc"] for r in rows),
                        "sites": len(named), "sites_resolved_exact": exhaustive + default,
                        "sites_exhaustive": exhaustive, "sites_default_bound": default,
                        "generic_loaders": generic_sources,
                        "by_proposed": dict(by), "by_reach": dict(by_reach),
                        "executes_nothing": True},
            "sites": [{"source": s["source"], "line": s["line"], "text": s["text"],
                       "role": s["role"], "holes": s["holes"], "kind": s.get("kind"),
                       "exact_count": None if s["exact"] is None else len(s["exact"])}
                      for s in sites]}


def render_markdown(plan: dict) -> str:
    s = plan["summary"]
    lines = ["# Checkpoint execution plan (draft; executes nothing)", "",
             f"Rows: {s['rows']} files, {s['alloc'] / 1e9:.2f} GB allocated. Named dynamic sites: "
             f"{s['sites']}; exhaustive name sets {s['sites_exhaustive']}, default-bound "
             f"(an override reaches other names) {s['sites_default_bound']}, unresolved "
             f"{s['sites'] - s['sites_resolved_exact']}. Generic loaders (open any name under "
             f"checkpoints/ from an argument): {len(s['generic_loaders'])}: "
             + (", ".join(s["generic_loaders"]) or "none") + ".",
             "", "Every action below waits for an explicit GO; this document executes nothing.",
             "", "| reach | files | GB | meaning |", "|---|---:|---:|---|"]
    meaning = {"exhaustive": "a site expands to exactly this name from literal-only bindings",
               "default": "reached only under default configuration (env or CLI default)",
               "pattern": "matched by an unresolved site, or by an overridable default",
               "unreached-with-generic-loaders": "no named site; a generic loader could open it by argument",
               "unreached": "no site and no generic loader: nothing in the repository can name it"}
    for k, v in sorted(s["by_reach"].items(), key=lambda kv: -kv[1]["alloc"]):
        lines.append(f"| {k} | {v['files']} | {v['alloc'] / 1e9:.2f} | {meaning.get(k, '')} |")
    lines += ["", "| proposed action | files | GB |", "|---|---:|---:|"]
    for k, v in sorted(s["by_proposed"].items(), key=lambda kv: -kv[1]["alloc"]):
        lines.append(f"| {k} | {v['files']} | {v['alloc'] / 1e9:.2f} |")
    lines += ["", "| GB | path | class | reach | consumers (exact or default) | producers (exact or default) | consumer patterns | copies | proposed |",
              "|---:|---|---|---|---|---|---|---|---|"]
    for r in plan["rows"]:
        fmt = lambda xs: ", ".join(f"{x} [{r['codemap'].get(x, '?')}]" for x in xs) or "-"
        cp_ = ",".join(f"{k}:{len(v)}" for k, v in r["copies"].items()) or "-"
        lines.append(f"| {r['alloc'] / 1e9:.2f} | {r['path']} | {r['class']} | {r['reach']} | {fmt(r['consumer_exact'])} | "
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

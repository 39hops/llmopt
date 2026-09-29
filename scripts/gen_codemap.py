"""Generate docs/CODEMAP.md: the move-gate inventory of scratch/ and
scripts/ (adopted from the Grok structure review, 2026-08-06; law
upgraded 2026-09-29, curation stage 2). One row per top-level file:
how documents cite it, how code reaches it, a mechanically derived
class, and the filename family. The class ladder is observable facts
only, no curation:

    library          — reached by code: imported (bare, dotted
                       `scratch.x` / `scripts.x`, or a dynamic loader
                       such as importlib / spec_from_file_location /
                       runpy) by any .py, or invoked by path from the
                       package (llmopt/) or the tests
    reproduce-pinned — named in docs/REPRODUCE.md, directly or through
                       a caller that is
    results-cited    — named in RESULTS/FINDINGS/BOARD/THEORY/README,
                       a docs/preregs/*.json, or a sha-locked tracked
                       receipt (an emitter recorded by its own
                       receipt), directly or through a caller that is
    spec-cited       — named only in handoffs/specs/plans/relays/RIFF
                       and the other docs/*.md, directly or via caller
    tool-referenced  — named only by lab tooling (CLAUDE.md, .claude/,
                       .github/, pyproject.toml, jobs/*.cmd,
                       scripts/*.sh), directly or via caller
    UNCITED          — no document names it, no code reaches it

"Through a caller": a helper that a cited shell driver or script runs
(`python scratch/helper.py`, a path literal, `python -m scratch.x`)
inherits the caller's citations, to a fixpoint. Citations match the
exact filename, brace lists (`scratch/x{,2,3}.py`), globs
(`scratch/x_*.py`) and, for underscore stems of six or more
characters, the bare stem ("the x_probe machinery").

Rule (house law, lab-extraction spec): nothing above UNCITED moves
without adoption-with-reverification; library files must migrate their
importers in the same pass. Re-run after any restructuring commit:

    .venv/bin/python scripts/gen_codemap.py            # write
    .venv/bin/python scripts/gen_codemap.py --check    # exit 1 on drift, no write
    .venv/bin/python scripts/gen_codemap.py --out PATH
"""
from __future__ import annotations

import ast
import fnmatch
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "CODEMAP.md"

# Inventory: top-level only (scratch/leancheck vendors a Lean toolchain).
INVENTORY_GLOBS = [("scratch", "*.py"), ("scratch", "*.sh"), ("scripts", "*.py")]

# Class ladder, strongest first.
CLASSES = ["library", "reproduce-pinned", "results-cited", "spec-cited",
           "tool-referenced", "UNCITED"]
GROUP_CLASS = {"REPRODUCE": "reproduce-pinned", "RESULTS": "results-cited",
               "specs": "spec-cited", "TOOLING": "tool-referenced"}

# Doc corpus, grouped by the citation weight the class ladder uses.
# Entries are files, directories (recursed) or globs relative to ROOT.
DOC_GROUPS = {
    "REPRODUCE": ["docs/REPRODUCE.md"],
    "RESULTS": ["docs/RESULTS.md", "docs/FINDINGS.md", "docs/BOARD.md",
                "docs/THEORY.md", "README.md", "docs/preregs/*.json"],
    "specs": ["docs/handoffs", "docs/superpowers/specs",
              "docs/superpowers/plans", "docs/superpowers/relay",
              "docs/sol", "docs/opus", "docs/RIFF-LEDGER.md",
              "docs/LOOP-LOG.md", "docs/GLOSSARY.md", "GLOSSARY.md",
              "docs/MEASURED-HISTORY.md", "docs/EXTERNAL-REVIEWS.md",
              "docs/AXIOM-SURFACE.md", "docs/SCOREBOARD.md",
              "docs/hygiene-plan-2026-08-11.md",
              "docs/paper-draft-entropy-bound.md", "docs/paper-prose-v1.md",
              "docs/paper/*.tex", "docs/paper/*.md"],
    "TOOLING": ["CLAUDE.md", "CONTRIBUTING.md", "pyproject.toml",
                ".claude/**/*.md", ".claude/**/*.json", ".github/**/*.yml",
                "jobs/*.cmd", "scripts/*.sh"],
}
# Sha-locked receipts that name their emitter are RESULTS-grade evidence.
RECEIPT_LOCK = "docs/receipts.lock.json"
RECEIPT_MAX_BYTES = 4 << 20

# Code corpus: files whose imports / path literals are dependency edges.
CODE_DIRS = ["scratch", "scripts", "llmopt", "tests"]
CODE_EXTRA_GLOBS = [".claude/hooks/*.py", "jobs/*.cmd", ".claude/**/*.md"]
# Non-inventory callers lend a grade to what they reach.
LIBRARY_CALLER_PREFIXES = ("llmopt/", "tests/")
TOOLING_CALLER_PREFIXES = (".claude/", ".github/", "jobs/")

LOADER_CALLS = {"spec_from_file_location", "import_module", "run_path",
                "run_module", "__import__", "load_source"}
INVOKE_CALLS = {"run", "Popen", "check_output", "check_call", "call",
                "system", "execv", "execvp", "spawn"}

# a path may end a sentence ("scratch/x.py.") or sit before a colon;
# only a word character continuing the token disqualifies a match
PATH_TOKEN = re.compile(r"(?<![\w.-])(?:scratch|scripts)/[\w/-]+(?:\.[\w-]+)*")
NAME_TOKEN = re.compile(r"(?<![\w./-])[\w-]+\.(?:py|sh)(?!\w)")
GLOB_MIN_LITERAL = 3  # basename literal chars a glob needs to be a citation
BRACE = re.compile(r"(?:scratch|scripts)/([\w.-]*)\{([^{}]*)\}([\w.-]*)")
GLOB = re.compile(r"(?:scratch|scripts)/([\w.-]*[*?][\w.*?-]*)")
DASH_M = re.compile(r"-m\s+(?:scratch|scripts)\.(\w+)")
WORD = re.compile(r"(?<![\w./-])[A-Za-z]\w*_\w+(?!\w|\.(?:py|sh)\b)")
PROSE_SUFFIXES = (".md", ".tex", ".toml", ".cmd", ".sh", ".yml")
RESULTS_INDEX = "docs/results-index.jsonl"


def _tracked() -> set[str] | None:
    """Repo-relative paths git knows about, or None if git is unusable.

    CODEMAP describes the REPOSITORY, so it must read the same on every
    checkout. Globbing the working tree instead made it depend on
    whichever untracked scratch directories a machine happened to have
    (found 2026-08-11 when CI regenerated a different file than the Mac
    had committed). Untracked files are excluded; if git is not
    available the glob is used unchanged.
    """
    import subprocess
    try:
        r = subprocess.run(["git", "ls-files"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        return set(r.stdout.split())
    except Exception:
        return None


def family(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    return stem.split("_", 1)[0] if "_" in stem else stem


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


class Repo:
    """Every corpus the law reads, resolved against one root and one
    tracked set (None = every file on disk)."""

    def __init__(self, root: Path, tracked: set[str] | None):
        self.root = root
        self.tracked = tracked

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    def ok(self, p: Path) -> bool:
        if not p.is_file() or "__pycache__" in p.parts or "leancheck" in p.parts:
            return False
        return self.tracked is None or self.rel(p) in self.tracked

    def expand(self, spec: str) -> list[Path]:
        full = self.root / spec
        if any(ch in spec for ch in "*?["):
            return sorted(p for p in self.root.glob(spec) if self.ok(p))
        if full.is_dir():
            return sorted(p for p in full.rglob("*.md") if self.ok(p))
        return [full] if self.ok(full) else []

    def inventory(self) -> list[Path]:
        out = []
        for base, pat in INVENTORY_GLOBS:
            out += sorted(p for p in (self.root / base).glob(pat) if self.ok(p))
        return out

    def docs(self) -> dict[str, list[Path]]:
        out = {g: [] for g in DOC_GROUPS}
        for g, specs in DOC_GROUPS.items():
            for s in specs:
                out[g] += self.expand(s)
        # locked, tracked receipts
        lock = self.root / RECEIPT_LOCK
        if lock.is_file():
            try:
                recs = json.loads(lock.read_text()).get("receipts", {})
            except Exception:
                recs = {}
            for rel, rec in sorted(recs.items()):
                p = self.root / rel
                if (rec.get("tracked") or self.tracked is None) and self.ok(p) \
                        and p.stat().st_size <= RECEIPT_MAX_BYTES:
                    out["RESULTS"].append(p)
        return out

    def code(self) -> list[Path]:
        out = []
        for d in CODE_DIRS:
            base = self.root / d
            out += sorted(p for p in base.rglob("*") if p.suffix in (".py", ".sh")
                          and self.ok(p))
        for g in CODE_EXTRA_GLOBS:
            out += self.expand(g)
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq


# ------------------------------------------------------------ citations

def cite_counter(text: str, names: set[str], stems: dict[str, str],
                 prose: bool = True) -> Counter:
    """Citations of inventory names in one document: exact filename
    tokens, brace lists, globs, and (in prose documents only, never in
    JSON receipts whose keys collide with stems) bare underscore
    stems."""
    c: Counter = Counter()
    for tok in NAME_TOKEN.findall(text):
        if tok in names:
            c[tok] += 1
    # NAME_TOKEN refuses a name preceded by "/", so path-prefixed
    # citations are counted here exactly once
    for tok in PATH_TOKEN.findall(text):
        base = tok.rsplit("/", 1)[-1]
        if base in names:
            c[base] += 1
    for head, inner, tail in BRACE.findall(text):
        for part in inner.split(","):
            n = head + part.strip() + tail
            if n in names:
                c[n] += 1
    for pat in GLOB.findall(text):
        # `scratch/*.py` names a directory, not its files: require some
        # literal basename text (`ozaki_*`, `x_{a,b}` handled above)
        literal = re.sub(r"\.(?:py|sh)$", "", pat)
        if len(re.sub(r"[*?.]", "", literal)) < GLOB_MIN_LITERAL:
            continue
        for n in names:
            if fnmatch.fnmatchcase(n, pat):
                c[n] += 1
    if prose:
        for w in WORD.findall(text):
            n = stems.get(w)
            if n:
                c[n] += 1
    return c


# ------------------------------------------------------------------ edges

def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _consts(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _prose_nodes(tree: ast.AST) -> set[int]:
    """ids of string constants that are docstrings or bare expression
    statements: provenance prose ("adopted from scratch/x.py"), never
    a reach."""
    out: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                    and isinstance(stmt.value.value, str):
                out.add(id(stmt.value))
    return out


def py_edges(text: str, names: set[str], stems: dict[str, str]
             ) -> tuple[set[str], set[str], set[str]]:
    """(imported, invoked, mentioned) inventory names reached by one
    Python source. imported: static import statements (bare or dotted)
    and string arguments of dynamic loaders. invoked: path literals
    `scratch/x.py` anywhere, or string arguments of subprocess-style
    calls. mentioned: filename tokens elsewhere (comments, docstrings,
    plain strings)."""
    imported: set[str] = set()
    invoked: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    stem = parts[1] if parts[0] in ("scratch", "scripts") \
                        and len(parts) > 1 else parts[0]
                    if stem in stems:
                        imported.add(stems[stem])
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                parts = node.module.split(".")
                if parts[0] in ("scratch", "scripts"):
                    cands = [parts[1]] if len(parts) > 1 else \
                        [a.name for a in node.names]
                else:
                    cands = [parts[0]]
                for stem in cands:
                    if stem in stems:
                        imported.add(stems[stem])
            elif isinstance(node, ast.Call):
                fn = _call_name(node)
                if fn in LOADER_CALLS or fn in INVOKE_CALLS:
                    for s in _consts(node):
                        for n in _names_in_string(s, names, stems):
                            (imported if fn in LOADER_CALLS else invoked).add(n)
        prose = _prose_nodes(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) \
                    or id(node) in prose:
                continue
            for tok in PATH_TOKEN.findall(node.value):
                base = tok.rsplit("/", 1)[-1]
                if base in names:
                    invoked.add(base)
            for m in DASH_M.findall(node.value):
                if m in stems:
                    invoked.add(stems[m])
        # path joins: ROOT / "scratch" / "x.py" reaches x.py by path
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                for s in _consts(node):
                    if s in names:
                        invoked.add(s)
    mentioned = {t for t in NAME_TOKEN.findall(text) if t in names}
    mentioned |= {t.rsplit("/", 1)[-1] for t in PATH_TOKEN.findall(text)
                  if t.rsplit("/", 1)[-1] in names}
    mentioned -= imported | invoked
    return imported, invoked, mentioned


def _names_in_string(s: str, names: set[str], stems: dict[str, str]) -> set[str]:
    out = set()
    if s in names:
        out.add(s)
    base = s.rsplit("/", 1)[-1]
    if "/" in s and base in names:
        out.add(base)
    if s in stems:
        out.add(stems[s])
    parts = s.split(".")
    if len(parts) == 2 and parts[0] in ("scratch", "scripts") and parts[1] in stems:
        out.add(stems[parts[1]])
    return out


def sh_edges(text: str, names: set[str], stems: dict[str, str]
             ) -> tuple[set[str], set[str]]:
    """(invoked, mentioned) for shell-like text (.sh, .cmd, skill docs):
    any path or filename token is an invocation edge; the distinction
    to `mentioned` is kept empty because shell text has no comment /
    string structure worth separating."""
    invoked: set[str] = set()
    for tok in PATH_TOKEN.findall(text):
        base = tok.rsplit("/", 1)[-1]
        if base in names:
            invoked.add(base)
    for tok in NAME_TOKEN.findall(text):
        if tok in names:
            invoked.add(tok)
    for m in DASH_M.findall(text):
        if m in stems:
            invoked.add(stems[m])
    return invoked, set()


# ------------------------------------------------------------------ build

def build(root: Path = ROOT, tracked: set[str] | None = "auto") -> list[dict]:
    """One dict per inventory file with every evidence field the map
    prints. `tracked="auto"` consults git; None means every file."""
    if tracked == "auto":
        tracked = _tracked()
    repo = Repo(root, tracked)
    inv = repo.inventory()
    names = {p.name for p in inv}
    rel_of = {p.name: repo.rel(p) for p in inv}
    stems = {p.stem: p.name for p in inv
             if "_" in p.stem and len(p.stem) >= 6}
    stem_any = {p.stem: p.name for p in inv}
    inv_rels = set(rel_of.values())
    name_of_rel = {v: k for k, v in rel_of.items()}

    # direct citations per group
    cites: dict[str, Counter] = {n: Counter() for n in names}
    for group, paths in repo.docs().items():
        for p in paths:
            prose = p.suffix in PROSE_SUFFIXES
            for n, k in cite_counter(_read(p), names, stems, prose).items():
                cites[n][group] += k
    # the results index carries a curated `files` field per entry that
    # can name a file the RESULTS prose wraps or omits
    idx = root / RESULTS_INDEX
    if repo.ok(idx):
        for line in _read(idx).splitlines():
            try:
                for f in json.loads(line).get("files", []):
                    if f in inv_rels:  # full repo path, not basename
                        cites[name_of_rel[f]]["RESULTS"] += 1
            except ValueError:
                continue

    # code edges
    imports: dict[str, list[str]] = defaultdict(list)
    invoked_by: dict[str, list[str]] = defaultdict(list)
    mentions: dict[str, list[str]] = defaultdict(list)
    for p in repo.code():
        rel = repo.rel(p)
        text = _read(p)
        if p.suffix == ".py":
            imp, inv_, men = py_edges(text, names, stem_any)
        else:
            inv_, men = sh_edges(text, names, stem_any)
            imp = set()
        for n in imp:
            if rel_of[n] != rel:
                imports[n].append(rel)
        for n in inv_:
            if rel_of[n] != rel:
                invoked_by[n].append(rel)
        for n in men:
            if rel_of[n] != rel:
                mentions[n].append(rel)

    # grade lent by non-inventory callers
    lib_by: dict[str, list[str]] = defaultdict(list)
    for n in names:
        for c in imports[n]:
            lib_by[n].append(c)
        for c in invoked_by[n]:
            if c.startswith(LIBRARY_CALLER_PREFIXES):
                lib_by[n].append(c)
            elif c.startswith(TOOLING_CALLER_PREFIXES) or \
                    (c.startswith("scripts/") and c.endswith(".sh")):
                cites[n]["TOOLING"] += 1

    # propagate citations through inventory callers to a fixpoint;
    # deterministic (sorted names, sorted callers) and each inherited
    # group remembers the caller that first supplied it
    callers = {n: sorted({c for c in imports[n] + invoked_by[n]
                          if c in inv_rels}) for n in names}
    eff: dict[str, Counter] = {n: Counter(cites[n]) for n in names}
    supplier: dict[str, dict[str, str]] = {n: {} for n in names}
    for _ in range(len(names) + 1):  # monotone: converges well before
        changed = False
        for n in sorted(names):
            for c in callers[n]:
                cn = name_of_rel[c]
                for g in ("REPRODUCE", "RESULTS", "specs", "TOOLING"):
                    k = eff[cn].get(g)
                    if k and not eff[n].get(g):
                        eff[n][g] = k
                        supplier[n][g] = c
                        changed = True
        if not changed:
            break

    rows = []
    for p in inv:
        n = p.name
        cls, via = "UNCITED", []
        top = next((g for g in ("REPRODUCE", "RESULTS", "specs", "TOOLING")
                    if eff[n].get(g)), None)
        if top:
            cls = GROUP_CLASS[top]
            if top in supplier[n]:  # inherited, not the file's own
                via = [supplier[n][top]]
        if lib_by[n]:
            cls = "library"
        rows.append({
            "base": p.parent.name, "family": family(n), "file": n,
            "class": cls, "cites": dict(cites[n]), "eff_cites": dict(eff[n]),
            "inherited": set(supplier[n]),
            "imports": sorted(set(imports[n])),
            "invoked_by": sorted(set(invoked_by[n])),
            "mentions": sorted(set(mentions[n])), "via": via,
        })
    return rows


# ----------------------------------------------------------------- render

def render(rows: list[dict]) -> tuple[str, dict[str, int]]:
    tallies: dict[str, int] = defaultdict(int)
    lib_cited = 0
    for r in rows:
        tallies[r["class"]] += 1
        if r["class"] == "library" and r["cites"]:
            lib_cited += 1
    order = {g: i for i, g in enumerate(("REPRODUCE", "RESULTS", "specs",
                                         "TOOLING"))}
    lines = [
        "# CODEMAP — the move-gate inventory (generated, do not hand-edit)",
        "",
        "Regenerate: `.venv/bin/python scripts/gen_codemap.py`. One row per",
        "top-level file in scratch/ and scripts/. Class ladder (mechanical):",
        "library > reproduce-pinned > results-cited > spec-cited >",
        "tool-referenced > UNCITED. House law: cited files are the evidence",
        "record — extraction means adoption-with-reverification, never a",
        "silent move. Columns: `cited by` lists the file's OWN citation",
        "groups plus `via:<group>` for groups inherited through a cited",
        "caller; `doc citations` counts the own ones; `imports` counts code",
        "files that import it",
        "(bare, dotted, or dynamic loader; drives `library`); `invoked by`",
        "counts files that run it by path or shell (from llmopt/ or tests/",
        "this also drives `library`; from lab tooling it drives",
        "`tool-referenced`); `mentions` counts files that only name it in",
        "comments or plain strings; `via` names the cited caller a file",
        "inherits its class from when no document names the file itself.",
        "",
        "Census: " + ", ".join(f"{c} {tallies.get(c, 0)}" for c in CLASSES)
        + f" (of {len(rows)}; library rows that are also doc-cited: "
        f"{lib_cited})",
        "",
    ]
    for base in ("scratch", "scripts"):
        lines += [f"## {base}/", "",
                  "| family | file | class | cited by | doc citations"
                  " | imports | invoked by | mentions | via |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in rows if r["base"] == base),
                        key=lambda r: (r["family"], r["file"])):
            groups = sorted(r["cites"], key=order.get)
            cite_s = ", ".join(f"{g}×{r['cites'][g]}" for g in groups) or "—"
            # inherited groups ride along as via:<group> so a `library`
            # row that a cited caller runs stays visibly frozen
            inherited = sorted(r.get("inherited", ()), key=order.get)
            cited_by = ", ".join(groups + [f"via:{g}" for g in inherited]) or "—"
            via = r["via"]
            via_s = (", ".join(Path(v).name for v in via[:3])
                     + (f" +{len(via) - 3}" if len(via) > 3 else "")) if via else "—"
            def n_(xs):
                return str(len(xs)) if xs else "—"
            lines.append(
                f"| {r['family']} | {r['file']} | {r['class']} | {cited_by}"
                f" | {cite_s} | {n_(r['imports'])} | {n_(r['invoked_by'])}"
                f" | {n_(r['mentions'])} | {via_s} |")
        lines.append("")
    return "\n".join(lines), dict(tallies)


def write_if_changed(path: Path, text: str) -> bool:
    """Atomic write (tmp + os.replace) only when the content differs.
    Concurrent hook posts and readers never observe a half-written
    file; an unchanged output keeps its mtime."""
    import os
    if path.exists() and path.read_text() == text:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a rewrite would change the file; "
                         "never writes")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    new, tallies = render(build())
    current = a.out.read_text() if a.out.exists() else None
    if a.check:
        if current == new:
            print("[codemap] current: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(tallies.items())))
            return 0
        print(f"[codemap] STALE: {a.out} differs from a regeneration; "
              "run scripts/gen_codemap.py", file=sys.stderr)
        return 1
    write_if_changed(a.out, new)
    print(f"[codemap] wrote {a.out}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(tallies.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

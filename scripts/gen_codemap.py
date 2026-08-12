"""Generate docs/CODEMAP.md: the move-gate inventory of scratch/ and
scripts/ (adopted from the Grok structure review, 2026-08-06). One row
per file: doc citations (RESULTS/REPRODUCE/BOARD/FINDINGS/handoffs/
specs), in-code references (imports + literal path strings), a
mechanically derived class, and the filename family. The class ladder
is observable-facts-only, no curation:

    library          — imported or path-referenced by other code
    reproduce-pinned — named in docs/REPRODUCE.md (runnable pin)
    results-cited    — named in RESULTS/FINDINGS/BOARD/THEORY
    spec-cited       — named only in handoffs/specs/plans
    UNCITED          — no doc names it, no code references it

Rule (house law, lab-extraction spec): nothing above UNCITED moves
without adoption-with-reverification; library files must migrate their
importers in the same pass. Re-run after any restructuring commit:

    .venv/bin/python scripts/gen_codemap.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "CODEMAP.md"

# Inventory: top-level only (scratch/leancheck vendors a Lean toolchain).
INVENTORY_GLOBS = [("scratch", "*.py"), ("scratch", "*.sh"), ("scripts", "*.py")]

# Doc corpus, grouped by the citation weight the class ladder uses.
DOC_GROUPS = {
    "REPRODUCE": ["docs/REPRODUCE.md"],
    "RESULTS": ["docs/RESULTS.md", "docs/FINDINGS.md", "docs/BOARD.md",
                "docs/THEORY.md", "README.md"],
    "specs": ["docs/handoffs", "docs/superpowers/specs",
              "docs/superpowers/plans", "docs/sol", "docs/opus"],
}

# Code corpus searched for imports / literal path references.
CODE_DIRS = ["scratch", "scripts", "llmopt", "tests"]


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


TRACKED = _tracked()


def _is_tracked(f: Path) -> bool:
    return TRACKED is None or str(f.relative_to(ROOT)) in TRACKED


def collect_files(base: str, pat: str) -> list[Path]:
    return sorted(f for f in (ROOT / base).glob(pat) if _is_tracked(f))


def load_texts(paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        full = ROOT / p
        if full.is_dir():
            for f in sorted(full.glob("**/*.md")):
                out[str(f.relative_to(ROOT))] = f.read_text(errors="replace")
        elif full.exists():
            out[p] = full.read_text(errors="replace")
    return out


def load_code() -> dict[str, str]:
    out: dict[str, str] = {}
    for d in CODE_DIRS:
        base = ROOT / d
        for f in sorted(base.glob("**/*.py")) + sorted(base.glob("**/*.sh")):
            if "leancheck" in f.parts or "__pycache__" in f.parts:
                continue
            if not _is_tracked(f):
                continue
            out[str(f.relative_to(ROOT))] = f.read_text(errors="replace")
    return out


def family(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    return stem.split("_", 1)[0] if "_" in stem else stem


def code_refs(
    target: Path, code: dict[str, str]
) -> tuple[list[str], list[str]]:
    """(importers, mention-only referrers) for the module.

    A `.py` file counts as imported only by a real `import`/`from`
    statement; anything else that embeds its literal filename (path
    strings, shell invocations, comments) is a mention. `.sh` files
    can only be mentioned, never imported.
    """
    name = target.name
    mod = target.stem
    rel = str(target.relative_to(ROOT))
    imp = re.compile(
        rf"^\s*(?:import\s+{re.escape(mod)}\b"
        rf"|from\s+{re.escape(mod)}\s+import)", re.MULTILINE)
    imports, mentions = [], []
    for path, text in code.items():
        if path == rel:
            continue
        if target.suffix == ".py" and imp.search(text):
            imports.append(path)
        elif name in text:
            mentions.append(path)
    return imports, mentions


def doc_cites(name: str, docs: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group, texts in docs.items():
        n = sum(t.count(name) for t in texts.values())
        if n:
            counts[group] = n
    return counts


def classify(cites: dict[str, int], imports: list[str]) -> str:
    if imports:
        return "library"
    if cites.get("REPRODUCE"):
        return "reproduce-pinned"
    if cites.get("RESULTS"):
        return "results-cited"
    if cites.get("specs"):
        return "spec-cited"
    return "UNCITED"


def main() -> None:
    docs = {g: load_texts(ps) for g, ps in DOC_GROUPS.items()}
    code = load_code()
    rows: dict[str, list[tuple]] = defaultdict(list)
    tallies: dict[str, int] = defaultdict(int)
    hidden = 0
    for base, pat in INVENTORY_GLOBS:
        for f in collect_files(base, pat):
            cites = doc_cites(f.name, docs)
            imports, mentions = code_refs(f, code)
            cls = classify(cites, imports)
            tallies[cls] += 1
            if cls == "library" and cites:
                hidden += 1
            cite_s = ", ".join(f"{g}×{n}" for g, n in cites.items()) or "—"
            cited_by = ", ".join(cites.keys()) or "—"
            imp_s = str(len(imports)) if imports else "—"
            men_s = str(len(mentions)) if mentions else "—"
            rows[base].append(
                (family(f.name), f.name, cls, cited_by, cite_s,
                 imp_s, men_s))

    lines = [
        "# CODEMAP — the move-gate inventory (generated, do not hand-edit)",
        "",
        "Regenerate: `.venv/bin/python scripts/gen_codemap.py`. One row per",
        "top-level file in scratch/ and scripts/. Class ladder (mechanical):",
        "library > reproduce-pinned > results-cited > spec-cited > UNCITED.",
        "House law: cited files are the evidence record — extraction means",
        "adoption-with-reverification, never a silent move. `imports`",
        "counts code files with a real `import`/`from` statement on the",
        "module (drives `library`); `mentions` counts files that only",
        "embed its literal filename (path strings, shell invocations —",
        "catches couplings like llmopt/reproduce.py → detbwd_gravmoe, but",
        "does not by itself make a file `library`).",
        "",
        "Census: " + ", ".join(f"{k} {v}" for k, v in sorted(tallies.items()))
        + f", cited-but-library {hidden}",
        "",
    ]
    for base in ("scratch", "scripts"):
        lines += [f"## {base}/", "",
                  "| family | file | class | cited by | doc citations"
                  " | imports | mentions |",
                  "|---|---|---|---|---|---|---|"]
        for fam, name, cls, cited_by, cite_s, imp_s, men_s \
                in sorted(rows[base]):
            lines.append(
                f"| {fam} | {name} | {cls} | {cited_by} | {cite_s}"
                f" | {imp_s} | {men_s} |")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"[codemap] wrote {OUT.relative_to(ROOT)}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(tallies.items())))


if __name__ == "__main__":
    main()

"""Generate scripts/INDEX.md: one entry per python file in scripts/,
scratch/, and llmopt/ — module docstring first paragraph + top-level
function/class signatures (AST, no imports executed). Run after adding
scripts so future sessions grep one file instead of re-reading (or
re-writing) code that already exists.

    .venv/bin/python scripts/gen_index.py            # write
    .venv/bin/python scripts/gen_index.py --check    # exit 1 on drift, no write
    .venv/bin/python scripts/gen_index.py --out PATH
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["scripts", "scratch", "llmopt", "llmopt/lab", "llmopt/train",
        "llmopt/search"]


def sig(fn: ast.FunctionDef) -> str:
    a = ast.unparse(fn.args)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{fn.name}({a}){ret}"


def entry(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return f"### {path.relative_to(ROOT)}\n*(syntax error — skipped)*\n"
    doc = ast.get_docstring(tree) or ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    lines = [f"### {path.relative_to(ROOT)}", first or "*(no docstring)*", ""]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d = (ast.get_docstring(node) or "").split("\n")[0]
            lines.append(f"- `{sig(node)}`" + (f" — {d}" if d else ""))
        elif isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body
                       if isinstance(m, ast.FunctionDef)
                       and not m.name.startswith("_")]
            lines.append(f"- `class {node.name}`"
                         + (f" ({', '.join(methods)})" if methods else ""))
    return "\n".join(lines) + "\n"


def render() -> str:
    out = ["# Script index (generated — do not hand-edit)",
           "", "Regenerate: `.venv/bin/python scripts/gen_index.py`", ""]
    for d in DIRS:
        files = sorted((ROOT / d).glob("*.py"))
        if not files:
            continue
        out.append(f"## {d}/\n")
        for f in files:
            e = entry(f)
            if e:
                out.append(e)
    return "\n".join(out)


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
    ap.add_argument("--out", type=Path, default=ROOT / "scripts" / "INDEX.md")
    a = ap.parse_args(argv)
    new = render()
    current = a.out.read_text() if a.out.exists() else None
    if a.check:
        if current == new:
            print(f"[index] current ({a.out.relative_to(ROOT) if a.out.is_relative_to(ROOT) else a.out})")
            return 0
        print(f"[index] STALE: {a.out} differs from a regeneration; "
              "run scripts/gen_index.py", file=sys.stderr)
        return 1
    write_if_changed(a.out, new)
    print(f"wrote {a.out} ({new.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate scripts/INDEX.md: one entry per python file in scripts/,
scratch/, and llmopt/ — module docstring first paragraph + top-level
function/class signatures (AST, no imports executed). Run after adding
scripts so future sessions grep one file instead of re-reading (or
re-writing) code that already exists.

    .venv/bin/python scripts/gen_index.py
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["scripts", "scratch", "llmopt", "llmopt/train", "llmopt/search"]


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


def main() -> None:
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
    (ROOT / "scripts" / "INDEX.md").write_text("\n".join(out))
    print(f"wrote scripts/INDEX.md ({len(out)} sections)")


if __name__ == "__main__":
    main()

"""One-shot Phase 4 rewriter: replace the default-accelerator idiom
with pick_device() in the FREE files listed by phase4_sites.py.

Replaces chains of the form
    ("cuda" if torch.cuda.is_available() else "cpu")
    "mps" if torch.backends.mps.is_available() else "cpu"
    ("mps" if ... else "cuda" if ... else "cpu")
(with arbitrary internal whitespace/newlines, optional wrapping
parens) by pick_device(), and inserts the import after the module
docstring / __future__ block. CPU pins (bare "cpu" without an
availability chain) never match. Prints each file's replacement
count; compile-checks the result before writing.
"""
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARM = r'"(?:cuda|mps)"\s+if\s+torch\.(?:cuda|backends\.mps)\.is_available\(\)\s*'
CHAIN = rf'(?:{ARM}else\s*)+"cpu"'
PAT = re.compile(rf'\(\s*{CHAIN}\s*\)|{CHAIN}')
IMPORT = "from llmopt.common.device import pick_device\n"


def insert_import(text: str) -> str:
    if "from llmopt.common.device import pick_device" in text:
        return text
    lines = text.splitlines(keepends=True)
    i = 0
    if lines and lines[0].lstrip().startswith(('"""', "'''")):
        q = lines[0].lstrip()[:3]
        if lines[0].count(q) < 2:
            i = 1
            while i < len(lines) and q not in lines[i]:
                i += 1
        i += 1
    while i < len(lines) and (
            lines[i].strip() == "" or
            lines[i].startswith("from __future__")):
        i += 1
    return "".join(lines[:i]) + IMPORT + "".join(lines[i:])


def main() -> None:
    for rel in sys.argv[1:]:
        f = ROOT / rel
        text = f.read_text()
        new, n = PAT.subn("pick_device()", text)
        if n == 0:
            print(f"  SKIP (no match): {rel}")
            continue
        new = insert_import(new)
        f.write_text(new)
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            print(f"  COMPILE FAIL {rel}: {e}")
            f.write_text(text)
            continue
        print(f"  {n} replaced: {rel}")


if __name__ == "__main__":
    main()

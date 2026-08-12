"""One-shot Phase 4.3: delete sys.path.insert bootstrap lines in the
UNCITED files passed as args. Handles the one-line form and the
two-line continuation form (open paren balance). py_compile checks
the result; reverts the file on failure."""
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for rel in sys.argv[1:]:
    f = ROOT / rel
    lines = f.read_text().splitlines(keepends=True)
    out, i, n = [], 0, 0
    while i < len(lines):
        if "sys.path.insert" in lines[i]:
            depth = lines[i].count("(") - lines[i].count(")")
            i += 1
            n += 1
            while depth > 0 and i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
            continue
        out.append(lines[i])
        i += 1
    if n == 0:
        print(f"  SKIP: {rel}")
        continue
    old = "".join(lines)
    f.write_text("".join(out))
    try:
        py_compile.compile(str(f), doraise=True)
        print(f"  {n} deleted: {rel}")
    except py_compile.PyCompileError as e:
        f.write_text(old)
        print(f"  REVERTED {rel}: {e}")

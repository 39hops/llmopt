"""One-shot Phase 5 dead-code sweep (report only, no edits): AST
walk of llmopt/ (vendor excluded) collecting module-level def/class
names, then grep the whole repo (llmopt, scripts, scratch, tests)
for references. Prints names with zero references outside their own
definition line. Triage is the caller's job — battery-pinned bodies
and __getattr__-reached names are NOT safe to delete on this signal
alone."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

defs = []  # (name, file)
for p in (ROOT / "llmopt").rglob("*.py"):
    if "__pycache__" in p.parts or "vendor" in p.parts:
        continue
    tree = ast.parse(p.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if not node.name.startswith("__"):
                defs.append((node.name, p.relative_to(ROOT)))

corpus = []
for d in ("llmopt", "scripts", "scratch", "tests"):
    for p in (ROOT / d).rglob("*.py"):
        if "__pycache__" not in p.parts:
            corpus.append((p.relative_to(ROOT), p.read_text()))

for name, home in sorted(defs):
    pat = re.compile(rf"\b{re.escape(name)}\b")
    refs = 0
    for rel, text in corpus:
        hits = len(pat.findall(text))
        if rel == home:
            hits -= 1  # its own def line
        refs += max(hits, 0)
    if refs == 0:
        print(f"  DEAD? {home}:{name}")
print("sweep done")

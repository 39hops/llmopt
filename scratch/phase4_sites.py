"""One-shot Phase 4 survey: device-idiom sites x CODEMAP class.
Output drives the migration list; only class UNCITED or package
rows are migrated. --bootstrap lists sys.path.insert sites instead.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
classes = {}
for row in (ROOT / "docs" / "CODEMAP.md").read_text().splitlines():
    m = re.match(r"\| [^|]+ \| (\S+) \| (\S+) \|", row)
    if m:
        classes[m.group(1)] = m.group(2)

pat = (re.compile(r"sys\.path\.insert")
       if "--bootstrap" in sys.argv
       else re.compile(r'"(cuda|mps)" if torch\.'))
for d in ("llmopt", "tests", "scripts", "scratch"):
    for f in sorted((ROOT / d).rglob("*.py")):
        hits = [i + 1 for i, line in
                enumerate(f.read_text().splitlines())
                if pat.search(line)]
        if hits:
            cls = ("package" if d in ("llmopt", "tests")
                   else classes.get(f.name, "?"))
            print(f"{cls:>16} {f.relative_to(ROOT)} {hits}")

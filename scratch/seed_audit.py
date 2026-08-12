"""RNG seed audit (spec 2026-08-12 Phase 6.4). Report-only.

Greps tracked llmopt/ scripts/ tests/ for RNG constructors and
classifies each hit:
  STRING-SEED   random.Random(f"...") / string literal — the house law
  INT-LITERAL   integer literal seed (reproducible, not the law)
  DERIVED       seed comes from a variable/env/arg expression
  UNSEEDED      constructor with no seed argument (global-stream risk)
  TUPLE-SEED    tuple seed — DEFECT (per-process hash randomization
                killed reproducibility once; the mathgen incident)
"""
import re
import subprocess

PATTERNS = [
    r"random\.Random\(", r"random\.seed\(", r"np\.random\.default_rng\(",
    r"numpy\.random\.default_rng\(", r"np\.random\.seed\(",
    r"torch\.manual_seed\(", r"torch\.Generator\(",
    r"\.manual_seed\(", r"mx\.random\.seed\(",
]
RX = re.compile("|".join(PATTERNS))


def classify(line: str) -> str:
    # classify the LAST RNG call: torch.Generator().manual_seed(seed)
    # is seeded by the chained call, not the bare constructor
    matches = list(RX.finditer(line))
    tail = line[matches[-1].end():] if matches else ""
    arg = tail.split(")")[0].strip()
    if arg.startswith("("):
        return "TUPLE-SEED"
    if arg.startswith(("f\"", "f'", "\"", "'")):
        return "STRING-SEED"
    if arg == "" or arg.startswith(")"):
        return "UNSEEDED"
    if re.fullmatch(r"-?\d+", arg):
        return "INT-LITERAL"
    return "DERIVED"


def main() -> None:
    files = subprocess.run(
        ["git", "ls-files", "llmopt", "scripts", "tests"],
        capture_output=True, text=True, check=True).stdout.split()
    rows = []
    for path in files:
        if not path.endswith(".py"):
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if RX.search(line) and not line.lstrip().startswith("#"):
                rows.append((classify(line), f"{path}:{i}", line.strip()[:90]))
    counts: dict[str, int] = {}
    for cls, loc, src in sorted(rows):
        counts[cls] = counts.get(cls, 0) + 1
        print(f"{cls:12} {loc:55} {src}")
    print("\n== totals ==")
    for cls, n in sorted(counts.items()):
        print(f"{cls:12} {n}")


if __name__ == "__main__":
    main()

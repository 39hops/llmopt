#!/usr/bin/env python3
"""Rewrite generated regions in README.md from ledger truth.

Region syntax:
  <!-- llmopt:generated honesty-ledger:start -->
  ...replaced content...
  <!-- llmopt:generated honesty-ledger:end -->

--check exits 1 if a rewrite would change the file (CI drift gate).

Counts are derived the same way tests/test_docs_integrity.py counts
maturity tags: one `[TAG]` per bullet, from the controlled vocabulary
in that test's MATURITY tuple, matched on the bullet's header line
before the first `(`. This is not the `**TAG**` markdown-bold form —
FINDINGS.md tags are plain `[TAG]` markers.

The same counts appear in TWO places, so this script owns both:
README's generated region and the `honesty_ledger` figure in
docs/figures.json (part values plus the fence's claim total). Both
have their own test in tests/test_gen_readme.py.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llmopt.figures.spec import FIGURE_PARTS, MATURITY, findings_counts, recount  # noqa: E402


def counts() -> dict[str, int]:
    return findings_counts((ROOT / "docs" / "FINDINGS.md").read_text())


def render() -> str:
    c = counts()
    total = sum(c.values())
    neg = c["NULL"] + c["RETRACTED"]
    return (f"The {total} curated claims in FINDINGS by maturity: "
            f"{c['REPLICATED']} replicated, "
            f"{c['MECHANISM-CONFIRMED']} mechanism-confirmed, "
            f"{c['SINGLE-SEED']} single-seed, {c['NULL']} null, "
            f"{c['RETRACTED']} retracted. "
            f"{neg} of {total} ({100 * neg / total:.1f}%) are null or "
            f"retracted; {c['SINGLE-SEED']} of {total} "
            f"({100 * c['SINGLE-SEED'] / total:.1f}%) are single-seed.")


def render_figure(fig_text: str) -> str:
    """Rewrite the honesty_ledger part values and the claim count in
    its fence string through the shared serializer
    (llmopt.figures.spec), leaving every other key untouched."""
    return recount(fig_text, (ROOT / "docs" / "FINDINGS.md").read_text())


__all__ = ["counts", "render", "render_figure", "FIGURE_PARTS", "MATURITY"]


def main() -> int:
    figures = ROOT / "docs" / "figures.json"
    fig_text = figures.read_text()
    fig_new = render_figure(fig_text)

    readme = ROOT / "README.md"
    text = readme.read_text()
    pat = re.compile(
        r"(<!-- llmopt:generated honesty-ledger:start -->\n)"
        r".*?"
        r"(\n<!-- llmopt:generated honesty-ledger:end -->)", re.S)
    if not pat.search(text):
        print("no generated region markers in README", file=sys.stderr)
        return 2
    new = pat.sub(lambda m: m.group(1) + render() + m.group(2), text)
    if "--check" in sys.argv:
        drifted = [name for name, changed in
                   (("README", new != text),
                    ("docs/figures.json", fig_new != fig_text)) if changed]
        if drifted:
            print(f"ledger counts drifted in {', '.join(drifted)}; "
                  "run scripts/gen_readme.py")
            return 1
        return 0
    readme.write_text(new)
    figures.write_text(fig_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())

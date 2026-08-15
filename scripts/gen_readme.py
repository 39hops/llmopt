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
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATURITY = ("RETRACTED", "NULL", "MECHANISM-CONFIRMED", "REPLICATED",
            "SINGLE-SEED")


def counts() -> dict[str, int]:
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    bullets = re.findall(r"^- \[.*?(?=^- \[|^#|\Z)", text, re.S | re.M)
    result = {t: 0 for t in MATURITY}
    for bullet in bullets:
        head = bullet.split("(")[0]
        tags = re.findall(r"\[([A-Z][A-Z-]*(?::[^\]]*)?)\]", head)
        mats = [t for t in tags if t in MATURITY]
        if len(mats) == 1:
            result[mats[0]] += 1
    return result


def render() -> str:
    c = counts()
    total = sum(c.values())
    return (f"The {total} curated claims in FINDINGS by maturity: "
            f"{c['REPLICATED']} replicated, "
            f"{c['MECHANISM-CONFIRMED']} mechanism-confirmed, "
            f"{c['SINGLE-SEED']} single-seed, {c['NULL']} null, "
            f"{c['RETRACTED']} retracted.")


# figure part label -> maturity tag, the mapping tests/test_gen_readme.py
# asserts; a label renamed in figures.json must be renamed here too.
FIGURE_PARTS = {"Replicated": "REPLICATED",
                "Mechanism confirmed": "MECHANISM-CONFIRMED",
                "Single seed": "SINGLE-SEED",
                "Null": "NULL",
                "Retracted": "RETRACTED"}


def render_figure(fig_text: str) -> str:
    """Rewrite the honesty_ledger part values and the claim count in
    its fence string, leaving every other key and the file's
    formatting untouched."""
    c = counts()
    fig = json.loads(fig_text)
    ledger = fig["honesty_ledger"]
    for part in ledger["parts"]:
        tag = FIGURE_PARTS.get(part["label"])
        if tag is not None:
            part["value"] = c[tag]
    ledger["fence"] = re.sub(r"\b\d+ curated claims\b",
                             f"{sum(c.values())} curated claims",
                             ledger["fence"])
    return json.dumps(fig, indent=2, ensure_ascii=False) + "\n"


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

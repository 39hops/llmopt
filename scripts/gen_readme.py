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
"""
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


def main() -> int:
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
        if new != text:
            print("README ledger counts drifted; run scripts/gen_readme.py")
            return 1
        return 0
    readme.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())

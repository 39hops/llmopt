"""The one serializer and the one FINDINGS recount for docs/figures.json.

Two generators rewrite that file: scripts/gen_readme.py (README's
generated region shares the honesty-ledger counts) and
scripts/gen_figures_web.py (the SVG/PNG renderer recounts before it
draws). Until 2026-10-13 each carried its own json.dumps call with a
different ``ensure_ascii`` and its own tag-counting regex, so running
them in the other order could leave a byte-level diff. Both now call
the two functions here, so the file is a fixed point of either
generator in any order.
"""
from __future__ import annotations

import json
import re

MATURITY = ("RETRACTED", "NULL", "MECHANISM-CONFIRMED", "REPLICATED",
            "SINGLE-SEED")

# figure part label -> maturity tag; a label renamed in figures.json
# must be renamed here too (tests/test_gen_readme.py asserts the map).
FIGURE_PARTS = {"Replicated": "REPLICATED",
                "Mechanism confirmed": "MECHANISM-CONFIRMED",
                "Single seed": "SINGLE-SEED",
                "Null": "NULL",
                "Retracted": "RETRACTED"}

_BULLET = re.compile(r"^- \[.*?(?=^- \[|^#|\Z)", re.S | re.M)
_TAG = re.compile(r"\[([A-Z][A-Z-]*(?::[^\]]*)?)\]")


def findings_counts(text: str) -> dict[str, int]:
    """Curated FINDINGS bullets by maturity tag: one ``[TAG]`` per
    bullet, read from the header line before the first ``(``, from the
    controlled vocabulary in MATURITY (the same rule
    tests/test_docs_integrity.py enforces)."""
    result = {t: 0 for t in MATURITY}
    for bullet in _BULLET.findall(text):
        head = bullet.split("(")[0]
        mats = [t for t in _TAG.findall(head) if t in MATURITY]
        if len(mats) == 1:
            result[mats[0]] += 1
    return result


def dump(spec: dict) -> str:
    """The canonical bytes of docs/figures.json."""
    return json.dumps(spec, indent=2, ensure_ascii=False) + "\n"


def load(text: str) -> dict:
    return json.loads(text)


def apply_findings(spec: dict, counts: dict[str, int]) -> dict:
    """Write the recounted honesty_ledger part values and its fence's
    claim total into the spec (in place) and return it. Every other key
    is left untouched."""
    ledger = spec.get("honesty_ledger")
    if ledger is None:
        return spec
    for part in ledger["parts"]:
        tag = FIGURE_PARTS.get(part["label"])
        if tag is not None:
            part["value"] = counts[tag]
    ledger["fence"] = re.sub(r"\b\d+ curated claims\b",
                             f"{sum(counts.values())} curated claims",
                             ledger["fence"])
    return spec


def recount(spec_text: str, findings_text: str) -> str:
    """figures.json text -> figures.json text with the FINDINGS recount
    applied, serialized canonically. Idempotent."""
    return dump(apply_findings(load(spec_text), findings_counts(findings_text)))

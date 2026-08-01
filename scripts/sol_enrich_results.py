"""Build Sol's maturity-enriched, read-only copy of the results index.

Inputs are the house ledger and index; outputs live only in docs/sol/.
Every inferred label carries its provenance and evidence so the house can
adopt, revise, or reject the derivation independently.

Usage: .venv/bin/python scripts/sol_enrich_results.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

INDEX = Path("docs/results-index.jsonl")
RESULTS = Path("docs/RESULTS.md")
OUT = Path("docs/sol/results-index-enriched.jsonl")
SUMMARY = Path("docs/sol/MATURITY-SUMMARY.md")

MARKER = re.compile(
    r"^(?:PRE-REG|VERDICT|NULL|AMENDMENT|RETRACTION|RIDER(?:\s+ON)?|"
    r"RESTATEMENT|BOOKED|BANKED)\s+",
    re.I,
)
SINGLE_SEED = re.compile(
    r"\bn\s*=\s*1\b|\bsingle[- ]seed\b|\bone seed\b|\bseed[- ]1\b",
    re.I,
)
REPLICATED = re.compile(
    r"\breplicat(?:ed|ion)\b|\bcross[- ](?:device|lab|runtime)\b|"
    r"\b(?:two|three|2|3) (?:devices|labs|runtimes)\b",
    re.I,
)
MECHANISM = re.compile(
    r"\bmechanism(?:-confirmed| confirmed| lands?| verdict)\b|"
    r"\bcausal (?:mechanism|chain)\b|\bproved by arithmetic\b",
    re.I,
)
ADOPTED = re.compile(
    r"(?<!NOT )\bADOPTED\b|\bSHIPPED\b|\bSCOPED ADOPTION\b|"
    r"\bTAKES PRODUCTION\b",
    re.I,
)
OPEN_N3 = re.compile(
    r"(?:needs?|requires?|must have)[^.\n]{0,60}\bn\s*(?:>=|≥)\s*3\b|"
    r"\bn\s*(?:>=|≥)\s*3\b[^.\n]{0,90}"
    r"(?:before booking|still needed|pending|owed|required)",
    re.I,
)
OPEN_DEVICE = re.compile(
    r"(?:device leg|cross-device (?:leg|replication|verification|control)|"
    r"same-device [^.\n]{0,30}gate)[^.\n]{0,100}"
    r"(?:pending|queued|unrun|never ran|remaining|await)|"
    r"(?:pending|queued|unrun|never ran|remaining|await)[^.\n]{0,100}"
    r"(?:device leg|cross-device (?:leg|replication|verification|control)|"
    r"same-device [^.\n]{0,30}gate)",
    re.I,
)


def _sections(entries: list[dict]) -> dict[str, str]:
    lines = RESULTS.read_text().splitlines()
    ordered = sorted(entries, key=lambda e: e["line"])
    out = {}
    for i, entry in enumerate(ordered):
        start = entry["line"] - 1
        stop = ordered[i + 1]["line"] - 1 if i + 1 < len(ordered) else len(lines)
        out[entry["id"]] = "\n".join(lines[start:stop])
    return out


def _refs(value) -> list[str]:
    """The curated index contains both scalar and list link fields."""
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def _topic(title: str) -> str:
    head = title.split(":", 1)[0]
    head = MARKER.sub("", head).lower()
    head = re.sub(r"\([^)]*\)", " ", head)
    head = re.sub(r"\b(?:extends|amends|partial|closing|leg)\b.*$", "", head)
    return re.sub(r"[^a-z0-9]+", "-", head).strip("-")


def _resolved_preregs(entries: list[dict]) -> dict[str, str]:
    resolved = {}
    for i, entry in enumerate(entries):
        if entry["type"] != "prereg":
            continue
        topic = _topic(entry["title"])
        if not topic:
            continue
        for later in entries[i + 1:]:
            if later["type"] == "prereg":
                continue
            other = _topic(later["title"])
            if other and (other == topic or other.startswith(topic + "-")
                          or topic.startswith(other + "-")):
                resolved[entry["id"]] = later["id"]
                break
    return resolved


def _evidence(text: str, pattern: re.Pattern, fallback: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    match = pattern.search(flat)
    if not match:
        return fallback
    lo = flat.rfind(". ", 0, match.start())
    hi = flat.find(". ", match.end())
    hi = hi + 1 if hi >= 0 else min(len(flat), match.end() + 140)
    return flat[lo + 2:hi].strip()[:300]


def _impact(entry: dict) -> int:
    """Transparent ranking proxy, not a scientific importance judgment."""
    score = min(entry["line"] // 500, 30)  # recent ledger position
    score += min(len(entry.get("threads", [])), 3) * 4
    if entry.get("verdict"):
        score += 40                         # hand-curated result summary
    if re.search(r"\b(?:VERDICT|LAW|DOCTRINE|SHIPPED|ADOPTED)\b",
                 entry["title"], re.I):
        score += 20
    if entry["maturity"] == "adopted":
        score += 10
    return score


def enrich() -> list[dict]:
    entries = [json.loads(line) for line in INDEX.read_text().splitlines()]
    sections = _sections(entries)
    resolved = _resolved_preregs(entries)
    strong_superseded = {}
    for later in entries:
        if re.search(r"\b(?:RETRACT(?:ED|ION)?|RETIRED|RESCOPE|SUPERSEDE)\b",
                     later["title"], re.I):
            for target in _refs(later.get("amends")):
                strong_superseded[target] = later["id"]
    out = []
    for original in entries:
        entry = dict(original)
        text = sections[entry["id"]]
        title = entry["title"]
        flags = []
        if SINGLE_SEED.search(text):
            flags.append("single-seed")
        if REPLICATED.search(text):
            flags.append("replicated")
        if MECHANISM.search(text):
            flags.append("mechanism-confirmed")
        if ADOPTED.search(title):
            flags.append("adopted")
        if entry["type"] == "null" or re.search(
                r"\b(?:NULL|NO-ADOPT|VOID|DIES|DEAD)\b", title, re.I):
            flags.append("null")

        if entry.get("superseded_by"):
            maturity, source = "superseded", "explicit"
            evidence = "superseded_by metadata"
        elif entry["id"] in resolved:
            maturity, source = "superseded", "inferred"
            entry["inferred_superseded_by"] = [resolved[entry["id"]]]
            evidence = f"later entry with matching heading label: {resolved[entry['id']]}"
        elif re.search(r"\b(?:RETRACT(?:ED|ION)?|RETIRED|WITHDRAWN)\b", title, re.I):
            maturity, source = "retracted", "explicit"
            evidence = _evidence(title, re.compile(r"RETRACT|RETIRED|WITHDRAWN", re.I), title)
        elif entry["type"] in {"prereg", "banked"}:
            maturity, source = "in-flight", "inferred"
            evidence = f"index type={entry['type']} and no matched later result"
        elif entry["type"] == "null" or re.search(
                r"\b(?:NULL|NO-ADOPT|VOID|DIES|DEAD)\b", title, re.I):
            maturity = "null"
            source = "explicit" if re.search(
                r"\b(?:NULL|NO-ADOPT|VOID|DIES|DEAD)\b", title, re.I) else "inferred"
            evidence = f"title/index type={entry['type']}"
        elif entry["id"] in strong_superseded:
            maturity, source = "superseded", "explicit"
            entry["inferred_superseded_by"] = [strong_superseded[entry["id"]]]
            evidence = f"later strong amendment: {strong_superseded[entry['id']]}"
        elif ADOPTED.search(title):
            maturity, source = "adopted", "explicit"
            evidence = _evidence(title, ADOPTED, title)
        else:
            maturity, source = "measured", "inferred"
            evidence = f"completed index type={entry['type']}; no stronger explicit label"

        open_match = OPEN_N3.search(text) or OPEN_DEVICE.search(text)
        entry.update({
            "maturity": maturity,
            "maturity_source": source,
            "maturity_evidence": evidence,
            "maturity_flags": flags,
            "replication_open": bool(open_match),
        })
        if open_match:
            entry["replication_evidence"] = _evidence(
                text, OPEN_N3 if OPEN_N3.search(text) else OPEN_DEVICE, open_match.group(0))
        out.append(entry)

    for entry in out:
        entry["impact_score"] = _impact(entry)
    return out


def write_summary(entries: list[dict]) -> None:
    counts = Counter(e["maturity"] for e in entries)
    sources = Counter(e["maturity_source"] for e in entries)
    singles = sorted(
        (e for e in entries if "single-seed" in e["maturity_flags"]
         and e["maturity"] not in {"retracted", "superseded"}),
        key=lambda e: (-e["impact_score"], -e["line"], e["id"]),
    )[:10]
    lines = [
        "# Maturity summary", "",
        "Generated by `scripts/sol_enrich_results.py` from the house ledger.",
        "Labels marked inferred are deterministic heuristics, not house verdicts.", "",
        "## Counts per status", "",
        "| Status | Count |", "|---|---:|",
    ]
    lines += [f"| {status} | {count} |" for status, count in sorted(counts.items())]
    lines += ["", "Provenance: " + ", ".join(
        f"{key}={value}" for key, value in sorted(sources.items())) + ".", "",
        "## Ten highest-impact live single-seed entries", "",
        "Impact is a reproducible triage proxy: curated verdict +40, explicit "
        "VERDICT/LAW/DOCTRINE/SHIPPED/ADOPTED +20, adopted +10, up to three "
        "thread tags +4 each, and ledger recency up to +30.", "",
        "| Score | Date | Status | Entry |", "|---:|---|---|---|",
    ]
    for e in singles:
        lines.append(f"| {e['impact_score']} | {e.get('date') or 'undated'} | "
                     f"{e['maturity']} | `{e['id']}` |")
    SUMMARY.write_text("\n".join(lines) + "\n")


def main() -> None:
    entries = enrich()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    write_summary(entries)
    print(f"{len(entries)} entries -> {OUT}")
    print(Counter(e["maturity"] for e in entries))
    print(f"replication_open: {sum(e['replication_open'] for e in entries)}")


if __name__ == "__main__":
    main()

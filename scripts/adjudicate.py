"""Deterministic bar adjudication from a machine-readable pre-reg.

    .venv/bin/python scripts/adjudicate.py docs/preregs/<name>.json \
        <observations.json>

The pre-reg names the bars; the observations document carries
measurement validity, per-arm admissibility, and the measured
metrics (see llmopt/lab/prereg.py for both schemas). Output is one
line per bar — FIRE / NO-FIRE / UNRESOLVED with the reason chain —
and the process exits 0 only if every bar reached FIRE or NO-FIRE,
so a wrapper can tell "cleanly adjudicated" from "something was
inadmissible" without parsing prose.

A MetricContractError here is NOT an outcome: it means the
measurement handed to a bar is not the registered quantity (wrong
metric, population, or aggregation). Fix the pipeline; never book it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.prereg import (adjudicate_prereg,  # noqa: E402
                               adjudicate_refutation, load)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prereg", help="docs/preregs/<name>.json")
    ap.add_argument("observations", help="observations JSON")
    a = ap.parse_args()
    prereg = load(a.prereg)
    obs = json.loads(Path(a.observations).read_text())
    r = adjudicate_refutation(prereg, obs)
    if r is not None:
        print(f"REFUTED-IF: {r}")
    unresolved = 0
    for o in adjudicate_prereg(prereg, obs):
        line = f"BAR {o.bar_id} {o.bar_name}: {o.outcome}"
        if o.reasons:
            line += " [" + "; ".join(o.reasons) + "]"
            unresolved += 1
        print(line)
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerate the Parquet lake (data/lake/) from the lab's jsonl/file exhaust.

The lake is regenerable EXHAUST (gitignored; logs doctrine 2026-08-06:
regenerate-don't-download). jsonl stays the write format; this is the query
layer. See llmopt/lab/lake.py for schemas and the gate-row device doctrine.

Usage:
  .venv/bin/python scripts/gen_lake.py [--tables runs,results,models,gates]
                                       [--lake-dir data/lake]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab import lake  # noqa: E402

ALL_TABLES = ("runs", "results", "models", "gates")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default=",".join(ALL_TABLES),
                    help="comma list of tables to (re)build")
    ap.add_argument("--lake-dir", default=str(lake.DEFAULT_LAKE_DIR),
                    help="output dir override (tests)")
    args = ap.parse_args()

    lake_dir = Path(args.lake_dir)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = set(tables) - set(ALL_TABLES)
    if unknown:
        ap.error(f"unknown tables: {sorted(unknown)} (choose from {ALL_TABLES})")

    for t in tables:
        if t == "runs":
            print("runs ->", lake.build_runs(lake_dir=lake_dir))
        elif t == "results":
            for p in lake.build_results(lake_dir=lake_dir):
                print("results ->", p)
        elif t == "models":
            print("models ->", lake.build_models(lake_dir=lake_dir))
        elif t == "gates":
            print("gates ->", lake.build_gates(lake_dir=lake_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

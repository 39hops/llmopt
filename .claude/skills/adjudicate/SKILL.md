---
name: adjudicate
description: Use when scoring a rung's measured numbers against its registered bars — turns FIRE/NO-FIRE/UNRESOLVED into a deterministic program run instead of prose judgment. Covers writing the machine-readable pre-reg (docs/preregs/*.json), the observations document, and reading the adjudicator's output.
---

# Deterministic adjudication (/adjudicate)

Origin: STREAM-WDISTILL-0 (2026-08-16). Two incident classes in one
thread came from adjudicating prose bars by hand — a layer bar
scored against one expert, and an over-budget scalar entering a
"matched-bytes" contrast. The repair promoted a law; this skill runs
the law as code:

    bar_adjudicable = measurement_valid
                      AND every named arm admissible
                      AND the contrast admissible

## The flow

1. **At pre-reg time** (alongside the prose entry, same commit):
   write `docs/preregs/<rung-kebab>.json`. Schema is documented in
   `llmopt/lab/prereg.py`; validate it immediately:
   `.venv/bin/python -c "from llmopt.lab.prereg import load; load('docs/preregs/<name>.json')"`.
   The prose entry in RESULTS.md remains the registration of record;
   the JSON is its executable projection. Set `results_id` to the
   prose entry's stable id from docs/results-index.jsonl.
   A JSON written AFTER receipts exist is not a pre-reg — it must
   carry a `note` saying RETROSPECTIVE, like the first fixture
   (docs/preregs/stream-wdistill-0.json).

2. **Declare receipts structurally.** The `receipts` list names the
   exact repo-relative paths the run will write. These feed
   `scripts/gen_receipt_lock.py` directly — no prose scraping, so
   bare filenames cannot slip through (the known item-2 gap).

3. **At verdict time**: write the observations JSON (measurement
   validity + per-arm admissibility WITH reasons + measured metrics)
   and run

       .venv/bin/python scripts/adjudicate.py \
           docs/preregs/<name>.json <observations.json>

   Exit 0 = every bar FIRE/NO-FIRE. Exit 2 = at least one
   UNRESOLVED (book it honestly with its reason chain). A
   MetricContractError = the measurement is not the registered
   quantity (wrong metric/population/aggregation) — that is a
   pipeline bug, NEVER a bookable outcome.

4. **Book the adjudicator's words.** The RESULTS verdict quotes the
   tool's per-bar lines verbatim; the prose may add reading but may
   not overrule an outcome. If prose and program disagree, one of
   them is wrong — resolve before booking.

## Fences

- The bar's `value` must be a number already on the page at
  registration (a bar you can reword after data is not a bar); the
  schema refuses non-numeric values and unknown/typoed keys loudly.
- UNRESOLVED is a final scientific outcome, not a retry ticket.
  Repair happens in a NEW registered rung.
- Descriptive cross-population looks stay outside this flow —
  `cross_population_difference()` returns a type the adjudicator
  structurally refuses.

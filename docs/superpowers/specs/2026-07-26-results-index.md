# SPEC (banked, not implemented): RESULTS.md index + programmatic query

Artin's ask (2026-07-26): "an easy way to index/grep it for you +
an easier/programmatic way to query/confirm certain results...
proper spec/plan/implement." Banked after a full end-to-end read
of RESULTS.md (~6,600 lines, ~250 entries).

## Why NOT a quarterly file split (decision, from the full read)

The file's value is its AMENDMENT CHAINS: entries are corrected,
superseded, and retracted by later entries (streaming v1 -> 4
amendments -> the 2x2 close; the sigma-grid -> its amendments;
the Muon close -> the CE-gate retraction). Splitting by date cuts
exactly those chains. The right structure is the file as-is + a
machine-readable INDEX layered on top.

## Design

1. **`docs/results-index.jsonl`** — one line per entry:
   `{"id": "2026-07-26-muon-close", "date", "title", "line",
   "type": "prereg|verdict|amendment|null|retraction|instrument",
   "threads": ["streaming", "optimizer"], "verdict": "<one line>",
   "amends": ["<id>"], "superseded_by": ["<id>"], "numbers":
   {"gate": 34, "comparator": 45}}`
   - BACKFILL: semi-automatic — a script extracts id/date/title/
     line from headers; type/threads/verdict/links are a manual
     pass, done incrementally (newest 50 entries first — the ones
     queries actually hit), never blocking.
   - FORWARD: every new entry adds its index line in the same
     commit (one line of discipline per booking).
2. **`scripts/results_query.py`** — greps the index, not the
   prose: `--thread streaming`, `--type null`, `--live` (entries
   not superseded), `--chain <id>` (walk amendment links both
   directions, print the full verdict chain). Output = id, date,
   one-line verdict, line number into RESULTS.md for the jump.
3. **Supersession hygiene**: the `--live` view is the killer
   feature — "has this idea been run and what's the CURRENT
   verdict" becomes one command instead of a 6,000-line read.
4. Optional later: same index schema for THEORY rows and
   RIFF-LEDGER banks (three files, one query tool).

## Costs / risks

- Backfill of ~250 entries is the real cost (est. 2-3 focused
  hours for the newest 100; the pre-micro-model era can stay
  header-only).
- Index drift is the failure mode — mitigated by the same-commit
  rule + a CI-ish check (every `^## ` header has an index line;
  gen_index.py pattern).

## Status: BANKED. Implement on Artin's GO at a natural freeze
point (post-ZX-column booking is a good one).

## Step-3 additions (Artin GO 2026-07-26)

- (a) `git tag results-pre-index` at the freeze point — the
  permanent snapshot of claims-as-written; no file copy.
- (b) bidirectional amendment links in the index (amends /
  superseded_by both directions; --chain walks either way).
- (c) generated SCOREBOARD.md from the index (--live view
  materialized; regenerated like scripts/INDEX.md, never
  hand-edited) — the curated "good RESULTS" layer; RESULTS.md
  keeps its name and its append-only log role. Rename revisited
  only after living with the scoreboard.
- (d) PER-PROBLEM INSTRUMENTATION AS DEFAULT (Artin 2026-07-26):
  every gate/probe in the CLI wrapper dumps per-problem outcomes
  + full greedy chains (jsonl sidecar), so any two cells can be
  diffed problem-by-problem after the fact — scores hide
  complementarity (the duo-wave +5 precedent) and wandering/
  identity/length signatures cost nothing to keep.

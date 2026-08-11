# Log hygiene plan — 2026-08-11 (print-only run)

Produced by `scripts/log_hygiene.py` (reviewer design, handoff
2026-08-11-0). The tool never deletes; `--apply` is double-gated
(ARTIN_GO=1 AND zero FROZEN/UNKNOWN rows) and even then only moves
SWEEPABLE files to `logs/archive/<today>/` preserving relative paths.
This run was plan-only; nothing was moved. Bulk deletion remains
Artin-GO by hand (logs doctrine 2026-08-06; pairs with the banked
51GB triage thread).

Command: `.venv/bin/python scripts/log_hygiene.py --json plan.json`

## Class summary (Mac checkout, logs/, --age-days 14)

| class | files | bytes |
|---|---:|---:|
| FROZEN | 238 | 1,160,290,281 (~1.08 GB) |
| PRESERVE-AS-RECEIPT | 0 | 0 |
| SWEEPABLE | 643 | 309,775,719 (~295 MB) |
| UNKNOWN | 167 | 17,120,844 (~16 MB) |

Notes:
- FROZEN = path or parent dir cited in docs/RESULTS.md (126 `logs/`
  citation lines, one grep pass; a directory-level citation freezes
  the whole dir). The doubled path `logs/archive/logs/archive/`
  (RESULTS 2421 cites `.../ceiling_probe.log`) classifies FROZEN
  unconditionally — indexed in place, never re-sorted.
- PRESERVE-AS-RECEIPT is empty on the Mac: the `*_oomkilled*`
  receipts from the micro-star OOMs live on the 3080 side, not in
  this checkout. The class is live in the classifier and tested.
- UNKNOWN (uncited, newer than 14 days) gets no action by design.

## Top 20 SWEEPABLE by size

All 20 largest are under `logs/archive/wsl-2026-07-26/` — the
2026-07-26 WSL pullback. See dedup section below before sweeping.

| path | bytes |
|---|---:|
| logs/archive/wsl-2026-07-26/data/zx_farm1_train.jsonl | 101,783,316 |
| logs/archive/wsl-2026-07-26/data/gen9_diet_B.jsonl | 80,957,495 |
| logs/archive/wsl-2026-07-26/data/regret_trace_labels.jsonl | 33,628,067 |
| logs/archive/wsl-2026-07-26/wsl_archive_2026-07-26.tar.gz | 31,441,966 |
| logs/archive/wsl-2026-07-26/data/merged_diet.jsonl | 26,369,671 |
| logs/archive/wsl-2026-07-26/data/micromodel_gen4_sidecar.jsonl | 4,562,356 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard7.jsonl | 2,196,195 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard11.jsonl | 2,150,811 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard10.jsonl | 2,144,247 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard8.jsonl | 2,130,593 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard9.jsonl | 2,120,101 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard6.jsonl | 2,117,147 |
| logs/archive/wsl-2026-07-26/data/zx_farm1_held.jsonl | 2,082,132 |
| logs/archive/wsl-2026-07-26/data/practice_rows_v5.jsonl | 1,965,148 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard1.jsonl | 1,125,184 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard3.jsonl | 1,108,764 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard0.jsonl | 1,098,338 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard4.jsonl | 1,098,171 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard2.jsonl | 1,088,874 |
| logs/archive/wsl-2026-07-26/data/micromodel_chains_shard5.jsonl | 1,076,020 |

## Dedup candidates (sha256-verified this run)

1. **Confirmed byte-identical duplicates (~183 MB reclaimable)** —
   `logs/archive/wsl-2026-07-26/data/` holds exact copies of two
   live `data/` files:
   - `gen9_diet_B.jsonl` (80,957,495 B) — sha256
     `92c22242…4fd0c4da` matches `data/gen9_diet_B.jsonl` exactly.
   - `zx_farm1_train.jsonl` (101,783,316 B) — sha256
     `ef0e0f33…c9224dbb` matches `data/zx_farm1_train.jsonl` exactly.
   (Also size-matched but not part of the ~183MB claim:
   `merged_diet.jsonl`, `zx_farm1_held.jsonl` — sha not yet run.)
2. **NOT duplicates — name-twin autopsy candidate**:
   `checkpoints/mathnative_110m_gen4_std.pt` and
   `checkpoints/mathnative_110m_gen4_std_3ep.pt` share byte size
   453,338,827 but their sha256 DIFFER
   (`2d2db625…5316c2fe` vs `287df077…cb7299b`). Same architecture,
   different weights — do not dedup; book as a name-twin autopsy
   candidate (which training run produced which?).

## COMPLETION-SIGNAL CONSOLIDATION MAP (print-only)

Writers of ad-hoc `.DONE` / `.rc` / `.marker` / `.ep` sentinels in
scripts/ and scratch/ top level (scratch/leancheck excluded; the
planner excludes itself). Proposed target: the unimplemented
`llmopt/lab/runfiles.py` contract (write_marker one-JSON-line,
is_done/rc_of shared by both machines, require_resume_marker).

| file | signal(s) | migration note |
|---|---|---|
| scripts/bench_syndrome_head.py | .ep | replace ad-hoc .ep with write_marker(); .ep currently means two different things (grow writes "-1", train reads N+1) |
| scripts/grow_mathnative.py | .ep | same .ep ambiguity — adopt runfiles resume gate FIRST (proposed adoption order) |
| scripts/train_mathnative.py | .ep | same; require_resume_marker(ckpt) closes the "ckpt without marker" ambiguity that nearly ate a checkpoint |
| scripts/rjob.py | .rc | writes `killed` into a .rc consumers may int(); rc_of() gives one decoding |
| scratch/wsl.sh | .rc | rjob and wsl.sh use two incompatible done-encodings; unify on rc_of/is_done |
| scratch/night31b_cuda.sh | .DONE | Spark-_SUCCESS-style marker via write_marker(); completion markers fire on SUCCESS only (remote-ops doctrine) |

## Recommended next step (needs Artin GO)

The 167 UNKNOWN rows block `--apply` by design. Path to a clean
apply: triage UNKNOWN (most are recent run exhaust that will age
into SWEEPABLE), then run
`ARTIN_GO=1 .venv/bin/python scripts/log_hygiene.py --apply` at a
natural freeze point under the BOARD housekeeping gate. The two
confirmed duplicate files above are the cheapest ~183 MB, but they
sit under a wsl-2026-07-26 pullback dir — verify no RESULTS entry
cites that dir at the directory level before any manual action
(the planner classed them SWEEPABLE, i.e. it found no citation).

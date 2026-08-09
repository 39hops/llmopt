# Relay 2026-08-09-1 (house -> axiom): ENGINE-SCALE-1 export SHIPPED — 30 cells armable, GO to run

WHO IS WRITING: Fable, llmopt seat. Artin GO'd 2026-08-09
afternoon. The fence is satisfied: this fires FIRST, ahead of any
further exact-training comparison work.

## What shipped (this pull)

`docs/superpowers/specs/engine_scale_cells/` — 9 unique artifacts
(3 param sizes x 3 window counts; const/SCHED pairs share bytes)
+ `manifest.jsonl`, one row per cell (30), each carrying bin name,
bin sha256, windows-ids sha256, measured param count, and the full
contract (V=40, DIM=64, DHEAD=16, FFN, NBLK, SHIFT=14, SEED=17,
T=32, NWIN, STEPS, SCHED, LR convention).

Artifact format = the DIET-BRIDGE layout your engine already
consumes: init params (param_items order, seed 17) then window
token ids (int64 [NWIN, 33]). Exporter:
scratch/engine_scale_export.py (committed), which SELF-VERIFIES:
the 60k-w8 bin reproduces the certified DIET-BRIDGE artifact sha
880e4e295f1e9544... and windows sha 99caaa646925d150... exactly —
the export path is receipt-anchored before you spend a cycle.

## Contract pins worth restating

- SCHED=1 means lrd *= 2 at ABSOLUTE steps 250/500/750 (lrd start
  1000) — the certified s4000-sched convention (traj sha
  15934bb8...); it does NOT scale pro-rata with STEPS. At
  s16000 the schedule therefore acts entirely in the first 750
  steps; that is the registered design, not an oversight.
- Windows are prefix-nested by construction (w8 subset of w32
  subset of w128, file-order draw) — shas in the manifest.
- Params measured: 31,424 / 60,224 / 109,376.

## Verification contract (from the pre-reg, unchanged)

Before any verdict books house-side, three spot cells must match
already-frozen house receipts (zero new house compute):
  60k-w8-s1000-const  -> DIET-BRIDGE final traj sha 8b443b68...
  60k-w32-s4000-const -> PLATEAU-BREAK arm B 561e28c5...
  60k-w8-s4000-sched  -> P-STEP-BOUND-2 15934bb8...
Emit per cell: milestone losses + FINAL trajectory sha, one jsonl
row (the pre-reg's format). Bars are booked at RESULTS L22317
(P-JOINT <= 11,266 at the top corner; P-DIET-FLOOR names WINDOWS
by the leave-one-small pattern).

## Sequencing

EXACT1-SMALL cells are running on the WSL box (CPU, 1-2 cores) —
plenty of headroom for the grid there or wherever you prefer;
your timing estimate was minutes-to-tens-of-minutes per cell.
Nothing else contends.

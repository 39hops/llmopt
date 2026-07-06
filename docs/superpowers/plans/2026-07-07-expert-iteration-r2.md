# Expert Iteration Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One full expert-iteration turn: harvest frontier derivations only the current engine can find, retrain the proposer on them, measure whether the next engine's frontier solve rate rises.

**Architecture:** mathgen gains level 4 (with a collision-rate guard); a harvest script runs baseline vs full engine over L4 problems and keeps full-engine-only winning paths; retraining reuses `train_proposer.py` unchanged (data grows); measurement is a before/after frontier race plus an L1-3 regression guard.

**Tech Stack:** sympy, torch, existing scripts.

**Spec:** `docs/superpowers/specs/2026-07-07-expert-iteration-r2-design.md`

## Global Constraints

- L4 collision guard BEFORE trusting any split: <1% duplicate sreprs over 1k draws per kind; widen generator if violated.
- Frontier = baseline (full+HCE, budget 200) fails; harvest = full engine (prop3+HCE, budget 400; +3 k1-restarts on failure) solves. (prop3+HCE is the measured champion config — NNUE stacking excluded per the wall-clock finding.)
- Retrain: existing recipe, only data grows (old 1429 rows + frontier rows).
- Measure: held-out L4 frontier problems (exclude-guarded), engine-r1 vs engine-r2, budgets 100/200/400; regression guard: L1-3 race totals within noise.
- Pre-registered null: near-zero harvest yield => guidance isn't the constraint, new rules are (rung 3).

---

### Task 1: mathgen level 4 + collision guard

**Files:**
- Modify: `llmopt/mathgen/problems.py` (`_expression` gains level 4)
- Test: `tests/test_mathgen_l4.py`
- Measure: inline collision script (goes in the commit message)

Level-4 shapes (all containing x, drawn with string-seeded rng):
- depth-3 composition: outer(inner_poly + inner_fn) e.g. sin(x**2 + cos(x)), exp(sqrt(x**2+1))
- composed product: atom(2) * outer(inner_poly) * atom(2)
- chained-usub integrand shape: poly'(x) * fn(poly(x)) * fn2(fn(poly(x))) is too wild; use fn(poly) * poly' * (fn(poly))**n
- sums of two of the above

Test: 200 L4 draws per kind are all valid sympy exprs containing x; diff of each is finite; collision rate over 1000 draws < 1%.

- [ ] Write failing test; implement level-4 branch in `_expression`; run guard; commit with measured collision rate.

### Task 2: frontier harvest script

**Files:** Create `scripts/harvest_frontier.py`

Per problem (seeds `eir2-harvest-{kind}-{level4}-{i}`): run baseline (full+HCE, 200 nodes, 120s wall) — if solved, skip (not frontier). Else run full engine (prop3+HCE, 400 nodes, 300s wall; on failure, 3 k1-random restarts at 133 nodes each). If solved: verify terminally (diff-based check), then replay winning path via `successors` and emit winning-path rows (same JSONL schema as gen_proposer_data). Print yield stats: n_frontier, n_harvested, yield%. Output `data/frontier_r1.jsonl` + roots json.

- [ ] Write script; run on WSL box (model inference needs GPU — the 3080 serves the proposer; CUDA works there); commit data with yield numbers. Null check: if yield < ~5% of frontier, STOP and report (pre-registered null).

### Task 3: retrain + before/after measurement

- Concatenate old + frontier rows; rerun `train_proposer.py` on the 3080 (reads both files via a --extra-data flag added to the script); save `checkpoints/proposer_lora_r2.pt`.
- `scripts/bench_frontier.py`: held-out L4 problems (exclude harvest roots), engine-r1 (old ckpt) vs engine-r2 (new ckpt), prop3+HCE, budgets 100/200/400, n=20/cell.
- Regression guard: bench_proposer-style quick race on L1-3 with r2 ckpt; totals within ±10 of r1's.
- [ ] Commit each with numbers; README + roadmap curve point; memory.

## Self-review

Spec covered: L4+guard (T1), frontier def + harvest + yield null (T2), retrain + frontier delta + forgetting guard (T3). Champion-config choice documented (prop3+HCE, per race). GPU placement: harvest + retrain on the 3080 (proposer inference); Mac stays free for the adaptive re-race.

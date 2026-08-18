# Handoff 2026-08-17-4: teacher LOCKED + gated, CUDA tower talks at 8.8 tok/s, precision gradient observed

Seat: Fable (main session model), Mac. Continues -3 (same day).
Session closes for a CLEAR; resume = this file -> BOARD -> spec
`docs/superpowers/specs/2026-08-18-scorer-tree-projection.md` ->
RESULTS tail.

## What landed (commit shas in today's git log)

- **VERDICT QWEN-TEACHER-0-LOCK** (7817e28): teacher v2d ACCEPTED.
  Commit pin 0ca4151 exact; sidecar cached-v-uncached gate PASSED
  — token equality 4722/4722, max rel L2 2.135e-4 v 5e-3 (fp16-
  symmetric quantization deviation disclosed; raw-fp32 reading
  ~3-4e-4 by auditor estimate, PASS robust); margin census booked
  from committed producer with recorded shas — SMALL-N FENCE LIVE
  on scored streams (corpus+prefixes bins < 0.2 nats all under
  30; EVERY prefix bin under 30 — no directional prefix-only
  margin claim exists). Both auditors ran pre-booking; census
  first emission REFUSED (uncommitted producer, unfalsifiable
  boolean) and re-emitted clean. qrope fence carried verbatim.
- **CUDA ladder rungs 0-4** (OBSERVATIONS -LADDER-0, -RUNG4,
  -S16-B + AMENDMENTS -SCOPE, -S16-B-PAIRED): Triton toolchain
  proven on WSL; VQ2 decode kernel bit-exact v qcodec (7 fixtures
  + real tensor); fused GEMV 176 GB/s effective; full tower
  resident, forward1 0.93s, **8.82 tok/s** (A), **7.76** (B via
  adopted S16Rows + s16 GEMV kernel); free VRAM measured
  8.86-9.51 GiB (C cannot fit under the rung-3/4 residency plan);
  top-5 identical across both CUDA backends and CPU reference.
- **Precision gradient (qualitative, paired at 6982ab3, fires
  nothing)**: same QM prompt, same commit — A tight-loops by ~400
  tokens; B executes the algebra, computes an impossible 2.998,
  SAYS SO, restart-cycles. Qualitatively consistent with T1's
  preregistered direction. Teacher under identical protocol =
  the registered adjudicator.
- Ledger hygiene: -SMOKE-LOCK amendment (bare-filename receipts
  never locked — standing order: full paths, force-add BEFORE
  lock regen); test count corrected 40->24; -RUNG4 prose/receipt
  commit mismatch booked; index observation-type producer bug
  being fixed at close (agent, see below).
- Riffs: cross-model KV transfer (arXiv:2608.03893), momentum-
  space phase portraits (PHASE-PORTRAIT-2 residue; PP-1 stored
  norms not vectors — verified), 2-bit-as-router (+ same-day
  correction: v vendor teacher, coupled FFN channel, R_k; GPT's
  E_k reconstruction extension in the spec).
- Instruments: scripts/arena_qwen.py (paired-run discipline);
  compactness figure (chat artifact).

## Conditions that bite next session

- **Open with the scorer** — the spec is written; the next
  interesting commit is X_A..K_C. Enforce n>=30 PER REPORTED
  STRATUM in the scorer (pooled census does not license strata).
- `docs/preregs/qwen-model1-tree.json` still does not exist.
- Sidecar-v2 cleanup required before ROLLOUT-based evidence only
  (4-step, 48/16 in receipt, raw-fp32, revision assert).
- mlx-lm/llama.cpp DO carry qwen3_5 arch code now (corrects this
  session's guess) — Q8 baseline starts with a conversion smoke,
  separately preregistered, no branch in the frozen tree.
- Teacher .npy logit records are Mac-local (shas in manifest);
  small receipts tracked + locked.
- 3080: idle at close, artifacts resident, both checkouts at
  origin/main.

## Close-out state

- Adoption at close: llmopt/lab/qcuda.py (CUDA primitives,
  source-identity-guarded to scratch/qwen_cuda_rung4.py) +
  results-index observation-type fix — landed by directed Opus
  write-agents, session-model-verified before commit.
- Suite green at close (pytest, real exit code), check_source
  green, pushed.

## Open decisions for Artin

1. README front-door Qwen paragraph (carried from -2).
2. Overnight 3080 allocation (scorer needs the Mac, not the
   3080 — box is free for anything).
3. MTP exclusion confirm (carried).
4. Q8 baseline route: external conversion smoke (now cheap) v
   house-compiled 8-bit arm.

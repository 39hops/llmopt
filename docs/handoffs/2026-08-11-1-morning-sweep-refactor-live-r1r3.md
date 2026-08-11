# Handoff 2026-08-11-1: morning review sweep adopted, refactor infra live, R1/R3 mid-flight

Session: Fable 5 seat (swapped back from Opus 5 ~09:30). Previous
handoff: 2026-08-11-0 (crown resolution + star-profile arc +
micro-ladder). Resume chain: this file -> RESULTS tail -> BOARD.

## BOOKED SINCE 2026-08-11-0 (all pushed)

1. VERDICT MICRO-STAR-1 (26301) — both bars fire; ignition between
   d32 and d48; d64 30/120; births in minutes. PLUS AMENDMENT
   MORNING-SWEEP-0811: params column corrected (ignition ~295k not
   ~180k), validity fences, budget knob named, d48-at-exactly-bar
   single-seed fence.
2. VERDICT LOSS-FLOOR-1 (26376) — floor 0.348 = 0.502 x corpus
   entropy H_full 0.175; knee at k=32; floor sits between H_16 and
   H_32. PLUS AMENDMENT LOSS-FLOOR-1-ARCH: the 0.348 belongs to the
   d512/L12 GROWN crown line, not "the d256 star" (verified against
   weights; the catalog was the independent cross-check). Reading
   STRENGTHENS (bigger star, same H_16 floor).
3. VERDICT SATURATION-1-CELL-B (26420) — widened diet +2 (75 v 73),
   P-SATURATED-CLASS fires; star is full at fixed mass. Sweep
   corrected the verdict's "3 epochs" wording: it was ONE epoch on
   the crown copy (log receipt "resuming at epoch 3"); experiment
   design was right, prose was wrong.
4. COUNTER-BOOK NPRIMES-LADDER (26450) — P-EXACT-TIE verified from
   raw receipts (signature ring-invariant 256/512/1024p; prefix
   digests byte-match; slope 1.94x/2.11x per doubling).
5. PRE-REG COFACTOR-WITNESS-2 (26485) — denominator-ledger design
   gate; relay 2026-08-11-0 DELIVERED to mac-axiom (census probe
   only; witness build gated; honest-death clause registered).
6. VERDICT REFACTOR-NIGHT-1 (26536) — see infra below; booked BY
   scripts/book.py from its own run marker (the dogfood).
7. PRE-REG MERGE-SPACE-1 (26602, R1) + PRE-REG FLOOR-HK-1 (26641,
   R3) — both RUNNING (below). FLOOR-HK-1 bar 1 re-anchored
   ladder-only by the LOSS-FLOOR-1-ARCH amendment; fresh d256 cell
   queued as its own run after d512.
8. RIFF banks: Anthropic zeta-bound run (external validation of
   oracle-verified + adversarial-subagent method; Lean-certificate
   tier is the piece to steal); grok-seat architecture cross-check
   (verified, priority stack matches extraction-spec items 5-7).

## INFRASTRUCTURE (REFACTOR-NIGHT-1 + morning hardening; pytest 597)

Shipped by a 14-agent workflow (7 build + 7 review), then hardened
by a 4-reviewer morning sweep. All in llmopt/lab/ unless noted:
- catalog.py + scripts/gen_catalog.py — 392 checkpoint rows, sha256
  (full pass done: rjob catalog_sha rc=0), shape-only arch,
  parent_ids from filenames (ep0->stem edge dropped — mtimes showed
  the bare stem can be the ROLLING file), RESULTS-citation flag.
  data/catalog/models.jsonl = exhaust, regenerable.
- lake.py + scripts/gen_lake.py — Parquet: runs(82) results(906+)
  result_edges models(392) gates. gates schema REQUIRES
  device+n_seeds+weights_sha non-null; dict-is-checksum enforced at
  write; atomic append. data/lake/ gitignored exhaust.
- merge.py — average (refuses without shared_lineage; cites 9135/
  12356), task_vector (probe-grade; cites 11197 + the flagged
  stale spec-INDEX claim), shell_graft (refuses ternary BY NAME
  HINT — house RAT_Q ckpts store fp32 latents, 4091 uniques
  verified, the lattice check cannot see them; anchors to
  blocks.N FFN keys only; writes the "-1" .ep sidecar). NEVER
  overwrites any existing file. gate_cmd returns commands, never
  runs them (device doctrine).
- scripts/book.py — mechanical booking from runfiles markers.
  Refuses: killed/nonzero marker, gate-dict-sum mismatch, sha-less
  gate (graduates RESULTS 13463), n=1 sub-sigma verdict without
  fence. Morning sweep closed real bypasses (d=512 shadowed the
  delta regex; "+3 solves" phrasing never matched; missing n_seeds
  skipped the fence) and added rollback-on-refusal.
- runlog.py — per-step streamed receipts (axiom row shape: digest
  chain, fb counters, wall_s, aborted-as-row). Mode "x" (refuses
  existing receipt); marker named per log stem.
- runfiles.py — marker contract + require_resume_marker (wired into
  train_mathnative: ckpt-without-.ep now refuses).
- backends/intbirth_native.py — axiom's prebuilt .so (exact int64
  GEMM x3 + rdiv) with intmath fallback + parity tests; repo-
  relative default path; non-integral floats REFUSED.
- vendor/axiom/ — nn_exact_ref (.axnn read/write), divergence,
  classify_sample; verbatim + provenance headers + source-identity
  guards. CAUTION: divergence.py runs argparse at import (upstream
  shape; documented in vendor __init__).
- scripts/log_hygiene.py — print-only planner; morning sweep added
  the bare-basename citation pass (3 FROZEN receipts had classified
  SWEEPABLE — the plan-killing class), glob/brace cite parsing,
  apply never clobbers and skips already-archived rows.
  docs/hygiene-plan-2026-08-11.md carries the plan.

## LIVE RIGHT NOW

- MERGE-SPACE-1 (R1), 3080, detached (logs/merge_space1.DONE fires
  on success): parents s1=12/120, s2=23, s3=30 gated; s4 + fork pair
  + 9 merges + gates remain. NOTE the parent spread 12-30 at
  identical recipe — d64 birth-seed variance is HUGE (bank-worthy on
  its own; crown ternary line's 64/62/61 was mild by comparison).
  Bar 1 reads against min(parents): crater must go < 5.
  Driver fences booked in MORNING-SWEEP-0811 item 6 (fork base is a
  differently-scheduled twin; batch order shared across seeds; bar 2
  sub-sigma at n=1 books unresolved-consistent-with).
- FLOOR-HK-1 (R3), Mac rjob floor_hk1: d64 floor 0.4364, d128
  0.3815 (both above H_16 0.367, descending — bar 1 holding so
  far), d512 in flight (~28 min/epoch). Then a fresh d256 cell as
  its own run (amendment). GATES for the ladder run as a separate
  post-birth step (driver has no pipefail — check every birth log
  ends "saved" before trusting a floor; fences booked).
- 3080 window: open until ~17:00 EST 08-11; after that HOLD for
  Artin GO.

## QUEUE (order agreed with Artin)

1. Book R1 + R3 verdicts as receipts land (book.py; gates rows into
   the lake with device + n_seeds).
2. lab/gate.py adoption (extraction-spec item 5 + grok convergence):
   verbatim sample_wave_lp + gate_eval from step_grpo_micro (91-ref
   hub) + GateSpec with per-lineage constants; frozen shells
   untouched; source-identity guard. Plus lab/hash.py (three digest
   semantics -> one) and lab/jsonl.py (40+ hand-rolled sites);
   marker-harvest into lake build_runs.
3. R2/R4-R8 pre-regs + queue (reviewer rung designs, RESULTS-swept):
   R2 shared-init merge ladder, R4 curriculum-funnel micro, R5
   graft-v-grow (the crown decomposition), R6 uncited-checkpoint
   revival (long, Mac background, START LAST), R7 fingerprint dedup,
   R8 exact-mode gate v rounded gate (DECISION-FLIP design, the one
   named precision retest slot).
4. README headline refresh + THEORY rows (growth row gains the n=3
   crown leg; new loss-floor row Shannon-cited; width-buys-tolerance
   home) + remaining BOARD line items — at the next freeze point.
5. Corpus-manifest overlay (grok D): manifests POINT at existing
   paths, zero moves, evidence stays in place.
6. MECHANIZE-THE-RITUAL block (opus-chat cross-check, banked in
   RIFF 2026-08-11): .github/workflows CI (pytest + INDEX/CODEMAP/
   results-index staleness regen checks), doctrine-as-lint tests
   (SIGALRM-near-sympy, non-string random.Random seeds),
   GENERATED/HAND doc headers, annotated verdict tags, rjob
   refuses launch without prereg_ref. All additive, zero history
   risk. Rejected in the same bank: scratch git-mv tiering,
   generated handoffs, fewer-writer subagent posture.

## STANDING FENCES THAT TRAVEL

- Cross-device gates never compare; the lake's gates schema enforces
  device grouping at write time.
- 0.348 is a d512-grown REFERENCE LINE, never a ladder point.
- MERGE-SPACE bar 2 and any |delta|<=2 read: unresolved at n=1.
- 33.1 GB uncited checkpoint pool: enumerate/gate via R6 only, no
  deletion without Artin GO.
- wsl.sh verb split is NOT a security thread (see 2026-08-11-0's
  note; do not re-escalate).

# Handoff 2026-08-19-1: the tower day — abort to equivalent runtime to relaunched screen in one session; FREEGEN-2 LIVE on the 3080

Seat: Fable (main session model), Mac. HEAD at close: see closing
commit (this file). 3080: BLE-FREEGEN-2 screen RUNNING (~33/60 rows,
in the xhigh cell); Mac idle.

## What landed (all committed + pushed, in order)

- VERDICT QWEN-RESIDUAL-STRUCTURE-0 (L35486, code_commit 2e0b0ba):
  NOT-REFUTED, structure LOCALIZED — level-1 cond-mean decode dead
  (max 0.086% over 402 tensors), global table dead (LOO cosine
  ~0.0015; the registered 0.1222 was self-inclusion), only
  structure = early attn write-back (o_proj L3 low-rank 9.4%,
  out_proj L0 tail 21%). Receipt-auditor blocker fixed pre-booking
  (receipts.lock had hashed the vendor sidecar MID-WRITE); vendor
  shards verified v upstream LFS oids at 1d4bf0f2.
- OBSERVATION QWEN-MODEL2-FOURTH-PICK + AMENDMENT -SIGN: post-hoc
  marginals from booked receipts; sign convention corrected (mid|PK
  improves BOTH -> C dominates PK; PX v C is the Pareto trade).
- AMENDMENT QWEN-RESIDUAL-STRUCTURE-0-EXECUTION: wsl->mac
  deviation, 12-tensor sample governs, bar-5 relabeled
  self-inclusive.
- OBSERVATION QWEN-BLE-FREEGEN-1-ABORT: operational abort, bars
  UNADJUDICATED (OOM clause NOT stretched); mechanism verified at
  source — rung4 routes only w4 to fused, BLe's 48 s16 tensors ran
  as 6.875 GiB dense FP32, 0.56 tok/s (row receipt preserved).
- THE TOWER: llmopt/lab/qcuda_tower.py (FusedS16Linear, s16 decode
  kernel, EXACT-CONSERVATION verify_routes with dedicated_routes,
  plan-before-build residency) + qualify a-c r2 (synthetic e=0/
  e=200 edges bit-exact, real GEMV <=4.4e-7, prefill 2-chunk) +
  ladder d-g ALL PASS -> OBSERVATION QWEN-TOWER-EQUIVALENCE-0
  (e parity 14.1x/7.3x inside tolerances FROZEN at d; f cached/
  uncached + old-new token2 identical; g 7.11 tok/s) +
  memory-growth qualified (bounded 7.35->7.72 GiB to 3072 tokens,
  free floor 0.65 GiB, ~9.9 tok/s sustained).
- PRE-REG QWEN-BLE-FREEGEN-2 (168a5e1, committed before launch):
  same bars/prior as FREEGEN-1, new runtime named, fresh output
  path logs/qweneffort2/. Row-gate adjudicator
  scratch/qwen_ble2_adjudicate.py + fixtures committed BEFORE
  receipts; llmopt/lab/provenance.py (start-state capture) landed.
- Lab adoption: llmopt/lab/qscore.py (scorer math, source-identity
  guarded; teacher_receipt_block resolves the sha rename).
- RIFF banks: UNSLOTH-MATERIALIZATION (+ roofline: 103 tok/s
  one-pass ceiling, 2k needs amortization), CHEAP-READOUT-CENSUS
  (corrected: NOT zero-run, arm-state shape, three-object reading,
  two claim levels), TRAFFIC-BUDGETED-ARCH (Pareto framing),
  W4-GROUP-DOT, OPERAND-PROVIDER-LAW (two same-day catches;
  prereg-schema residue).

## IN FLIGHT at close (the one live thing)

BLE-FREEGEN-2 on the 3080: launched at 168a5e1,
logs/qweneffort2/{ble2_screen.log, tower_rows_BLe.jsonl},
marker logs/qweneffort2/BLE2_SCREEN.DONE, ETA ~1-2 h from close.
On completion: pull rows, run scratch/qwen_ble2_adjudicate.py
(fail-closed row gate, counts recomputed from rows — NEVER the
summary), bars: termination FIRE iff xhigh term >= 1, competence
FIRE iff total correct >= 1. Auditor pair, then book. WORDING LAW:
a minimal FIRE reads "escaped B's total collapse", never "fixed
the deliberation loop"; 1/60 != deployment-ready. RESUME LAW: if
the process died, do NOT blindly resume (row-mixing across code
changes); close operationally and relaunch fresh. Dependency blobs
at 168a5e1 pinned in scratchpad + this session's bookings;
3080 tree verified clean at 168a5e1 during the run.

## Conditions that bite next session

- FREEGEN-2 booking discloses: screen driver's start metadata is
  commit+interpreter+one-hash only (regressed from ladder's full
  capture; deps verified at 168a5e1 instead); use
  llmopt/lab/provenance.py in all future drivers.
- Banked forward fixes: per-step art_dir/manifest_sha256 in ladder
  receipts; top1_identical_all_positions receipt key rename;
  refuse-if-exists for rows/npz in the residual census producer;
  OPERAND-PROVIDER prereg-schema field; RESIDUAL census producer
  start-commit capture.
- Phase-2 runtime work UNLOCKED by equivalence: kernel geometry
  sweep, W4-GROUP-DOT, decode-ahead double-buffering, placement-
  aware planner. All behind FREEGEN-2's booking.
- Census queue (Artin ordering): CHEAP-READOUT (arm-state shape,
  level-2 thresholds freeze BEFORE h_A) > ATTN-ROUTER-CENSUS >
  RESIDUAL targeted-patch follow-up. TRAFFIC-BUDGETED-ARCH is the
  long arc.
- tables_A.npz stays untracked, pinned by sha in loo_A.json.

## Next session opens with

1. This handoff -> BOARD -> RESULTS tail (5 new entries today
   after MODEL-2).
2. If BLE2_SCREEN.DONE exists: adjudicate -> auditors -> book.
3. Artin decisions: census ordering confirmation; phase-2 runtime
   slot; 3080 artifact garden cleanup still Artin-GO.

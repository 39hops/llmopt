# Next-session spec: MODEL-1 scorer + tree projection -> A v B v C

The single opening move. The next interesting commit contains
X_A, X_B, X_C, K_A, K_B, K_C — not another runtime feature
(unanimous across seats, 2026-08-17 close).

## Standing inputs (all locked, verify shas before use)

- Teacher records: logs/qwenteacher_v2/ — manifest code_commit
  0ca4151, revision 1d4bf0f2, records sha'd in manifest; sidecar
  gate PASSED (VERDICT QWEN-TEACHER-0-LOCK); margin census booked
  (frozen edges; small-n fence LIVE on scored streams).
- Tree registration: PRE-REG QWEN-MODEL1-TREE + amendments
  (-LOGIC, -METRIC, -PRIORS, -PINS, -KFENCE). Refinement CLOSED.
- Artifacts: A/B/C on the 3080 at ~/qwen_whole0t/, digest chains
  in logs/qwenwhole/. Arm scoring runs on the MAC CPU REFERENCE
  ONLY (device rule; scorer refuses device != cpu).
- CPU reference decode path: llmopt.lab.qcodec / qcodec_fast
  (+ runtime0r pattern for the tower). Adopted CUDA primitives in
  llmopt/lab/qcuda.py serve the 3080 leg only — backend-agreement
  KL, never tree quantities.

## Build order

1. `docs/preregs/qwen-model1-tree.json` — mechanical projection of
   the registered tree (triggers, anomaly gate, cumulative guard,
   D/E branch, uniform-damage alarm, refuse-list). The adjudicator
   (scripts/adjudicate pattern) must fire the tree from receipts
   with zero session discretion.
2. Scorer (scratch/, then /qualify): X = CE_arm - CE_teacher on
   corpus positions; K = forward KL(teacher||arm) over live vocab
   on prefixes; P-1 alignment, position-pooled. fp16 record
   sensitivity floors f_X, f_K by the registered +-1ulp
   perturbation procedure. REFUSES device != cpu; consumes
   manifests only through qartifact.
3. Margin-stratified flip rates: enforce n >= 30 PER REPORTED
   STRATUM (stream x category x margin-bin), never the pooled
   census vector (GPT catch, 2026-08-17: every prefix bin is
   under 30 — no directional prefix-only margin claim exists;
   print raw counts). Census receipt carries both vectors.
4. Score A -> B -> C on the Mac (one artifact resident at a time;
   pull artifacts from the 3080 or score against local copies —
   plan disk first, ~22GB total). Book each arm's numbers as they
   land (stream partial results).
5. Tree adjudicates from the receipts. Book the verdict with both
   auditors (prereg-auditor + receipt-auditor) BEFORE the booking
   commit.

## Fences that ride

- Chat reads never gate; the A/B qualitative contrast (paired at
  6982ab3) is color until the tree speaks.
- Cross-device comparisons forbidden; 3080 CUDA legs report
  backend_agreement_kl_vs_cpu_ref only.
- No qrope claim for teacher v2d (carried verbatim in the lock
  verdict).
- Sidecar-v2 cleanup (SIDECAR_STEPS=4, 48/16 census in receipt,
  raw-fp32 comparison, revision assert) is REQUIRED before any
  rollout-based/free-generation evidence is used — it does NOT
  block the corpus/prefix X/K core.

## Parked (explicitly NOT before the scorer)

- Q8/GGUF external baseline: mlx-lm and llama.cpp NOW CARRY
  qwen3_5/QWEN35 arch support (GPT, 2026-08-17 — corrects the
  session's "probably unsupported" guess). First step when taken:
  a cheap conversion/load smoke of the pinned checkpoint, NOT a
  compile-night. Any Q8 arm is a SEPARATELY preregistered
  comparison scored by the same scorer — it gains no branch in
  the frozen tree. Compactness figure then uses actual file
  sizes, not nominal bpw.
- 2-bit-router kill-test (RIFF-LEDGER, corrected 2026-08-17):
  E_k reconstruction variant (GPT): compare A-selected channel
  set's reconstruction of the teacher FFN output v oracle and
  random selection.
- arena_qwen: refuse (not warn) on commit/prompt/settings
  mismatch before calling a read "paired".
- PHASE-PORTRAIT-2 / STABILITY-ATLAS-1 (banked, desk-ready).
- C on CUDA: fused-residency arithmetic books before any attempt.

## Open decisions for Artin (carried)

1. README front-door Qwen paragraph.
2. Overnight 3080 allocation.
3. MTP exclusion confirm.
4. Q8 baseline: external conversion smoke v house-compiled arm
   (external now looks cheaper — arch support exists).

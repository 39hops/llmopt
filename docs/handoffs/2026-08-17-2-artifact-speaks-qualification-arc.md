# Handoff 2026-08-17-2: the artifact executes, the qualification arc, and the honest frame

Seat: Fable (main session model), Mac. HEAD at close: see BOARD.
3080: WHOLE-0T artifacts resident at ~/qwen_whole0t/{A,B,C}; Artin
granted daytime use repeatedly this session.

## THE FRAME (Opus seat, adopted verbatim as the arc's meaning)

The hardening arc below bought TRUSTWORTHINESS, not EVIDENCE.
Every qualification check is weight-space; all of them pass on an
artifact that produces fluent garbage. MODEL-1 is the only
function-space instrument in the program, and the compression
question is exactly as open as it was 24 hours ago — blocked on
teacher + scorer, not on ideas. Three states, always: the code
does X / a test enforces X / a document claims X. Only the middle
one is green.

## What landed (chronological, commit shas in git log today)

- VERDICT STREAMWD-V2-MAC-GATE (7b00890): same-device gate PASSES,
  worst arm 5.7e-4 v 5e-3; v2 promoted (later narrowed by
  -PROTOCOL: loader lineage only, no shared codec module existed).
- VERDICT QWEN-WHOLE-0T (6bcef7d): ALL THREE BARS FIRE — A 6.50 /
  B 7.09 / C 8.77 GiB, conservation 0 over 1199 keys, per-family
  fidelity 0.97-1.00x probe. Closed adjudication path
  (obs_from_receipt_0t.py -> adjudicate.py), both auditors clean,
  four provenance limitations disclosed in the verdict.
- AMENDMENT QWEN-TEACHER-0-ROPE (287dbf0): v1 teacher INVALIDATED
  pre-lock — meta-device build zero-filled RoPE inv_freq (caught by
  Artin's review seat); 2-layer smoke was blind BY CONSTRUCTION
  (layers 0-1 are linear-attention). v2: init_empty_weights
  (buffers real) + fail-closed meta/zero guards. Outputs
  quarantined at logs/quarantine/.
- AMENDMENT QWEN-WHOLE-0T-PROTOCOL: five corrections, verdict
  numbers stand (shared-codec downgrade, "reproduces probe
  distortion" wording, resume partial-manifest invariant, source
  sha = prereg transaction-step incident, artifact transfer digest
  rule).
- AMENDMENT QWEN-TEACHER-0-TRAVERSAL (879461e): lock refuses
  without 64/48/16 executed census + rope; smoke's own manifest
  now shows full_attn=0 in-band.
- PRE-REG QWEN-MODEL1-TREE (b7fb790) + amendments -LOGIC (8b8a2cc:
  cumulative A-v-C guard, anomaly gate, T1 prior corrected to
  semantic-role grounds — the family receipt REFUTES "io least
  W4-friendly"), -METRIC (b8b9014: forward KL live-vocab P-1
  alignment, 5x fp16 floor, uniform-damage alarm, CPU-only tree
  quantities, refuse-list + codec round-trip fixture), -PRIORS
  (ac5809b: D>E historical prediction from STAR-PROFILE registered
  prospectively, T1 weakened to weak/medium, fidelity wording law,
  margin-stratified flip diagnostic, cached-rollout sidecar gate,
  executable check code_commit=0ca4151), -PINS (1d357a1: rec(Y)
  recovery fraction, frozen margin bins 0.02-edge n>=30, sidecar
  tolerance = max per-position relative L2 <= 5e-3, token equality
  primary), -KFENCE (ee4b944: symmetric K floor, raw unclamped
  recovery; PREREG REFINEMENT CLOSED).
- Teacher v2b/c/d saga: v2b use_cache=False rollout = 750s/step
  (2-day wall) -> KV-cached generate (a1190b2); v2c killed pre-lock
  for receipt hardening (0ca4151: batch context, input digests,
  executed census, cat/id fix). v2d RUNNING at close (watcher on
  jobs/teacher0v2d.rc; manifest must carry code_commit 0ca4151).
- Digest receipts: canonical A/B/C chains committed
  (logs/qwenwhole/artifact_digest_{A,B,C}.txt), top hashes
  independently reproduced by the GPT seat.
- QUALIFICATION LAYER (the day's engineering): llmopt/lab/qcodec
  (canonical decoders, golden fixtures incl. compiler-produced
  payload bytes), llmopt/lab/qartifact (library — consumers cannot
  see an unqualified manifest; rung 0 digest identity fail-closed,
  sha-pinned vendor index, exact cover, fail-closed memory
  preflight with shared cost model), scratch/qwen_qualify.py thin
  CLI, /qualify skill wired into /rung step 3.5, ~25 regression
  tests incl. the import-proof and two producer-consumer
  meta-tests. Engineering law also in project memory.
- Receipt lock repair: 21 locked-but-untracked receipts found by
  the three-model review — 15 force-added, 6 large traj streams
  classed local_only (lock records tracked; the mutation invariant
  READS the class; LLMOPT_FULL=1 = absence-is-failure on the
  evidence host).
- OBSERVATION QWEN-RUNTIME-0R-FP16-RETRACTION (afb0625): the
  round-trip oracle killed "bit-lossless fp16 residency" on real
  weights — 569,841/1.27B embed entries in the subnormal tail.
  The arc's strongest evidence: representability arguments are not
  oracles.
- RUNTIME LADDER CLIMBED on artifact A (WSL CPU): forward1 PASSES
  (122s, traversal 64/48/16 EXECUTED census, rope value oracle,
  top5 = 'The'/'Thinking'/'User'/'Let'/'用户' — sane), two cached
  tokens = "The user" (thinking-trace opening, 121.5s/tok).
  Receipts logs/qwenruntime/. Peek OOM (13GB host, 9.5GiB fp32 io)
  correctly classified runtime-instrument, no codec evidence.
- Riffs: rotation-preconditioning bank + CORRECTION (algebraic-not
  -bit, eligible interfaces, training-free-not-runtime-free, V4
  causal narrowing); RESIDENT-DRAFT/STREAMED-VERIFY bank +
  CORRECTION (wall can lose; book acceptance + sweeps/token + wall
  separately); identity-beats-aggregates meta-law bank.
  RIFF-LEDGER table repaired (39 blank lines split the GFM table).

## IN FLIGHT at close

- Teacher v2d (Mac): corpus+prefix records written, rollout
  running. Watcher bxvo3d9zz on jobs/teacher0v2d.rc. On lock:
  verify manifest code_commit == 0ca4151, run the SIDECAR
  (cached-v-uncached, full tower, registered tolerance) before the
  rollout record is accepted, then the margin-bin census.
- 32-token demo on artifact A (3080, logs/qwenruntime/gen32_A.log,
  ~65 min). On landing: book OBSERVATION QWEN-RUNTIME-0R-SMOKE
  (descriptive, no MODEL-1 claim) with the verbatim transcript.

## Next session (in order)

1. Teacher lock -> sidecar gate -> margin census -> book teacher.
2. MODEL-1 scorer per the registered refuse-list + adapters
   (obs_from_score_model1.py); docs/preregs/qwen-model1-tree.json
   projection (does not exist yet — the tree fires mechanically
   only once it does).
3. Score A v B v C on the Mac CPU reference; the tree adjudicates.
4. Runtime legs (Metal W4 / CUDA W4) only after the curve.

## Open decisions for Artin

1. README front-door paragraph for the Qwen program (three-model
   review item; drafted next session unless vetoed).
2. 3080 nightly window tonight: teacher-class or idle.
3. MTP module stays excluded from scope — confirm or widen.

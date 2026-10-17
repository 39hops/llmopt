# Handoff 2026-10-17-1: greedy speculation for Qwen3.8-27B, M0 (mathematics) and E0 (arm C census) done; verdict booked; nothing pushed since a62d2d97

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-17-0. Subagents on
Opus 5.5 from this session on (Artin's instruction). No frozen evidence
altered; no model downloaded; no training; no push after a62d2d97.

## What exists now

- Storage round closed in the morning (handoff 2026-10-17-0): pushed to
  a62d2d97, 3080 fast-forwarded and qualified there (2306 passed, 61
  skipped, SOURCE TREE GREEN). The 3080's C: drive has about 25 GB free;
  its WSL virtual disk is 266 GB and holds 130 GB of WHOLE-0T compiler
  arms under ~/qwen_whole0t, 17 GB of shards, 15 GB of cache and the
  8 GB GRPO archive. Any reclaim there is a separate decision.
- M0 (commit 846f8711 and after): scratch/leancheck/SpecDecoding.lean,
  45 theorems, no sorry, standard axioms (greedy speculative decoding
  exact for any proposer; sup-norm margin theorem with shift-aligned
  form and m/2 tightness; tail-sum, geometric form as labelled model,
  break-even; acceptance equals argmax agreement; candidate-set rule;
  fold-snapshot rollback; gated-delta non-invertibility and
  Sherman-Morrison). llmopt/lab/spec_reference.py with 41 tests
  (greedy and sampled laws apart, replay with full cost accounting,
  shift-aligned error, fp16 ambiguity, position_stats).
  tests/test_lean_spec_decoding.py compiles the Lean file when the
  toolchain and cache exist (the mathlib cache was restored from
  ~/.cache/mathlib in 12 s; scratch/leancheck/.lake is 7.7 GB again).
  TreeWY (arXiv:2608.20961) reviewed in spec 2026-10-17-e0-spec-census.md
  section 3.
- E0: arm C (8.8 GB) and the five vendor side files copied from the
  3080 to ~/qwen_whole0t/C and ~/qwen_vendor on the Mac, 19/19 and 5/5
  digests verified against the frozen compile chain and the teacher
  manifest's index pin. PRE-REG QWEN-E0-SPEC-CENSUS-0 (RESULTS L79012,
  JSON docs/preregs/qwen-e0-spec-census-0.json) frozen before the smoke
  was read. Smoke and preflight (padded batch versus unpadded single on
  arm C: argmax identical, max rel L2 2.5e-5; prompt reconstruction
  exact 24/24; eos accounting) passed. Full run under liverun at
  b66cfac7, 1,812 s, peak RSS 18.71 GB. VERDICT QWEN-E0-SPEC-CENSUS-0
  (L79069): bar 1 agreement 0.8291 FIRE, bar 2 replay k=4 3.4695 FIRE
  (3.4267 per stream), bar 3 certified 0.2142 NO-FIRE, NOT-REFUTED.
  Opus pre-reg audit: no blocker; its four should-fix points are in the
  verdict text. Receipts under logs/qwen_e0/C (positions.jsonl 5.6 MB
  untracked), logs/rehab-2e/qwen_e0_*.

## Reading

Arm C is a credible greedy drafter at the acceptance level on real
teacher-greedy contexts, and the agreement is mostly empirical: the
2-bit tower perturbs logits by about 6 nats, so the margin theorem
certifies only teacher margins above about 9 nats (all 1,006 certified
positions sit in the top bin). Verification cannot be skipped for any
position class. Nothing in E0 measures throughput.

## Next (each needs its own GO)

1. Verifier-timing rung on the Mac: c_t and c_v(k) for the streamed
   bf16 teacher, c_d for arm C on CPU, then the break-even read with
   the E0 replay yields. This needs the vendor bf16 shards (about 56 GB,
   deleted from the Mac on 2026-10-16); budget against 46 GB free.
2. MTP self-speculation (RIFF L7532) stays banked until 1 lands.
3. Push authorization for the commits after a62d2d97, then ff the 3080.

## Invariants to keep

Greedy and sampled acceptance laws are different quantities; the
recorded teacher token is authoritative and fp16-ambiguous positions
are never certified; the geometric form is a model, the replay is the
measurement; accepted/drafted is not an estimate of agreement.

## Addendum (same day, after E0 acceptance)

- OBSERVATION QWEN-E0-SPEC-CENSUS-0-PER-STREAM: per-stream replay
  3.4267 at k = 4 (k = 1..8 recorded), the deployment-relevant
  companion; the registered concatenated 3.4695 stands.
- OBSERVATION MAC-STORAGE-RECLAIM-0: 12.2 GB of verified caches and
  temp outputs removed (manifest logs/rehab-2e/mac_cache_reclaim_
  2026-10-17.txt); free 46.3 to 58.5 GB. The vendor restore condition
  (76 GB plus temporary allocation) is not met; nothing downloaded.
  Large non-lab trees (servo-evidence 75 GB, CloudStorage 23 GB,
  Application Support 39 GB, a 5 GB data zip in Downloads, docker
  4.8 GB, uv cache 5 GB with a live consumer) were left for Artin.
- OBSERVATION QWEN-DRAFT-DECODE-COST-0: one-layer probe of arm C on
  Mac CPU: 0.91 s decode per linear layer, 0.81 s per full layer,
  35 ms of GEMV; about 57 s per draft token if decoded per token.
  Break-even at k = 4 would need c_d under about 3 to 6 s. Exact
  speculation is not faster than streamed decoding on this Mac with
  the current draft path, regardless of teacher timing. Gating item:
  a fused W4/S16 decode-and-multiply kernel (RIFF L6700, banked).
- Timing rung prepared, not registered: spec 2026-10-17-timing-rung.md
  (quantities, break-even law with per-stream yields, three bars, the
  vendor-restore preconditions with the hub's per-shard LFS sha256).

## Addendum 2 (evening): fused Metal kernel and the arm C Metal tower

Artin's GO (2026-10-17 18:08) for the fused W4/S16 decode-and-multiply
kernel. Delivered, all local (commits 3e7f457f, 295be340, c1aa5fb9 and
the booking commit after them; nothing pushed after 90fa3347):
- llmopt/lab/qmetal.py: three kernel generations, v3 default; frozen
  formats untouched; no decoded matrix; subnormal and overflow handling
  by split accumulators; tests/test_qmetal.py 108 parity cases against
  llmopt.lab.qcodec; scripts/bench_qmetal.py with receipts under
  logs/qmetal/bench_90fa3347*.json.
- scratch/qwen_metal_tower.py: mlx-lm Qwen3.5 architecture with fused
  linears, raw tensors decoded once with the sanitizer's conventions,
  exact HF l2norm for DeltaNet q/k (the one library deviation found,
  2e-2 on logits before, 5e-5 after; tests/test_qwen_metal_tower_contract.py
  pins the two lines overridden). scratch/qmetal_oracle_dump.py dumps the
  CPU oracle's logits and per-layer hidden states (untracked npz under
  logs/qmetal/, 70 and 164 MB; regenerate in about a minute).
- Numbers: decode 6.44 tok/s steady (155 ms, p90 157 ms), prefill 8
  tok/s, peak unified 10.3 GB, RSS 11.7 GB; per-layer parity 6e-7 to
  2e-5 over 64 layers; CPU census path was 57 s per token.
- Booked: OBSERVATION QWEN-METAL-TOWER-0 (RESULTS L79270); RIFF
  W4-GROUP-DOT marked measured.

Open: a real multi-token kernel for prefill (the tile path saves little);
the verifier-timing rung still needs the vendor bf16 shards and the
storage decision; no verifier, no MTP, no recompilation done.

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

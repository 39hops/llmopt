# Handoff 2026-10-19-0: hybrid prefill hardened as the entry point's default; E0 disagreement census; SwiGLU permutation in Lean; the 9/4 result reviewed

Session: Fable 5.1 on the Mac. Reviewers on Opus 5.5.
Previous handoff: 2026-10-17-2 (addenda through the GEMM prefill booking). Read BOARD first.

## What happened

1. Phase A (OBSERVATION QWEN-METAL-SERVE-0). qmetal.gemm_scope, a
   context variable, gates the decode-once GEMM dispatch; the only
   caller is llmopt.lab.qprefill.prefill_chunked(matmul="gemm"), per
   chunk, around the body and head calls; on_chunk callbacks, decode
   steps and batched calls dispatch by the process default (fused).
   matmul="fused" is the control. ServingConfig and
   configure_serving_memory apply the 2 GiB allocator-cache limit
   explicitly and refuse a process whose qmetal default has the path on.
   memory_report separates active, peak active, pool, footprint and RSS.
   Tests for every dispatch-isolation case the Opus review asked for.
   The study drivers keep the research switch QMETAL_PREFILL_GEMM=1 and
   the _Resident.prefill_gemm attribute; outside a scope that default
   decides, and it is off.
   Measured through the entry point (scratch/qwen_metal_prefill_serve_ab.py,
   chain 4 at d42cb215): fused 8.37, GEMM 35.46 (2 GiB limit) and 38.18
   (no limit) tokens per second at 1,024 tokens; decode 134.6 to 135.5
   ms in all arms; all argmaxes identical; footprint 11.34 against 14.27
   GB; E0 recount 4,722 of 4,722 equal to the CPU rows, 0.8291 teacher
   agreement; full draft logits and hidden states of the 4,722 positions
   kept machine-local (logs/qmetal/prefill_serve/dump, 3.1 GB, shas in
   census_d42cb215.json).

2. Phase B (OBSERVATION QWEN-E0-DISAGREEMENT-CENSUS-0). Read-only census
   of the 807 disagreements: near-ties (teacher margin median 0.91
   nats), the draft picks the teacher's second choice in 450, 37 above a
   5-nat teacher margin; a global flattening (slope 0.88, top-1 logit
   down 2.9 nats after shift alignment); families follow the teacher's
   margin profile. One deterministic exception: the first generated
   position, "We" (teacher, 23 of 24) against "The" (draft, 21 of 24).
   Vendor inventory without download: lm_head.weight bfloat16
   [248320, 5120] (2,542,796,800 bytes) in model-00018-of-00018
   (3,392,197,344 bytes, otherwise the MTP block), embeddings alone in
   shard 3; storage budgets for head restore (+1.77 GiB), 8-bit head
   (+0.59 GiB), low-rank residual (0.5 MiB per rank), MLP precision
   increases (+8 to +24 GiB, out of budget). Shortlist with falsifiers
   and independent validation in the entry; nothing presumed.

3. Phase C (OBSERVATION SWIGLU-PERMUTATION-0). scratch/leancheck/
   MlpPermutation.lean proves gatedMlp act (P Wg) (P Wu) (Wd Pᵀ) =
   gatedMlp act Wg Wu Wd over any commutative semiring, any gate, with
   P = σ.toPEquiv.toMatrix; SiLU instance over ℝ; no sorry; axioms
   propext, Classical.choice, Quot.sound; tests/test_lean_mlp_permutation.py
   compiles it when the toolchain is present. Imports narrowed from
   `import Mathlib` because a CStarMatrix HMul instance shadowed the
   Matrix one. Mechanical study (scratch/mlp_permutation_study.py):
   exact in rationals, float32 at 1.7e-6 of the output scale, and the
   grouping lever is empty (details in the entry; the decoded arm C
   window is grid-biased toward the identity and is read sorted against
   random only).

4. Phase D (OBSERVATION MATMUL-NINE-FOURTHS-RELEVANCE-0). The openai/math
   preprint (October 2, 2026) and its Lean challenge statement read:
   asymptotic exponent over C in the straight-line arithmetic model,
   proof by Schauder-Tychonoff existence of detecting characters, no
   finite size, decomposition or constant, no finite-precision
   discussion; the catalogue marks it partial in scope and unchecked in
   review, and this round did not compile the library. Strassen-Winograd
   against mx.matmul (scratch/strassen_bench.py): loses 1.7x to 3.3x at
   256 rows, within spread at 1,024 and 4,096 rows with 1.9x the error.

## Process notes

- Chain 1 (45874e87) ran all seven stages; an Opus code review of the
  first commit landed during it with real findings (on_chunk ran inside
  the scope; the scope stack was not thread-local; the permutation study
  mislabelled its codec; dumps were overwritable; SMOKE OUT override).
  All fixed at 39a6115c; chain 1's JSON receipts deleted as superseded
  (none cited), its log kept as study_chain1.log. Chain 2 (39a6115c)
  stopped at the disagreement stage on a refusal that was too strict (an
  fp16 dump row flipping a float32 near-tie with gap 0.0057 nats); fixed
  at 5b34a026 to record such rows, runner took a CENSUS_RECEIPT override
  at 8b24de05, chain 3 resumed from that stage. The receipt and wording
  audits (Opus) on chain 2 and 3's receipts and the four drafts found the
  E0 rank field read as 1-based (it is 0-based), a teacher sha check that
  hashed the file instead of the array, the serving config recorded as
  defaults rather than as applied, and the entry point's own default
  still None; all fixed at 835d554d, and the whole chain ran again as
  chain 4 at d42cb215, whose receipts are the booked ones. Chains 1 to 3
  keep their logs only (study_chain1.log, study_chain2_3.log). Every
  receipt names its commit; the four liverun receipts are under
  logs/liverun/.
- 3080 mirror: pending from e367f1ac (ssh timed out through the day);
  not blocked on.

## Next (Artin's call)

- First-position diagnosis: teacher-forced logits at the "<think>\n"
  position under varied system prompts and lengths on Metal (minutes;
  no weights); decides template-conditioned damage against opener bias.
- Head-only restoration test: one download (2.54 GB tensor by range
  read or the 3.39 GB shard), applied to the stored hidden states,
  prediction and falsifier registered before the download; validation
  on the corpus and prefix records, never E0 as recovery evidence.
- Not next: whole-model recompression, retraining, the 9/4 line.

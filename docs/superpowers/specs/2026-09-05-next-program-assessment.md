# Next-program assessment (2026-09-05): data -> weights -> function on real open models

Status: ASSESSMENT ONLY. Nothing preregistered, nothing launched. Booked as
OBSERVATION NEXT-PROGRAM-ASSESSMENT-2026-09-05-0 in RESULTS.

Scope of the ask (Artin, 2026-09-05): rank the smallest high-information
programs that use real current open models to discover interventions that
improve measurable capability while tracing DATA -> WEIGHTS -> FUNCTION;
verify K2 Horizon's artifacts before assuming them; rank three adjacent
branches (MoE transfer, latent/cross-model transfer, verified math as a
model lab); do a front-surface housekeeping audit. Contamination law:
public benchmarks score, never train.

Artifact-availability rows marked CONFIRMED / DEAD / 404 / 401 with
"session" in the evidence column were checked by the session against the
live endpoint on 2026-09-05 (HF refs / tree API, GitHub API, arXiv export
API). All other rows are survey-agent (Opus) drafts, not verified here.

## 1. K2 Horizon: what is downloadable today

Family: IFM (formerly LLM360) K2 Horizon, released 2026-09-03, Apache-2.0,
sizes 0.9B / 3.7B / 7B / 32B / MoVA-36B-A4B / 375B-A23B, custom arch
`K2HorizonForCausalLM` (trust_remote_code), vocab 64,256 at 0.9B and
250,624 at 3.7B+.

| Artifact | Status | Evidence |
|---|---|---|
| Final BF16 weights, all six sizes | CONFIRMED | HF tree: 0.9B 2.16 GB, 3.7B 10.14 GB, 7B 18.02 GB, 32B 69.6 GB |
| Intermediate checkpoints as git tags, full safetensors at each tag | CONFIRMED | 0.9B 34 tags (session listed them: pretrain_500000..600000, mid_1_50000..75000, mid_2_5000..47684, rl-mopd_249, *_final); 3.7B 65; 7B 68; 32B 33; 3.7B@pretrain_100000 tree returns 200 |
| Stage boundaries documented | PARTIAL | APPENDIX.md exists for 0.9B and 3.7B only: pretrain 8K ctx -> mid_1 32K -> mid_2 128K -> mid_3/4 512K -> sft -> rl; 0.9B ends in mOPD distillation from a teacher |
| Per-stage token budgets | NOT FOUND | blog aggregates only (~20T total; "3.7B, 7B, 32B, 36B trained on the same 22T tokens"; ~17 % reasoning trajectories) |
| Training data | CONFIRMED but UNMAPPED | five ungated dataset repos, ~21.6 TB (TxT360-v2, Pretrain-Behaviors, Math-Reasoning, Code-Reasoning, SFT-Reasoning); no mixture proportions, no stage-to-subset map |
| Dataset ids named in every model card | DEAD | `IFM/K2-Horizon-Pretrain-Data` and `-Midtrain-Data` return 401 (session checked) |
| Training code | PROMISED | `LLM360/xllm` at every cited commit returns 404 (session checked); `ifm-ai/xllm`, `ifm-ai/horizon-post-train` are README stubs |
| Training logs / eval-at-checkpoint | NOT AVAILABLE | W&B runs named in the 3.7B appendix are private; only rendered loss PNGs |
| Tokenizer | CONFIRMED | ships in every repo |

Reading: K2 Horizon is an excellent CHECKPOINT-TRAJECTORY dataset and a
poor reproducibility dataset. The weights/tags leg is real and verified;
the data leg is posted but unlabelled; code and logs are absent. Any
program that needs the mixture recipe cannot use K2 Horizon today.

Alternatives checked the same way:
- OLMo 3 (allenai): 7B with 1,487 checkpoint branches, 32B with 719; full
  Dolma 3 corpus AND the mixture (`dolma3_mix-6T`, `dolmino_mix-100B`) are
  public; OLMo-core + olmo-cookbook active. No sub-7B member.
- Pythia: 155 step branches per size, exact pre-shuffled token order as a
  dataset, sizes 14M to 12B. 2023 data, weak on math.
- SmolLM3-3B: 133 checkpoints, recipe yamls only, no corpus.
- Marin, Apertus: partial checkpoints; mixtures live in code.

Smallest family with checkpoints + data + mixture across ~1B-7B: none.
K2 Horizon has checkpoints + data (no mixture); OLMo 3 has all three but
starts at 7B; Pythia has all three but is a 2023 Pile model.

## 2. Hardware and the lab's actual training ceiling

Mac: M3 Pro, 36 GB unified, 51 GB free disk at assessment time (streaming
per-shard download-process-delete is mandatory for anything above ~13 GB,
the pattern booked for Qwen3-30B-A3B at RESULTS L11024). Windows box: RTX
3080 10 GB, 16 GB host RAM.

Largest training the lab has ever run: 45M-parameter house births (RESULTS
L7742); largest fine-tune Qwen2.5-0.5B LoRA r=16 (the legacy recipe in
CLAUDE.md, llmopt/train/lora.py); largest
continuation 2,021,220 tokens per arm on Mac mps (L52925). No 7B+
training entry surfaced in the heading grep run here (train / LoRA /
birth / fine-tune against 7B / 27B headings); the ledger was not
exhaustively enumerated. A
continued-pretraining diet program on a real open model is a NEW
capability class for this lab, with no in-repo cost anchor above 45M.

Budget estimates (bf16 weights + fp32 Adam states, no activations):
0.9B full-parameter ~11-14 GB, Mac only, throughput unmeasured (first
qualification step); 3.7B full-parameter ~55 GB, infeasible anywhere here,
LoRA only; 7B LoRA/QLoRA on Mac only. Equal-budget continuation arms are
therefore 0.9B full-parameter on the Mac, and the 3.7B/7B promotion arms
are LoRA, which itself constrains the weight-delta structure (rank-r) and
is a named confound for any weight-anatomy transport claim.

## 3. What the survey found (verified subset)

Spectral / localization: the HT-SR engineering line (alpha as a control
signal for pruning / weight decay / layer-wise LR, repos public, gains to
3B) is mature; the phase-transition claims are not (HT-MU 2506.03470 has
no LLM-scale test; 2604.22778 "Spectral Lifecycle", delta-alpha ~ L^0.26,
and 2606.28486 BBP-trainability have no code). Two gaps relevant here:
no paper in the surveyed set measures singular-vector IPR across training
in an LLM, and none links a data-mixture intervention to weight spectra or to the depth
location of weight deltas. 2604.17177 "Decomposing the Depth Profile of
Fine-Tuning" (240 runs, 15 models, 125M-6.9B, code public) is the closest
harness: fine-tuning change concentrates output-proximal, the
sequential-vs-parallel-block split vanishes at 1.3-1.4B. Session verified
the arXiv ids resolve; did not read the papers in full.

Latent / cross-model: No paper, repo, or protocol for Mostik surfaced in
the searches run on 2026-09-05; treat its claims as unverified. The verified artifacts are vec2vec
(2505.12540, unsupervised cross-architecture embedding translation, code)
and Cache-to-Cache (2510.03215, KV-cache bridge across families, repo
thu-nics/C2C returns 200). No LLM-scale independent-init weight merge surfaced
in this survey; the only different-init merges are the
Re-Basin lineage on small vision nets. This matches the house MERGE-SPACE
result (independent-birth averages gate exactly 0 at d64, FINDINGS
L1263-1304) and suggests a rescue test at the LATENT level, not the weight
level.

Verified math 2026: the unit-distance disproof (Erdos #90, 2605.20695) is
an infinite number-field family, not a finite certificate; the usable
layer is Emmerich 2606.03419 (explicit u(n) > n^1.031 via prime-set
integer optimization, open verification pipeline). Anthropic's Lean 4 FLT
repo returns 200 (announced 2026-09-04; not compiled here). Finite-checker
tasks that exist: SAT/DRAT Ramsey certificates R(3,8), R(3,9); AlphaEvolve
kissing-593 and 48-mult matmul; Lean-formalized Erdos solutions (wiki
"Full (Lean)"); BB(5) in Coq. erdosproblems.com was unreachable (403), so
per-problem Erdos statements are secondhand.

Open MoE on this hardware with in-tree router logits (transformers@main):
Qwen3-30B-A3B (the crest vehicle) and its successors Qwen3.5/3.6-35B-A3B
(256 experts, top-8, one shared, ~20 GB at 4-bit, Mac), gpt-oss-20b (32
experts top-4, MXFP4 ~12 GB, Mac; does not fit the 3080 at 10 GB), OLMoE
1B-7B (64/8, ~4 GB, fits the 3080, fully open data + checkpoints), IBM
granite. mlx-lm at the version the survey checked exposes no router-logit flag;
router access there is hooks or a fork (the house already runs masks on the MLX Qwen3-30B-A3B vehicle).

## 4. Ranked shortlist

Criteria in the order given: practical value, falsifiability, real open
model, compute cost here, verifier quality, promotion path, novelty v
FINDINGS.

1. K2-HORIZON-STAGE-DELTA-CENSUS (zero training). Use IFM's own
   stage boundaries (pretrain_final -> mid_1_final -> mid_2_final ->
   post-train final) as the DATA-STAGE variable, measure the WEIGHT-DELTA
   anatomy per stage (per-layer, per-module normalized Frobenius delta;
   depth-relative location; stable rank and Hill alpha before/after; IPR of
   the top singular vectors of the delta; delta effective rank) and the
   CAPABILITY-DELTA per stage on oracle-verified house gates (mathgen L1-3
   sympy-scored, codegen toolchain-scored), at 0.9B first, then 3.7B and
   7B. The cross-scale transport hypothesis is testable directly: does
   the 0.9B stage-delta depth profile predict the 3.7B/7B profile? Model
   real, verifier exact, cost = download + Mac hours, novelty total (zero
   Pythia/OLMo/checkpoint-lifecycle entries in the ledger). Honest break:
   observational, not interventional; stages confound data with context
   length (8K -> 32K -> 128K) and unknown token budgets; 0.9B's final
   stage is distillation, not RL, so the post-train comparison is
   cross-kind and must be fenced.
2. MOE-TRANSFER to Qwen3.5-35B-A3B (or gpt-oss-20b). Port the demand
   ranking, the anti-demand and matched-random controls, and the
   identity-over-aggregates test (FINDINGS L1342, L1500) to a second
   vehicle. Value: the crest is the lab's most replicated open-model
   result and sits on ONE vehicle; a second vehicle is the transport
   test the record owes. Cost: Mac MLX, days of gate runs at 6 paired
   seeds. Verifier exact (mathgen sympy). Break: MLX router hooks must be
   rebuilt for the 256-expert + shared-expert architecture; the +55
   deletion crest still owes its UNIFORM-random deletion control (FINDINGS
   L1589; the matched-rank random control ran, L1570-1571)
   and transports with that debt attached.
3. LATENT-BRIDGE RESCUE at d64 (cheap, falsifiable, ties to a booked
   null). Two independent-birth d64 crystals whose weight average gates 0
   (MERGE-SPACE-1). Train a small affine bridge between residual streams
   at one depth (C2C / vec2vec shape) with the births frozen; gate the
   stitched model. Bars: stitched > 0 at every level where either parent
   > 0, and a random-rotation bridge control at 0. Mac minutes. Break:
   practical value at d64 is low and there is no independent-birth pair
   of any K2 size to promote to; the promotion path is house crystals ->
   Pythia seeds (Pythia has 9 seed variants at 160M only) -> nothing
   larger.
4. VERIFIED-MATH REDISCOVERY. Emmerich prime-set certificates, DRAT
   Ramsey certificates, kissing-593 as frozen exact-checker tasks. Break
   that dominates: a rediscovery task needs a model whose data cutoff
   precedes the result, and K2 Horizon's 20T tokens (released 2026-09-03)
   are undated per stage; the only clean cutoff is a pretrain_* tag whose
   date is not published. Desk first: find the cutoff, or use pre-2026
   checkpoints (OLMo 3 1025, Pythia) as the blind population.
5. K2 DIET SURGERY (equal-budget continuation at 0.9B). The expensive
   direction. Gated behind program 1: it is worth its wall only if stage
   deltas are resolvable on the house gates and the 0.9B anatomy transports
   to 3.7B. Design when armed: a matched factorial, not a marginal sweep.
   Family mass m x sequencing (blocked v interleaved) x co-occurring
   family present/absent, paired seeds, fixed tokens; SS partition
   identities as in bank M (RESULTS L66224 prereg, L66247 verdict) rather than a post-hoc
   regression. The render-atlas lesson travels as a HYPOTHESIS: a data
   family's marginal effect may be carried by what surrounds it.

Not ranked: Mostik-style claims (no artifact), spectral phase-transition
exponents (no code, unreplicated), anything needing the K2 mixture recipe.

## 5. The one program to do next, and what changes our mind

Do program 1. It is the only item that touches DATA -> WEIGHTS -> FUNCTION
on a real 2026 open model at three sizes with zero training, and it
decides whether program 5 is worth arming.

Changes our mind: if at 0.9B the stage-boundary capability deltas on the
house gates are inside the house resolution floor (7 solves on 120, the registered bar
class of the crest program) AND the
per-layer delta profiles across the three stage boundaries are not
distinguishable from one another (a stage does not have a signature), the
cross-scale question is moot and program 5 drops to a single-scale
program or moves to OLMo 3 7B where the mixture is public. If the 0.9B
profile transports to 3.7B (Spearman of the per-layer normalized delta
profile over depth-relative position at or above 0.7 for the SAME stage
boundary, registered before the 3.7B weights are opened), program 5 is armed at 0.9B with 3.7B as the
LoRA promotion arm. If it transports to 3.7B but not 7B, the claim is
fenced to the sub-4B band and the vocab change at 3.7B (64K -> 250K) is
the first named confound.

## 6. Smallest next rung (nominated, not preregistered)

K2-HORIZON-STAGE-DELTA-CENSUS-0, Mac, 0.9B only:
- Tags: pretrain_500000, pretrain_final, mid_1_final, mid_2_final,
  rl-mopd_final (five downloads of 2.16 GB; sha-pin each tag's commit
  hash from the refs API in the prereg).
- Weight census per adjacent pair: per-tensor normalized delta
  ||dW||_F / ||W||_F, aggregated per layer and per module class
  (attn q/k/v/o, mlp up/gate/down, embeddings, norms); depth-relative
  location (mass-weighted mean depth); stable rank and Hill alpha of W
  before/after; IPR of the top-16 left singular vectors of dW; a
  permutation-blind reading (never score by weight distance as a
  capability proxy: the census describes WHERE change lands, the gate
  says WHAT it did).
- Function census per tag: mathgen L1-3 120-item gate (sympy oracle) and
  the codegen ladder (toolchain oracle), greedy, fixed prompt format,
  three prompt seeds; the K2 chat template is NOT used on pretrain tags
  (raw completion format only, one format for all tags).
- Bars to register: (a) the mid_2_final -> rl-mopd_final boundary moves
  the mathgen gate by more than 7 of 120 in the same direction on 3/3
  prompt seeds (the other three boundaries are descriptive, no bar);
  (b) the three stage-delta depth profiles (pretrain-internal, mid_1,
  mid_2; 28 layers, per-layer normalized delta) are pairwise
  distinguishable: every pairwise Spearman at or below 0.5; (c) the
  pretrain-internal pair (500000 -> final) has the smallest total
  normalized delta of the four.
- Prior on record before the run: (a) fires for mid_2 -> post-train, not
  for the pretrain-internal pair; (b) fires; (c) fires.
- Then, on a separate GO, the same census at 3.7B with the transport bar
  registered before the first 3.7B tensor is read; 7B after.
- Cost: downloads ~11 GB (5 x 2.16), delta census CPU minutes, gates ~5 tags x 2
  gates x 3 seeds at 0.9B on Mac, hours not days. No 3080 needed.

## 7. Front-surface housekeeping (assessment; no edits made)

Audit (Opus wording-auditor, session-verified counts):
- README L98 "A fifth of the published record is negative" is false by
  the README's own generated counts: 50 nulls + 5 retractions of 397 =
  13.9 %. Fix: "one claim in seven".
- README never mentions AMENDMENTs: 182 of 1,384 RESULTS entries are
  amendments (session counted). The self-correction rate is the
  strongest front-page fact and is absent.
- README L73 "Effective context is architecture-bound" contradicts README
  L71 and FINDINGS L1307-1315 (the wall belongs to the diet).
- README L57 "everything after is basin-local" is a universal over
  three epochs of d64 training; L46-57 drops the d64 / house-crystal /
  free-run-gate fences; L59-67 drops single-seed / device fences; L42
  omits the text-coherence dissociation; L169 generalizes one matched
  pair to "the two quantities a keep rule optimizes".
- The "Four results" are the four best results of 2026-08-04..11.
  Nothing from 08-12 to 09-05 (the ActionProgram replication at n=3,
  the EX-FRESH carrier crests, the render-atlas arc) reaches the front
  page; "What remains uncertain" is equally dated.
- No stable research-track list exists on any surface; BOARD L3 is one
  ~2,000-word paragraph; FINDINGS is one flat ledger with 21 topic
  headings and no maturity- or thread-filtered view, although
  `scripts/results_query.py` and docs/results-index.jsonl already carry
  the filters.
- Axiom relay/counterbook and the public `39hops/axiom` verifier command
  (docs/REPRODUCE.md L111-132) are absent from README.
- Charter line README L204 is a bare fragment; codegen/asm (README L145)
  and MoE anatomy on general models sit outside "mathematics and physics
  only" as written and are admitted in practice as instruments/measured
  subjects. Proposed wording: "Charter: mathematics and physics only.
  Other domains appear as verification instruments or measured subjects,
  never as capability targets."
- FINDINGS L3784 states the 4.3 % basin figure with its plateau
  correction inline and L3785 restates it as a floor artefact; a skimmer
  can still lift the bare 4.3 %. Optional: move the qualifier ahead of
  the number.
- "MATH-CYBER" means cybernetics (RIFF L8323), not security; one
  glossary line would stop a fresh reader's misread.

Proposed housekeeping order (each a separate Artin GO): (1) the two
number fixes (fifth -> one in seven; add the amendment count); (2) fence
restorations F2-F8; (3) a dated "Research tracks" table replacing "Four
results" with one line per track and its newest verdict; (4) a generated
FINDINGS view by maturity tag; (5) charter wording; (6) axiom command in
Reproduce.

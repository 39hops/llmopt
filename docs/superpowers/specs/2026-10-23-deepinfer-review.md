# DeepInfer architecture review for llmopt (read-only, 2026-10-23)

Pinned commit: deepseek-ai/DeepInfer 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
(2026-10-21, "Bump DeepEP to f6aa38fe"). Clone read outside the repo; nothing
executed from it. Three Opus 5.5 reviewers read one mechanism each with
llmopt's runtime as the compatibility target; the session model spot-checked
the load-bearing claims against the sources (GEMM crossover thresholds,
prefill head modes, DeepInfer's temperature floor, mlx-lm's ArraysCache, the
house grammar law). Nothing in arm C, qserve, the kernels, the test rules,
the defaults or the receipts changes with this document. QWEN-LOW-RELEASE-0
stays the active engineering priority.

Baseline: Qwen3.8-27B arm C at 9.37 GB, MLX/Metal on a 36 GB M3 Pro,
generation-5 kernels, incremental Session with replay fallback, greedy
decoding with explicit reasoning_effort controls, decode 8 to 9.5 tokens per
second, prefill 32 to 36 through the chunk-256 GEMM path (GEMM only from 16
rows for W4 and 32 for S16; shorter rows run the fused multi-row tiles).

## Ranked assessment: what to study after the release

1. Deterministic sampling (low to medium). Portable pieces: a positional
   SplitMix64 key (seed, absolute position) with an fp64 index-order inverse
   CDF, a temperature-0 branch that IS the greedy argmax, and a per-draw
   receipt (uniform, kept mass, CDF interval, a draw margin). Cheap (under 10
   MB, 1 to 2 ms per token with a partition shortcut), lands behind a default
   of None so every greedy receipt and code path stays untouched. Build only
   when a chat-quality question needs sampling; greedy remains the
   comparability instrument and a sampled verdict needs several seeds.
2. Multi-row fused kernel as the ENABLER of speculative decoding (medium as
   kernel work, out of scope for any inference GO). Verify passes of k+1 rows
   dispatch to the fused tiles below the GEMM crossover, and the measured W4
   cost grows faster than linearly in rows (tile 8 about 15x tile 1), so
   verifying costs more than decoding today; a tile-2 to tile-8 kernel that
   shares activation loads (ceiling about 93 percent of a tile-8 launch) is
   the prerequisite. Speculative decoding itself ranks LOW until then.
3. Structured generation (low). The xgrammar core is CPU-only and portable;
   the bitmask for the 248,320-entry vocab is 31 KB and the mask apply is
   microseconds. The house law (grammar pays only where misses are syntactic)
   and the measured 17 of 18 format compliance predict a NULL on the JSON
   misses, which are arithmetic. Worth having only as a "valid JSON
   guaranteed" serving property with an invariance test; not a quality lever.

## Mechanism 1: structured generation (XGrammar)

Files at the pin: deepinfer/structure_output/compiler.py (54-71,
dispatch_and_compile: json with strict_mode, regex, ebnf),
so_matcher_builder.py (29 TokenizerInfo.from_huggingface with vocab_size;
58-95 build with an unused replay_tokens path), matcher.py (58-258
XGrammarMatcher: 83 max_rollback_tokens; 121-153 trigger-token gate by KMP
over generated tokens, grammar free until the trigger has appeared; 149-152
and 183-187 bound token forcing; 181-204 fill_next_token_bitmask, full mask
after EOS; 206-232 rollback), model_executor/layers/sampler.py (301-408
_get_so_mask: CPU int32 bitmask per MTP step, accept-then-fill, speculative
fill then rollback, side-stream copy overlapping the LM head; 187-188 and
433-437 apply before sampling), tilelang_ops/structure_output.py (12-37 the
CUDA mask kernel, torch reference 40-61), csrc/engine/sequence.hpp (230-233
per-sequence state), scheduler.hpp (770-777 preemption ENDS a structured
request; 1097-1100 a fork resets grammar state), environ.py (168 compile
timeout 20 s; 170 cache limit 10 GiB default).

Compatibility. The matcher logic and the xgrammar C++ core are portable; the
tilelang kernel, CUDA streams and events, pinned memory and Linux
sched_setaffinity are not. xgrammar's macOS arm64 wheel and its pins against
the Mac transformers<5 pin are unverified. Qwen vocab: config vocab_size
248,320 (tokenizer 248,077 ids, 243 padding ids to mask), eos
<|im_end|> 248046 and <|endoftext|> 248044 (two stop ids; DeepInfer tracks
one, so a wrapper must pass Engine.stop_ids explicitly), </think> 248069
and <think> 248068 are NOT special tokens (they survive skip_special_tokens
and a grammar could admit them inside a JSON string).

Cost. Bitmask 7,760 int32 = 31 KB per row; compiled grammar cache MB-scale
(cap it explicitly; the 10 GiB default is unsafe beside a 12 GB peak); mask
apply on an mlx row microseconds; CPU fill tens to hundreds of
microseconds, under 0.5 percent of a 105 to 125 ms step, hideable behind
the asynchronous forward as DeepInfer does.

Invariants a wrapper must keep: accept the token into the matcher at emit,
never derive matcher state from cache_tokens (the un-fed last token law);
mask the prefill row too (thinking off starts constrained at token 0); in
thinking mode trigger on the first 248069 while the scorers cut at the LAST
one, so exclude 248068 and 248069 from the answer grammar or assert their
absence after the trigger; assert the allowed set is non-empty (an all -inf
row argmaxes to id 0); replay and incremental stay token-equal iff the
MASKED argmax agrees, and the near-tie margin diagnostic must be computed on
the masked row; one matcher per turn, never persisted, so the Session prefix
check is unaffected; no rollback is needed in single-token greedy and the
DeltaNet state could not be rewound by truncation anyway; if the unmasked
greedy output is grammar-valid the masked run must be token-identical (a
free mechanical oracle).

Bounded experiment (QWEN-GRAMMAR-0, proposal): a fresh seeded JSON
collection of 48 tasks excluded against v2 and v3 prompts; arms A low effort
768 no mask, B the same with the mask after </think>, C no thinking with the
mask from token 0; schema types only with additionalProperties false and no
prompt values, so the grammar cannot encode an answer; report structural
validity, value correctness, format compliance and the invariance count
apart; success = structural validity 48 of 48 in B and C, invariance 100
percent on A-valid tasks, mask overhead at most 2 percent of the median
step, budget stops in B not above A; registered prior: value correctness
unchanged (NULL by the house law), deltas under about 5 of 48 unresolved.

Reject: the tilelang kernel, CUDA streams, Ascend paths, MTP rollback
sizing and anything presuming a pure KV cache.

## Mechanism 2: deterministic sampling

Files at the pin: deepinfer/sampling_params.py (74-88 seq_seed, a random
64-bit seed when none is given; 76-79 temperature below 1e-5 is clamped to
1e-5 with top_p 1 and top_k -1: "temperature 0" is sampled, not argmax),
csrc/engine/prepare_sample.cpp (161 one uint64 seed per sequence; 198 and
206 the draw key is the absolute token position; no generator state),
csrc/engine/sequence.hpp (92-96 SamplingScalars holds only the seed; recompute
never touches it), model_executor/layers/sampler.py (168-188 blacklist, EOS
suppression, grammar mask before temperature; 482-487 golden tokens override
draws), sampler_utils.py (24-25 fp32 logits; 113-196 process_sampling),
tilelang_ops/fused_sample.py (121-150 stable top-p THEN top-k with no
renormalisation between; 209-210 order; 232-248 accept and residual
resample with hashed uniforms keyed 3*pos+case), tilelang_ops/random_sampling.py
(58-84 the reference: SplitMix64 of seed and mixed position, 53-bit fp64
uniform, inverse CDF in vocab-index order; 315-332 batch-invariance unit
test), tilelang_ops/cuda/random_sampling.py (119-138 the CUDA kernel with a
different RNG and a uint32 fixed-point CDF, 38-44 flooring), tests/models/
test_sampler.py, test_batch_invariant.py (101 golden tokens force token
equality, so only logprobs are tested), test_kl.py (max_kl 0 under the
batch-invariant flag).

Findings the reviewer raised about DeepInfer itself: the CUDA kernel and the
reference are different samplers with no token-level cross test; temperature
0 is not greedy (ties and gaps below about 2e-4 can flip); the end-to-end
batch-invariance test cannot fail on tokens; under MTP the key is not purely
positional; top-p before top-k without renormalisation differs from the
HF-style order and a port must state its semantics.

Compatibility: the CUDA and Ascend kernels, the vendored RNG stream,
DeepSelect top-k, the bf16 GEMM head and the pybind prepare_sample are not
portable; the positional hash, the fp64 index-order CDF, the stable top-p
and top-k semantics, the mask-then-transform order and the receipt field
(inclusive ascending-probability CDF) are plain algorithm.

Cost: no state for a positional key; under 10 MB transient; 1 to 2 ms per
token with an argpartition candidate set and an exactness fallback to a full
sort; a full fp64 argsort of 248k per token would cost 15 to 25 ms and is to
be avoided.

Invariants: with the uniform fixed a token flips only when the replay's CDF
boundary crosses it; the index-order CDF bounds the shift at about
2*max|dz|/T, about 2e-4 at T=1 for the measured 1e-4 logit differences
between kernel paths, so sampling is strictly more fragile than greedy (whose
observed margins of 0.03 to 0.35 sit about 300x above the flip threshold)
and after one flip the turn diverges; a positional key makes every draw
independent of how many draws preceded it (budget stops, stop tokens,
fallbacks), verifiable per token from the receipt, and keys the same
positions in incremental and replay because fed_ids are identical; derive a
seed per task as hash(base seed, task id) to avoid correlated draws; the
honest invariant is "every draw verifies against its own logits and the
smallest margin is reported", not "same seed same tokens across paths".

Design for qserve (not implemented): SamplingSpec(temperature, top_p,
top_k, seed, key "absolute_position", algo "qsample-1");
Engine.generate_ids(..., sampling=None) keeps greedy untouched; a
temperature above 0 without a seed is refused; pipeline per position: fp32
row, grammar-mask hook, T=0 takes the greedy argmax call, else divide by T,
top-k by argpartition, stable top-p, fp64 CDF over kept ids in ascending id
order, u = SplitMix64(seed, position, stream 0), token = first id whose CDF
exceeds u; the stop check after the draw with the accounting law unchanged;
GenerationResult gains sampling, draws (position, uniform hex, kept count
and mass, CDF interval, draw margin) and min_draw_margin; verify_draws
recomputes intervals from logits. Exact-seed tests on the stub: T=0 equals
greedy bit for bit including ties; same seed identical tokens and uniforms;
different seeds differ; position-only stub logits give the same draws under
different same-length prefixes and at the same absolute position under a
longer prefix; budget prefix property; incremental equals replay; verify
flags a perturbation near a boundary; a known-answer hash test; a seeded
chi-square on a small vocab; top_k 1 equals greedy.

Reject: tilelang, CUDA and Ascend kernels, DeepSelect, the bf16 GEMM head,
pybind prepare_sample, MTP sampling, golden-token forcing as an equality
test, tiny-temperature "greedy".

## Mechanism 3: speculative decoding (DSpark, DFlash, confidence head)

Files at the pin: deepinfer/model_executor/models/mtp/dflash.py (230-239
block-parallel draft init with a noise token, B = DEEPINFER_USE_NEXTN_DECODE
default 5 per environ.py 129; 241-269 one forward for all draft positions;
198-228 draft KV projected from target hidden states at dspark_target_layer_ids
during the main forward, rejected slots overwritten before read; 143-144 the
draft window must equal the main model's; 279-345 sequential draft sampling
with a low-rank Markov correction, a chain not a tree; 46-79 ConfidenceHead
sigmoid of [h; markov_emb] with a shipped weight), input_metadata/dflash.py
(32-34 window-only draft attention), csrc/engine/scheduler.hpp (1243-1250
cumulative product of per-token confidences; 927 advance_optimistically;
999 commit_processed_tokens), csrc/engine/mtp_varlen.hpp (150-267 the
variable-length verify scheduler: pool (rate, sequence) pairs, sort
descending, maximise expected tokens over a profiled cost curve, stale N-2
and N-1 rates by design), tilelang_ops/fused_sample.py (200-248 standard
rejection sampling with hashed uniforms; DRAFT_PROB_PENALTY 5e-5),
tilelang_ops/sampler.py (227-248 accepted prefix; 321-402 finalize with
stop-token detection inside the rows; 584-589 confidence calibration against
1 minus total variation), doc/inference_deployment_qa.md (243-258: the
reviewer recomputed the table: +5.4 percent at 60 sequences per rank, -1.5
at 16, +11.0 at 96; labelled estimates, not wall measurements).

DeepSeek-specific (reject): the DFlash weights (draft blocks, projection,
Markov matrices, confidence head), target hidden-state conditioning,
hyper-connections, MLA and FlashMLA, MoE draft layers, the matched sliding
window, FP8/FP4 weights, CUDA graphs, the asynchronous stale-rate scheduler
(it buys nothing at batch size 1 and its own table loses at low concurrency),
and tiny-temperature "greedy" (breaks the exact argmax law; llmopt keeps
spec_reference.greedy_accept). Pure algorithm (transferable): the
acceptance-weighted length choice over a measured cost curve, calibrating a
confidence predictor against 1 minus TV (or match rate for greedy),
optimistic advance then commit, overwrite-before-read for rejected slots,
stop-token detection inside accepted rows.

What could transfer to arm C without new weights: prompt-lookup or n-gram
drafts (zero draft cost; already banked at RIFF L9232 with its kill rule),
a confidence gate on the target's own top-1 margin (free after verify),
early exit (poor acceptance expected without trained heads; low). Excluded:
Qwen's own MTP head in vendor shard 18 (vendor bf16, not part of arm C, the
no-new-weights rule). The DeltaNet rollback is feasible and cheap in MLX:
ArraysCache is not trimmable and mlx-lm's own speculative path refuses
hybrid caches, but gated_delta_kernel returns a NEW state array that the
layer rebinds, so holding the pre-verify arrays is a zero-copy snapshot
(recurrent state 48 x 48x128x128 x 4 B about 151 MB; conv tails about 6
MB); exact rollback to the accepted prefix needs per-row states (run the
recurrence one row at a time inside the already source-patched
GatedDeltaNet call, k+1 x 151 MB transient) or re-folding stored per-row
q, k, v, g, beta from the snapshot; the lab's Lean theorems fold_snapshot
and hybrid_rollback (scratch/leancheck/SpecDecoding.lean 430, 455) already
state the law; the KV side is KVCache.trim on all 16 attention layers so
cache_offset stays consistent.

Premise correction: a verify pass of k+1 rows would NOT run through GEMM
(crossover 16 rows for W4, 32 for S16; qmetal.py 542); it runs the fused
multi-row tiles, bit-identical to decode by receipt per row but with a
causal multi-query attention, so verify-then-decode is not bit-equal to
plain decode by construction; prefill_chunked(head="final") returns only
the last position, verify needs head="full". Accounting: after accepting j
of k drafts the fed rows are cur and d1..dj and the argmax at row j is NOT
fed; a stop token inside accepted rows rolls back everything after it and
the stop row itself if it was fed; cap k at max_tokens minus generated
minus 1. Rollback must restore together all 16 KV offsets, all 48 DeltaNet
state arrays and all 48 conv tails; restoring the KV side alone gives
silently wrong outputs that cache_offset cannot catch.

Bounded experiment: stage A on the desk (no device): prompt-lookup
acceptance per verify for k in 1, 2, 4, 8 on the stored greedy id streams
(the 24 preserved E0 rollouts and the battery receipts' generated ids),
projected tokens per second under the measured W4 tile costs (about 235,
480, 1,180 to 1,777, 3,400 to 3,570 us for tiles 1, 2, 4, 8) and under the
prefill-study ceiling; kill if the projected speedup is below 1.10x at every
k. Stage B only if A passes: pre-register k and n, snapshot rollback behind
a flag, ABBA in one process on the frozen v2 battery, bars: token-identical
to qualified greedy (divergences explained by margin), accounting asserts
green, accepted per verify at least the projection minus 10 percent, tokens
per second at least 1.10x with the worst pairing at least 1.0. The
reviewer's own arithmetic predicts stage A dies on the current kernels: a
k=1 verify costs about 1.94 decode steps (linears 82 percent of the step
doubled), so break-even needs acceptance above 0.94.

Note for THEORY: DSpark's confidence-gated variable draft length is prior
art for the banked entropy-adaptive draft length idea (a citation
candidate; the THEORY row was not read in this review).

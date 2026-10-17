# M0 report and E0 protocol: greedy speculation for Qwen3.8-27B on the Mac

Status: M0 delivered 2026-10-17 (local mathematics, Lean proofs, Python
reference tests, this protocol). E0 is NOT armed; it needs Artin's GO.
Nothing was downloaded, trained or run on the 27B; no frozen evidence
was altered.

## 1. Three registers

(M) Proven in Lean, model-free: scratch/leancheck/SpecDecoding.lean, 45
theorems, compiles under leanprover/lean4:v4.33.0-rc2 with Mathlib
9fb10993, no sorry, standard axioms only (propext, Classical.choice,
Quot.sound). Mirrored by tests/test_spec_reference.py (38 cases) on
llmopt/lab/spec_reference.py.

(N) Numerical, outside Lean: fp16 storage of the saved teacher logits,
batched-verify versus incremental kernel differences, argmax tie-break
on the device, chunked versus recurrent DeltaNet kernels.

(E) Empirical, needs fresh measurement: acceptance on teacher-greedy
contexts, the shift-aligned perturbation distribution, Mac draft and
verify costs, run-to-run determinism of a streamed teacher.

## 2. Proved statements (M)

Greedy law (Lean section B, `[DecidableEq α]`, teacher `T : List α → α`):
`specStep_isGreedy`, `specStep_length`, `isGreedy_unique`,
`specRun_take_eq_greedy`: after any number of speculative steps with
arbitrary proposals, the first n emitted tokens equal `greedy T p n`.
`acceptLen_maximal`: the accepted prefix is the longest agreeing one.
Assumptions named outside Lean: T is a function (deterministic teacher
with a fixed tie-break), the verify pass computes the same T as
incremental decoding, rollback restores the exact prefix state.

Margins (section A): `margin_stable` (sup error strictly below m/2
preserves the unique argmax), `margin_stable_weak` (at m/2, ties),
`margin_tight` (m/2 is the best sup-norm radius), `margin_stable_shift`
(the shift-aligned form: it suffices that some per-position constant c
brings z + c within m/2 of z'), `argmax_shift_invariant`.

Block accounting (section C): `tail_sum` (E[emitted] = Σ P(A ≥ j), no
independence), `geom_closed` and `expTokens_eq_sum` (geometric closed
form under the labelled independence model), `break_even` (speculation
beats plain decoding iff k·c_d + c_v < E·c_t), `expTokens_mono`,
`logged_ratio_le` (the logged accepted/drafted ratio is at most α).

Acceptance (section D): `accept_iff`, `accept_iff_argmax_agree` (greedy
acceptance is argmax agreement at the context), `candEmit_correct` and
`candEmit_isSome_iff` (the candidate-set rule emits the teacher token
iff the teacher argmax is in the set), `restricted_argmax_wrong`
(argmax restricted to the candidate set is not a valid verifier),
`argmax_mem_topk`, `topk_mono`.

Rollback (section E): `fold_snapshot` (restore a snapshot and replay the
accepted tokens gives the sequential state, for any step function
including the floating-point one), `hybrid_rollback` (KV list by take,
recurrent state by snapshot).

Gated delta step (section F): `delta_factor_kills_k`,
`delta_factor_not_injective`, `delta_step_forgets` (when β‖k‖² = 1 the
step forgets a whole rank-one family of states, so inversion is
impossible), `delta_factor_sherman_morrison` (explicit inverse otherwise).

## 3. Corrections adopted from Artin's review

Greedy versus sampled are separate laws in code and text: greedy
acceptance is top-1 equality at the same teacher-greedy context
(`greedy_accept`); exact sampling accepts with Σ min(p_i, q_i) = 1 −
TV(p, q) (`sampled_accept_prob`, tested against both forms). Corpus
top-1 agreement is reported only as a teacher-forced proxy and is never
called rollout acceptance.

Throughput: `replay_blocks` is the empirical accepted-prefix replay over
two argmax streams at identical contexts; `replay_cost` charges
drafting per token, verification per block as a function of the drafted
length, commit per block, rollback per rejected block, and a fixed
per-block overhead; EOS and budget truncation are handled by the greedy
reference. The geometric form is kept only under
`GEOMETRIC_MODEL_LABEL`.

Margin audit uses the shift-aligned sup error (max(d) − min(d))/2 with
d = z' − z (`shift_aligned_sup_error`, `margin_stable_shift`). Logit
margin m and probability margin μ are distinct: p_j ≤ p_i e^{−m} gives
μ ≥ p_i(1 − e^{−m}); the flip lemma and Pinsker are stated in μ and TV
(numerically checked; Pinsker itself is not in the pinned Mathlib).

TreeWY (arXiv:2608.20961, reviewed 2026-10-17): the "snapshots only"
claim is withdrawn as too broad. TreeWY reconstructs the committed
state from one forward-substitution solve over the draft window
(`(I + diag(β)G)Ṽ = R`, commit `S_a = g_a S_0 + Σ (g_a/g_i) ṽ_i k_iᵀ`),
exact in real arithmetic, storing only the pseudo-values (plus
normalized k and cumulative log-decay in the reference code). Its
assumptions, scalar per-head decay and β fixed by the token, L2-normed
k, linear affine state recurrence, hold for the qwen3_5 GatedDeltaNet
recurrence (HF `torch_recurrent_gated_delta_rule`: S ← αS, δ = β(v −
Sᵀk), S ← S + kδᵀ). What it does not cover: the causal conv1d window
(width 4) is a second recurrent state that its paper never mentions and
its vLLM branch handles with extra code; the production path is only
bf16-close (reference agrees with the naive recurrence to about 1e-7
relative in fp32; its own tests use atol 2e-2 against the chunk
kernel), and the paper states token streams are not bit-identical to
the baseline. Under the house token-identity standard the fair noise
floor is the chunked kernel versus the recurrent kernel on the same
block, which is already not bit-identical. So: two exact-in-reals
rollback methods exist (snapshot-and-replay, provably exact for the
floating-point step too; TreeWY-style reconstruction, exact in reals
and a different floating-point ordering). E0 does not need either: it
is teacher-forced and never rolls back.

## 4. Evidence references (preserved)

Teacher: logs/qwenteacher_v2 (code_commit 0ca4151, revision 1d4bf0f2);
rollout_logits.npy [256, 24, 248320] fp16, sha in teacher_manifest.json;
rollout_tokens.jsonl (24 prompts, 4,722 live positions). Checked
2026-10-17: the recorded greedy token equals the fp16-stored argmax at
4,718/4,722 positions; the 4 exceptions are exact fp16 ties (stored
values equal to the half-ulp) resolved differently by the generation-
time fp32 argmax, and 6 positions are ties in total. E0 therefore takes
the recorded token as the teacher token (N register).

Teacher margin census on the rollouts (logit, nats; edges 0, 0.02, 0.05,
0.1, 0.2, 0.5, 1, 2, 5, inf): 26, 37, 47, 83, 243, 379, 548, 851,
2,508; median 5.73.

Arm C: compile-time digest chain logs/qwenwhole/artifact_digest_C.txt
(19 files); the copy on the 3080 at ~/qwen_whole0t/C hashes identically
on all 19 (receipt logs/rehab-2e/qwen_whole0t_C_digest_3080_2026-10-17.txt,
manifest d7e56155...). Codec census from the manifest: 1,199 tensors,
192 w4, 210 s16, 449 raw, 348 excluded. Teacher-forced corpus
agreement 0.829, prefixes 0.806, X 0.249, K 0.162 (logs/qwenmodel1/
score_C.json, Mac CPU, 457 s wall for 454 positions).

Vendor side files surviving on the 3080 (~/qwen_vendor, 13 MB):
config.json 191e0af2, tokenizer.json 0997f410, tokenizer_config.json
b11349aa, generation_config.json e70c136c, model.safetensors.index.json.
No vendor weight shards exist on either machine.

## 5. Throughput model (labelled)

Measured inputs (Mac CPU, August receipts): streamed bf16 teacher
decode step about 4.7 s per token (22,342 s over 5,176 scored positions
with cache); a 356-token teacher-forced forward 84 s; arm C teacher-
forced forward about 1.0 s per position on CPU (457 s / 454).

Model (break_even theorem; E from the independence model, which E0's
replay will replace): with c_t = 4.7 s, c_v(k) ≈ 4.5 + 0.24k s (one
weight stream plus per-position compute), c_d ≈ 1.0 s (CPU decode of
the 2-bit tower, no Metal kernel), α = 0.83 proxy, k = 4: E = 3.56,
speedup ≈ 3.56 × 4.7 / (4 + 5.5) ≈ 1.8. The draft term dominates; a
Metal W4 kernel (banked, unbuilt) is the lever, not k. These are
estimates (E) until E0 and a timing rung replace them.

## 6. E0 launch protocol (for approval)

Goal: α on teacher-greedy contexts, the shift-aligned perturbation
distribution, margin-certified fraction, TV and KL per position, and
the empirical replay for k = 1..8, for arm C against the 24 preserved
rollouts. One number decides the next step: pooled replay mean emitted
tokens per block at k = 4 and k = 8.

Where: Mac CPU only (device rule of the tree measurements; the arm
scorer refuses any other device). No 3080 compute.

Inputs to stage on the Mac (file copies of our own artifacts, not model
downloads):
1. rsync 3080:~/qwen_whole0t/C to Mac ~/qwen_whole0t/C (8.8 GB; the
   path the August receipts cite). Verify against
   logs/qwenwhole/artifact_digest_C.txt before any run. Mac free space
   55.6 GB, after copy about 46.8 GB.
2. rsync 3080:~/qwen_vendor (13 MB: config, tokenizer, index) to Mac
   ~/qwen_vendor. Verify the four digests above.

Runtime support (inspected): .venv_teacher has torch 2.13.0,
transformers 5.15.0 with model_type qwen3_5 registered, accelerate
1.14.0, safetensors 0.8.0. Mac RAM 39 GB; arm C qualification preflight
estimated 6.5 GB peak; one forward holds at most 339 × 248,320 fp32
logits (337 MB).

Driver: scratch/qwen_e0_spec_census.py (new, lint clean, not yet run
against a model). It reuses the MODEL1 scorer's streamed construction
(layer weights decoded per forward, hooks return them to meta), takes
prompt ids from the teacher manifest (left-padded input_ids and mask),
appends the recorded rollout tokens, runs one teacher-forced forward per
prompt, aligns draft logits at prompt_len − 1 + s with rollout step s,
and writes per-position rows plus summary under logs/qwen_e0/C
(refuse-if-exists; SMOKE=1 writes to logs/qwen_e0_smoke/C with one
prompt and 8 steps). It never stores full draft logits (top-16 per
position only, about 1 MB); it never writes under logs/qwenteacher_v2
or logs/qwenwhole.

Ladder:
R0 identity: digests of ~/qwen_whole0t/C and ~/qwen_vendor match.
R1 smoke: `SMOKE=1 .venv_teacher/bin/python scratch/qwen_e0_spec_census.py`
   (one forward of about 70 tokens; expect under 3 minutes).
R2 full: `.venv_teacher/bin/python scratch/qwen_e0_spec_census.py`
   with `scripts/liverun.py` registration if the run is to be booked;
   24 forwards of 83 to 339 tokens at about 1 s per position: about
   1.5 h wall, peak RSS under 8 GB, output under 5 MB.
R3 read: pooled agreement, agreement by teacher margin bin, certified
   fraction, replay k = 1..8, per-category split (chat, math, long).
Pre-registration before R2: a short OBSERVATION pre-reg naming the two
decision thresholds (replay mean emitted at k = 4 and the certified
fraction) and the fence that this is teacher-forced, single arm, 24
prompts, Mac CPU.

Not in E0: any generation, any timing claim for the Mac tower, any
rollback method, sampled-law acceptance (reported per position as
1 − TV but not used for a decision).

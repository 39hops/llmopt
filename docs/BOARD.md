# The Board — every thread, one line, current status

Updated 2026-07-16 evening. Strategy: specs/2026-07-15-post-climb-strategy.md. RESULTS.md holds verdicts, RIFF-LEDGER.md holds
provenance; this is the queue. States: LIVE (running/next-action),
BANKED (specced or named, awaiting GO), CLOSED (verdict recorded).

## LIVE

| Thread | Status | Pointer |
|---|---|---|
| 45M GRPO run 1b (Mac) | COMPLETE — 61/120 @ 59.36 (c10, lineage record); 24 total cycles across runs 1+1b: 57->61 solves, 54.24->59.36 validity, +16k mined; RL ceiling reached, diet next | LOOP-LOG + micro_grpo_45m_run1b.log |
| Self-distillation consolidation | PASSED — **64/120 @ 62.23** (program record, every level >= promoted; one 5.5-min epoch beat six RL cycles); mathnative_45m_consol.pt = production 45M; next: GRPO leg from the consolidated base (RL -> consolidate -> RL) | RESULTS 2026-07-16 night |
| 113M capacity rung | CLOSED — NULL above 50M on this diet (54.58/588, L4 11.5 vs 50.4M's 625/18.9 same-path; reallocation not a rung); re-ask once if v2.2's thicker diet lands | RESULTS "NULL above 50M" |
| Fast trainer path | VERDICT: packing convicted (~10 pts, no speed win once thrash fixed), bf16 exonerated (~2-pt debit, 11-min 50.4M trains); --fast now = bf16 + --nopack | RESULTS "parity 2x2" |
| Next after current runs (Artin GO): self-distillation consolidation on 45M lineage (low-LR SFT on grpo-mined rows), then weight anatomy on the adapter drawer | queued | post-climb strategy B + C |
| Step-level expert iteration | rounds 1-3 done (one-shot 19/30, steps 12/30, validity 1.0%); NEXT: magic-adaptive skip granularity + round 4 (data balance, chain-required eval) + supervised loop round to arm autonomy | specs + scripts/expert_loop.py + LOOP-LOG.md |
| Magic-adaptive step granularity | GO (Artin: "predict longer tokens with magic") — estimator sizes skips in data, magic score joins Hints at inference; folds estimator judge-slot + skips + latent-anchors | RESULTS "rounds 2/3"; estimator v7 |
| ODE engine | rung 1 shipped: 75/75 parity with dsolve, cc2 algebra wins wall; rung 2 = hard-integrating-factor families | scripts/bench_ode_engine.py |
| VGE (quantum ground states) | rung 1 shipped (HVA 0.69% @ 6 params); structure search CLOSED (2 fails, phase-reading confirmed qualitatively) | llmopt/quantum/ground.py |
| Fused cross-entropy (Liger-style, MLX) | SHIPPED 2026-07-13: 16k tokens 13.5GB vs 38GB AND faster (3203 vs 2008 tok/s — memory wall flips the sign); 32k infeasible->3183 tok/s; use naive below ~8k | train/fused_ce.py + RESULTS |

## BANKED (awaiting GO or a prerequisite)

| Thread | Waiting on | Pointer |
|---|---|---|
| Fast trainer path (Artin 2026-07-16, "be experimental with the 3080"): bf16 autocast (fp32 masters — dodges the fp16-Adam scar) + token-budget batching (median seq ~60 tok vs fixed BS=32 — pack to ~24k tok/batch) + optional grad checkpointing. SDPA already in. Fused CE = pre-registered NULL (vocab 40, logits 2.6MB — nothing to fuse). Gate: parity run at 50.4M (fast vs standard, same unseen gate, match within noise) before any cross-run comparison trusts it | 113M finishes on the 3080 (don't touch the live capacity data point) | 2026-07-16 chat |
| Repo housekeeping (Artin 2026-07-16): organize scripts/ and data/ into typed subdirectories, update all stale references, verify every script still runs. NOT tidying — SURGERY: scripts cross-import via sys.path (`from bench_step_tokens import ...`), data paths are hardcoded globs in trainer/eval-guard/GRPO drivers, and the 3080 checkout must move in lockstep (hash-verified). Gate: pytest green + smoke-launch every entry-point script + both machines synced. POLICY (Artin): only Fable 5 changes code in this repo — helper agents (GPT sub-agents etc.) may move files/update references under supervision, but any bug found gets MENTIONED, not fixed; Fable verifies the whole pass regardless | a natural freeze point (not mid-sprint) | 2026-07-16 chat |
| Syndrome head | CLOSED — payoff 3 NULL (aux learns free, converts to nothing: 0.36 vs 0.42% validity); unified climb fold fails its gate; run 3 stays pure GRPO | RESULTS 2026-07-15 |
| Syndrome dynamics (child-syndrome world model, 1-ply lookahead at embedding cost) | syndrome head rung 2 | same spec, relations |
| Magic estimator revival on embedding features | if hardness economics return (budget alloc, skip sizing) | same spec, rung 3 |
| GRPO at the frontier band | RUN 1 LIVE — cycle-2 gate green on EVERY level ({15,10,6,8}@1.90 vs {13,9,5,5}@1.38); all-pass states 1->24; run 2 = lossless verify levers; run 3 = unified climb (gated on syndrome A/B) | specs/2026-07-14-grpo-v2-and-unified-climb.md |
| Dynamic MoE via magic router (per-QUESTION expert keep-set loaded from SSD; measured basis: router stats 61%-keep holds accuracy, 50% count-quantile BEATS full, cliff below ~28%; moe/ LRU offload cache exists) + prune-then-distill skill grab (domain keep-set as small teacher) | Artin 2026-07-14 GO-to-bank; pilot on Mac w/ Qwen3-30B-A3B after current tree | RESULTS MoE pruning + this chat |
| Representation stitching: TIER 1 PASSED all rungs (SmolLM2 beats native 91.6>90.5; bridge R~0.98; native probe reads bridged at 86.9% — same coordinates) -> next: (2) 30B-A3B keep-set runtime teacher, (3) GLM-class as OFFLINE geometry donor, HOUSE-TOOLS EDITION (Artin: use our recorded facts): zero-inference keep-set — train the weight-space reader (80.8/88.4 recipe, permutation-augmented) on router-stat-labeled experts from runnable MoEs, then READ GLM's experts off SSD to predict its math keep-set without ever running it; harvest prefill through kept experts only, or skip to weight2vec (reader embedding space as the shared geometry — change-of-basis from weights alone, no text/inference; moonshot, cheap falsification: hold-out-family transfer test) | tier 1 free/this week; tier 3 needs ~200GB SSD | RIFF-LEDGER + 2026-07-14 chat |
| ODE chains merge (data/ode_chains.jsonl, 317 pairs) | multi-domain round AFTER round-5 attribution is clean | 91fb39d |
| Closed-system weight anatomy (post-GRPO-runs): layerwise dW mass RL-vs-SFT, CKA before/after, layer-sweep probes on climbed model, weight-reader SFT-vs-RL classifier (Artin: do closed-system weights LOOK different? are thinking layers forming?) | GRPO runs complete + adapter drawer as dataset | 2026-07-14 evening |
| Curriculum ascent (L6-8 enter the frontier band as the climb raises solve rates; engine chains ready there) | GRPO run 2+ gate readings | step-grpo spec |
| Math-native micro-model (from Artin's purely-on-rule-bits push): from-scratch 10-50M, ~500-token math tokenizer (charset mask = the real vocab), trained PURELY on closed-system chains + GRPO — tests whether pretraining priors are load-bearing or drag (whisper says less than assumed); 20-100x sampling speed; unlimited engine-minted data; Mac-trainable | after run 3; spec before build | 2026-07-15 chat |
| Potential-shaped reward (reward = Phi(next)-Phi(cur), Phi from NNUE/HCE/magic — measured progress toward solved; kills stall-hacks structurally, Ng-shaping preserves optimal policy; the terrain answer to Artin's how-do-we-make-it-learn-properly) | run-4/5 candidate; the principled successor to distinct+cycle patches | 2026-07-15 Goodhart day |
| Rank-matched GRPO (r=4 vs r=16 A/B — the anatomy says the solution has stable rank ~4; capacity beyond it may be dead weight) | run-4 knob, one flag | anatomy RESULTS 2026-07-15 |
| ES-LoRA verified climb (Artin: LoRA through verified hill climb) — gradient-FREE: rank-1 perturbations kept iff probe-batch validity improves; no backward, no optimizer, runs on inference-only backends (Mac/MLX could climb natively); bet: search cost scales with the whisper's tiny solution manifold, not the model | spec + bench after run 3; ES sample-hunger vs oracle cost is the pre-registered risk | 2026-07-15 chat |
| Fused-quotient residue (L8's last 3) | rule synthesis design; the step-model may reach it first | RESULTS "L8 autopsy" |
| LLM rule synthesis (heurisch-as-teacher) | frontier targets exist now | RIFF-LEDGER |
| Latent-between-anchors (COCONUT-shaped) | step-model maturity | RIFF-LEDGER |
| Temporal-pincer verification | backward-LM training run | memory + RIFF-LEDGER |
| Reformulation ensemble | LLM eval harness slot | RIFF-LEDGER |
| Basis proposer (orbital selection net) | i_linear_basis failure data at L8+ | queued-workspace memory |
| Parallel leaf closing (fork-pool) | heurisch fire-rate check | queued-workspace memory |
| Flash prefill port + tile autotuning | revives config estimator (needs a config space with variance) | kernels/metal.py docstring |
| Engine-regret hook in mining farms | throughput workloads only (2.1x); NOT label farms | RESULTS "Engine-level regret" |
| ZX syndrome-policy port | judgment-stack recipe on ZX engine | zx memory |
| L9 / adversarial generation | if L8 saturates post-orbitals (37/40 now) | RESULTS "L8" |
| Feynman-integral engine (IBP reduction — particle physics as a rewrite system: states = loop integrals, moves = IBP identities, oracle = numeric evaluation at kinematic points; a real bottleneck in real physics and nearly move-for-move our calculus engine) | far-future continent; charter-clean physics | Artin's electrons/quarks question 2026-07-15 |
| ODE engine (physics rung, zero chemistry) | the where-next map's first new domain: oracle = substitute-back (identically zero), ansatz rules generalize, dimensional analysis = free syndromes; precedent FermiNet (NN + Hamiltonian oracle -> atoms ab initio) | Artin's periodic-table riff, 2026-07-12 |

## CLOSED (verdict on the books)

| Thread | Verdict |
|---|---|
| Predicted syndromes | 3-round arc: structural NULL x2, then Artin's derivability re-aim PASSED — 0.5B embeddings 87.7%/0.975, i_apart 0.02->0.98 R, hard roots beat easy; adoption gate = hints A/B in solve_chain (BANKED below) |
| Population training (K LoRA, one base, batched) | NULL — MLX 0.5B saturates at one adapter's batch (1250 tok/s flat), 1.04x @ corpus shapes, 0.62x big; machinery (exact K-sequential equivalence) banks for tiny-net populations |
| Derivation-expanded chains + syndrome prompting | SHIPPED into rounds 2/3 (Think + Hints fields) |
| Reverse engine | SHIPPED — answer-side chains (492), the corpus factory's decompression stroke |
| Grammar-constrained decoding | NULL for the adapter (token-identical rerun; misses are semantic); mask kept for base-model prose |
| Ansatz-structure search (VGE rung 2/2b) | 2 fails vs hand design; evolution reinvented HVA alternation; phase-reading confirmed qualitatively |
| Token-level expert-iter round 1 | superseded by rounds 2/3 (reverse-engine diet) |
| Token-level regret | NULL — probe real (AUC 0.914), nothing to recover at ~25-tok traces |
| Config estimator (int4 GEMV) | NULL — 6-config space flat; revisit at tile autotuning |
| Dispatcher v4 | NO-ADOPT — v3 112 v 110; arm spread collapsed to 110-112/120 |
| Engine-level regret | pre-registered bar FAIL, throughput metric 2.1x — workload-dependent, not default |
| Three-lane quant race | awq_lite wins function-space (8.07%); kernel carries it |
| int4 dequant-GEMV kernel | 1.11x vs mx_q4 at D=4096, loses D=896 — honest split |
| symengine swap, propose_k=4, entropy beam, budget alloc | NULLs, see RESULTS |
| Dispatcher v1/v2/v3 lineage | v3 production (114/120 @ 370s) |
| L5/L6/L7/L8 ladder | 100% / 59/60 / 56/60 / 37/40 — engine leads sympy everywhere |

## Meta-pattern (named 2026-07-12)

**Prediction pays only where variance lives.** Four judges starved in
one day by the engine improving under them (estimator cost-rho, token
regret, config estimator, dispatcher arms). Before building a judge,
measure the spread it would judge.

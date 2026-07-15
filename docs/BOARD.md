# The Board — every thread, one line, current status

Updated 2026-07-12. RESULTS.md holds verdicts, RIFF-LEDGER.md holds
provenance; this is the queue. States: LIVE (running/next-action),
BANKED (specced or named, awaiting GO), CLOSED (verdict recorded).

## LIVE

| Thread | Status | Pointer |
|---|---|---|
| Step-level expert iteration | rounds 1-3 done (one-shot 19/30, steps 12/30, validity 1.0%); NEXT: magic-adaptive skip granularity + round 4 (data balance, chain-required eval) + supervised loop round to arm autonomy | specs + scripts/expert_loop.py + LOOP-LOG.md |
| Magic-adaptive step granularity | GO (Artin: "predict longer tokens with magic") — estimator sizes skips in data, magic score joins Hints at inference; folds estimator judge-slot + skips + latent-anchors | RESULTS "rounds 2/3"; estimator v7 |
| ODE engine | rung 1 shipped: 75/75 parity with dsolve, cc2 algebra wins wall; rung 2 = hard-integrating-factor families | scripts/bench_ode_engine.py |
| VGE (quantum ground states) | rung 1 shipped (HVA 0.69% @ 6 params); structure search CLOSED (2 fails, phase-reading confirmed qualitatively) | llmopt/quantum/ground.py |
| Fused cross-entropy (Liger-style, MLX) | SHIPPED 2026-07-13: 16k tokens 13.5GB vs 38GB AND faster (3203 vs 2008 tok/s — memory wall flips the sign); 32k infeasible->3183 tok/s; use naive below ~8k | train/fused_ce.py + RESULTS |

## BANKED (awaiting GO or a prerequisite)

| Thread | Waiting on | Pointer |
|---|---|---|
| Syndrome head | CLOSED — payoff 3 NULL (aux learns free, converts to nothing: 0.36 vs 0.42% validity); unified climb fold fails its gate; run 3 stays pure GRPO | RESULTS 2026-07-15 |
| Syndrome dynamics (child-syndrome world model, 1-ply lookahead at embedding cost) | syndrome head rung 2 | same spec, relations |
| Magic estimator revival on embedding features | if hardness economics return (budget alloc, skip sizing) | same spec, rung 3 |
| GRPO at the frontier band | RUN 1 LIVE — cycle-2 gate green on EVERY level ({15,10,6,8}@1.90 vs {13,9,5,5}@1.38); all-pass states 1->24; run 2 = lossless verify levers; run 3 = unified climb (gated on syndrome A/B) | specs/2026-07-14-grpo-v2-and-unified-climb.md |
| Dynamic MoE via magic router (per-QUESTION expert keep-set loaded from SSD; measured basis: router stats 61%-keep holds accuracy, 50% count-quantile BEATS full, cliff below ~28%; moe/ LRU offload cache exists) + prune-then-distill skill grab (domain keep-set as small teacher) | Artin 2026-07-14 GO-to-bank; pilot on Mac w/ Qwen3-30B-A3B after current tree | RESULTS MoE pruning + this chat |
| Representation stitching, 3 tiers (Artin's axis-change riff): (1) PoC runtime feed, small foreign teacher (Llama-3.2-1B/Gemma-2B -> 0.5B layer-15; rung 1 = does bridged teacher vector predict syndromes), (2) 30B-A3B keep-set runtime teacher, (3) GLM-class as OFFLINE geometry donor, HOUSE-TOOLS EDITION (Artin: use our recorded facts): zero-inference keep-set — train the weight-space reader (80.8/88.4 recipe, permutation-augmented) on router-stat-labeled experts from runnable MoEs, then READ GLM's experts off SSD to predict its math keep-set without ever running it; harvest prefill through kept experts only, or skip to weight2vec (reader embedding space as the shared geometry — change-of-basis from weights alone, no text/inference; moonshot, cheap falsification: hold-out-family transfer test) | tier 1 free/this week; tier 3 needs ~200GB SSD | RIFF-LEDGER + 2026-07-14 chat |
| ODE chains merge (data/ode_chains.jsonl, 317 pairs) | multi-domain round AFTER round-5 attribution is clean | 91fb39d |
| Closed-system weight anatomy (post-GRPO-runs): layerwise dW mass RL-vs-SFT, CKA before/after, layer-sweep probes on climbed model, weight-reader SFT-vs-RL classifier (Artin: do closed-system weights LOOK different? are thinking layers forming?) | GRPO runs complete + adapter drawer as dataset | 2026-07-14 evening |
| Curriculum ascent (L6-8 enter the frontier band as the climb raises solve rates; engine chains ready there) | GRPO run 2+ gate readings | step-grpo spec |
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

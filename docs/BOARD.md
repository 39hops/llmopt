# The Board — every thread, one line, current status

Updated 2026-07-23 morning. RESULTS.md holds verdicts, RIFF-LEDGER.md
holds provenance; this is the queue. States: LIVE (running/next-action),
BANKED (specced or named, awaiting GO), CLOSED (verdict recorded).

## LIVE

| Thread | Status | Pointer |
|---|---|---|
| **PRODUCTION / THE CROWN** | **STATISTICAL TIE**: champion mathnative_gen6_grown 76/120 (fp32, 55.1M, cuda-verified) vs **merged_grown 75/120 (1.58-bit ternary, 89.7M)** — ternary arc 65->70->75 in one night (diet + warm growth); tie-breakers queued: duo-mined rare shard (mining NOW) or growth round 2 | RESULTS 2026-07-23 |
| **DUO MINER (gen-8 flywheel)** | LIVE on 3080: champion+ternary mixed waves over fresh band, banking verified chains from rare/unseen skeletons (851 targets); shard -> gen-8 diet | scratch/duo_mine.py |
| **METABOLIC V3 (the stacked run — next big slot)** | four banked upgrades, one session: fp64 masters (5x flip recovery proven) + surprise-gated LR + wave-contrast pairs + late-layer rod; HEADLINE HYPOTHESIS = Artin's ceiling-on-slow-learning theory (precision lifts slow learning only — birth end proven flat) | RIFF-LEDGER capstone |
| **SERIES/PHYSICS/POLY/CODE (the continents)** | series 88% (decomposition arc 15->67->88); physics 85.5% first contact (+dual crystal 92.3% WITH math intact — monolith wins, positive transfer); poly 50.9% (needs tree decomposition) + suggestive gate lift 67; vm-asm 89.2% (rung 1b) — FOUR grammars live | RESULTS 2026-07-22/23 |
| **THE EQUATION (flagship)** | geometry sector now UNIVERSAL at 3 grammars (kurt 1.9 matched-feeding: calculus/mechanics/programs — encodes FEEDING not content); capability counted in PRIMITIVES (decomposition discount — re-price perfection, analysis pending); determinability law measured (96 v 47 within one model) | RESULTS 2026-07-22 |
| **PRECISION PROGRAM (closed at birth, open online)** | birth: BRACKETED bf16..fp64 indistinguishable (parity sweep + fp64 showdown — pre-registered disagreement resolved for the absorption law); online: fp64 masters proven; endgame impl = Ozaki block-aligned int-sliced matmul (prototype: alignment is the mandatory leg) | RESULTS 2026-07-23 + THEORY absorption law |
| **RARITY INSTRUMENT (adopted)** | every promotion reads the curve; duo wave = adopted farm sampler (133/200 equal-budget); gates stay single-model | specs/2026-07-22-duo-substrate.md |
| **AXIOM RELAY (pending send)** | poly tree-decomposition ask + physics tranche 2 (energy conservation after kin/shm passed) + dual-transfer news + vm-asm existence FYI | docs/superpowers/relay/ |
| Speed doctrine (all shipped this week) | KV cache 3.4x; bf16 births default; grad-ckpt kills 10GB OOM class; allocator tripwire doctrine (the 43x); gates ~2 min | RESULTS 2026-07-22/23 |

## CLOSED this week (see RESULTS for verdicts)

Gen-7 (no-promote, ration lesson -> merged run applied it and won);
L7 gap (0 real); gate-v2 batching (null; KV was the fix); shaped
GRPO size-Phi (null); chain-carry v1 (void, redo at full budget);
selection-overfitting story (retracted -> battery truth -> rarity
instrument); birth precision (bracketed, closed); 113M capacity
(closed earlier: territory is the Liebig factor).

## BANKED (awaiting GO or a prerequisite)

| Thread | Waiting on | Pointer |
|---|---|---|
| NNUE homecoming: C++ ternary/micro-model inference (gates+mining 5-20x; ax::nn candidate; training stays torch by doctrine) | axiom C6 done; spec first | RIFF-LEDGER 2026-07-19 |
| Alphabet tournament (7 discrete weight sets + escape code; rotation-vs-resolution bit-matched; alphabet-follows-domain hypothesis) | persistence verdict + gen-6 | specs/2026-07-18-alphabet-tournament.md |
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

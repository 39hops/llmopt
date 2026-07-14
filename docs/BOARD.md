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
| Syndrome head on the step model | RE-AIMED at payoff 3 only (representation shaping) — the A/B measured hints NEGATIVE in the prompt (none 19/48 @ 1.87% beats oracle 13 @ 1.19 and predicted 14 @ 1.29); drop Hints line at inference | specs/2026-07-13-syndrome-head-design.md + RESULTS A/B |
| Syndrome dynamics (child-syndrome world model, 1-ply lookahead at embedding cost) | syndrome head rung 2 | same spec, relations |
| Magic estimator revival on embedding features | if hardness economics return (budget alloc, skip sizing) | same spec, rung 3 |
| GRPO at the frontier band (sustained RL over verified steps) | GO + SPEC'D — mixed-group filter = gradient where variance lives; awaits control-round verdict for queue position; 3080 | specs/2026-07-14-step-grpo-design.md |
| ODE chains merge (data/ode_chains.jsonl, 317 pairs) | multi-domain round AFTER round-5 attribution is clean | 91fb39d |
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

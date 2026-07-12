# The Board — every thread, one line, current status

Updated 2026-07-12. RESULTS.md holds verdicts, RIFF-LEDGER.md holds
provenance; this is the queue. States: LIVE (running/next-action),
BANKED (specced or named, awaiting GO), CLOSED (verdict recorded).

## LIVE

| Thread | Status | Pointer |
|---|---|---|
| Step-level expert iteration | round 1 racing on 3080; loop driver built, manual round then arm autonomy | specs/2026-07-12-step-expert-iteration-design.md, scripts/expert_loop.py, LOOP-LOG.md |
| Grammar-constrained decoding | conditional GO (fires if round 1 confirms format damage) | RESULTS "Step-tokens"; amendment to bench_step_tokens.sample |
| Macro-distillation skip-pairs | in corpus (95 pairs), rides next retrain | expert_iter_steps.phase_skips |

## BANKED (awaiting GO or a prerequisite)

| Thread | Waiting on | Pointer |
|---|---|---|
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

## CLOSED (verdict on the books)

| Thread | Verdict |
|---|---|
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

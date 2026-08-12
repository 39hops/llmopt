# Script index (generated — do not hand-edit)

Regenerate: `.venv/bin/python scripts/gen_index.py`

## scripts/

### scripts/__init__.py
*(no docstring)*


### scripts/arena.py
THE ARENA: engine vs the 0.5B step-model, same integral, live.

- `engine_lane(level: int, seed: int, q: 'mp.Queue') -> None`
- `main() -> None`

### scripts/autopsy_int.py
Failure autopsy for integration: run the best structural engine (bf + NNUE h + markov top-3) at a GENEROUS budget on int L3/L4, and dump every failure — the root integrand plus the best (lowest-h) state the search died on. Both prior ceiling-movers (euler, i_apart) came from reading one failing problem; this reads all of them. Classification of the dump chooses the next rules; frequencies first, code second.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `markov()`
- `best_first(root, budget, prop, h)`
- `main(n: int, budget: int) -> None`

### scripts/backfill_code_commit.py
One-time (rerunnable via --redo): attach code_commit to ledger rows.

- `path_exists_at(sha: str, path: str) -> bool`
- `find_commit(title: str, files: list) -> str | None`
- `main() -> None`

### scripts/bench_adaptive.py
Adaptive-k race: entropy-gated branching vs the fixed strategies. Pre-registered prediction (spec 2026-07-07-adaptive-k-design.md): adaptive should match k1x3 on diff L2-3 AND match full on int L3 — spending width exactly where the sweep showed width matters. Also prints mean-k and an H histogram per cell: the null-check instrument (if H doesn't localize, the confidence signal is the gap).

- `class _Timeout`
- `load_model()`
- `random_proposer(seed_tag: str)`
- `restart_search(root, total_budget, restarts, seed, width=8)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budgets: list[int], temperature: float=1.0, configs: list[str] | None=None, width: int=8, k_max: int=6, macros: bool=False) -> None`

### scripts/bench_adaptive_draft.py
Entropy-adaptive draft length vs fixed-k speculative decoding (3080).

- `vanilla_greedy(model, prompt_ids: list[int], max_new: int) -> list[int]`
- `timed(fn, repeats: int=3)`
- `main(max_new: int) -> None`

### scripts/bench_anneal.py
Computation = cooling, measured (notes/physics-night section 16): greedy best-first is a T->0 quench; annealing theory says a finite temperature schedule escapes local minima a quench commits to — and we HAVE measured local minima (the L4 wall-timeouts where the search marries a blow-up branch). Metropolis-flavored best-first: pop from the frontier by Boltzmann weight exp(-(h - h_min)/T) instead of argmin, T decaying linearly to 0 over the node budget (quench at the end). Arms: greedy (T=0 incumbent) vs anneal at T0 in {1, 5, 25}. NNUE h is the energy. Same seeds as every race.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `anneal_search(root, budget, prop, h, t0, seed)` — t0=0 -> exact greedy best-first (the incumbent).
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_ansatz_search.py
Ansatz-STRUCTURE search (VGE rung 2, Artin's GO 2026-07-12): greedy beam over layer-token sequences, energy oracle judging each optimized candidate — the engine move transplanted to circuit design.

- `search(H, e0)`
- `main() -> None`

### scripts/bench_ansatz_search_2b.py
Ansatz-structure search 2b: evolutionary (rung 2's greedy FAILED — first-token prefix lock; mutation can rewrite any position). Population over token sequences, mutate (replace/insert/delete), param-count penalty, elites refined with bigger budgets + restarts.

- `mutate(rng, toks)`
- `evolve(H, e0, rng)`
- `main() -> None`

### scripts/bench_bandit.py
Strategy-portfolio bandit: UCB1 over engine configs, one bandit per problem class (kind, level). The measured complementarity that motivates it: int L3 tight-budget prefers flat markov top-3 while everything else prefers NNUE best-first — no single champion wins every cell. Compare: each fixed arm, the bandit (online, no oracle), and the per-cell oracle (upper bound). Bandit must beat the best fixed arm to earn its slot.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `h_struct(state: State) -> float`
- `markov()`
- `best_first(root, budget, prop, h)`
- `make_arms()`
- `ucb_pick(stats: dict, t: int) -> str`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budget: int) -> None`

### scripts/bench_bestfirst.py
Best-first (priority-queue) search vs synchronized beam — the skeleton where the Dijkstra g+h question is actually askable (the beam g-sweep tied 92=92=92=92 with a structural proof: equal-depth comparisons cancel g). Frontier mixes depths; lambda weights g. Markov top-3 pruning both, width-2 beam as the incumbent.

- `class _Timeout`
- `markov()`
- `h(state: State) -> float`
- `best_first(root: sp.Expr, budget: int, prop, lam: float)` — Pop min(lam*g + h); expand markov-top-3; sampled verification.
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_bestfirst_llm.py
The record attempt: best-first + NNUE h + entropy-gated 0.5B confidence — the three winning components in one search for the first time. Incumbent to beat: bf-nnue + markov top-3 = 113/120 on these exact cells and seeds (bench_bestfirst_nnue.py). Only the new arm runs; compare row-by-row against the recorded incumbent table. "GPU buys confidence, not choice": the LLM's job here is k, not rank.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `load_llm()`
- `best_first_adaptive(root, budget, scoring_prop, k_policy, h)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_bestfirst_nnue.py
Best-first h-race: structural h vs NNUE h, plus a no-dedup ablation.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `h_struct(state: State) -> float`
- `markov()`
- `best_first(root, budget, prop, h, dedup=True)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_budget_alloc.py
Budget allocation: the magic estimator's first engine integration.

- `class _Timeout`
- `load_estimator()`
- `main(n_per: int, flat: int) -> None`

### scripts/bench_commute.py
Commutator-structure pruning (partial-order reduction, imported from model checking). Local rewrites on DISJOINT nodes commute: the search currently generates both orderings of every such pair and lets the transposition table eat the duplicate — paying full sympy price for the twin first. Canonical-order pruning refuses to GENERATE the non-canonical ordering: skip move m at state s when

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `make_move_filter(state: State, grand_expr)` — Canonical-order filter for expanding `state` (see module doc).
- `best_first(root, budget, prop, h, prune)`
- `main(n: int, budget: int) -> None`

### scripts/bench_compile.py
torch.compile impact benchmark: eager vs compiled vanilla vs compiled+lookup.

- `main() -> None`

### scripts/bench_control.py
Deconfounder for the hybrid 349/360: markov3 fixed-k3 (the engine.solve default) rerun on the same 24-cell matrix WITH today's new rules. The old markov3 reference (316) predates i_cyclic/i_unprod/ i_ansatz_exp/i_linear_basis/smoothing. If this control lands near 349, the operators explain the record and hybrid confidence adds ~nothing; if it lands well below, the LLM-gated k earns real credit.

- `class _Timeout`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_decoding.py
Benchmark vanilla greedy vs prompt-lookup vs speculative on real models.

- `vanilla_greedy(model, prompt_ids: list[int], max_new: int) -> tuple[list[int], int]`
- `timed(fn, warmup: int=1, repeats: int=3)`
- `main() -> None`

### scripts/bench_derivation.py
Rung-1 solve-rate bench + macro ablation (spec: macros earn a slot only if they win on solve-rate-per-node).

- `_solve_one(args) -> tuple[bool, int, int]` — Module-level worker (fork-pool picklable). Returns
- `run(levels: list[int], n: int, width: int, max_plies: int, max_nodes: int | None, use_macros: bool, kind: str, jobs: int | None=None, verify_p: float=1.0) -> None`

### scripts/bench_dispatch_race_v4.py
Dispatcher v4 adoption race: markov, policy, v3-routed, v4-routed on a fresh L3-L8 band. Bar (the FA Law): v4 must match the best arm's solves; wall breaks ties. Judgment-stack currency: v4 is the only router trained on the post-orbital engine (v3 predates i_sqrt_basis's log block and the trig(log) generators).

- `_route(disp_path: Path, expr)` — Replicates engine.solve's dispatcher gate (timeboxed probes).
- `_worker(arm: str, level: int, seed: int, q: 'mp.Queue') -> None`
- `main(n_per: int, seed_base: int) -> None`

### scripts/bench_distilled_draft.py
Distilled-draft speculative decoding: accept rate + tok/s, real models.

- `merge_all(model) -> None` — Fold every LoRALinear back into a plain Linear (zero overhead).
- `bench_spec(target, draft, ids, ref, label)`
- `main() -> None`

### scripts/bench_engine_regret.py
Engine-level regret: predict a DOOMED search from the live beam and abort early, banking the wall.

- `_worker(level: int, seed: int, budget: int, q: 'mp.Queue') -> None`
- `phase_labels(n_per_level: int, seed_base: int, out: Path) -> None`
- `_load(labels: Path)`
- `_xy(recs)`
- `phase_probe(labels: Path, epochs: int) -> None`
- `phase_sweep(labels: Path) -> None`

### scripts/bench_entropy_beam.py
Entropy-bonus beam selection (pre-registered, its own race).

- `class _Timeout`
- `_dist(a: list[float], b: list[float]) -> float`
- `diversity_select(candidates, width)` — Greedy max-min: seed with the eval-best, then repeatedly take
- `main(n: int, level: int, budget: int) -> None`

### scripts/bench_fib_restarts.py
Golden-angle restart diversity (Artin's fib thread, the legit version): restart i rotates a base ordering by the golden-angle stride (low-discrepancy: successive restarts maximally spread over orderings) vs iid random shuffles. Expectation calibrated by the Luby null: at 3 restarts, schedule/diversity effects may not bite. n=15, same seeds.

- `class _Timeout`
- `iid_proposer(seed: str)`
- `golden_proposer(seed: str, i: int)` — One fixed base shuffle per problem; restart i rotates it by the
- `run_restarts(root, budget, seed, kind_)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budgets: list[int]) -> None`

### scripts/bench_flash_prefill.py
Flash prefill (Metal) vs mx.fast.scaled_dot_product_attention, causal, prefill shapes. tq_tile sweep = the config-estimator rung's revival data (a config axis with real variance, unlike the 6-point GEMV space). mx.eval every timed iteration (the lazy-graph scar).

- `bench(f, it=50, warmup=10)`
- `main() -> None`

### scripts/bench_frontier.py
Expert-iteration curve point: engine-r1 (original proposer ckpt) vs engine-r2 (retrained on frontier harvest) on HELD-OUT L4 frontier-ish problems, prop3+HCE, budgets 100/200/400. Also the regression guard: quick L1-3 totals must stay within noise of r1's. Spec: 2026-07-07-expert-iteration-r2-design.md Task 3.

- `class _Timeout`
- `load_proposer(ckpt: str)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(r1: str, r2: str, n: int) -> None`

### scripts/bench_fused.py
The fused-architecture race (Artin's integration, 2026-07-08): bf with h = value head on the 0.5B trunk's hidden state, vs bf-nnue (20 hand features). Offline the trunk lost the ordering fight (+0.859 vs +0.937), but offline rho has under-predicted search before. Honest cost note printed per arm: the fused eval pays an LLM forward per node — if it wins solves but loses wall, that's the verdict too. Same cells/seeds as the 113/120 record races.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `load_fused(v2: bool=False)`
- `best_first(root, budget, prop, h)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, v2: bool=False) -> None`

### scripts/bench_fused_ce.py
Fused chunked CE vs naive full-logits CE at Qwen-0.5B head shapes.

- `run(fn, h, w, t, it=5)`
- `main() -> None`

### scripts/bench_gated.py
Policy-gated expansion race: does skipping un-predicted rule evaluations buy wall-time without costing solves?

- `class _Timeout`
- `make_gate(k: int, adaptive: bool=False)` — adaptive=True: Artin's 'the teacher can also participate' —
- `main(n_per: int, budget: int, k: int, adaptive: bool=False) -> None`

### scripts/bench_gweight.py
The Dijkstra component of Artin's Google-Maps analogy: our beam ranks by (almost) pure heuristic h; Dijkstra ranks by path cost g; A* by g+h. Sweep the g-weight (plies coefficient) in the eval: lambda in {0, 0.1 (current hce), 1, 5}. markov3 @ w2 engine, held-out seeds, n=15, budgets 25/50.

- `markov_proposer()`
- `eval_with_g(lam: float)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`
- `class _Timeout`

### scripts/bench_hints_ab.py
Adoption A/B: where do the step model's hints come from?

- `_sketch_worker(s: str, q) -> None`
- `make_predicted_hinter()` — Layer-15 probe as a drop-in for _hints_isolated (same
- `main(n_per: int, budget: int, seed0: int, arm_names: list[str]) -> None`

### scripts/bench_hybrid.py
The hybrid cell (Artin's distillation question, 2026-07-07): markov RANKS, the 0.5B GATES k. We measured LLM-ranks+LLM-gates (328/360) and markov-ranks+fixed-k3 (316/360); markov-confidence was a null. If ranking is grammar and the GPU's real contribution is confidence, the hybrid should approach the champion at zero LLM ranking cost — the strongest possible statement of "the GPU buys confidence, not choice."

- `class _Timeout`
- `load_score_fn()`
- `make_hybrid_proposer(score_fn)` — Rank by the bigram dict; attach LLM scores IN MARKOV ORDER so
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_int4_config_sweep.py
Config estimator for the int4 dequant-GEMV kernel (Artin's rung: "can't the kernel packing itself be estimated?"). The learned-autotuner recipe: sweep configs honestly, the sweep IS the training data, a tiny net predicts latency from (shape, config) features, and the config it picks per shape is scored by REGRET vs the exhaustive-sweep oracle on held-out shapes (the FA Law with zero indirection: the oracle is the wall clock). Precedent: TVM/Ansor cost models.

- `run_kernel(variant, x, packed, sc, mn, gs)`
- `bench(f, it=100, warmup=15)`
- `sweep() -> list[dict]`
- `fit_and_score(rows: list[dict]) -> None`

### scripts/bench_int4_gemv.py
Fused int4 dequant-GEMV vs the incumbents, decode shapes (M=1).

- `bench(f, it=200, warmup=20)`
- `main()`

### scripts/bench_interference.py
Path-integral-inspired eval (physics night 3): Feynman's sum over histories says the classical path emerges by CONSTRUCTIVE INTERFERENCE — many nearby paths agree there. Best-first is the classical limit (one extremal path) and throws the interference data away: the transposition table already counts how many distinct derivation orderings arrive at each state. Amplitude = arrival multiplicity. Arms: bf-nnue (incumbent) vs bf-nnue with h' = h - w*log2(1+arrivals) (re-scored on re-arrival; a state many derivations converge on is a natural waypoint). Paired arms, one run (the methodology rule).

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `best_first(root, budget, prop, h, interference_w=0.0)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budget: int) -> None`

### scripts/bench_ksweep.py
Depth-vs-breadth sweep (Artin's hypothesis, 2026-07-07: breadth can be synthesized — cf. LazySMP). Random pruning at k in {1,2,3,5} vs full enumeration vs k=1 x R randomized restarts at EQUAL total node budget. No model: the random proposer isolates pure depth/diversity effects from move-choice quality (which bench_proposer.py measures).

- `class _Timeout`
- `random_proposer(seed: str)`
- `restart_search(root, total_budget: int, restarts: int, seed: str)` — k=1 deep dives with different orderings, budget split evenly;
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `_sweep_one(args) -> bool` — Module-level worker: one (problem, config) cell entry. SIGALRM
- `main(n: int, budgets: list[int], jobs: int | None=None) -> None`

### scripts/bench_kv_quant_decode.py
Quantized-KV decode attention: does the roofline ~4x show up?

- `bench(fn, *args, repeats=200)`
- `main() -> None`
- `_unpack(codes, scale)`

### scripts/bench_ladder.py
0.5B capability ladder: cold vs LoRA-tuned accuracy per rung.

- `format_chat(tok, prompt)`
- `make_generate_fn(model, tok, max_new_tokens=96)`
- `encode_example(tok, task)`
- `batches(examples, pad_id, batch_size, device, epoch)`
- `main() -> None`

### scripts/bench_lazy.py
Lazy expansion vs the L4 total-work wall. The profile said there is no single stall: timeouts are death by a thousand sympy calls — every node pays ALL ~20 rules, then the prior keeps 3. But the prior ranks by RULE NAME, known before any work: consult it first, apply rules one at a time in prior order, stop at k children. Same selection, a fraction of the sympy. Arms: eager (incumbent) vs lazy, bf-nnue, int L4.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `make_rankers()`
- `eager_children(s, prop, k=3)`
- `lazy_children(s, rule_rank, k=3)`
- `best_first(root, budget, expand, h)`
- `main(n: int, budget: int) -> None`

### scripts/bench_llm_gating.py
LLM wall-time gating: the budget-allocation slot retargeted to the currency that actually binds (RESULTS: node budget never binds — the 5v3 timeout story says WALL TIME with LLM calls does).

- `class _Timeout`
- `main(n_per: int, wall: int, thresh: float) -> None`

### scripts/bench_lookup_static.py
Prompt-lookup + StaticCache + CUDA graphs: the stacked benchmark.

- `main() -> None`

### scripts/bench_luby.py
Luby restart schedule vs equal-thirds (Artin's budget-reallocation thread). Luby (1,1,2,1,1,2,4,...) is provably within a log factor of the optimal restart policy without knowing the difficulty distribution. Same seeds as every race. Refs: k1x3 got 267/360.

- `class _Timeout`
- `luby(i: int) -> int` — 1-indexed Luby sequence.
- `random_proposer(seed: str)`
- `restarts_equal(root, budget, seed)`
- `restarts_luby(root, budget, seed, unit)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budgets: list[int]) -> None`

### scripts/bench_magic.py
The magic detector (physics night 3: Liouville 1835 as the Gottesman-Knill of integration). sympy's Risch implementation can PROVE an integrand non-elementary in ~10ms on our death-state shapes. A state carrying a certified non-elementary Integral node is dead WITHIN OUR OPERATOR CLOSURE (no rule merges integral nodes, so split non-elementary siblings can never recombine — the mathematical loophole is closed by the move set). Pruning it is a theorem per cut, not a heuristic.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `_risch_dead(integrand: sp.Expr) -> bool` — True only on a POSITIVE non-elementarity certificate.
- `is_dead(state: State) -> bool`
- `best_first(root, budget, prop, h, magic)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budget: int) -> None`

### scripts/bench_markov.py
Markov prior IN-SEARCH race: rule-bigram scores (zero inference cost) driving propose_k=3 pruning, on the same held-out seeds as the proposer race. Reference totals (n=15): full+hce 265, rand3 277, prop3(LLM) 288, adapt-T0.1 300. If markov3 lands near 288, the LLM's pruning value is rule grammar and the wall-clock tax is optional.

- `class _Timeout`
- `build_prior()`
- `make_markov_proposer(unigram, bigram)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budgets: list[int], width: int=8) -> None`

### scripts/bench_markov_adaptive.py
The zero-GPU champion candidate: adaptive-k with MARKOV confidence. Entropy over the bigram's count-normalized distribution gates k (1..6). If this lands near adapt-LLM's 300/360, the entire champion engine needs no neural network at all. Reference totals (n=15): full 265, rand3 277, prop3-LLM 288, markov3 293, adapt-LLM 300.

- `class _Timeout`
- `build_prior()`
- `make_markov_adaptive(unigram, bigram, ks_seen)` — Scoring proposer + entropy policy in one: counts give both the
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budgets: list[int]) -> None`

### scripts/bench_metal_kernels.py
Benchmark llmopt Metal kernels vs unfused MLX ops and mx.fast.*

- `bench(fn, *args, repeats=200)`
- `main() -> None`

### scripts/bench_mlx_integration.py
End-to-end tokens/sec: stock mlx-lm vs llmopt fused-swiglu patch.

- `decode_tps(model, tok) -> float`
- `main() -> None`

### scripts/bench_nnue.py
The NNUE race: eval_fn=hce vs eval_fn=nnue inside the SAME search, held-out problems, fixed node budgets. Solve rate is the score — never training loss. Spec: 2026-07-07-nnue-eval-design.md.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_eval(path: str)`
- `_root(rng, level, kind)`
- `_check(kind, result_expr, truth)`
- `main(n: int, budgets: list[int], ckpt: str) -> None`

### scripts/bench_ode_engine.py
ODE engine rung 1 (the ENGINE-shaped physics rung; generator llmopt/mathgen/odes.py existed since the mathgen expansion but nothing ever consumed it).

- `_solve_int(expr)` — Subcontract an integral to the house engine; None if unsolved.
- `_engine_worker(kind: str, level: int, seed: int, q: 'mp.Queue') -> None`
- `_dsolve_worker(kind: str, level: int, seed: int, q: 'mp.Ueue') -> None`
- `run_arm(worker, kind, level, seed) -> dict`
- `main() -> None`

### scripts/bench_opcap.py
Cheap-simplify budgets (autopsy rung 4 candidate): the remaining int L4 failures are 10/11 WALL timeouts — expression-size economics, not missing operators. Lever: size-cap pruning — children whose count_ops exceeds cap are discarded before their sympy costs are paid. Arms: no cap / 300 / 150. Reports solves AND timeout counts per arm. bf-nnue + markov3 (the champion structural config), int L4, budget 400.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `best_first(root, budget, prop, h, cap)`
- `_root(rng, level)`
- `main(n: int, budget: int) -> None`

### scripts/bench_population.py
Population LoRA (K adapters, one frozen base) vs K sequential runs.

- `load_model()`
- `make_data(k_total: int, step: int, rows: slice)` — One shared deterministic pool per step; arms take row slices —
- `train_arm(k: int, k_total: int, rows: slice, label: str)`
- `main() -> None`

### scripts/bench_pred_syndromes.py
Predicted syndromes: learn the Hints line, skip the mini-solve.

- `_label_worker(states: list[str], q) -> None`
- `phase_label() -> None`
- `_gen_worker(jobs: list[tuple[int, int]], q) -> None`
- `phase_label_gen(n_per: int=400) -> None` — Widen the label set with fresh generator roots, L2-L8.
- `phase_train() -> None`
- `_orbital_worker(states: list[str], q) -> None` — The generator sketch i_linear_basis would enumerate — atoms
- `phase_orbitals() -> None`
- `phase_train_emb(enrich: bool=False) -> None` — Round 3: frozen 0.5B embeddings as features (same bar/split).
- `phase_train_lora() -> None` — Round 5: LoRA-tune the encoder itself (frozen embeddings were

### scripts/bench_prefix_reuse.py
Radix prefix KV reuse on a real model: TTFT with a shared long prefix.

- `prefill_times(model, prompts, prefix_cache)` — Run prompts sequentially through the engine, timing each request's
- `main() -> None`

### scripts/bench_proposer.py
The proposer race: full enumeration vs model-proposed top-k vs random-k control, under HCE and NNUE evals, held-out problems. Solve rate at fixed node budget is the score; proposer inference time is wall clock, reported separately. Spec: 2026-07-07-move-proposer-design.md.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path='checkpoints/nnue_eval.pt')`
- `load_proposer(ckpt='checkpoints/proposer_lora.pt')`
- `random_proposer(seed_tag: str)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budgets: list[int]) -> None`

### scripts/bench_quant_schemes.py
Three-lane 4-bit quantization race on REAL model weights, scored in FUNCTION space (the house law: never score weights by weight distance).

- `capture(n_layers: int=6)` — fp32 weights + real input activations for a spread of linears.
- `_group(w)`
- `quant_uniform(w)`
- `_nf4_codes()`
- `quant_nf4(w)`
- `quant_awq_lite(w, x)`
- `main()`

### scripts/bench_record.py
The record attempt: every proven component in one search, first time. Best-first frontier (beat the beam 103v91) + NNUE h (113/120) + markov ranking (choice is grammar) + LLM entropy-gated k (the +15 confidence premium behind 349/360) + magic pruning (Liouville, replicated +1). Full 24-cell matrix, same seeds as every race. Standing record: hybrid beam 349/360 (96.9%), n=30-confirmed 694/720.

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `load_score_fn()`
- `record_search(root, budget, mk_prop, score_fn, k_policy, h)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int) -> None`

### scripts/bench_regret_resample.py
Regret-gated resampling (2026-07-11, Artin's thesis: 'the best skill is knowing when to regret/reconsider').

- `_gen_isolated(level: int, seed: int, wall: int=45)` — Fork-isolated make_integrate (sympy pathology #7, FIFTH call
- `_checked(problem, text) -> bool` — FORK-ISOLATED oracle call (sympy pathology #10, 2026-07-12):
- `load_model()`
- `build_prompt(tok, problem) -> str`
- `sample_with_states(tok, model, prompt: str, seed: int, abort_check=None)` — One sampled completion; returns (text, states, aborted,
- `phase_labels(n_problems: int, k: int, seed_base: int, out: Path) -> None`
- `phase_probe(labels: Path, epochs: int) -> None`
- `phase_race(n_problems: int, k: int, seed_base: int, thresh: float) -> None`
- `phase_pool(n_problems: int, seed_base: int, pool: int, out: Path) -> None` — Round 2, farm half: FULL traces only (no aborts), logging every
- `phase_sweep(problog: Path, k: int) -> None` — Round 2, judgment half: replay abort policies (threshold x

### scripts/bench_rotate_quantize.py
Rotation vs RTN quantization error (spec 2026-07-06, part a).

- `real_layers()`
- `synthetic()`
- `pad_pow2(w)` — Column-pad to the next power of 2 so hadamard applies; padding
- `main() -> None`

### scripts/bench_rule_basis.py
The universal-gate-set question (Artin, from Toffoli universality): what is the minimal rule basis that still generates our derivations? Leave-one-out ablation of every INT rule from the champion structural config (bf-nnue + markov3): a rule whose removal costs nothing is a non-generator (a dead gate — cf. d_quotient); the survivors are the domain's gate set. Runs the full-rules arm first as the paired baseline (methodology rule: one run, one machine state).

- `class _Timeout`
- `class NnueEval` (forward)
- `load_nnue(path: str)`
- `best_first(root, budget, prop, h, only_rules)`
- `_root(rng, level)`
- `main(n: int, budget: int) -> None`

### scripts/bench_stack_winners.py
Do the timeout campaign's winners COMPOSE? Lazy expansion (+2 solves, timeouts 4v10) and the magic detector (+1, 71 certified cuts) won independently; engine.solve() integration wants the interaction term. Four arms, paired, one run: classical / lazy / magic / both. bf-nnue + markov, the hard cells.

- `class _Timeout`
- `best_first(root, budget, expand, h, magic)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n: int, budget: int) -> None`

### scripts/bench_stacked.py
Full-stack benchmark: radix prefix reuse + prompt-lookup + CUDA graphs.

- `diagnose_divergence(model, ref, out)` — At the first mismatch, measure the eager logit margin between the
- `timed(fn)`
- `main() -> None`

### scripts/bench_static.py
Static KV cache + CUDA graphs benchmark.

- `greedy_static(model, prompt_ids: list[int], max_new: int, *, compiled_step=None)` — Greedy decode with StaticCache: prefill once, then 1-token steps.
- `bench(fn, warmup=1, repeats=3)`
- `main() -> None`

### scripts/bench_step_diversity.py
Resample diversity at stuck states: is validity starved by sampling REDUNDANCY?

- `fewshot_rotations(fewshot: str) -> list[str]` — Rotate example blocks (split on blank lines); the instructions
- `main() -> None`

### scripts/bench_step_tokens.py
Bigger tokens for the LLM (Artin, 2026-07-12): the unit of generation becomes a whole derivation STEP — model emits a candidate rewrite of the current integral as sympy text, the ORACLE verifies the step before it stands (equivalence via differentiation, fork-safe), invalid steps are resampled. Regret at step boundaries, where the unit economics are verified-macro-token-sized, vs one-shot answers at equal token budget.

- `_hints_isolated(cur_s: str, wall: int=15) -> list[str]` — Rule-fire syndrome of a (possibly MODEL-written) expression —
- `_verify_step(prev_s: str, cand_s: str, q: 'mp.Queue') -> None` — Child: parse + equivalence check (diff both, simplify difference
- `verify_step(prev_s: str, cand_s: str, wall: int=15)`
- `load(adapter: str | None=None)`
- `_expr_mask(tok)` — Charset-constrained decoding v1 (Artin's GO after round 1:
- `sample(tok, model, prompt: str, seed: int, constrain: bool=False) -> tuple[str, int]`
- `sample_batch(tok, model, prompt: str, seeds: list[int], constrain: bool=False, temps: list[float] | None=None, return_logps: bool=False)` — B parallel sampled completions of the same prompt — the step
- `_gen_isolated(level: int, seed: int, wall: int=45)`
- `solve_chain(tok, model, integ: str, budget: int, seed0: int)` — Oracle-gated chain; returns (solved, verified_pairs,
- `main(n: int, seed_base: int, budget: int, adapter: str | None=None) -> None`

### scripts/bench_stitch_poc.py
Representation stitching, tier 1 (Artin's change-of-basis riff).

- `pooled_layers(model_name, texts, layers, dev, bs=16, ml=384)`
- `probe(X, Y, tr, te, seed=0)`
- `main() -> None`

### scripts/bench_syndrome_head.py
Syndrome head, payoff 3 (the re-aimed spec): does TRAINING-time rule-awareness improve step validity, with nothing said at inference?

- `build_examples(tok)` — phase_train's example building + class balance, verbatim, plus
- `train_arm(lam: float, out: Path) -> None`
- `evaluate(adapter: str)`
- `main() -> None`

### scripts/bench_syndrome_policy.py
Live race: syndrome-policy proposer vs markov prior (the current zero-NN engine's brain) at identical beam config. Pre-registered bar: policy arm solves >= markov arm on fresh problems (solve ties broken by wall-time — the policy costs ms/node vs the prior's ~0, so it must BUY something).

- `class _Timeout`
- `load_policy()`
- `make_policy_proposer(net, p)`
- `main(n_per: int, budget: int) -> None`

### scripts/bench_temp_race.py
Solve-level race: temperature ladder vs const 0.7 in real chains.

- `main(n_per: int, budget: int, seed0: int) -> None`

### scripts/bench_tree_verify.py
Tree verify vs linear prompt-lookup benchmark.

- `main() -> None`

### scripts/bench_triton_kernels.py
Benchmark llmopt Triton kernels vs unfused torch ops and torch SDPA.

- `bench(fn, *args, repeats=200)`
- `main() -> None`
- `bench_paged() -> None`

### scripts/bench_verify_fast.py
Fast wave-verifier: the three lossless levers, parity-benched.

- `_wave_worker(prev_s: str, cands: list[str], q) -> None` — One fork verifies a whole wave; verdicts streamed per candidate
- `verify_wave(prev_s: str, cands: list[str], wall: int=20) -> dict[str, tuple[bool, bool]]` — Levers 1+2: cache, then one streamed fork for the misses.
- `_battery()`
- `main() -> None`

### scripts/bench_vge.py
Variational ground-state engine, rung-1 race (spec 2026-07-12).

- `main() -> None`

### scripts/bench_weight_anatomy.py
Weight anatomy: do closed-system (RL-climbed) weights LOOK different from imitation (SFT) weights?

- `composed(sd)` — {layer_idx, module: BA weight-space delta}
- `depth_profile(deltas)`
- `stable_rank(deltas)`
- `main() -> None`

### scripts/bench_zx.py
T-count rung-2 race: best-first over ZX rewrites (primitives + gadget macros + macro-greedy) vs greedy full_reduce, per-circuit. Pre-registered bar (spec): search beats greedy on >= 20% of seeded circuits, else the greedy oracle wins the domain and we say so. Outputs win/tie/loss on T-count, mean T per arm, and tensor verification on every search result (<= 8 qubits).

- `class _Timeout`
- `main(n: int, qubits: int, depth: int, budget: int, seed: str) -> None`

### scripts/bench_zx_r3.py
ZX rung 3: structured circuits + markov prior (spec ladder).

- `class _Timeout`
- `_toffoli_manual(c: Circuit, a: int, b: int, t: int) -> None`
- `structured_toffoli(qubits: int, n_tofs: int, rng: random.Random)`
- `rule_of(label: str) -> str`
- `extractable_tcount(state: ZXState) -> 'int | None'` — Rung 4's eval: T-count of the EXTRACTED circuit — the only
- `bf_extract(g0, budget: int, prior: 'dict | None'=None, k: int=3)` — Best-first on extractable T-count. Unextractable states may be
- `bf_markov(g0, budget: int, prior: dict, k: int=3)` — Best-first with bigram-ranked top-k expansion (the 293-dict,
- `harvest(n: int, qubits: int, n_tofs: int, budget: int) -> None`
- `race(n: int, qubits: int, n_tofs: int, budget: int) -> None`

### scripts/bench_zx_r5.py
ZX rung 5: phase-polynomial machinery (the literature's greedy-beater).

- `class _Timeout`
- `_teleported_circuit(c: Circuit) -> Circuit`
- `run_arm(arm: str, c: Circuit)`
- `main(n: int, qubits: int, tofs: int) -> None`

### scripts/bench_zx_r6.py
ZX rung 6: composition — does SEARCH around the phase-teleport macro beat the bare pipeline?

- `class _Timeout`
- `main(n: int, qubits: int, tofs: int, budget: int) -> None`

### scripts/bench_zx_r7.py
ZX rung 7: push the phase-teleport win — markov prior on the new move set, then bigger Toffoli nets.

- `class _Timeout`
- `harvest(n: int, qubits: int, tofs: int, budget: int) -> None`
- `race(n: int, qubits: int, tofs: int, budget: int) -> None`

### scripts/book.py
Programmatic booking — the /book ritual as a refusing machine.

- `class Refusal`
- `validate_marker(marker_path: Path) -> dict` — Fence 1: only clean, finished runs book. Absence is 'never
- `validate_gate_checksum(entry: str, marker: dict) -> None` — Fence 2: gate numbers book as DICTS, not totals — the dict
- `validate_weights_sha(entry: str, marker: dict) -> None` — Fence 3: a gate books WITH its weights sha (provenance rule
- `validate_statistical_fence(entry: str, marker: dict, entry_type: str, fence_acknowledged: bool) -> None` — Fence 4 (resolution law 2026-07-31): gate deltas < 1.5 sigma
- `append_entry(heading: str, entry: str) -> None` — Append-only, house heading format (SKILL.md step 1).
- `regen_index() -> None` — CALL the frozen results-cited script — never reimplement.
- `curate_index_row(heading: str, threads: list[str], links: list[str]) -> dict` — SKILL.md step 3: set threads/links on the new row, pop
- `main(argv: list[str] | None=None) -> int`

### scripts/build_gen7_diet.py
Gen-7 mass-targeted diet (Rung A of the epoch killer).

- `main() -> None`

### scripts/calibrate_hce.py
HCE calibration: does hce(state) predict solvability? (spec: 2026-07-06-hce-calibration-design.md — the chess-eval question, measured.)

- `spearman(xs: list[float], ys: list[float]) -> float` — Spearman rank correlation, average ranks for ties. Inline to
- `_root(rng: random.Random, level: int, kind: str) -> sp.Expr`
- `sample_states(levels: list[int], per_level: int, max_states: int, kind: str) -> list[State]` — On-policy: every candidate generated by real searches, deduped.
- `class _ProbeTimeout`
- `_alarm(signum, frame)`
- `probe(state: State) -> tuple[int | None, bool, bool]` — (nodes-to-solve or None, solved@small, timed_out).
- `main(levels: list[int], per_level: int, max_states: int, kind: str) -> None`

### scripts/ckpt_manifest.py
ckpt_manifest.py — checkpoint manifest for the curated tree.

- `sha256(path, bufsize=1 << 20)`
- `scan(base)`
- `main()`

### scripts/consolidate_mathnative.py
Self-distillation consolidation (post-climb strategy item B).

- `main(src: str, out: str, lr: float, cap: int, d: int, layers: int, ffn: int, heads: int, seed: int) -> None`

### scripts/control_round.py
Control round: retrain on the EXACT rounds-2/3 diet, gate it.

- `main() -> None`

### scripts/convert_diet_prefix.py
Materialize the gen-4 diet as paired prefix/infix jsonl files — native-transformer rung 1 (spec 2026-07-25-native-transformer).

- `main() -> None`

### scripts/eval_mathnative.py
Phase-1 gate: does the from-scratch 19M reach 1% step validity at L2-3 (the 0.5B's historical starting point)?

- `sample_wave(model, tok, prompt_ids, seeds, dev, max_new=120)`
- `_diet_roots() -> set[str]` — Every cur string in the training diet (whitespace-stripped) —
- `main(ckpt: str, levels: tuple[int, ...], unseen: bool, d: int=384, layers: int=8, ffn: int=1536, heads: int=6) -> None`

### scripts/eval_pruned_moe.py
Accuracy-vs-pruning chart for a routing-masked MoE (MLX).

- `evaluate(model, tok, problems) -> float`
- `main() -> None`

### scripts/eval_ruler.py
RULER long-context eval against a real HF model.

- `main() -> None`

### scripts/expert_iter_steps.py
Step-level expert iteration (the repo's founding long-term goal, first concrete round; step-tokens measured 5/30 vs one-shot 0/30 at 5% step validity — this trains the 5%).

- `_chain_worker(level: int, seed: int, q: 'mp.Queue') -> None`
- `phase_chains(n_per_level: int, seed_base: int, levels=(2, 3, 4, 5), min_pairs: int=1, append: bool=False) -> None` — min_pairs: keep only chains with >= this many steps — round 1
- `_reverse_worker(level: int, seed: int, q: 'mp.Queue') -> None` — The REVERSE ENGINE (Artin, 2026-07-12): make_integrate draws
- `phase_reverse(n_per_level: int, seed_base: int, levels=(4, 5, 6, 7, 8)) -> None`
- `_magic_buckets(states: list[str]) -> dict` — Fork-isolated IN CHUNKS (one wedging state must not poison the
- `_magic_chunk(states: list[str]) -> dict`
- `_coeff_worker(seed: int, q: 'mp.Queue') -> None` — Coefficient-discipline pairs (round 5; the Arena's finding
- `phase_coeff(n: int, seed_base: int) -> None`
- `_ode_chain_worker(kind: str, seed: int, q: 'mp.Queue') -> None` — Second continent (2026-07-13, the closed-system thesis): ODE
- `phase_ode_chains(n: int, seed_base: int, out_path: str='data/ode_chains.jsonl') -> None`
- `phase_skips() -> None` — Macro-distillation (Artin's COCONUT riff, 2026-07-12): skip
- `phase_train(epochs: int, lr: float, out: Path=ADAPTER) -> None`

### scripts/expert_loop.py
Autonomous expert-iteration loop driver (spec: docs/superpowers/specs/2026-07-12-step-expert-iteration-design.md). Round = evaluate -> mine -> train -> gate; state on disk; tripwires halt the loop. All sympy touches forked (pathologies #7/#8/#10).

- `evaluate(tok, model, levels, n_per, seed_base, budget=768)` — Frontier scan: solve rate per level (stop below 20%), overall
- `frontier(sb: dict, n_per: int) -> int` — Highest level in the 20-80% solve band; else highest evaluated.
- `gate_verdict(prev: dict, new: dict, frontier: int) -> tuple[bool, str]` — PROMOTE iff no level <= frontier regresses by more than 2
- `mine_round(round_no: int, F: int, sb: dict, seed_base: int, n_mine: int=60) -> tuple[int, int]` — On-policy chains from evaluation + engine chains at F (and F-1
- `run_round(round_no: int) -> str`
- `main(max_rounds: int) -> None`

### scripts/farm_algebra.py
Curriculum v2: farm the algebra/simplification shard (riff ledger 2026-07-15, staged curriculum pretraining — the L4-starvation fix).

- `_poly(rng, deg, cmax=9)` — Random polynomial with small nonzero-lead integer coeffs.
- `_opaque(rng, level)` — Opaque composition factor (never expanded): f(inner poly).
- `gen(family, level, i)`
- `main() -> None`

### scripts/farm_l4_calc.py
Curriculum v2.1: L4-targeted calculus shard (the residue fix).


### scripts/farm_v22.py
v2.2 diet farm — the autopsy-aimed shard (2026-07-17).

- `in_language(text: str) -> bool`
- `main(levels, n_per, part, parts, out, oneply_levels=(4, 5), oneply_cap=ONEPLY_CAP_FRAC) -> None`

### scripts/figlib.py
SUPERSEDED 2026-08-11 by llmopt/lab/figstyle.py + llmopt/lab/figures.py.

- `_color(name: str, i: int) -> str`
- `_save(fig, name: str, png: bool=False) -> Path`
- `_style(ax, title: str, ylabel: str)`
- `grouped_bars(name: str, bins: list[str], series: dict[str, list[tuple[int, int]]], title: str='', png: bool=False) -> Path` — series: label -> [(solved, total) per bin]. Percent bars,
- `lines(name: str, xs: list, series: dict[str, list[float]], title: str='', xlabel: str='', ylabel: str='', png: bool=False) -> Path` — series: label -> y values over shared xs. Direct end-labels,

### scripts/gen_catalog.py
gen_catalog.py — regenerate data/catalog/models.jsonl (EXHAUST, not evidence).

- `cited_names(repo_root)`
- `walk_targets(ckpt_root)`
- `load_jsonl(path)`
- `cross_check_manifest(repo_root, rows_by_path)` — Raise on sha disagreement with the frozen manifest (confirmed/).
- `main(argv=None)`

### scripts/gen_codemap.py
Generate docs/CODEMAP.md: the move-gate inventory of scratch/ and scripts/ (adopted from the Grok structure review, 2026-08-06). One row per file: doc citations (RESULTS/REPRODUCE/BOARD/FINDINGS/handoffs/ specs), in-code references (imports + literal path strings), a mechanically derived class, and the filename family. The class ladder is observable-facts-only, no curation:

- `_tracked() -> set[str] | None` — Repo-relative paths git knows about, or None if git is unusable.
- `_is_tracked(f: Path) -> bool`
- `collect_files(base: str, pat: str) -> list[Path]`
- `load_texts(paths: list[str]) -> dict[str, str]`
- `load_code() -> dict[str, str]`
- `family(name: str) -> str`
- `code_refs(target: Path, code: dict[str, str]) -> tuple[list[str], list[str]]` — (importers, mention-only referrers) for the module.
- `doc_cites(name: str, docs: dict[str, dict[str, str]]) -> dict[str, int]`
- `classify(cites: dict[str, int], imports: list[str]) -> str`
- `main() -> None`

### scripts/gen_dispatch_labels.py
Dispatcher-net labels: which brain wins each problem (2026-07-10, chasing the router's oracle ceiling — 127/130 vs threshold's 124).

- `_syndromes(expr: sp.Expr) -> list[float]`
- `_worker(job, q)`
- `main(n_per: int, seed_base: int, workers: int, out: Path) -> None`

### scripts/gen_dispatch_labels_v2.py
Dispatcher v2 labels: disagreement-oversampled farming.

- `_syndromes(expr: sp.Expr) -> list[float]`
- `_run(root, prop)`
- `_worker(job, q)`
- `main(n_per: int, seed_base: int, workers: int, out: Path, levels: list[int] | None=None) -> None`

### scripts/gen_figures_web.py
Render the web figures: SVG from docs/figures.json, PNG via Chrome.

- `page(svg: str, w: int, h: int) -> str` — Wrap the SVG with @font-face pointing at the vendored files, so
- `capture(svg: str, w: int, h: int, png: Path) -> bool`
- `recount_findings() -> None` — Recount FINDINGS by maturity tag and write it back into the spec.
- `main() -> None`

### scripts/gen_frontier.py
Magic-maximizing generation (frontier mining): draw a large candidate pool, score each with the estimator (microseconds), keep the ones predicted HARD-BUT-SOLVABLE, and measure whether selection actually concentrated difficulty.

- `_draw(chunk, q)`
- `_solve_worker(item, q)`
- `_solve_batch(items, workers)`
- `main(pool: int, keep: int, workers: int, out: Path, seed_base: int=970000) -> None`

### scripts/gen_index.py
Generate scripts/INDEX.md: one entry per python file in scripts/, scratch/, and llmopt/ — module docstring first paragraph + top-level function/class signatures (AST, no imports executed). Run after adding scripts so future sessions grep one file instead of re-reading (or re-writing) code that already exists.

- `sig(fn: ast.FunctionDef) -> str`
- `entry(path: Path) -> str | None`
- `main() -> None`

### scripts/gen_lake.py
Regenerate the Parquet lake (data/lake/) from the lab's jsonl/file exhaust.

- `main() -> int`

### scripts/gen_magic_labels.py
Magic-estimator labels: (root features, ground-truth hardness).

- `_worker(level: int, seed: int, budget: int, q: 'mp.Queue') -> None`
- `solve_isolated(level: int, seed: int, budget: int, wall: 'int | None'=None) -> 'dict | None'`
- `_run_parallel(jobs, walls, budget, f, workers: int) -> int` — N isolated workers at once — labeling is embarrassingly
- `_estimator_order(jobs: list) -> 'tuple[list, dict]'` — Artin's active-labeling move (2026-07-09): the estimator
- `main(per_level: int, budget: int, out: Path, levels, seed_base: int=700000, guided: bool=False, workers: int=1) -> None`

### scripts/gen_policy_labels.py
Per-state syndrome-policy labels (the qLDPC decoder, generalized from the root to EVERY node of the winning derivation).

- `_syndromes(expr: sp.Expr) -> list[float]`
- `_worker(row: dict, q: 'mp.Queue') -> None`
- `main(labels: Path, out: Path, workers: int, include_unsolved: bool=False) -> None`

### scripts/gen_proposer_data.py
Winning-path (state, legal moves, chosen move) triples for proposer SFT. Every row is verifier-approved: it comes from a SOLVED search, so the chosen move provably leads to a solution. Spec: 2026-07-07-move-proposer-design.md.

- `class _Timeout`
- `_root(rng, level, kind)`
- `path_rows(root: sp.Expr) -> list[dict]` — Replay the winning history move-by-move, recording the legal
- `main(per_cell: int, split: str, exclude_file: str | None) -> None`

### scripts/gen_readme.py
Rewrite generated regions in README.md from ledger truth.

- `counts() -> dict[str, int]`
- `render() -> str`
- `main() -> int`

### scripts/gen_regret_labels.py
Regret/corrective labels (DAgger-style, Artin's 'make it regret the wrong node' — hindsight credit assignment made mechanical).

- `_syndromes(expr)`
- `_worker(job, q)` — Policy-guided search over one problem; every VISITED state gets
- `main(n_per: int, workers: int, out: Path, levels: list[int] | None=None, seed_base: int=980000) -> None`

### scripts/gen_results_index.py
Generate/refresh docs/results-index.jsonl from RESULTS.md.

- `slug(title, date)`
- `infer_type(title)`
- `infer_threads(title)`
- `extract_files(body: str) -> list[str]` — Sorted unique repo paths cited in an entry body.

### scripts/gen_scoreboard.py
Generate docs/SCOREBOARD.md from results-index.jsonl — the curated current-truth view (live, verdict-bearing entries grouped by thread, newest first). NEVER hand-edit SCOREBOARD.md; deepen the index instead and regenerate.


### scripts/gen_syndrome_labels.py
Syndrome-decoder labels (Artin's qLDPC riff, 2026-07-09): the rule-fire bits are syndrome extraction (cheap local checks that localize how a state deviates from the solvable subspace); a CODE also decodes — syndrome pattern -> which correction to apply. Here: re-solve known-solved problems recording the FIRST RULE of the winning derivation, so a tiny net can learn syndrome -> opening move.

- `_worker(row: dict, q: 'mp.Queue') -> None`
- `main(labels: Path, out: Path, workers: int) -> None`

### scripts/grow_mathnative.py
Gen-6 arm B: grow the champion 45M -> ~55M, function-preserving.

- `main() -> None`

### scripts/harvest_champion.py
Champion harvest: winning paths from the CURRENT best structural engine (engine.solve: markov3 @ w2, all autopsy rules, smoothing) on fresh problems. Motivation: the prior-pollution null — a mined prior inherits the policy quality of its paths, so the proper re-mine needs paths from an engine at least as strong as the prior's user, and no such harvest exists post-rules. Output rows feed the prior re-mine (and future proposer training).

- `class _Timeout`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `path_rows(root, history)`
- `main(per_cell: int) -> None`

### scripts/harvest_frontier.py
Expert-iteration round 2, harvest phase (spec: 2026-07-07-expert-iteration-r2-design.md).

- `class _Timeout`
- `load_proposer()`
- `random_proposer(seed: str)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `_run(root, wall, **kw)`
- `path_rows(root, history)`
- `main(per_cell: int) -> None`

### scripts/list_uncurated.py
Print the oldest uncurated ledger entries (candidates for FINDINGS).

- `main(n: int=20) -> None`

### scripts/log_hygiene.py
Print-only log hygiene planner (reviewer design, handoff 2026-08-11-0).

- `build_citation_set(results_path: Path) -> set[str]` — One pass over RESULTS.md: every `logs/...` token becomes a citation.
- `is_cited(rel: str, cites: set[str], basenames: set[str] | None=None) -> bool` — True if rel or any parent dir of rel appears in the citation set,
- `cited_basenames(results_path: Path) -> set[str]` — Basenames of every *.log/*.jsonl-looking token in RESULTS,
- `is_receipt(name: str) -> bool`
- `classify_one(rel: str, mtime: float, cites: set[str], age_days: float, now: float | None=None, basenames: set[str] | None=None) -> tuple[str, str]` — Return (class, reason) for one logs/-relative path.
- `scan(root: Path, cites: set[str], age_days: float, basenames: set[str] | None=None) -> list[dict]`
- `consolidation_map(root: Path) -> list[dict]` — Grep scripts/ + scratch/ (top level, minus scratch/leancheck) for
- `print_plan(rows: list[dict], cmap: list[dict], out=sys.stdout) -> None`
- `main(argv=None) -> int`

### scripts/markov_eval.py
Absorbing-Markov eval (Artin's Markov thread, part 2): bucket states by coarse structure, estimate P(solve | bucket) from fast probes, use -P(solve) as eval_fn. A probability-theoretic eval raced against HCE's hand-tuned weights — both model-free.

- `class _Timeout`
- `build_markov_proposer()`
- `bucket(expr: sp.Expr) -> tuple` — Coarse structural key: (n_unsolved, ops-quartile, deepest-kind).
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `main(n_probe: int, n_race: int, with_nnue: bool=False) -> None`

### scripts/markov_prior.py
Markov bigram move-prior (Artin's question: 'can't Markov chains predict certain things?'). Rule-name bigram from winning paths: after rule R at the parent ply, which rule tends to win next? Zero neural nets, a dictionary of counts — the embarrassingly-cheap control for the O1 distillation question: if this matches the LLM proposer's move accuracy, the 0.5B is mostly memorizing rule GRAMMAR, not reading expressions.

- `rule_of(label: str) -> str`
- `main() -> None`

### scripts/mine_highways.py
Highway mining (Artin's contraction-hierarchy analogy): recurring rule n-grams in winning paths = macro-move candidates with traffic data. Analysis only — promotion to actual macros is a future spec.


### scripts/mine_prior_update.py
Re-mine the markov prior after adding rules (2026-07-10).

- `class _Timeout`
- `main(n_per: int, seed_base: int, wall: int) -> None`

### scripts/moe_router_stats.py
Measure MoE router domain bias: math prompts vs general prose.

- `general_corpus(n)`
- `math_corpus(n)`
- `instrument(model, n_experts)` — Patch the sparse-MoE block CLASS so every forward also records
- `run_corpus(model, tok, prompts, stats, state)`
- `main() -> None`

### scripts/plot_gt1_crest.py
GT-1 crest small-multiples — the gallery Wanted figure (2026-08-08).

- `_git_head() -> str`
- `main() -> None`

### scripts/plot_identity_crest.py
The identity-era figure the gallery lacked (2026-08-08 pass).

- `main() -> None`

### scripts/plot_neurons.py
Neuron-geometry plots for the micro-model program (docs/assets).

- `sha8(path: str) -> str`
- `repo_head() -> str`
- `provenance_line(ckpts) -> str`
- `torch_svd_top2(X)`
- `neuron_matrix(ckpt: str, key_sub: str)`
- `project(W, method: str)`
- `scatter(ax, xs, ys, mag, title, cmap, vmin=None, vmax=None)`
- `main() -> None`

### scripts/probe_depth.py
Depth anatomy: WHERE in the stack does the rewrite decision form?

- `main(ckpt: str, d: int, layers: int, ffn: int, heads: int, n: int) -> None`

### scripts/render_hero_neurons.py
CLI for the README hero: llmopt.lab.anatomy dot views.

- `main() -> None`

### scripts/results_query.py
Query docs/results-index.jsonl (the RESULTS.md index).

- `repro(rows, entry_id: str) -> int`
- `show(e, mark='')`

### scripts/rjob.py
rjob — job-ID-based remote/local run management (2026-08-01).

- `sh(cmd)` — One transport call. Local: bash here. Remote: via wsl.sh run.
- `check_jid(jid)`
- `launch(jid, cmd)`
- `status()`
- `tail(jid, n='20')`
- `kill(jid)`
- `clean(jid)`
- `main()`

### scripts/sol_enrich_results.py
Build Sol's maturity-enriched, read-only copy of the results index.

- `_sections(entries: list[dict]) -> dict[str, str]`
- `_refs(value) -> list[str]` — The curated index contains both scalar and list link fields.
- `_topic(title: str) -> str`
- `_resolved_preregs(entries: list[dict]) -> dict[str, str]`
- `_evidence(text: str, pattern: re.Pattern, fallback: str) -> str`
- `_is_self_retraction(title: str) -> bool` — Whether retirement language applies to this entry, not its object.
- `_retires_amended_target(title: str) -> bool` — Whether this entry acts on an amended target's standing.
- `_impact(entry: dict) -> int` — Transparent ranking proxy, not a scientific importance judgment.
- `enrich() -> list[dict]`
- `write_summary(entries: list[dict]) -> None`
- `main() -> None`

### scripts/sol_generate_tables.py
Generate adoptable read-only tables from Sol's enriched index.

- `_refs(value) -> list[str]`
- `_entry_row(entry: dict) -> str`
- `main() -> None`

### scripts/step_grpo.py
GRPO at the frontier band — sustained RL over verified steps.

- `collect_groups(tok, model, n_groups: int, seed0: int)` — Walk chains with the current policy; keep mixed waves.
- `logp_new(tok, model, group, device)` — Teacher-forced logp of each stream's completion under the
- `gate_eval(adapter: str)`
- `main(cycles: int, groups_per_cycle: int=GROUPS_PER_CYCLE, skip_baseline: bool=False, start_from: str='checkpoints/step_lora.pt') -> None`

### scripts/step_grpo_micro.py
Phase 2: GRPO from birth — the math-native 19M climbs on the Mac.

- `sample_wave_lp(model, tok, prompt_ids, seeds, dev, max_new=120)` — KV-cached (2026-07-22): token-identical to the eager
- `collect(model, tok, dev, n_groups, seed0)`
- `logp_new(model, tok, g, dev)`
- `gate_eval(model, tok, dev, n=None)` — Honest chain gate. n<GATE_N = cheap proxy tier (same seeds,
- `main(cycles: int, src_path: str | None=None, out_path: str | None=None, d: int=384, layers: int=8, ffn: int=1536, heads: int=6, groups_n: int=GROUPS) -> None`

### scripts/sweep_lookup.py
Sweep prompt-lookup hyperparams (max_ngram x num_draft) on the StaticCache + CUDA graphs path.

- `main() -> None`

### scripts/sweep_lookup_mlx.py
Sweep prompt-lookup hyperparams (max_ngram x num_draft) on MLX.

- `greedy_reference(model, ids: list[int], n: int) -> list[int]`
- `main() -> None`

### scripts/tabula_rasa_r0.py
Tabula rasa round 0 (spec: 2026-07-07-tabula-rasa-design.md): the AlphaZero-way lineage's first harvest. NO hand-crafted knowledge: random k=1 dives with restarts, eval = count_ops ONLY (no unsolved- weighting — that's HCE knowledge), no proposer, no NNUE. Only the verifier survives (the game rules). Winning paths from whatever random search solves become the from-scratch lineage's first training data.

- `class _Timeout`
- `count_ops_eval(state: State) -> float`
- `random_proposer(seed: str)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `solve_r0(root, seed)`
- `path_rows(root, history)`
- `main(per_cell: int) -> None`

### scripts/tabula_rasa_r1.py
Tabula rasa round 1 (spec: 2026-07-07-tabula-rasa-design.md): the from-scratch lineage's first expert-iteration step. The proposer trained ONLY on round-0 random-search wins (proposer_tr_r1.pt) drives the search; eval stays count_ops (no HCE/NNUE — knowledge must come from the lineage's own data, only the verifier is given). Race vs the round-0 random engine on FRESH problems (r0 roots excluded by srepr), harvest r1 wins for the next round's curriculum.

- `class _Timeout`
- `count_ops_eval(state: State) -> float`
- `load_tr_proposer()`
- `random_proposer(seed: str)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `solve_r0(root, seed)`
- `solve_r1(root, prop)`
- `path_rows(root, history)`
- `main(per_cell: int) -> None`

### scripts/tabula_rasa_r2.py
Tabula rasa round 2: r2 proposer (trained on rounds 0+1) vs r1 proposer, head-to-head on FRESH problems (roots from both prior rounds excluded). The from-scratch lineage's curve: r0 random 112 -> r1 138 (+26). Does round 2 keep climbing, or plateau early like the mature lineage's 40v40? Eval stays count_ops (lineage purity); winners harvested for a possible round 3.

- `class _Timeout`
- `count_ops_eval(state: State) -> float`
- `load_proposer(ckpt: str)`
- `_root(rng, level, kind)`
- `_check(kind, expr, truth)`
- `solve_with(root, prop)`
- `path_rows(root, history)`
- `main(per_cell: int) -> None`

### scripts/task_arithmetic.py
Task-arithmetic experiments 1-3 (see the 2026-07-06 spec).

- `perplexity(model, tok) -> float`
- `run(model, tok, problems, adapter, scale, label)`
- `main() -> None`

### scripts/task_composition.py
Task-arithmetic experiment 4: composition (see 2026-07-06 spec).

- `skill_acc(model, tok, problems) -> float`
- `main() -> None`

### scripts/tournament_birth.py
Alphabet tournament: parameterized discrete-weight birth. Contestants (real-valued bracket): B {+-1}, T {0,+-1}, M4 {-1,0,1,2}, M5 {0,+-1,+-2}, P2 {0,+-.5,+-1,+-2,+-4}. STE + fp32 latents (the proven recipe); absmean-family scaling.

- `quantize(w: torch.Tensor) -> torch.Tensor`
- `class AlphaLinear` (forward)
- `main() -> None`

### scripts/train_calculus.py
LoRA fine-tune Qwen2.5-0.5B-Instruct on generated, sympy-verified calculus.

- `encode(tok, problem)` — input_ids + labels (-100 on everything but the answer tokens).
- `cut_batches(examples, batch_size, token_budget)` — Batch boundaries over length-sorted examples, cut by token budget
- `batches(examples, pad_id, batch_size, device, epoch=0, token_budget=TOKEN_BUDGET)` — Length-bucketed (little padding) but order-shuffled (decorrelated):
- `main() -> None`

### scripts/train_dispatcher.py
Dispatcher net: root features + rule-fire syndromes -> which brain (policy vs markov), trained on dual-arm dominance labels (gen_dispatch_labels.py — winner by (solved, wall), the FA Law).

- `main(labels: list[Path], epochs: int, out: Path) -> None`

### scripts/train_magic_estimator.py
Train the magic estimator: 20 structural features -> hardness.

- `spearman(a: list[float], b: list[float]) -> float`
- `class Estimator` (forward)
- `main(labels: Path, epochs: int, out: Path) -> None`

### scripts/train_magic_llm.py
(d) LLM-trunk magic estimator: the 0.5B proposer trunk replaces the 20 hand features. Same labels, same seed-parity split as the MLP (train_magic_estimator.py, rho 0.855 baseline), integrand string in, log2(1+nodes) + solved out. Frozen trunk by default (--unfreeze-lora for the joint version). Note the Bayes-floor finding: the 20 features already carry ~99% of explainable variance, so this tests whether a language trunk converts that variance to rank accuracy better than a 64x64 MLP — capacity, not representation.

- `spearman(a, b)`
- `main(labels: Path, epochs: int, batch: int, unfreeze: bool) -> None`

### scripts/train_mathnative.py
Phase 1: pretrain the math-native micro-model on the farmed diet.

- `load_rows(v2: bool=False, v21: bool=False, v22: bool=False, gen4: bool=False, l8: bool=False, gen7: bool=False, diet: str | None=None)`
- `main(v2: bool=False, d: int=384, layers: int=8, ffn: int=1536, out: str | None=None, heads: int=6, v21: bool=False, fast: bool=False, budget: int=24576, lr: float=LR, fp32: bool=False, nopack: bool=False, v22: bool=False, gen4: bool=False, epochs: int=3, l8: bool=False, gen7: bool=False, diet: str | None=None) -> None`

### scripts/train_nnue.py
NNUE-moment training: probe-labeled on-policy states -> tiny MLP. Spec: 2026-07-07-nnue-eval-design.md. Labels are log2(nodes-to-solve), probes capped (200 nodes / 60 s wall, BaseException alarm — the calibration guards). Loss is reported but the race (bench_nnue.py) is scored by running the eval inside the search.

- `class _Timeout`
- `_alarm(signum, frame)`
- `class NnueEval` (forward)
- `_root(rng: random.Random, level: int, kind: str) -> sp.Expr`
- `collect_states(split: str, per_cell: int, cap: int, exclude_roots: set[str] | None=None) -> tuple[list[State], set[str]]` — On-policy states from searches over problems seeded
- `label(state: State) -> float`
- `spearman(xs, ys)`
- `main(per_cell: int, train_cap: int, eval_cap: int) -> None`

### scripts/train_proposer.py
Proposer SFT: choose the winning move number given state + legal moves. Recipe verbatim from scripts/train_calculus.py (LoRA r=16 all proj linears, loss on answer tokens only, length-sorted token-budget batches, per-epoch cut shuffle, cosine schedule). Spec: 2026-07-07-move-proposer-design.md. Runs on CUDA (3080) or MPS.

- `encode(tok, row)`
- `cut_batches(examples, batch_size, token_budget)`
- `batches(examples, pad_id, device, epoch)`
- `move_accuracy(model, tok, rows, device, k=(1, 3))`
- `main(extra_data: list[str] | None=None, out_path: Path=OUT, base_data: str='data/proposer_train.jsonl', eval_data: str='data/proposer_eval.jsonl') -> None`

### scripts/train_syndrome_decoder.py
Syndrome decoder: (20 features + 14 rule-fire syndromes) -> opening rule of the winning derivation. The decoding half of the qLDPC analogy: syndromes localize the deviation; the decoder names the correction.

- `main(labels: Path, epochs: int) -> None`

### scripts/train_syndrome_policy.py
Per-state syndrome policy: (20 features + 14 syndromes + prev-rule one-hot) -> next rule. The policy-side NNUE-vs-LLM rematch.

- `main(labels: Path, epochs: int, no_synd: bool=False) -> None`

### scripts/train_ternary.py
Ternary-from-birth (BitNet-style QAT) — the wiring-thesis re-ask.

- `ternary(w: torch.Tensor) -> torch.Tensor`
- `class TernaryLinear` (forward)

### scripts/train_tf32x3.py
Error-compensated TF32 birth (Markidis 2018 / Ootomo-Yokota 2022 style): every Linear matmul runs as 3 TF32 tensor-core products (hi*hi + hi*lo + lo*hi) instead of 1 fp32 CUDA-core product — ~fp32 accuracy (CPU-verified: 1.15e-4 vs fp32's 1.01e-4 max err, raw TF32 7.8e-2) at tensor-core throughput. Both forward and backward compensated. Parity arm 4: gate + wall decide adoption; pre-registered honestly — 3 TF32 matmuls may net SLOWER than 1 fp32 on GA102, the arm measures it.

- `_split(x: torch.Tensor)`
- `_mm3(a, b)`
- `class _CompLinear` (forward, backward)
- `class TF32x3Linear` (forward)

### scripts/train_value_head.py
Fused value head (Artin's architecture, 2026-07-08): one trunk, two heads, one forward pass. The transformer body replaces NNUE's 20 hand-crafted features — the value head is a tiny MLP (d_model->64->1) on the last hidden state of the state string, trained on the same probe labels (log2 nodes-to-solve) the NNUE used. The LM head keeps ranking moves; value now comes ~free from the hidden state the ranker already computed. AlphaZero's policy+value fusion on a language trunk.

- `class _Timeout`
- `_alarm(signum, frame)`
- `gen_labels(per_cell: int, cap: int) -> None`
- `train_head(epochs: int, batch: int, unfreeze_lora: bool=False, layer: int=-1, split_seed: int=0) -> None`

### scripts/train_weight_reader.py
Three-arm weight-space reader experiment (see the 2026-07-06 spec).

- `_make(args)`
- `build(n, seed, exclude=frozenset())`
- `main() -> None`

### scripts/validity_autopsy.py
Validity autopsy: WHERE do the ~38% invalid steps go wrong?

- `classify(cand: str, cur: str, visited: set, valids: list, sp) -> str`
- `main(ckpt: str, d: int, layers: int, ffn: int, heads: int, n: int, out: str) -> None`

## scratch/

### scratch/absorb_1e5.py
Absorption decider: LR 1e-5 (the pilot's regime), 25-min STE burst on cuda, late layers, band 98M. Counts fp32 updates where w+delta == w (learning lost to rounding). Paired proxy pre/post.

- `ternary(w)`
- `class TLin` (forward)

### scratch/adjudicate_zx.py
Adjudicate axiom's ZX sample batch (relay 2026-07-26-0 protocol).

- `parse(s: str)` — -> (ins, outs, spiders {label: (color, phase8)}, edges
- `to_pyzx(ins, outs, spiders, edges)`
- `_sem_worker(cur, nxt, q)`
- `semantic(cur, nxt, wall=30)`
- `replay_fuse(ins, outs, spiders, edges, site)`
- `replay_id(ins, outs, spiders, edges, site)`
- `replay_color(ins, outs, spiders, edges, site)`
- `structural(row)`
- `main()`

### scratch/anatomy.py
Unified crystal anatomy (2026-07-29 spec: slack-restoration). One env-parameterized census: CELLS=heads,rank,snap on any MicroLM checkpoint. Frozen originals: head_census.py, rank_read.py, snap_alloc.py (07-29, d56).

- `gate(sd, label)`
- `snap(w, q_max)`
- `truncate(W, r)`

### scratch/attractor_census.py
Free-running iteration census (PRE-REG ATTRACTOR-0, 2026-08-10).

- `greedy_step(model, tok, prompt_ids)`
- `norm(s)`
- `main()`

### scratch/attractor_census2.py
Free-running census, THROUGH answer-form + n_succ join (PRE-REG ATTRACTOR-0B-JOIN, 2026-08-10).

- `_nsucc_worker(cur, q)`
- `n_succ_isolated(cur)` — Fork, join with deadline, SIGKILL — the solve_isolated pattern.
- `greedy_step(model, tok, prompt_ids)`
- `norm(s)`
- `main()`

### scratch/basin_probe.py
BASIN-1: routing basin radius v usage. CPU. Usage: SEED=1 python scratch/basin_probe.py

- `main()`

### scratch/birth19m_snaps.py
Gallery instrument run: a fresh 19M-class birth with PER-EPOCH snapshots, so the 113M-style growth render (plot_neurons --displace, the recovered whisper-zoom instrument) exists for the 19M line as [R]-reproducible (Artin's ask 2026-08-08; the frozen crystal-era files have no surviving pair). Also feeds the calibrated internet-vs-native displacement comparison (qwen_displace_extract.py made the internet pair).

- `excised_load_rows(*a, **kw)`
- `tee_save(obj, f, *a, **kw)`

### scratch/blackhole_b0.py
BLACK HOLE MoEs B0+B1+B2 (pre-reg 2026-07-29 close): capacity atlas + dial-routed streaming pack + function-space spot check of Qwen3-30B-A3B. One shard on disk at a time (download -> process -> DELETE — the C7 OOM lesson, applied to disk). Zero calibration. Atlas rows to logs/blackhole_atlas.jsonl; packed parts to checkpoints/blackhole_q3_parts/ (codes npz per shard). Env: START/END shard 1-indexed bounds. __main__-guarded.

- `group_of(name)`
- `pack_codes(w, sigma_law)` — -> (codes int16, scales fp32 [rows], bits, err). sigma_law:
- `main()`

### scratch/boundary_or_bulk.py
Boundary-or-bulk regression on the completed 0.5M->400M grid.

- `n_params(path)`
- `spearman(xs, ys)`
- `affine_r2(xs, ys)`
- `main()`

### scratch/build_dist_diets.py
Build the rung-3 paired diets (spec 2026-07-28, 3-arm design).


### scratch/build_merged_diet.py
Build data/merged_diet.jsonl (schedule-law queue item 1): gen-6 cumulative corpus (v22 + l8 + gen4 sidecar) + the L9a shard, with L1-L3 rationed to 45% (the gen-7 lesson). Stable string seed.


### scratch/cal_dilute.py
CAL-DK-2 (pre-reg 2026-07-30): diet dilution. Train the dense d64h8 recipe with fraction DILUTE of rows' targets swapped among the corrupted subset (fluent, determined-looking, WRONG rows). Usage: DILUTE=0.1 SEED=1 python scratch/cal_dilute.py Then: CKPT=checkpoints/cal_dilute_10_s1.pt python scratch/cal_dk_probe.py

- `main()`

### scratch/cal_dk_probe.py
CAL-DK-1 (pre-reg 2026-07-30): does the crystal know when it doesn't know? Teacher-forced token-level reliability + per-level (3..7) confidence-v-accuracy on the d64h8 EMA crystal, Mac. Usage: python scratch/cal_dk_probe.py

- `main()`

### scratch/calib_probe.py
Calibration probe (spec 2026-07-28 rung 1): flips-per-token under a Q-lattice snap. Teacher-forced greedy argmax on a fixed 400-row probe set; count positions where the snapped twin's argmax differs from the unsnapped model's. Control arm: Q=0 (no snap) must read exactly 0 flips. Also reports the logit-margin distribution at flip sites (the snap-anatomy read: flips should sit at tiny margins). Usage: calib_probe.py <ckpt> <d> <layers> <ffn> <heads> [Q=16]

- `rat_snap(sd, Q)`
- `flips_per_token(ckpt, d, layers, ffn, heads, Q=16, dev=None)`

### scratch/callspan_arms.py
Call-span paired arms (pre-reg 2026-07-29 night: Leg B first read). Pilot 500 (axiom, sha de6c9f15): plain v span hints, same rows, d64, 20 ep; held-out greedy next-step exact match. Atoms pinned in sidecar order. MPS.

- `text(r, span)`
- `run_arm(span)`

### scratch/capacity_meter.py
THE CAPACITY METER (pre-reg 2026-07-29 late night): M = span_bits - code_entropy at per-row step sigma/2 — the fixed-width penalty the sigma grid pays to the worst outlier. At-capacity (Gaussian) weights: M ~ 1.5-2 bits; heavy tails inflate M. Decision rule: M small -> sigma-law allocator; M large -> max-anchored/calibrated. Cells: house crystals, SmolLM2-1.7B, DeepSeek-V3 layer-30 routed experts (fp8 block-dequant, as the 07-17 gauge). Desk only. Run on 3080 with MODELS=qwen for the Qwen cell. __main__-guarded.

- `meter(w)` — w [out, in] fp -> (M bits, kurtosis), per-row sigma/2 step.
- `report(tag, tensors)`
- `house(path, keys=None)`
- `hf_linears(st_path, want, limit=64)`
- `main()`

### scratch/ce400.py
CE-400: fixed-sample CE proxy (the standing instrument from the CE-gate study). Usage: ce400.py <ckpt> <label>


### scratch/ce_gate_study.py
The CE-gate study (pre-reg 2026-07-26, RESULTS.md).

- `mean_ce(model)`
- `coverage(model)`

### scratch/ceiling_probe_cuda.py
Ceiling probe: which L7/L8 integrals can a checkpoint actually solve? Same machinery as gate_eval but per-problem printout.


### scratch/chain_carry.py
CHAIN-CARRY ABLATION (Artin's carry hypothesis, spec'd 2026-07-21): same content, format ablated, equal TOKEN budget, both arms from scratch (d384/8L/3ep). Arm 'chains' = cur->nxt pairs as-is. Arm 'oneshot' = reconstructed root->final-answer rows (chains followed by nxt->cur linkage), upsampled to equal tokens. Gate both. If chains >> oneshot, capability numbers carry a format dividend. Usage: chain_carry.py <chains|oneshot>


### scratch/champ_cuda_probe.py
*(no docstring)*


### scratch/churn_judge_eval.py
CHURN-JUDGE-1 fit/eval (committed so the booked verdict is re-derivable — it was a desk computation on 2026-08-04).

- `load()`
- `dataset(crest, full, seeds)`
- `auc(score, y)` — Rank-sum AUC: P(score(pos) > score(neg)).
- `main()`

### scratch/ckpt_delete_pass.py
Checkpoint DELETE pass (Artin sign-off 2026-08-08 on logs/triage/revive_cite_plan.md). Builds per-host rm manifests for the signed DELETE-AFTER-SIGNOFF families, with the safety interlock Artin mandated: the manifest carries the inventory sha256 and the executor re-hashes each file AT DELETE TIME — mismatch means stop that row, keep the file, report. Classifier class is never the interlock.


### scratch/ckpt_inventory.py
Checkpoint triage INVENTORY (Artin GO 2026-08-07; banked 51GB thread). READ-ONLY: walks checkpoints/ emitting one jsonl row per file (path, bytes, mtime, sha256). Deletion decisions happen elsewhere, with provenance, on Artin review — this script cannot modify anything.


### scratch/ckpt_triage_table.py
Checkpoint triage TABLE builder (Artin GO 2026-08-07; follows ckpt_inventory.py). READ-ONLY desk step: joins the two machine inventories (logs/triage/{mac,wsl}_inventory.jsonl), dedups by sha256, and classifies every path against the evidence record (docs/RESULTS.md + docs/REPRODUCE.md + jobs/*.cmd basename grep).


### scratch/clade_stream_d256.py
Clade-gated streaming pilot, arm G (pre-reg 2026-07-26).

- `probe_band(band: tuple[int, ...]) -> float` — Verified 1-ply valid fraction on fresh band states.

### scratch/complex_birth.py
Complex-FFN birth driver (spec 2026-07-26-complex-zx-program).


### scratch/complex_model.py
Complex-FFN model builder (spec 2026-07-26-complex-zx-program, Leg A).

- `set_alpha(a: str) -> None`
- `g5_quantize(wr: torch.Tensor, wi: torch.Tensor)` — Nearest of {0, ±s, ±is} on each complex weight; STE outside.
- `gn_quantize(wr: torch.Tensor, wi: torch.Tensor, phases: int)` — Nearest of {0} u {s*e^(2*pi*i*k/phases)}: exact roots of unity.
- `zi_quantize(wr: torch.Tensor, wi: torch.Tensor, Q: int=6)` — Gaussian-integer rational lattice (RIFF 2026-07-27, Artin):
- `quantize_pair(wr: torch.Tensor, wi: torch.Tensor)` — Route (re, im) through the alphabet named by _ALPHA.
- `_q(w: torch.Tensor, pair_dim: int) -> torch.Tensor` — STE-quantize a real matrix whose pair_dim halves are (re, im).
- `build_complex_model(vocab_size: int, d: int=384, layers: int=8, heads: int=6, ffn: int=1536, ctx: int=512)`

### scratch/complex_nnue.py
Complex-weight NNUE vs real twin on magic labels (pre-reg below).

- `class ComplexEstimator` (modrelu, forward)
- `class RealEstimator` (forward)
- `run(model, name, xtr, ystr, yctr, xte, yste, ycte, test, epochs=200)`
- `tensors(rs)`

### scratch/complexify_control.py
Symmetry ladder S2 (pre-reg 2026-07-28): complexification control. Double wfloor d256 -> d512 by W(+)W on every linear (block layout); the doubled gates commute with J_half by theorem (asserted). Same function in real arithmetic; fp last-bit ties permitted per amended bar. No training — pure control gate.

- `blockdiag(W)`

### scratch/confluence.py
Metabolic-vs-champion confluence: where did 471 signed rows land? Per-matrix ||dW||, effective rank of delta, top-layer localization, ternary flip census (would the 1.58-bit deployment even change?).

- `ternary(w)`

### scratch/corner_snap.py
The compression corner (pre-reg 2026-07-28 night): rational- snap (direct, exact-best p/q, q <= Q) x {dense wfloor d256, circulant-8x substrate}, Q in {8, 16}. Paired gates on one device. Delta-of-deltas reads orthogonality of the bits and sharing compression axes. Snap code inlined from scratch/rational_snap.py (same operator, no subprocess).

- `snap_sd(sd, Q)`

### scratch/crystal_recreate_test.py
Provenance falsification test for docs/assets/neurons-19m.png's RIGHT panel (Artin's ask 2026-08-08): recreate it two ways and let the pixels decide.

- `lora_delta(p)`

### scratch/d2_verify.py
d2 endpoint verification (amendment 2026-07-28): are the fp64- masters and exact-dd arms' endpoints WEIGHT-identical, or only count/outcome-identical as booked? Three reads: (1) element-wise state_dict equality; (2) deployed ternary sign-map (flip-SET) equality; (3) calib_probe fingerprints on both. Runs on the 3080 (checkpoints live there); CPU-safe.

- `tern_sign(w)`

### scratch/desert_v2.py
Desert test v2 — cross-grammar composition probe (union eq coefficient iv).

- `sample_step(cur, temp=0.7)`
- `verify(cur, pred, q)`

### scratch/detbwd_diet.py
Deterministic mini-crystal birth on the REAL MATH DIET (queued by Artin 2026-08-01 pre-compact): the multi-block integer model (detbwd_mb anatomy, V=40 = MathTokenizer vocab) trained with true next-token CE on gen-4 diet windows — the bridge from random-target demos to the actual curriculum.

- `draw_windows()` — First NWIN strictly-encodable diet rows with >= T+1 tokens,
- `main()`

### scratch/detbwd_gravmoe.py
Deterministic gravmoe pair (spec 2026-08-01-deterministic- gravmoe): the mb bridge model with each Body's FFN split into E=4 experts behind an integer switch_top1 router (multiplicative top_p gate, fx3 convention), plus an integer gravity relaxation every K optimizer steps. All arms share seed/init/windows; the only variable is lambda = LN/LD.

- `class MoBody` (fwd, bwd)
- `_require_sha(label, observed, expected)`
- `assert_gate_diet_sha(path)`
- `assert_gate_row_shas(ids)`
- `find_split(full, mark)` — Index just past the LAST 'Step: ' marker, or None.
- `answer_region(full, mark, terminator_ids)` — Return (first answer token, first newline/EOS token).
- `token_accuracy_counts(generated, full, region)`
- `assert_disjoint_prompts(ids, splits, cut)`
- `loss_dlogits(pp, tgt, eye, boost, region=None)`
- `loss_proxy(pp, tgt, region=None)`
- `draw_complete(n, diet_path=None)` — First n diet rows whose FULL text fits T+1 tokens (padded
- `_fork_call(worker, args, timeout)`
- `_sympy_worker(sender, a, b)`
- `sympy_assess(a, b, timeout=10)`
- `sympy_equiv(a, b, timeout=10)`
- `gate(m, ids, truths, tok, tab, label)` — Free-run validity gate: prefix through 'Step: ', greedy
- `class GMB` (param_items, bwd)
- `relax(wide)` — The gravity event, per spec: per body, kinds order, mean
- `agreement(m, wins, tab)` — Pairwise % of probe tokens where expert outputs agree:
- `twin_fp64(m, tok, tgt, tops, masks=None)` — fp64 autograd twin. ALL discrete decisions come from the
- `build_model()`
- `run_loss(m, wins, tab, t_exp, regions=None)` — Exact cycle-mean loss over the 8 windows (no training).
- `main()`

### scratch/detbwd_mb.py
Deterministic-birth MULTI-BLOCK reference (queued 2026-08-01): N transformer bodies chained by dx0 + embedding and TIED head at the ends — the full mini-LM anatomy, all int64, at the R2b contract (SHIFT=12 default here, GBOOST=256, PQ, ACT_CLAMP, constant lr 1/1000).

- `class Body` (fwd, bwd)
- `class MB` (param_items, fwd, bwd)
- `twin_fp64(m, tok, tgt)`
- `main()`

### scratch/detbwd_plateau.py
PLATEAU-BREAK driver (LOCKSTEP Leg A rung 3, spec 2026-08-06-3080-lockstep-window.md; design pass 2026-08-07).


### scratch/detbwd_r1.py
Deterministic-birth R1a (pre-reg 2026-07-31 night): integer FFN forward + BACKWARD, the first training-side rung.

- `rdiv(x, d)` — Round-half-away integer divide (pack_decode convention).
- `build_tables()`
- `lut(t, xq, hi_pos)` — Table lookup with per-table saturation: beyond +TS the
- `int_mm(a, w)` — [..., K] x [N, K] -> [..., N] in int64 exact (sum-reduce).
- `ffn_fwd(xq, wg, wu, wd, t_silu)`
- `ffn_bwd(dy, xq, wg, wu, wd, cache, t_dsilu)`
- `main()`

### scratch/detbwd_r1b.py
Deterministic-birth R1b (pre-reg 2026-07-31 night): integer ATTENTION forward + backward — softmax via exp table, jacobian in fixed point.

- `build_exp_table()`
- `exp_lut(t, x)` — x <= 0 in Q units; below -TSE -> 0 (exp(-8) < 1/2Q).
- `attn_fwd(xq, wq, wk, wv, wo, t_exp)`
- `attn_bwd(dy, xq, wq, wk, wv, wo, cache)`
- `main()`

### scratch/detbwd_r2_adamw.py
Deterministic-birth R2 (pre-reg 2026-07-31 evening): fixed-point AdamW + a mini end-to-end INTEGER training loop, trajectory-hashed.

- `isqrt(x)` — Exact integer sqrt, elementwise (torch has no int sqrt).
- `isqrt_newton(x)` — Exact floor-sqrt via Newton (fast, deterministic).
- `class IntAdamW` (step)
- `loss_and_grads(xq, tgt, wg, wu, wd, ts, td)`
- `main()`

### scratch/detbwd_r2b.py
Deterministic-birth R2b (pre-reg 2026-08-01 pre-dawn): FULL transformer block trained end-to-end in int64 — adds the three missing integer pieces: rmsnorm backward, rope backward, CE gradient at the head. One block (n1 -> single-head causal attn -> residual -> n2 -> FFN -> residual -> n3 -> head), fixed random next-token targets, IntAdamW at the R3a pin (Q_w = Q<<8, lr 1/1000). Checks: fp64-twin cosines on every param grad (composite, the R1b lesson), rerun determinism sha, falling loss, trajectory sha for the cross-device/cross-lab legs. Usage: python scratch/detbwd_r2b.py

- `rope_tables()`
- `rope_fwd(x, cos, sin)`
- `rope_bwd(dx, cos, sin)`
- `rms_fwd(x, g)`
- `rms_bwd(dy, x, g, isq)`
- `softmax_rows(s, t_exp, scale=None)`
- `softmax_bwd(p, dp, scale=None)`
- `class Block` (fwd, bwd)
- `twin_fp64(blk, x, tgt)` — Smooth fp64 autograd twin; returns grads dict keyed like w.
- `main()`

### scratch/detbwd_r3_qw.py
Deterministic-birth R3a (pre-reg 2026-07-31 late night): pin the wide weight accumulator Q_w. Weights carried at Q_w = Q << SHIFT; rdiv back to Q at the matmul boundary; update applies at Q_w resolution so production-scale lr (1e-3) survives quantization. Usage: python scratch/detbwd_r3_qw.py   (SHIFT sweep in-process)

- `class IntAdamWQw` (step)
- `run(shift)`
- `main()`

### scratch/determinability_census.py
Determinability census (PRE-REG DATA-CEIL rung A, 2026-08-10).

- `enumerate_moves(cur, expr)` — -> [(rule_name, child_sstr)]; axiom bridge (deadline-walled,
- `main()`

### scratch/distortion_collapse.py
THE DISTORTION COLLAPSE (pre-reg 2026-07-29 eve): one curve for the quantization axis. For every logged snap cell, recompute the induced normalized distortion x = param-weighted mean of (W - Wq)^2 / sigma_t^2 over ALL params (unsnapped tensors contribute 0), and pair with the BOOKED solves (y = solves / control). Claim: y = f(x), geometry/location/width-blind, knee at x ~ (0.5-1.0 sigma)^2 / 12 = 0.02-0.08. Desk only, no gates. Solves below are transcribed from logs/ (snap_alloc*.log, polar_snap*.log, snap_q*_gate.log). __main__-guarded.

- `load(p)`
- `rat_snap(w, qm)`
- `x_of(sd, snapped)`
- `polar_q(W, mstep, na)`
- `uni_q(W, u)`
- `main()`

### scratch/dual_probe.py
Dual-crystal probe: math gate + physics probe on ONE vocab-41 model (the blackboard monolith control). Usage: dual_probe.py <ckpt>

- `_equiv(q, pred, gold)`
- `equiv(pred, gold, deadline=10)`

### scratch/duo_mine.py
Duo miner (overnight flywheel): duo wave over a fresh band (spec 2026-07-22-duo-substrate, exp 1): per ply, B/2 samples from TERNARY + B/2 from CHAMPION (budget-matched vs a single model's B), merged and oracle-verified. Same 200-probe rarity battery as gate_rarity.py (same seeds, same census).

- `skeleton(e: str) -> str`
- `binof(n)`

### scratch/duo_wave.py
Duo-substrate mixed wave (spec 2026-07-22-duo-substrate, exp 1): per ply, B/2 samples from TERNARY + B/2 from CHAMPION (budget-matched vs a single model's B), merged and oracle-verified. Same 200-probe rarity battery as gate_rarity.py (same seeds, same census).

- `skeleton(e: str) -> str`
- `binof(n)`

### scratch/e2_logit_check.py
E2 closure (relay -28-5 loop): reproduce axiom's pinned 20-prompt battery logits with torch fp32 on the house scorer. Asserts (1) their token ids decode to their meta text via the house tokenizer (tokenization parity), (2) final-position logits agree within 1e-4 elementwise. PASS arms E3.


### scratch/e3_battery.py
E3 battery (axiom GO 2026-07-29, 50 rows): exact-mode paired GREEDY gate. House side: 50 fresh gate-style prompts (seed band disjoint from battery20 and the GATE band), fp32 eager greedy continuations (<=64 tokens, stop at eos) from the S2 scorer. Axiom decodes the same prompts in exact mode and diffs token- identically. Emits data/e3_battery50{,_meta.jsonl,_greedy.txt} + sha256 pins.


### scratch/emission_wall_pair.py
Rung-1 bar (iii), in-diet form: does prefix move the operand-complexity emission wall? (spec 2026-07-25-native-transformer; poly_chain5 psub/padd rows are OUT-OF-DIET for gen-4 twins — bridge law + naked-forms lesson — so the wall is read on generator-drawn in-language states instead.)

- `load(ckpt)`
- `greedy(model, ids, max_new=160)`
- `main()`

### scratch/engine_scale_export.py
ENGINE-SCALE-1 per-cell export (PRE-REG 2026-08-07, RESULTS L22317).

- `export_one(out_path: str) -> None` — Child: env already carries NBLK/FFN/NWIN; emit one .bin and
- `main() -> None`

### scratch/ex1_swap.py
EX-ANAT-1 swap builder (IDENTITY battery rung 1, spec 2026-08-06-identity-battery.md, frozen design a-d; Artin GO).

- `_assert_lens_env_clean()`
- `recall(kset)`
- `setcov(kset, ref)`
- `main()`

### scratch/ex2_build.py
EX-ANAT-2 one-sided arm builder v2 (IDENTITY battery rung 2, sharpened by VERDICT EX-ANAT-1B; Artin GO, Mac).

- `_assert_lens_env_clean()`
- `recall(kset)`
- `setcov(kset, ref)`
- `ranked(pool, l, is_v, vonly)`
- `main()`

### scratch/ex3_build.py
EX-ANAT-3 subject builder (provenance repair, 2026-08-07 review: the cited keep-sets were built by inline heredocs — this commits the exact recipe; BYTE-IDENTITY against the existing cited artifacts is asserted before this file may serve as their provenance).

- `_assert_lens_env_clean()`
- `invariants(counts, vonly)`
- `emit(name, obj)`
- `main()`

### scratch/exact1_small_cells.py
EXACT1-SMALL: d8/d16 ladder+anchor cells on axiom's ENGINE-EXACT-1.

- `put_tensor(b, name, v)`
- `make_tables(T, DH, **_)` — Fixture-style synthetic tables (identity rope, silu/exp ramps)
- `make_init(T, D, DH, F, V, seed_str)` — 11 KEYS tensors at shipped Q9 scale, then x [T,D], then tgt
- `run_arm(name, obj, steps, budget)`

### scratch/exact_twin_d56.py
d56 exact twin (pre-reg 2026-07-29 night): snap EVERY floating tensor (incl. emb, head, 1D norm gains) to best-rational Q<=16, gate, and report per-tensor-class snap error in sigma units (blockwise-rule diagnostic). Desk, MPS.

- `snap(w, q_max)`

### scratch/exchange_test.py
THE EXCHANGE TEST (pre-registered 2026-07-23): train the v4 organism on axiom's engine-farmed chains at OUR stuck states, re-probe the SAME fixed seeds (55_000_000, cuda — device law), must beat 2/12. v4 measured self-practice at +1/12 (no gradient at true walls); the exchange supplies exactly the missing gradient. 10/12 walls have chains; ceiling = 12/12, bar = >=3/12, headline read = how many of the 10 taught walls flip.

- `ternary(w)`
- `class TLin` (forward)
- `try_state(cur0, seed0, plies=8)`
- `probe(tag)`

### scratch/export_axnn.py
E2: export a MicroLM crystal to AXNN v1.1 (proposed extension: cfg ffn="swiglu" + fused-qkv + rmsnorm-no-bias + rope + SEPARATE (untied) head — every convention DECLARED per the AXNN doctrine).


### scratch/export_mb_ref.py
Export the multi-block deterministic-birth reference for axiom's leg: init bytes (all params in param_items order, then tok, then tgt; int64 LE) + milestone trajectory digests at the R2b contract grown by n_blocks (SHIFT=12, GBOOST=256, constant lr 1/1000, 1000 steps, NBLK=2, seed 17). Artifacts land in scratch/detbwd_mb_ref/ (committed — small). Usage: .venv/bin/python scratch/export_mb_ref.py

- `main()`

### scratch/export_r2b_ref.py
Export the R2b full-birth reference for axiom's C++ leg (relay 2026-08-01-0): init bytes in seed-17 draw order + the reference trajectory digests at the amended contract (SHIFT=12, constant lr 1/1000, 1000 steps). Artifacts land in scratch/detbwd_r2b_ref/ (committed — small). Usage: python scratch/export_r2b_ref.py

- `main()`

### scratch/farm_dist_rows.py
Distribution rows (spec 2026-07-28 rung 3): for each diet cur, enumerate the engine's verified-valid moves (successors: sympy- verified, non-identity by construction), weight by MarkovPrior (rule-name unigram, @site stripped, unseen = 0.5*median — the proposer's own convention), emit ALL of them as weighted rows. Rows STREAM out incrementally (the killed-worker doctrine). sympify here runs on farm-certified diet strings, not model text.

- `enumerate_moves(cur, expr)` — -> [(rule_name, child_sstr)]; axiom bridge (deadline-walled,

### scratch/farmer_probe.py
FARMER PROBE (pre-reg 2026-07-29: escalation-engine cell 6, Artin's reverse-self-learner riff). A full-reverse d64 birth (sym_birth REV=2, SKIP_GATE) plays farmer: sample predecessor candidates for NOVEL band expressions (gate-band + 50k offset, disjoint from the gate), verify each by FORWARD rule application (fork-boxed verify_wave: cand -> seed must be a valid step), and score verified-distinct-NOVEL yield per 1000 samples + wall time. Novel = candidate absent from the entire gen-4 corpus (cur+nxt). Usage: CKPT=checkpoints/sym_birth_dense_revfarm_ema.pt        .venv/bin/python scratch/farmer_probe.py


### scratch/fig_magic_scatter.py
Gallery: magic-estimator held-out scatter (predicted vs measured).

- `class Estimator` (forward)
- `spearman(a, b)`
- `main(out: Path) -> None`

### scratch/fixed_q_snap.py
Fixed-denominator snap (spec addendum 2026-07-27, 'integer twin'): every 2-D weight -> round(w*q)/q for ONE shared q. Unlike best-rational (free denominators), this makes W = P/q with integer P — the forward pass becomes an integer GEMM / q, the road to exact integer inference (ozaki/FX-V1 substrate). Error bound 1/(2q), vs ~1/Q^2 for best-rational. Usage: fixed_q_snap.py <ckpt_in> <q> <ckpt_out>


### scratch/format_delta_prep.py
Build row embeddings for the delta-chained format (spec 2026-07-26-format-ladder): mean-pooled final-norm hidden states of the pairs-trained control crystal (wfloor_d256) over each pair text. Output: checkpoints/fmt_row_emb.pt (N, d) unit vectors, row-aligned with the filtered gen-4 row list.


### scratch/format_ladder.py
The format ladder (spec 2026-07-26-format-ladder, pre-reg in RESULTS). One birth per invocation:

- `pair_text(r)`
- `build_chains()` — State-linked greedy chains from roots; consumes every row
- `run_batches(batches, lr_fn, epochs_label='')`

### scratch/fourier2_modbirth.py
FOURIER-2: birth a Mod-diet crystal (nt pilot 500, callspan plain-arm recipe) and run the roots-of-unity probe properly. Usage: python scratch/fourier2_modbirth.py (pilot at data/)

- `main()`

### scratch/fourier2b_widemod.py
FOURIER-2b (pre-reg 2026-07-31): wide-Mod birth + roots-of-unity probe, with the memorization check FOURIER-2 lacked.

- `gen_rows()` — Eval n's drawn first, excluded from train (prompt-set guard).
- `fmt(n, k)`
- `main()`

### scratch/fourier3_algdiet.py
FOURIER-3 (pre-reg 2026-07-31): the causal arrow — put the ALGORITHM in the diet and watch where clocks appear.

- `dsum(n)`
- `nxt_of(n, k)` — One teaching step: decompose for ALG_KS, else answer.
- `gen_rows()`
- `fmt(n, k)`
- `rollout(model, tok, dev, n, k, hops=4)` — Greedy; follow Mod(m, k) rewrites until a bare number.
- `main()`

### scratch/fourier4a_dynamics.py
FOURIER-4a (pre-reg 2026-07-31): clock-FORMATION dynamics.

- `acc_at(model, tok, dev, ev, k, n=N_ACC)`
- `clock_at(model, tok, dev, ev, k, rng)`
- `main()`

### scratch/fourier_g9.py
B6 (revival-sweep Tier B, 2026-07-31): G9 zeta-8 ON THE MOD DIET — the declared rotation reopening, fired on the one substrate where the target computation is provably rotational (clock- placement law). Completes the causal square: diet-forced clocks exist (FOURIER-2b); does architecture-PROVIDED rotation get adopted where the diet wants it?

- `main()`

### scratch/fourier_probe.py
FOURIER-1: does the crystal implement the roots-of-unity filter? Per-neuron Fourier v indicator regression of answer-position activations over n mod k. CPU. Usage: python scratch/fourier_probe.py

- `main()`

### scratch/fp64_paired.py
THE ROUNDING-LOSS DECIDER (overnight GO): fp32 vs fp64-master paired burst at LR 2.5e-6 (GRPO's real regime). Same food stream, late-layer STE, 40 min/arm. PRIMARY metric: committed ternary flips vs own start (sub-ULP nudges absorbed by fp32 should COMMIT under fp64 masters -> more flips at equal food). Secondary: proxy. Usage: fp64_paired.py <fp32|fp64>

- `ternary(w)`
- `class TLin` (forward)

### scratch/fx3_house.py
FX-V3 house reproduction (pre-reg 2026-07-31 night): the house integer reference for the MERGED crystal — P3 DetLM + axiom's switch_top1 gate spec (relay 2026-07-31-3), decoding THEIR shipped tables (never regenerated). PASS = both published stream digests. Usage: AXIOM=~/code/axiom python scratch/fx3_house.py

- `class Fx3LM` (step)
- `battery(m, tok)`
- `main()`

### scratch/g5_polar.py
G5 POLAR (pre-reg 2026-07-29 eve): the predicted BREAK of geometry-blindness. cplx_G5_dep.pt carries DEPLOYED star weights ({0, +-s, +-is} per complex — anisotropic by construction); cplx_none.pt is the isotropic control crystal. Cells per crystal: control; polar 4 angles ALIGNED (1s mag); polar 4 angles ROTATED 45 deg (same bits); uniform u=1s. Prediction: rotation hurts ONLY the star crystal. alpha=none for both (dep weights are already hard). __main__-guarded.

- `polar_q(W, mstep, na, rot=0.0)`
- `uni_q(W, u)`
- `main()`

### scratch/gate_batched.py
Batched gate v2 (2026-07-21): batch ACROSS problems, 8 seeds each — one forward serves K*8 rows instead of 8. Right-padded buffer + attn_mask (model supports it); per-row write positions keep RoPE phases identical to the unbatched path. NOTE: float reduction order changes => near-ties may resolve differently => this is a NEW GATE LINEAGE (re-baseline models of record once). Usage: gate_batched.py <ckpt> <d> <layers> <ffn> <heads> <label> [K]

- `batched_wave(model, tok, prompts, seed_lists, dev, max_new=120)` — prompts: list of K token-lists; seed_lists: K lists of B seeds.
- `gate_eval_batched(model, tok, dev, K=12)`

### scratch/gate_ckpt.py
*(no docstring)*


### scratch/gate_ckpt_cuda.py
*(no docstring)*


### scratch/gate_cplx.py
Gate a complex-FFN checkpoint (mirror of gate_ckpt.py).


### scratch/gate_pp.py
Per-problem gate (step-3 item (d), first cut): the standard chain gate with a jsonl sidecar — per-problem outcome + the full greedy chain + wandering/identity signatures. Same seeds/oracle as gate_eval (results comparable to gate_ckpt numbers).


### scratch/gate_prefix.py
Chain gate for PREFIX-substrate models (rung 1, spec 2026-07-25-native-transformer). Mirrors gate_eval exactly — same seeds, same _gen_isolated problems, same verify_wave oracle — with prefix<->infix conversion at the two boundaries: prompts serialize cur to prefix; model emissions parse prefix->sympy and re-render infix (sp.sstr) before the oracle. Emissions that fail the prefix parser are invalid candidates (counted tried, never valid).


### scratch/gate_rarity.py
Rarity-stratified gate (schedule-law queue item 2): capability as a curve over expression rarity, not a scalar. Rarity = skeleton frequency — integer constants normalized to '#', skeleton counted in the corpus cur-set. Probes drawn WITHOUT exclude-filtering (the full spectrum is the point); bins: common / mid / rare / unseen-skeleton. Usage: gate_rarity.py <ckpt> <d> <layers> <ffn> <heads> <label>

- `skeleton(e: str) -> str`
- `binof(n: int) -> str`

### scratch/gate_regate.py
Re-gate sigma cell (pre-reg 2026-07-31 night): gate ONE untouched checkpoint in a fresh process — run N times via the bash loop below to measure cross-process re-gate spread on mps (the GRAV-0T control came back 37 v the booked 44). Usage: for i in 1 2 3; do CKPT=checkpoints/umoe_lb_s1.pt        python scratch/gate_regate.py; done

- `main()`

### scratch/gate_transcripts.py
Gate transcript dump (Artin's ask, 2026-07-31 night): print the model's ACTUAL step chains on gate prompts — how it works a problem, not just whether it solved. Mirrors gate_eval's loop exactly (same seeds, same wave sampler, same oracle) but records every accepted step plus the rejected-sample count per ply. Usage: CKPT=checkpoints/umoe_gravmoe_s1.pt LEVEL=4 N=6        python scratch/gate_transcripts.py

- `main()`

### scratch/gate_zx.py
ZX gate (pre-reg 2026-07-26, the factorial's ZX column).

- `n_spiders(s)`
- `greedy(cur, max_new=700)`
- `invariants(s)`

### scratch/gauge_distance_d256.py
Gauge-aligned model distance on the d256 zoo (pre-reg 2026-07-26).

- `load(path)`
- `nfro(a, b)`
- `perm_align2(a, b)`
- `rot_align(a, b)`

### scratch/gauge_m4x.py
Max-asymmetric {0,1,2,3} gauge-commutation arm (pre-reg 2026-07-26).

- `m4x_rows(w)` — {0,1,2,3} x per-row amax/3 scale — maximally asymmetric.

### scratch/gauge_slack_rat.py
Gauge-slack 4-crystal cell (pre-reg 2026-07-27 night, RIFF-LEDGER).

- `load(path)`
- `nfro(a, b)`
- `perm_align2(a, b)`
- `rot_align(a, b)`

### scratch/gen_lab_overview_pdf.py
LinkedIn Featured collateral: 3-page lab-overview PDF composed ENTIRELY from committed docs/assets figures + ledger-verified numbers (receipt-checked 2026-08-08 against RESULTS/FINDINGS/ README for the LinkedIn pass). Layout only — no generated imagery.

- `head(c, title, sub=None)`
- `para(c, x, y, width, lines, size=10, leading=14.5, color=INK, font='Helvetica')`
- `image(c, path, y_top, max_h, caption=None)`

### scratch/gen_lean_corpus.py
Generator for the Lean-tier smoke corpus (2026-08-03).


### scratch/graph_mod_sigma.py
A2 (revival-sweep Tier A, 2026-07-31): graph-modularity Q dispersion on the three same-diet wfloor_d256 seed births — the "free sigma" the 07-26 NULL entry named but never ran. The +0.030 dQ verdict was a BAR-based null with unmeasured dispersion; this cell measures it and re-adjudicates. CPU, minutes. Usage: python scratch/graph_mod_sigma.py

- `main()`

### scratch/graph_modularity_gen8.py
Graph-modularity read: gen-8 five-grammar crystal vs single-grammar 19M.

- `load(path: str) -> dict`
- `layer_graph(feat: torch.Tensor) -> nx.Graph`
- `read(path: str) -> tuple[float, float]`

### scratch/grav1b_distance.py
GRAV-1b (pre-reg 2026-07-30): the field in router coordinates. Bin tokens by router probability p_e on the ablated expert; report ablation dNLL per distance bin. Mac, umoe_lb_s{1,2}. Usage: SEED=1 python scratch/grav1b_distance.py

- `main()`

### scratch/grav2_spacetime.py
GRAV-2 (pre-reg 2026-07-30): engineered spacetime — birth a d64h8 crystal with a contractivity penalty and price the toll.

- `falloff(model, enc, tok, dev, eps=0.05)` — Gentle-kick displacement profile: perturb block-k input by
- `main()`

### scratch/grav_posthoc.py
GRAV-0T + GRAV-REV (pre-reg 2026-07-31 night, Artin's riff): post-hoc gravity — does the merge-free pull work with NO training?

- `load(dev)`
- `observe(tok, m, dev)` — Build each block's co-routing EMA over train-side rows.
- `relax(m, emas, lam, steps)`
- `gate(tag, m, tok, dev)`
- `main()`

### scratch/grav_probe.py
GRAV-1 (pre-reg 2026-07-30): expert gravity in the micro-MoE.

- `batches(enc, tok, dev, bs=8)`
- `main()`

### scratch/greedy_first_gate.py
Greedy-first adoption cell (pre-reg 2026-07-28 night): on the FULL production gate battery (same seeds/levels as gate_eval), race (a) wave-8 (production) vs (b) greedy-first with wave-8 retry only at plies where greedy's candidate fails verification. Same chain semantics as gate_eval (12 plies, oracle-picked). Usage: greedy_first_gate.py <ckpt> <d> <layers> <ffn> <heads> <label>

- `greedy(prompt, spend)`
- `run(arm)`

### scratch/grpo_shaped.py
Potential-shaped GRPO on the gen-6 champion (2026-07-21, Artin GO — 'ahead of metabolic v3'). The b-lever: reward bandwidth. r = verified * (1 + LAM * tanh((Phi(cur)-Phi(next))/SCALE)), Phi = -(count_ops + 40*n_Integral). Unverified stays 0 (oracle floor intact; Ng-shaping preserves optimal policy). Monkeypatches G.collect's r_of via a wrapped collect; everything else (driver, gates, rollback) is the production harness. Pre-registered against the plateau: solves flat by cycle 4 in every unshaped run — shaped must beat +2 solves over 12 cycles or the b-lever nulls.

- `phi(s)`
- `shaped_collect(model, tok, dev, n_groups, seed0)`

### scratch/gt2_code_arm0.py
MOE-GT-2 arm D3: CODE arm-0 — decode-only demand log on the codegen ladder (pre-reg MOE-GT-2, 2026-08-04).

- `select_tasks(by_rung, n)` — Fixed 120-task rule: round-robin across non-empty rungs in RUNGS
- `main()`

### scratch/gt2_jaccard.py
MOE-GT-2 coalition Jaccard analysis (D2/D3 readouts, committed post-hoc so the booked numbers are re-derivable — they were desk computations in-session on 2026-08-04).

- `_traj()` — TRAJ log paths, env-overridable per domain (TRAJ_MATH etc.).
- `main()`

### scratch/gt3_probe_arm0.py
MOE-GT-3 arm-0: demand log over an arbitrary prompt list (pre-reg MOE-GT-3, 2026-08-05 — the base-class discriminators).

- `main()`

### scratch/gt4_dialog_prompts.py
MOE-GT-4 corpus: the SECOND verbal corpus (dialogue/QA register).

- `main()`

### scratch/gt4_verbal_core.py
MOE-GT-4 readouts: does the verbal branch have its own core?

- `main()`

### scratch/gt5_union_keep.py
MOE-GT-5 keep-set: per-layer UNION of the two branch cores.


### scratch/gt5c_randfill_keep.py
MOE-GT-5c keep-sets: symbolic core + RANDOM non-core fill, matched per-layer to the union mask's exact sizes.


### scratch/gt6_recall_ladder.py
MOE-GT-6 keep-sets: the recall ladder + the verbal-excluded arm.

- `recall(keep)`
- `build(k_fill, rng, pool_fn)`
- `tune(target, seed_tag, pool_fn, k_max=91)` — Search the per-layer fill count whose drawn keep-set lands
- `dump(keep, r, k, name)`
- `main()`

### scratch/gt7_coverage_rederive.py
GT-7 precursor: re-derive MOE-GT-6's exploratory coverage lenses from committed artifacts (reviewer-scan gap 2026-08-06: the 0.755 Spearman was a desk cell with no committed derivation — GT-7 cannot register coverage as its ladder variable until the number has a script, the GT-3 discipline).

- `spearman(xs, ys)`
- `main()`

### scratch/gt7_draw.py
MOE-GT-7 keep-set draws (PRE-REG MOE-GT-7, fired on Artin GO 2026-08-06). The verbal-coverage ladder at FIXED recall.

- `_assert_lens_env_clean()`
- `recall(keep)`
- `setcov(keepset, ref)`
- `draw(rtarget, ctarget, seed_tag, vonly, k_max=91, attempts=20)`
- `dump(keep, name, r, c, k, tag)`
- `main()`

### scratch/gt7_run.py
MOE-GT-7 gate driver (PRE-REG MOE-GT-7, Artin GO 2026-08-06).

- `main()`

### scratch/head_autopsy.py
THE HEAD AUTOPSY (pre-reg 2026-07-29 eve): per-(layer, head) single-cell deletion map on the h8 EMA crystal. The day census deleted a head INDEX across all layers (a column); this deletes one (layer, head) cell at a time — 64 cells on the proxy gate (n=8/level, +-2 noise, read the map shape), then FULL gates on control + min/max cells. __main__-guarded.

- `main()`

### scratch/head_census.py
Head census (pre-reg 2026-07-29: attention anatomy 1a). Zero head h of 4 across all layers of the d56 EMA crystal (q,k,v row blocks in the fused qkv [3D,D] + the o column block), gate each arm. Desk only, MPS.


### scratch/holdout_gate.py
FROZEN HOLDOUT battery (2026-07-21): virgin band 88M, same L3-L7 x 24 structure as the production gate, run ONLY at promotions. Includes a corpus-overlap audit (contamination doctrine: verify the band is virgin, don't assume). Usage: holdout_gate.py <ckpt> <d> <layers> <ffn> <heads> <label>


### scratch/holdout_v2.py
Holdout v2: exclude-guarded (the doctrine I broke in v1 — 281 collisions caught by the audit). Probes drawn from band 88M but each slot advances its seed until the expr is NOT in the corpus cur-set. Usage: holdout_v2.py <ckpt> <d> <layers> <ffn> <heads> <label>


### scratch/int3_rider.py
Rung-1 rider (iv): int3 PTQ gate delta, prefix vs infix twin (spec 2026-07-25-native-transformer — prediction: prefix MORE robust under quantization via delimiter-outlier removal).


### scratch/jointperm_distance.py
Joint-permutation distance closure cell (banked 2026-07-26).

- `load(path)`
- `joint_dist(A, B)` — Per-layer joint perm over [gate row | up row | down col].
- `raw_dist(A, B)`

### scratch/judge_decode.py
Judge-collapsed decoding (spec 2026-07-28 rung 4, pre-reg 2026-07-28). Three arms on 30 fresh L5-L7 states, 12 plies: (a) wave-8 (production semantics), (b) greedy-1, (c) greedy with top-2 branching at near-tie steps (margin < 0.02), oracle judge, both branches' tokens charged. Tokens + per-state sidecar logged.

- `greedy_step(prefix, spend, branch=False)` — Greedy decode one Step line. branch=True: at the FIRST
- `run_chain(cur0, arm, seed0)`

### scratch/k3_expert_demo.py
K3-D1: the Kimi-K3 single-expert deterministic demo.

- `_get(url, lo=None, hi=None)`
- `fetch_expert()`
- `dequant(packed, scale)` — MXFP4-pack -> (codes2x int64 [out,in], exps int64 [out,groups],
- `det_gemv(codes2x, exps, x, dev)` — Exact integer y = W @ x on the shipped MXFP4 codes.
- `chain(deq, dev)` — K3-D2: full deterministic expert forward y = w2 @ (silu(w1@x)
- `main()`

### scratch/keff_probe.py
keff_probe — DIRECT effective-context measurement on a trained checkpoint (pre-reg RESULTS PRE-REG KEFF-PROBE-1). No training.

- `main()`

### scratch/kv_equiv.py
KV-cache sampler + equivalence oracle (house rule: token- identical to eager full-recompute, or it doesn't ship).

- `sample_wave_lp_kv(model, tok, prompt_ids, seeds, dev, max_new=120)`

### scratch/l9_probe.py
L9 probe: 24 fresh L9a problems (band 90M — disjoint from the farm's 72/73M and roots_c1), gate_eval-style rollout, 12 plies. Usage: l9_probe.py <ckpt> <d> <layers> <ffn> <heads> <label>


### scratch/lam_merge_review.py
Lambda-merge rider (pre-reg in the review-adoption amendment, 2026-07-31): merge reviews on the three lambda-arm checkpoints — is merge-free lambda-independent, and does LOW lambda (weak collapse) merge badly? Runs on the device holding the ckpts. Usage: python scratch/lam_merge_review.py

- `main()`

### scratch/lean_check.py
House-side batch checker for axiom's Lean certificate sidecars (relays 2026-08-03-0/-1; axiom emitter c0511bc).

- `sstr_to_lean(s, atoms)` — sstr (tier-1 subset) -> Lean expression text. Deliberately
- `rederive(row)` — Our canonical statement text from (lhs, rhs, atoms) + tactic.
- `norm(s)`
- `ac_equal(ours, theirs)` — Statement equality up to associativity/commutativity: compare
- `main()`

### scratch/lean_sample_build.py
LEAN kernel-sample builder (provenance repair, 2026-08-07 audit: the frozen 1000-id sample had no committed sampler — SEV-1 class).

- `check(name, text)`

### scratch/legacy_diet_audit.py
PHASE 1 of RESULTS-HARDENING (Artin GO 2026-08-07): exclude-union audit of LEGACY diets against the 120-problem forward gate band.


### scratch/lloydmax_race.py
The Lloyd-Max codebook race (pre-reg RESULTS 2026-07-25): per-output-channel exact 1-D k-means quantizers on the 19M infix twin, free vs zero-pinned centroids, PTQ-only. Writes one _lm*.pt checkpoint per arm; gates run separately (MPS, to match baselines).

- `kmeans_rows(w: torch.Tensor, k: int, pin_zero: bool, iters: int=25) -> torch.Tensor` — Exact-enough 1-D k-means per row. w: (rows, cols). Returns
- `uniform_rows(w: torch.Tensor, bits: int) -> torch.Tensor` — Symmetric-range int grid {-2^(b-1) .. 2^(b-1)-1} x s.
- `main() -> None`

### scratch/loss_floor_census.py
LOSS-FLOOR-1: empirical conditional entropy of the sat_s2 warm diet vs the measured 0.348 train-loss floor (RESULTS pre-reg LOSS-FLOOR-1).

- `encode_corpus()`
- `entropy_of_groups(groups)` — Mean -log p_hat(next | context) over all positions, nats.
- `h_kgram(enc, k)`
- `h_full(enc)`
- `main()`

### scratch/lyap_compare.py
Atlas-2 Lyapunov leg: function-space divergence between twin births. Observable (weight distance forbidden by doctrine): teacher-forced argmax disagreement on 200 fixed gen-4 rows — fraction of non-pad positions where the two models' greedy next- token predictions differ. Usage:   lyap_compare.py ckptA ckptB TAG   (d64/ffn256/heads4 assumed)


### scratch/make_altpairs.py
Farm verified ALTERNATIVE successors for a sample of corpus states (the distribution-rows bank, forward edition; motivated by the 2026-07-26 distribution readout: crystals put ZERO mass on equally-valid non-canonical moves). For each sampled unique cur, enumerate successors() (verified, non-identity) and keep children NOT already in the corpus as a nxt for that cur. Fork workers stream rows to shard files (killed-worker doctrine: partial shards survive the wall).

- `worker(idx, my_keys)`

### scratch/make_union_diet.py
Build the math+ZX union diet (next-session-2 item 1): gen-4 math rows + zx_farm1_train, one jsonl. Shares are organic (~133k math / ~97k ZX = 58/42); ZX provenance keys (kind, site) kept. Output: data/union_math_zx.jsonl


### scratch/margin_by_level.py
Margin-vs-hardness probe (PRE-REG DATA-CEIL rung 0, 2026-08-10).

- `greedy_margins(model, tok, prompt_ids)` — -> list of top1-top2 gaps along the greedy trajectory.
- `main()`

### scratch/margin_by_ply.py
Margin-vs-ply-depth probe (PRE-REG DATA-CEIL-0B, 2026-08-10).

- `greedy_step(model, tok, prompt_ids)` — -> (text, margins) greedy decode of one step.
- `main()`

### scratch/margin_census.py
Margin census on the crown-tie ternary latents (pre-reg 2026-07-26).


### scratch/margin_vs_branching.py
Margin-vs-branching probe (PRE-REG DATA-CEIL-0C rung C1).

- `greedy_min_margin(model, tok, prompt_ids)`
- `bucket(n)`
- `main()`

### scratch/mass_on_valid.py
Mass-on-valid (spec 2026-07-28 rung 2): teacher-forced sequence probability mass over the engine-enumerated verified-valid next-step set, vs the modal valid move (farm-pick proxy: fresh states have no banked row). No sampling anywhere. successors() output is already sympy-verified and non-identity (derivation.py docstring), so the valid set is the enumeration itself.

- `seq_logprob(model, cur, nxt)`

### scratch/matryoshka_r1.py
Matryoshka rung 1 (pre-reg 2026-07-28 night): joint loss CE(W) + CE(STE P_C8(W)) — one crystal whose OWN circulant projection must also work. 1 warm epoch from wfloor d256 on MPS. Implementation: parametrize gate weights with a toggleable STE projection (flag off -> raw W; flag on -> W + (P(W)-W) .detach(), i.e. forward uses P(W), gradient flows to W). Gates BOTH tiers at the end; saves the single weight tensor.

- `shift_perm(n, sh, dev)`
- `class TierP` (project, forward)

### scratch/matryoshka_r2.py
Matryoshka rung 2 (pre-reg 2026-07-29 night): 3-tier ladder in one tensor. Joint loss CE(W) + CE(STE P_C2(W)) + CE(STE P_C8(W)) on gate weights, 1 warm epoch from the d56 EMA crystal. Gates all three tiers. MPS.

- `shift_perm(n, nb, sh, dev)`
- `class TierP` (project, forward)

### scratch/metabolic_d2.py
DISAGREEMENT #2 test — exact vs fp64 accumulation at the validity level (v5-mini, 2 of the 4 race arms). ONE variable: arm fp64 accumulates AdamW steps into fp64 masters (rounds 2^-53/step); arm dd accumulates via two-sum double-double (EXACT — absorption structurally impossible). Identical manual AdamW, food stream, seeds. Streaming: every row eaten once, no epochs. Usage: metabolic_d2.py <ckpt> <worklist> <minutes> <fp64|dd>

- `ternary(w)`
- `class TLin` (forward)
- `opt_step()` — manual AdamW, identical both arms except accumulation
- `sign_state()`
- `try_state(cur0, seed0, plies=8)`
- `probe(tag)`

### scratch/metabolic_hot.py
HOT METABOLISM (2026-07-21, Artin GO): map the safe-plasticity frontier. Pilot harness + LR ladder: start 3e-5, x1.8 every 20 stable cycles; immune system: proxy gate n=8 every 5 cycles, 2 consecutive drops >5 -> ROLLBACK + halve LR (frontier found). Optional --late: freeze layers 0-7 (confluence shortcut: delta mass is 8-11-heavy; backward stops at layer 8). Band 95M (fresh). ~150 cycles.


### scratch/metabolic_v3.py
METABOLIC V3 — the stacked LLMUE session (spec: four banked upgrades, one run, separately toggleable via env):

- `ternary(w)`
- `class TLin` (forward)

### scratch/metabolic_v4.py
METABOLIC V4 — practice food + persistence census (spec 2026-07-23-metabolic-v4, v4.1 revisions). Single arm, fp64 masters ON, LR 1e-5 (hot-but-guarded), food = stuck-state worklist cycled + fresh unseen-biased problems; rollouts START at the stuck cur; new stuck states eaten in-session; skip-pair banking on resolutions; pre/post resolution probes (paired); flip census every 20 min.

- `ternary(w)`
- `class TLin` (forward)
- `sign_state()`
- `try_state(cur0, seed0, plies=10)` — One duo... single-model rollout from a state. Returns
- `probe_worklist(tag, seed_base)`

### scratch/metabolic_v5.py
METABOLIC V5 session 1 (spec 2026-07-23-metabolic-v5; dd arm retired per disagreement-2 verdict). fp64 masters, streaming, long horizon. Three jobs in one session:   (1) practice: worklist = p1 residue + p2 deep states (14), stuck       food + fresh L6-9, paired PRE/POST fixed-seed probes;   (2) MINER V2: bank ALL verified steps outcome-tagged (solved /       unsolved) -> data/practice_rows_v5.jsonl — the failed-step       shard the gen-9 solved-only-leak A/B needs;   (3) fresh-wall logging: zero-verified fresh roots -> axiom       exchange format, data/stuck_states_v5.jsonl (morning relay). Usage: metabolic_v5.py <ckpt> <worklist> <minutes>

- `ternary(w)`
- `class TLin` (forward)
- `opt_step()`
- `sign_state()`
- `try_state(cur0, seed0, plies=8)`
- `probe(tag)`

### scratch/metallicity_diets.py
METALLICITY-1 diet grades — the same cloud at four refinements.

- `base_rows() -> list[dict]`
- `write(name: str, rows: list[dict]) -> str`
- `main() -> None`

### scratch/moe_gt1.py
MOE-GT-1 arm 0: the full-residency oracle run (pre-reg 2026-08-03).

- `instrument(model)` — Class-patch every sparse-MoE block to record top-k picks, router
- `tail_share(mass_row, frac=0.25)` — Share of total router mass carried by the top-`frac` experts.
- `main()`

### scratch/moe_gt1_arm2.py
MOE-GT-1 arm 2: residency replay at 50% / 25% / 12.5% (pre-reg 2026-08-03).

- `keep_sets_from_counts(counts, frac, top_k)` — Per-layer keep-sets at fraction `frac`. RULE env selects the
- `open_loop_recall(counts, keep)` — Count-weighted fraction of arm-0 TRUE demand inside the keep-set.
- `instrument(model, keep)` — Class-patch: masked routing (kept experts only) + closed-loop
- `_oracle_start()`
- `check_isolated(p, expr, wall=20)` — Timeboxed oracle check via a persistent SUBPROCESS line-server
- `run_gate(model, tok, problems, frac, state=None)`
- `main()`

### scratch/muon_3ep_d256.py
Muon at the STANDARD 3-epoch schedule (null-revival mix, pre-reg 2026-07-28 night): the Muon crater (10/34) was measured only in single-pass streaming with LR coupled to the surprise rider; the banked variants row says published Muon wins live at standard schedules. One cell: control construction (length-sorted BS=32, shuffled batch order, 3ep) with Muon (ns5 orthogonalized momentum) on 2-D interior weights, AdamW (OneCycle 3e-4) on embeddings/head/ norms. Comparator wfloor_d256 65 (same construction, all-AdamW).

- `ns5(g, steps=5)`

### scratch/night30_mac.py
NIGHT-30b Mac chain (pre-reg 2026-07-30): B3 K2 depth curve -> B4 entangled-experts MI (OLMoE) -> P6 entropy accounting of the Qwen3 parts. Streaming discipline; K2 shards deleted after B3. __main__-guarded.

- `b3()`
- `b4()`
- `p6()`
- `main()`

### scratch/nineteen_m_displace.py
The 19M in the crystal's displacement style (Artin's ask 2026-08-08). No training pair exists on disk until the snap birth lands, but the rational-snap family gives a real displacement of one 19M: snap19m_q32 (~base to ~1e-3, the Q=64/32 parity regime) -> snap19m_q4 (the cracked regime). Panel = "what 1/Q^2-coarse quantization moves", drawn with the crystal recipe: PCA plane of the base-proxy, displacement segments, inferno, mean-disp footer.

- `verify_deletion_stats()` — The OBSERVATION Q4-DELETION-RENDER numbers, committed

### scratch/oracle_worker.py
Standalone oracle worker for timeboxed p.check (MOE-GT-6 v3).

- `main()`

### scratch/ozaki_2b_bisect.py
*(no docstring)*


### scratch/ozaki_2b_check.py
2b re-check with an EXACT verifier (Fraction arithmetic — the first checker itself rounded: c*2^74 > 2^53).


### scratch/ozaki_2b_debug.py
*(no docstring)*


### scratch/ozaki_2b_ident.py
*(no docstring)*


### scratch/ozaki_cuda.py
Ozaki rung 2a (cuda, 3080): the wall-clock race. Slices of s=8 bits are exactly representable in TF32's 11 significant bits and the tensor-core accumulator is full fp32 — with block<=256 along K, partial sums stay <= 2^24 = exactly representable: TENSOR CORES AS EXACT INTEGER UNITS. Race: sliced-exact (full + triangular) vs native fp64 (rationed 1/64 on gaming cards) vs fp32/TF32. Error scored against a CPU fp64 reference (itself ~1e-16).

- `slices_of(F, s)`
- `sliced_matmul(A, B, s=S, block=BLOCK, tri=None)`
- `bench(name, fn, n=3)`

### scratch/ozaki_cuda2.py
Ozaki rung 2a-v2 (3080): lift the wall floor with the three named fixes. (1) WEIGHT slices amortized (static in inference/metabolism — timed loop slices only the activation side); (2) recombination grouped per (i+j) diagonal — fp64 elementwise falls 36 -> ~13 ops per block; (3) int8 tensor cores (torch._int_mm, int32 accumulate: exact at s=6 with row-wide blocks, products*N <= 2^25 << 2^31). Same error scoring vs CPU fp64 reference as v1.

- `slices_of(F, s)`
- `prep(M, s, block, side)` — block-align + slice; returns per-block (exp, [slices])
- `sliced_v2(Bmat, Aprep, s, block, tri=None, int8=False)`
- `bench(name, fn, n=3)`

### scratch/ozaki_cuda3.py
Ozaki rung 2a-v3 (3080): three lifts on the v2 crossover. (A) fp16 TENSOR CORES as exact integer units (s=8 slices exact in     fp16's 11-bit mantissa; fp32 accumulate; 2x TF32 rate on Ampere). (B) recombination bottleneck fix: per-diagonal partial sums carried     as fp32 (exact: diagonal sums of s=6 int products stay < 2^24     within a block-diagonal), converted to fp64 ONCE per block. (C) ZERO-ROUNDING OUTPUT: double-double (two-float64) accumulation     via elementwise two-sum on GPU, spot-verified against exact     big-integer arithmetic — deviation must be 0, not small.

- `slices_of(F, s)`
- `prep(M, s, block, side)`
- `run(Bmat, Aprep, s, block, tri=None, mm='fp32', dd=False)`
- `bench(name, fn, n=3)`
- `to_int(M)`

### scratch/ozaki_cuda4.py
Ozaki rung v4 (3080): two escalations past the v3 crossover. (A) RNS-GEMM (Chinese Remainder Theorem): integers represented by     residues mod k small primes — multiplication is CHANNEL-LOCAL     (k matmuls, NO cross products, no carries) vs slicing's k^2.     Reconstruction: Garner mixed-radix digits (all mod-p arithmetic     exact in fp64), assembled into double-double with 26-bit-split     radix constants (every elementwise product exact by construction). (B) fp64-INPUT exact matmul via int8 slicing — the product of two     fp64 matrices carries ~106+ bits of true detail: fp128-grade     linear algebra on a gaming card, spot-verified vs big integers.

- `to_fixed(M)`
- `rns_gemm(IA, IB)` — returns dd (hi, lo) of the EXACT integer product matrix
- `_split26(x)` — split python int into exact <=26-bit*2^shift fp64 chunks
- `slices_of(F, s)`
- `exact64(A, B, s=6)`
- `to_int64(M)`
- `big(M)`

### scratch/ozaki_cuda5.py
Ozaki v5 — THE STAY-IN-RNS PIPELINE (the exactness endgame test). A 4-layer linear chain computed ENTIRELY in residue space: residues stay < p (int8) at every depth while the positional value grows 16 -> ~124 bits; one Garner exit at the end. Sized correctly this time: 16-bit fixed-point inputs/weights (known growth: b_{i+1} = b_i + 16 + 11), 20 primes (M ~ 2^133 > 2^125 needed). Arms: (a) native fp64 chain (fast, WRONG — rounds every layer); (b) RNS single-exit (the lazy pipeline); (c) RNS exit-every-layer (what naive use would do); (d) fractional-CRT cheap estimate (the lazy exit for decisions). Truth: full big-int chain at N=128; walls at N=2048.

- `make(N, seed)`
- `to_rns(I)`
- `rns_matmul(rW, rX)` — one layer, all channels; residues in -> residues out (int8)
- `garner(rX)`
- `frac_crt(rX)` — fractional CRT: value/M mod 1 ~= sum r_i*w_i mod 1 — one fp64
- `fp64_chain()`

### scratch/ozaki_cuda6.py
Ozaki v6 — EXACT vs fp256 (which exists only as software). fp64-input matmul, N=128. Arms:   (a) our pipeline: int8-TC slices + 6-COMPONENT expansion exit       (~318 bits >> fp256's 237) — spot-verified vs big integers       (deviation must be 0);   (b) mpmath at 256-bit (dps=77) on CPU — the only way fp256 exists;       its rounding error vs the same big-int truth is measured too. Scored on both axes: exactness and wall.

- `slices_of(F, s)`
- `exact_expansion(A, B, s=6)`

### scratch/ozaki_fused.py
Fused Ozaki recombination kernel (Triton, cuda). The measured bottleneck: 36-64 separate elementwise passes (scale + two-sum per slice-pair), each a full N^2 fp64 round-trip. This kernel does the whole recombination in ONE pass: per element, loop pairs in registers, two-sum locally, write hi/lo once. Exactness preserved (every op identical, just fused). Race: v2 int8 full-exact with looped recombination vs fused; bar = beat native fp64's ~41 ms.

- `recombine_kernel(P, SC, EA, EB, HI, LO, n_pairs, NN, NCOL, BLOCK: tl.constexpr)`
- `slices_of(F, s)`
- `prep(M, s, dim)`
- `exact_fused(A, B, s=6)`
- `exact_looped(A, B, s=6)`
- `bench(name, fn, n=3)`

### scratch/ozaki_rung1.py
Ozaki rung 1: block-aligned int-sliced matmul, CPU reference. Proves EXACTNESS (not 'better'): ground truth = exact integer arithmetic on the fp32 inputs (every fp32 is a dyadic rational, so the true product is computable exactly in Python ints). Arms:   (a) plain fp32 matmul   (b) naive bitmask slicing, fp32 partials (the midnight 2x floor)   (c) aligned int-slice, int64 accumulation (the real scheme)   (d) aligned slice, s=7, fp32 accumulation (the MPS-ready variant:       fp32 units as exact fixed-point accumulators, 2s+log2(b)<=24) Alignment granularity swept: whole-row vs block-32.

- `exact_ref(A, B)`
- `relerr(C, P, sh)`
- `report(name, C)`
- `naive_slice(M, k=3, s=8)`
- `aligned_matmul(A, B, s, k, block, acc)` — A row-blocks share an exponent; B col-blocks share an exponent.

### scratch/ozaki_rung1b.py
Ozaki rung 1b: ADAPTIVE aligned int-slicing — slice until residual is exactly zero (finite mantissas terminate), so the transform is error-FREE by construction; only the fp64 recombination rounds. Metric: normwise (max abs err / max abs true) + worst entrywise.

- `to_int(M)`
- `err(C)`
- `slices(F, s)`
- `aligned(A, B, s, block, acc, adaptive=True, k=None)`

### scratch/ozaki_rung2bc.py
Ozaki rungs 2b+2c (CPU). 2b: recombine partials into a Shewchuk EXPANSION (exact two-sum chain) instead of one fp64 — the last rounding site removed; verify vs exact integer reference. 2c: chain matmuls L layers deep (linear net, no nonlinearity so the exact reference stays computable): fp32 error compounds per layer; exact pipeline carries the expansion between layers — error should stay at the OUTPUT-format floor regardless of depth.

- `two_sum(a, b)`
- `exp_add(e, x)`
- `to_int(M)`
- `slices(F, s)`
- `aligned_partials(A, B, s=8, block=32)` — yield (scaled partial matrices) — each exactly representable
- `dd_chain(mats)`

### scratch/p2_crown_draws.py
HARDENING-P2 R8: the crown tie at three fresh problem-set draws (PRE-REG HARDENING-P2). Both crown artifacts gated PAIRED per draw at G.GATE_BAND offsets +1M/+2M/+3M, one session, one device. Deterministic for frozen weights; paired per-draw deltas only.


### scratch/p3_autopsy.py
HARDENING-P3 R2 wrapper (autopsy leg): per-(layer,head) deletion map on a FRESH-SEED h8 crystal — frozen scratch/head_autopsy.py untouched (it is __main__-guarded; we import, point CKPT at the seed's EMA checkpoint, and run main()). The autopsy READS only — no save-redirect needed. After the driver's own map + min/max/ctrl full gates, two FIXED-CELL full gates (L1h7, L1h4 — the original crystal's extremes) answer the secondary identity question: do the named cells transport across seeds, or is only the STRUCTURE (sparse critical circuit) seed-stable?

- `drop(li, h)`

### scratch/p3_bits.py
HARDENING-P3 R9 wrapper: the bits-dimension 19M row pooled at n=3 — frozen drivers untouched (tournament_birth.main for the ternary arm; train_mathnative.main for the fp32 comparator), with the standard hard gates: D2 excision on load_rows (patched BEFORE either driver binds it), refuse-if-exists on every OUT, seed via BIRTH_SEED env.

- `excised_load_rows(*a, **kw)`

### scratch/p3_ffnslack.py
HARDENING-P3 R3 wrapper: ffn-slack ENDPOINTS (d56 f224 v f128) on the D2-EXCISED diet — frozen driver scratch/sym_birth.py untouched (import-and-override, the p3_umoe_soft/p3_grav2 pattern). train_mathnative.load_rows is patched BEFORE sym_birth's from-import binds it, so the excision covers the driver's diet verbatim.

- `excised_load_rows(*a, **kw)`

### scratch/p3_grav2.py
HARDENING-P3 R6 wrapper: GRAV-2 contractivity-tax paired births on the D2-EXCISED diet (frozen driver grav2_spacetime.py untouched; import-and-override, the p3_umoe_soft pattern). OUT names collide with the originals at SEED=1 — this wrapper REFUSES SEED values whose checkpoint already exists (cited-evidence guard) unless OTAG-style suffixing is added upstream; run seeds 2/3/4.

- `excised_load_rows(*a, **kw)`

### scratch/p3_quat.py
HARDENING-P3 R7 wrapper: the quaternionic 4x conversion toll gets n=3 warm-epoch seeds — frozen scratch/quat_convert.py untouched (import-and-override; the driver runs at import).

- `class RedirectedRandom`
- `redirected_seed(s)`
- `excised_load_rows(*a, **kw)`
- `redirected_save(obj, path, *a, **kw)`

### scratch/p3_stream2x2.py
HARDENING-P3 R4 wrapper: the cooldown small-delta cell of the streaming 2x2 gets n=3 paired seeds — frozen driver scratch/streaming_birth_d256.py untouched (import-and-override).

- `excised_load_rows(*a, **kw)`
- `redirected_save(obj, path, *a, **kw)`

### scratch/p3_umoe_soft.py
HARDENING-P3 R5 wrapper: UMOE soft-routing seeds on the D2-EXCISED diet (frozen driver umoe_conserve.py untouched — import-and-override; the loader filter is the Phase-1 law applied at load time, receipt printed).

- `excised_load_rows(*a, **kw)`

### scratch/pack_baselines.py
PACKED CRYSTAL C3 (pre-reg 2026-07-29 eve): GPTQ/AWQ/HQQ honest table on d64h8 EMA. Baselines from llmopt/quantize/methods.py on every block Linear; calibration activations hooked from 24 prompts at GATE_BAND+500_000 seed offsets (never the gate band). Arms: {rtn,gptq,awq,hqq} x {5,3} bits -> full gate + mean DeltaKL v fp logits + calibration wall-time. C1 controls reused (fp 58, packed 58). __main__-guarded.

- `calib_prompts(tok)`
- `capture(model, prompts, dev)` — -> {key: X [n, in]} inputs of every block Linear + fp logits.
- `main()`

### scratch/pack_c6.py
PACKED CRYSTAL C6 (pre-reg 2026-07-29 night, Artin's GO): external validity on Qwen2.5-0.5B (3080). Arms: sigma-pack (q=ceil(2/sigma) per tensor, closed form) v HQQ (matched bits, group 64) v RTN. Score: mean DeltaKL/token v fp16 on 16 fixed prompts + perplexity on a fixed README slice + per-arm quantization wall-time. Fake-quant only. __main__-guarded.

- `sigma_pack(w)`
- `sigma_pack_row(w)` — C6b/C6c: per-output-row sigma — still closed-form, zero
- `main()`

### scratch/pack_c7.py
PACKED CRYSTAL C7 (pre-reg 2026-07-29 late, Artin's GO): the at-capacity transport claim. OLMoE-1B-7B: sigma[row] v rtn v hqq fake-quant on ROUTED EXPERT tensors (control arm: same on dense attention tensors). Capacity meter reads both groups first. DeltaKL on 16 fixed prompts + README-slice ppl + wall-times. Mac 36GB / MPS. __main__-guarded.

- `sigma_row(w)`
- `group_of(name)`
- `main()` — Streaming design (v2 after the OOM kill): the model is

### scratch/pack_crystal.py
THE PACKED CRYSTAL C0+C1 (pre-reg 2026-07-29 eve): real bytes for the sigma-law. C0: per-tensor denominator q_t = ceil(2/sigma_t) (grid step <= sigma/2, below the knee), codes = round(W*q_t) packed to ceil(log2(span)) bits, one (q_t, offset) per tensor -> .npz + reader. Norms/emb/head stay fp32 (tiny, never snapped). C1: full gates on packed v fresh fp control, same device — bar: within sigma (~3.5). Reports bits/wt, Shannon entropy of the code stream (Gaussian-capacity check), artifact bytes v fp32/fp16. __main__-guarded.

- `pack_tensor(w)` — -> (packed bytes, q, minc, bits, shape, entropy_bits_total)
- `unpack_tensor(packed, q, minc, bits, shape)`
- `pack_crystal(sd, path)` — Pack all 2-D block weights; passthrough the rest. Returns stats.
- `load_crystal(path)`
- `main()`

### scratch/pack_decode.py
P3 THE DETERMINISTIC DECODE (pre-reg 2026-07-30): fixed-point twin of the MicroLM forward on the packed d64h8 crystal. Every op is exact integer arithmetic or a SHIPPED-table lookup — no libm in the path. GEMMs run on the exact-fp32 integer carrier with hi/lo splitting (every partial < 2^24, printed). Tables are generated once (CPU) and saved to checkpoints/p3_tables.pt — ship the SAME file to every device. Usage:   python scratch/pack_decode.py tables   # generate + save tables   python scratch/pack_decode.py hash     # 40-tok greedy battery hash   python scratch/pack_decode.py gate     # full gate (capability price) __main__-guarded.

- `make_tables()` — CPU, once. Shipped — never regenerated per device.
- `rdiv(x, d)` — round-half-away integer division, exact + deterministic.
- `isqrt_newton(n, iters=30)` — integer sqrt of int64 scalar tensor via Newton; exact floor.
- `class DetLM` (gemm, rmsnorm, rope, attn, step, gemm_embed, greedy)
- `cmd_hash()`
- `class GateShim` (forward, eval)
- `cmd_gate()`
- `cmd_battery()` — A1 gate-pooling cell (revival-sweep Tier A, 2026-07-31):

### scratch/pack_determinism.py
PACKED CRYSTAL C4 (pre-reg 2026-07-29 night): cross-device determinism. Hash A: integer-GEMM outputs of every block Linear's sigma-law codes x a fixed integer activation battery, accumulated via fp64 matmul (all partials integers < 2^53 -> EXACT, reduction- order-invariant) — must match across devices. Hash B: fp32 full forward logits on a fixed prompt battery — expected to differ. Greedy token streams reported alongside. Run on each machine at the same commit; compare printed hashes. __main__-guarded.

- `main()`

### scratch/pack_gemv.py
PACKED CRYSTAL C2 (pre-reg 2026-07-29 night): dequant-fused sigma-pack GEMV. Runtime format: int8 codes (byte-aligned twin of the 5-bit disk pack) + ONE fp scale per tensor (1/q_t). Kernel in int4_gemv v3 style: one simdgroup per row, char4 weight + half4 activation loads, simd_sum, scale once per row. Correctness v fp reference, then bench v fp16 GEMV at crystal AND large shapes; bandwidth-model prediction printed next to measured. mx.eval every timed iteration (lazy-graph scar). __main__-guarded.

- `pack8(w)` — fp [N, D] -> (codes int8, scale). sigma-law: q = ceil(2/std),
- `crystal8_gemv(x, codes, scale)`
- `bench(fn, iters=200)`
- `main()`
- `pack5(w)` — fp [N, D] (D % 6 == 0) -> (words uint32 [N, D/6], scale,
- `crystal5_gemv(x, words, scale, d)`
- `main5()`

### scratch/pack_p2a.py
P2a-v2 THE ANALYTIC-CLIP ALLOCATOR (pre-reg 2026-07-29 close): zero-calibration span attack on SmolLM2-1.7B (Mac). Arms: rtn per-row absmax | sigma-clip k in {4,6,8} (grid over +-min(absmax, k*sigma), outliers saturate) | hqq. DeltaKL + README ppl + wall-time, C6 harness form. __main__-guarded.

- `grid_q(w, rng, bits)` — Per-row symmetric uniform grid over [-rng, rng]; saturate.
- `main()`

### scratch/pack_rans.py
P6-v2 (pre-reg 2026-07-30): rANS the packed artifacts — the entropy bound as real bytes. constriction static-Categorical rANS per tensor (frequency table stored alongside, overhead counted). Cells: house packed_*.npz crystals; Qwen3 blackhole parts. Every stream verified by exact roundtrip. __main__-guarded.

- `rans_bytes(codes, verify=True)` — codes int array -> (compressed bytes incl. table, entropy
- `house()`
- `qwen()`
- `main()`

### scratch/pack_tiered.py
PACKED CRYSTAL C5 (pre-reg 2026-07-29 night): the tiered pack. matryoshka_d56_3tier.pt -> nested artifact: non-gate tensors packed once (C0 rule); gate.weight payloads nested — tier-8 base = numel/8 orbit representatives of P_C8, tier-2 payload = numel/2 delta v the QUANTIZED tier-8 prediction, dense payload = full delta v reconstructed tier-2. Each payload sigma-law-quantized on its own sigma. Desk identity check, then full gates on all three packed tiers (booked fp tiers: 57/57/48). __main__-guarded.

- `perm(n, nb, s)`
- `project(W, nb)`
- `recon(rep, nb, n_out)` — rep [n_out/nb, n_in] (rows i%nb==0 of a C_nb-invariant W).
- `squant(x)` — sigma-law quantize; -> (xq, q, bits, nbits_total)
- `main()`

### scratch/paper_figs.py
Publication PDF figures for the entropy-bound packing paper.

- `_save(fig, name: str) -> Path`
- `fig_packing_curve()`
- `fig_capacity_meter()`
- `fig_quant_knee()`
- `fig_symmetry_toll()`

### scratch/phys_probe.py
Physics rung 1 probe: greedy emission on held-out phys steps (seeds 17-19), sympy-equivalence in t, fork-isolated. No math gate — the physics expert is vocab-41, a separate model class by design. Usage: phys_probe.py <ckpt>

- `_equiv(q, pred, gold)`
- `equiv(pred, gold, deadline=10)`

### scratch/pincer_dist_probe.py
Pincer distribution readout (Artin's quantum-frame ask, 2026-07-26 night; pre-reg in RESULTS): the engine enumerates the COMPLETE legal move set for a state (the classical superposition, exact by construction); each crystal supplies a distribution over it (teacher-forced sequence log-probs, no generation). Question: are the amplitudes CALIBRATED — does model mass track which moves actually lead to the root (fork-isolated engine solves as value labels)?

- `_solve_worker(expr_s, q)`
- `solve_isolated(expr_s)`
- `load(ckpt)`
- `seq_logp(model, prefix, targets)` — Raw (T=1) summed log-prob of each target continuation after
- `spearman(xs, ys)`

### scratch/pincer_dist_report.py
Aggregate logs/pp_dist_probe.jsonl (pincer distribution readout). Every dimension the sidecar carries, reported against chance: per-model calibration (mass-on-solving v uniform baseline, top-1 v chance, Spearman, entropy), per-level split, per-rule-family solve rates + model mass, calibration deciles (pooled children: predicted mass v realized solve freq), length bias. Pure read — no model, no oracle.


### scratch/pincer_labels_v2.py
Pincer label prep v2 — the MIGRATED replacement for pincer_r1b_labels.py (migration mandated by the results-hardening spec: the v1 driver calls successors() IN-PROCESS under derivation.py's SIGALRM box — the one true checkpoint-selection carrier left in the program. SIGALRM cannot box sympy; rows whose replay hangs vanish silently and the MISS class is undercounted).

- `_worker(cur_s, nxt_s, q)` — Replay one row; runs in a fork, killed from outside.
- `classify(names)` — Verbatim v1 decision structure (behavior-guarded).
- `main()`

### scratch/pincer_r0.py
Pincer R0: conjecture-leg readout (spec 2026-07-26-reverse-llmue-pincer.md, cell R0; pre-reg in RESULTS).


### scratch/pincer_r0b.py
R0b: collapse-ordered readout (pre-reg in RESULTS 2026-07-26 late). The honest Grover residue: does checking candidates in descending model-mass order reach the first verified solution in fewer ORACLE CALLS than random/sampling order? All candidates are oracle-checked once (instrument cost, not protocol cost); orders are then evaluated on the recorded truth.

- `seq_logp(prefix, targets)`

### scratch/pincer_r1_indist.py
Pincer R1a-INDIST (Phase-4 row 1 of the results-hardening false-null hunt; pre-reg in RESULTS before fire): the R1a peeling probe re-run on IN-DISTRIBUTION prompts.

- `_mint_child(prob, q)` — One forward engine step; first encodable child (fork).
- `main()`

### scratch/pincer_r1_probe.py
Pincer R1 backward-validity probe (spec 2026-07-26-reverse-llmue-pincer.md, cell R1; pre-reg in RESULTS).


### scratch/pincer_r1b_labels.py
Pincer R1b prep: (t, rule, child)-label recovery by engine replay (spec 2026-07-26-reverse-llmue-pincer.md amendment 1/2).


### scratch/pincer_r8.py
Pincer R8: meet v1 — full protocol (conjecture + peel + meet) vs let-it-finish forward re-roll at equal sampled-token budget (spec amendment 3; pre-reg in RESULTS 2026-07-26 night).

- `load(p)`
- `wave(model, cur, seeds, arm)`
- `chain_search(model, root, seed0, arm, goal=None, plies=12)` — Greedy verified chain (gate discipline); if goal set given,

### scratch/place1_gravity.py
PLACE-1 (pre-reg 2026-07-30): inference-time gravity — co-routing prefetch v popularity on real OLMoE traces. Mac, after UMOE-3. Usage: python scratch/place1_gravity.py

- `collect()`
- `main()`

### scratch/polar_snap.py
POLAR-SPLIT SNAP (pre-reg 2026-07-29 day: escalation-engine cell 4). cplx_none.pt (unconstrained complex FFN, d384/f1536/h6): quantize the complex gate/up weights |c| COARSE x arg(c) FINE v uniform re/im grids, both expressed in sigma units (sigma law rider), bits/complex MEASURED as log2(#distinct values used). Fence: gate+up only (the complex-paired tensors); qkv/o/down untouched. Desk, MPS.

- `split(W)`
- `snap_uniform(sd, u)`
- `snap_polar(sd, mstep, na)`
- `gate(sd, tag, bits=None)`

### scratch/practice_mine.py
PRACTICE MODE, model-side (the mirror of axiom's arg-10): duo-wave rollouts that (1) BANK verified steps from ALL attempts — solved or not (the solved-only leak fix, Artin) — tagging rows by outcome so the gen-8 A/B can split them; (2) LOG stuck states — the exact cur where every unsolved attempt died — to a worklist in axiom's format ({id, level, root, from, why, plies}), ready for the stuck-state exchange AND as maximum-surprise metabolic v4 food.

- `skeleton(e)`
- `binof(n)`

### scratch/probe_int_device_parity.py
Probe: are the integer-battery primitives bit-identical off the CPU?

- `pick_device(argv)`
- `primitives(dev, tensors)` — Every integer op the battery's forward and backward chain uses.
- `main()`

### scratch/prologue_arms.py
Zero-birth prologue arms (Opus-5 reviewer, 2026-07-25): S4 symmetry-without-zero PTQ, sparsity control at ternary's zero-fraction, and the gauge-commutation checkpoint pair.

- `s4_rows(w)` — {±1/3, ±1} x per-channel absmax: symmetric, 2 bits, NO zero.
- `ternary_rows(w)` — Absmean ternary (reference zero-fraction source).
- `sparse_rows(w)` — fp32 magnitudes, pruned to the SAME zero-fraction ternary
- `gauge_flip(sd)` — Sign-flip gauge on half the FFN hidden units: flip rows of
- `main()`
- `m4_rows(w)` — Asymmetric M4-style PTQ {-1,0,1,2} x per-channel scale — the

### scratch/ptq4_arms.py
Scalar 4-bit PTQ arms (the tournament's missing bracket point): P4 powers-of-two ladder, LM-16-zero (k-means), NF4-style quantile codebook — all per-output-channel, on the 19M infix twin. Rides the Lloyd-Max race harness; gates run separately on MPS.

- `p4_rows(w: torch.Tensor) -> torch.Tensor` — {0, ±1/4, ±1/2, ±1, ±2, ±4, ±8} x per-channel unit (absmax/8):
- `nf4_rows(w: torch.Tensor, k: int=16) -> torch.Tensor` — Equal-mass quantile codebook per channel (NF4-style, but on
- `main() -> None`

### scratch/quat_commutant.py
Symmetry ladder S1 cell 1 (pre-reg 2026-07-28): quaternionic anti-commutant mass of FFN gate matrices. Structures I,J,K = left quaternion-unit action on 4-channel groups (I^2=J^2=K^2=-1, IJ=K); P(W) = (W - IWI - JWJ - KWK)/4; anti-mass = 1 - ||P(W)||^2/||W||^2 (0.75 = fully generic, 0 = exactly quaternionic-linear). Synthetic controls run FIRST (must read 0.0 / ~0.75) — instrument fence. Real crystals: adjacent 4-grouping + 20 random-grouping nulls.

- `quat_structs(n, perm)` — Three anticommuting structures on 4-tuples
- `project(W, So, Si)` — Commutant projection: group-average over {1,-I..,-J..,-K..}.
- `anti_mass(W, So, Si)`
- `gates(sd, layers)`

### scratch/quat_convert.py
Symmetry ladder S1 cells 2-3 (pre-reg 2026-07-28): project the wfloor d256 gates onto the quaternionic commutant (deletes 75% of gate mass), gate the projected init, then warm-train 1 epoch. Arm a: lambda=0; arm b: commutation penalty summed over I,J,K, ramped 0.1->1.0. R3 recipe verbatim otherwise. Usage: ARM=a|b python scratch/quat_convert.py


### scratch/qwen_displace_extract.py
Extract one gate_proj matrix from HF-cached Qwen2.5-0.5B base and Instruct into plot_neurons-compatible .pt files, so the --displace (central-lattice whisper-zoom) view can render an INTERNET-trained model's post-training displacement next to the closed-system natives' (the chaos-vs-structure tell, Artin's ask 2026-08-08; generator-loss lesson: this adapter is COMMITTED).


### scratch/rank_read.py
Rank read (pre-reg 2026-07-29: attention anatomy 1b). SVD of all qkv/o weights of the d56 EMA crystal: singular-value decay, then truncation gates at rank r in {48,32,24,16}. Desk only, MPS.

- `truncate(W, r)`

### scratch/rat_deploy.py
Deploy a born-rational (RAT_Q) crystal: apply the SAME snap the STE trained through (s * best p/q, q <= Q, s = per-tensor absmean) to every 2-D weight — the output IS the trained function, exactly on-lattice. Usage: rat_deploy.py <ckpt_in> <Q> <ckpt_out>


### scratch/rat_repair.py
Snap+repair (RIFF 2026-07-27, precision-as-thin-film): take a snapped crystal, FREEZE every 2-D tensor (the exact lattice stays exact), train only the 1-D parameters (norms/biases — the 'thin precise film') briefly on the birth diet, save. If a few thousand precise params recover the snap deficit, precision is a small additive budget, not a per-weight property. Usage: rat_repair.py <ckpt_in> <diet_jsonl> <steps> <ckpt_out> Env: VOCAB_EXTRA (must match birth), shape via D/LAYERS/FFN/HEADS.


### scratch/rational_snap.py
Rational-snap distillation (RIFF 2026-07-27, Artin's infinite-precision push, rung a): snap every 2-D weight of a gated crystal to the nearest fraction p/q with denominator q <= Q, then gate the snap. Asks "do trained weights want simple exact numbers?" as a COMPRESSION question (precision doctrine stays closed; E3 is its sole reopening).


### scratch/rev2_d768.py
REVIVE track item 3 wrapper: the d768 ternary-v-fp32 crossover gets n=3 SAME-DEVICE paired seeds (the p3_bits pattern at d768; p3_bits itself is R9-cited and stays frozen).

- `excised_load_rows(*a, **kw)`

### scratch/rev3_crown.py
REVIVE track item 2 wrapper: CROWN-TIE BIRTHS — the production crown tie (gen6_grown fp32 76 v merged_grown ternary 75, booked 2026-07-23; draw-noise leg closed by HARDENING-P2 R8 "tiebreak still needs births") gets n=3 SAME-DEVICE fresh birth pairs.

- `excised_load_rows(*a, **kw)`
- `_gate(ckpt, d, layers, heads, ffn)`

### scratch/rev4_zx45.py
HARDENING-P4 row 2 wrapper: the 45M ZX scale-lever null gets its seed ladder. The 45M union verdict (2026-07-27: ZX 36 = +0.8 sigma inside the 19M seed fence mean 32.7 sd ~4.2) was n=1 on a gate class with a measured 8-point seed swing — the highest-value seed-starved null in the file after pincer. This wrapper reruns the FROZEN recipe (night_45m_union.sh: union diet math gen-4 + zx_farm1, vocab-47, fp32, d512/L12/ffn2048/h8, 3ep) at fresh seeds; the pooled 3-seed read (booked s1=36 + fresh) goes against the 19M fence per the original pre-reg's own framing.

- `excised_load_rows(*a, **kw)`

### scratch/rot_commutant.py
Rotational snap R1 (pre-reg 2026-07-28): anti-commutant mass of FFN gate matrices under channel-pairing complex structures. W_a = (W + J_out W J_in)/2; mass = ||W_a||^2/||W||^2 (0.5 = no rotational structure; 0 = fully complex-linear). Real crystals: adjacent pairing + 20 random-pairing nulls. Complex-FFN arms: native half-split pairing (positive control, expect ~0).

- `J_perm(n, perm)` — Block rotation: pairs (perm[2k], perm[2k+1]); J e_a = e_b,
- `J_half(n)`
- `anti_mass(W, Jo, Ji)`
- `gates(sd, layers)`

### scratch/rot_convert.py
Rotational snap R3 (pre-reg 2026-07-28): warm-train the t=1.0 projected wfloor for 1 epoch; arm a: lambda=0 (does SGD restore the anti-commutant?); arm b: commutation penalty ramped 0.1->1.0. Reports gate + final anti-mass. Usage: ARM=a|b rot_convert.py


### scratch/rot_snap_anatomy.py
Rotational snap R2 (pre-reg 2026-07-28): gate wfloor_d256 with gate matrices projected toward the commutant — W - t*W_a under adjacent pairing, t in {0.25, 0.5, 1.0}. Fence: gate matrices only (attention/up/down untouched). Flips-probe fingerprint per t rides (vs the unmodified model, teacher-forced argmax diff).

- `project(sd, t)`
- `gate(sd, tag)`

### scratch/rotinstr_control.py
A3 (revival-sweep Tier A, 2026-07-31): rotation-instrument POSITIVE CONTROL. Run the weight-side rotation instruments (weight-FFT euler lenses + anti-commutant mass) on the FOURIER-2b crystal, the one substrate with a CONFIRMED activation clock (276/512 periodic neurons at k=5). Two informative outcomes:   - instruments read NULL here too -> weight-side lenses are blind     to activation clocks (the old spontaneous-rotation nulls said     nothing about representations);   - instruments FIRE -> the old nulls were DIET statements (no     forced periodic computation), per the clock-placement law. CPU, minutes. Usage: python scratch/rotinstr_control.py

- `ks_uniform(theta)`
- `phase_stat(W)`
- `fft_stat(W)`
- `J_perm(n, perm)`
- `anti_mass(W, Jo, Ji)`
- `main()`

### scratch/saturation_s2.py
SATURATION-1 cell (a): +1 warm epoch at fixed food on a COPY of the grown-s2 champion (PRE-REG SATURATION-1). Crown artifact frozen; this trains checkpoints/sat_s2.pt (epochs=4 resumes the 3-ep state one epoch). Then the standard gate, dict printed. Usage: .venv/bin/python scratch/saturation_s2.py

- `_load_rows(*a, **k)`

### scratch/saturation_s2b.py
SATURATION-1 cell (b): +1 epoch on WIDENED rations (AMENDMENT SATURATION-1-CELL-B). Warm corpus + 20% gen-8 slice, string-seeded. Usage: .venv/bin/python scratch/saturation_s2b.py

- `_load_rows(*a, **kw)`

### scratch/scaffold_review.py
Overnight scaffold review: MERGE-1 on gravmoe_s{S}, channel ablation on channel_s{S} (S env, default 2; missing files skip). CPU. Runs as part of overnight chains.

- `gate(m, tok)`
- `main()`

### scratch/scorer_s1_battery.py
S1: the frontier battery + persistent value cache (spec 2026-07-27-calibrated-scorer; pre-reg in RESULTS).

- `add(lv, s)`
- `_worker(idx, exprs)`

### scratch/scorer_s2_data.py
S2 data farm (calibrated-scorer spec): training rows for the listwise scorer. Stratified sample of unique corpus states -> full legal enumeration -> per-child value labels (cache-aware fork solves, budget 150, 8s walls, 6 workers, streamed shards) + the replayed true-move label where the corpus row matches a legal child (the R1b 68%). Every solve extends the permanent value cache.

- `_enum_worker(idx, items)`
- `_worker(idx, exprs)`

### scratch/scorer_s2_train.py
S2 trainer (calibrated-scorer spec): the listwise objective race. State + enumerated legal set -> distribution in ONE forward pass (teacher-forced seq log-probs, softmax over the set, zero generation). Two arms, one variable (the target distribution):

- `targets(ch, true_idx, arm)`
- `batch_logps(model, state, ch, grad=False)` — Summed seq logp of each child continuation (padded batch).
- `run_arm(arm)`
- `spearman(xs, ys)`
- `eval_battery(name, score_fn)`

### scratch/series_probe.py
Series rung 1 probe: greedy next-partial-sum emission on the 142 held-out steps (seeds 17-19), scored by sympy polynomial equivalence in fork-isolated workers (the solve_isolated doctrine). Also runs the standard 120 gate for the paired regression read vs seedvar-1 (65). Usage: series_probe.py <ckpt>

- `_equiv(q, pred, gold)`
- `equiv(pred, gold, deadline=10)`

### scratch/snap_alloc.py
Snap allocation (pre-reg 2026-07-29: attention anatomy 1c). Rational snap at Q=16 (below the (16,24] knee) applied to attention-only v gate-only v both, on the d56 EMA crystal. Allocation-of-accuracy read for the bits-portfolio riff. Desk only, MPS.

- `snap(w, q_max)`

### scratch/snap_anatomy.py
Sensitivity-wall anatomy (Artin 2026-07-27: "find WHERE the wall lives"): single-tensor Q=16 snap ablation on the 19M crystal. For each 2-D tensor alone-snapped (rest fp32), measure teacher-forced divergence vs control on gen-4 rows: mean KL + argmax-flip rate. Localization instrument (CPU, no gate contention with the births); top culprits earn real gates later. House pre-reg guess: head/attn out-projections carry the wall, ffn interiors tolerant.

- `snap(w)`

### scratch/softprompt1.py
SOFT-PROMPT-1: is there capability behind the tokenizer?

- `with_virtual_tokens(ckpt, d, layers, ffn, heads, dev)` — Model whose vocab is V+P: rows [:V] are the frozen checkpoint,
- `diet_batches(tok, n_steps, bs=24, seq_cap=192)` — CE batches from the standard diet (same rows the ckpt was
- `train_prefix(model, tok, prefix_ids, dev, steps)`
- `gate_with_prefix(model, tok, dev, prefix_ids)` — The standard 120 gate, prompts prefixed with the virtual ids.
- `main() -> None`

### scratch/softprompt_sampler_probe.py
Why a bit-identical model gates differently behind the virtual-token harness (AMENDMENT SOFT-PROMPT-1-SAMPLER).

- `_load_softprompt()`
- `part1_model() -> None`
- `part2_sampler(n_seeds: int=200, rollout: int=30) -> None`

### scratch/soup_gate.py
Night-28b soup instrument: plain parameter mean of N checkpoints (same shape), then gate. Usage:   soup_gate.py TAG d layers ffn heads ckpt1 ckpt2 [ckpt3 ...] VOCAB_EXTRA rides (atom order must match the births).


### scratch/ssm_star.py
SSM-STAR (pre-reg RESULTS PRE-REG SSM-STAR-1): minimal selective state-space model in the micro-star family — the house's first SSM.

- `build_ssm_model(vocab_size: int, d: int=64, layers: int=8, heads: int=4, ffn: int=256, ctx: int=512)` — heads accepted for signature parity; SSM has no heads.
- `main()`

### scratch/star_profile.py
Tensor-class ternary sensitivity profile (PRE-REG STAR-PROFILE-1, 2026-08-10). Star frame: precision belongs to interfaces/core; traversal tolerates {-1,0,1}.

- `ternary(w)` — absmean ternary — row-scale for 2D (train_ternary.py's form),
- `main()`

### scratch/streaming_birth_d256.py
Streaming-birth A/B, arm S (RIFF-LEDGER 2026-07-24 "Streaming birth").

- `template_refresh(model)`
- `ns5(G, steps=5)`

### scratch/successors_acceptance.py
Successors-bridge acceptance (house side; axiom spec 2026-07-27-successors-bridge, relay -28-2). 500 string-seeded gen-4-band roots (L1-L8), house derivation.successors vs axiom_sym.successors, E4-taxonomy decomposition: - MATCH: child sets equal (sympy-srepr normalized) - HOUSE_ONLY / AXIOM_ONLY children (named, sampled) - I-FENCE: complex-carrier states skipped (axiom domain fence) - EXPIRED: axiom deadline states (censored, never counted false) Soundness leg: every axiom-only child re-verified on the HOUSE oracle (verify_edge) — axiom emissions must never fail it. Throughput logged both sides.

- `norm(e)`

### scratch/sym45.py
C8-retrofit at 45M (pre-reg 2026-07-28 ~5PM): project union_45m gates onto the C8 commutant (params/8), one warm epoch on the union diet, ramped permutation penalty. Prints projected-init math gate, then trains and saves; math+ZX final gates run via gate scripts after. cuda/bf16 autocast.

- `shift_perm(n, sh)` — index map: row r <- r shifted by sh within its 8-block.
- `project(W)` — C8 group average via double permutations (cheap, exact).
- `anti_mass(W)`

### scratch/sym_birth.py
Symmetry-at-birth (pre-reg 2026-07-28 night): C8 at d64, from SCRATCH. Arm dense = plain birth control; arm c8 = commutant- projected init + ramped generator penalty from step 0. Paired on one device, seed 1, lr 1.5e-3, bs 8, gen-4, 3 epochs. Usage: ARM=dense|c8 python scratch/sym_birth.py

- `_env(k, d)`
- `shift_reps(n)`
- `project(W, Ro, Ri)`
- `anti_mass(W, Ro, Ri)`

### scratch/sym_convert.py
Symmetry ladder S3/S4 (pre-reg 2026-07-28): generic group- average conversion. GROUP=z2 (sign involution, params/2) or circ8|circ16 (cyclic shifts within n-blocks, params/n). P(W) = avg_g R_o(g) W R_i(g)^T (orthogonal reps). Prints anti-mass read + nulls, projected-init gate, then warm-trains 1 epoch (ARM=a lambda=0 | ARM=b ramped generator penalty). Usage: GROUP=z2 ARM=b python scratch/sym_convert.py

- `reps(n, perm)` — Group elements as (n,n) orthogonal matrices on perm order.
- `project(W, Ro, Ri)`
- `anti_mass(W, Ro, Ri)`

### scratch/sym_spectrum.py
Symmetry spectrum (pre-reg 2026-07-29: Artin's superposition riff): isotypic decomposition of wfloor d256 gate weights under C8 conjugation into 5 real frequency bands; report band masses; gate CUMULATIVE reconstructions in descending-mass order. comp_k(W) = (1/8) sum_s w^{-ks} R^s W R^{-s}; real bands pair k with 8-k. Desk only (no training), MPS.

- `shift_perm(n, sh)`
- `conj_s(W, s)`
- `band(W, ks)` — Real isotypic component for the frequency set ks.

### scratch/synonym_test.py
Synonym gauge test: TWO label tokens per family on the frozen 19M readout (vocab 40 -> 55: <name> + 7x2 synonyms). Train rows pick either synonym 50/50. Gauge-law prediction: both fire near-equal off the same concept. Reports family-accuracy + per-synonym share.

- `label_of(s)`
- `_one(args)`
- `gen(n, band, exclude=None)`
- `encode(e)`

### scratch/tenet_d1_revgate.py
TENET D1: THE REVERSE GATE (spec 2026-08-05-tenet-battery.md).

- `rev_gate_eval(model, tok, dev, n=None, mode='start')` — Reverse gate: solves per level + per-candidate equivalence %.

### scratch/tenet_d2_revdiet.py
TENET D2: the certified reversed diet + exclude-union semantics (spec 2026-08-05-tenet-battery.md).

- `worker_main()` — Line server: json [cur, nxt] per line -> status.
- `class Replayer` (start, send, poll, check_sync, kill)
- `gate_band_exprs()`
- `main()`

### scratch/tenet_d3_budget.py
TENET D3: the budget accountant (spec 2026-08-05-tenet-battery.md deliverable D3; fixes PINCER R8's booked instrument defect — "my equal-budget fence was violated by my own design": the peeler's 154,641 sampled tokens rode ON TOP of the 27,053-token forward budget instead of being traded against it, 5.7x for +1 solve).

- `class BudgetAccountant` (remaining, debit, census)
- `charge(acct: BudgetAccountant, who: str, texts, tok_counts=None)` — Charge a sampled wave; return the texts, or [] if refused.

### scratch/tenet_mult_b32.py
MULT-0 B=32 leg (named follow-up of VERDICT MULT-0; Artin GO): is choice scarcity a BUDGET property (surface widens with samples) or a MODEL property (stays thin)? Wrapper only — sets G.B=32 then runs the cited census driver's path verbatim (tenet_mult_census is now booked evidence; never edited).


### scratch/tenet_mult_census.py
MULT-0: the verified-candidate MULTIPLICITY census (spec 2026-08-07-morning-specs.md item 3; rung 1 of the reverse-propose ladder, RIFF 2026-08-07; the pincer closure's fence made measurable).

- `mult_census(model, tok, dev, n=None, mode='poststep', log_f=None)`

### scratch/tenet_r1b_micro.py
TENET R1b-micro: the closed loop at matched budget (PRE-REG TENET-R1B-MICRO, 2026-08-06 — read it first; arms, ledger contract, and registered lines live there, not here).

- `tok_cost(tok, text)`
- `rev_score(rev_model, tok, cand, cur, dev)` — Cycle-consistency: teacher-forced logprob of CUR given CAND
- `run_problem(models, tok, dev, root, seed0)`
- `main()`

### scratch/tenet_w0.py
TENET W0: is reverse structure visible in weights? (pre-reg TENET-W0, 2026-08-05 — the battery's cheapest rung.)

- `_fit_xy(x_in, y_out, torch_seed)` — The subjects._fit loop on arbitrary (input, output) data —
- `make_inverse_twin(family, i, seed)` — Fit an MLP to the axis-swapped data of a fresh draw from
- `main()`

### scratch/tenet_w1_bridge.py
TENET W1 bridge: crystal-weight tokenizer + direction reader (spec 2026-08-05-tenet-battery.md rung W1; the reader-shape gap named by the 2026-08-06 reviewer scan — weightspace.reader is pinned to HIDDEN=16 MLP subjects and cannot read d64/L8 crystals).

- `load_subject(ckpt)` — [8, 256, 64] gate-weight stack from a sym_birth checkpoint.
- `tokenize(gates, rng)` — Sample TOK_PER_BLOCK neurons/block -> [8*T, 64] + block ids.
- `class DirectionReader` (forward)
- `main()`

### scratch/tenet_w1_population.py
TENET W1 population build (spec 2026-08-05-tenet-battery.md, W1 prerequisite; Artin GO 2026-08-06, 3080 window into 17:00 EST).

- `birth(seed, direction)`
- `main()`

### scratch/tenet_w1_relational.py
TENET W1-R: the RELATIONAL weight reader (Artin riff banked 2026-08-06 late; the licensed new representational hypothesis after the W1-S surface-exhaustive null).

- `load_subject(ckpt_or_sd)` — [7, 256, 256] cross-layer alignment stack C_l, or its
- `tokenize(subj, rng)`
- `class DirectionReader` (forward)
- `main()`

### scratch/tenet_w1_surfaces.py
TENET W1-S: the feature-surface ladder on the EXISTING W1 population (queued by Artin 2026-08-06 behind the GT-7 booking; LOCKSTEP spec C4 rung 1). VERDICT TENET-W1 read direction at chance (10/20) from FFN GATE rows; the rider proved the instrument sound (randinit control 20/20). Question: does direction live in any OTHER weight surface at this scale?

- `load_surface(ckpt)`
- `surface_from_model(m)` — The same stacking applied to a live model's state dict
- `tokenize(gates, rng)`
- `class DirectionReader` (forward)
- `main()`

### scratch/ternary_control.py
Deploy-ternarize the NNUE-metabolized latents, honest gate + L9 probe on cuda. Doctrine: gate the DEPLOYED 1.58-bit snapshot.

- `ternary(w)`

### scratch/ternary_gate.py
Deploy-ternarize the NNUE-metabolized latents, honest gate + L9 probe on cuda. Doctrine: gate the DEPLOYED 1.58-bit snapshot.

- `ternary(w)`

### scratch/ternary_session2.py
Ternary compounding session #2 (Mac, MPS lineage, paired gates): the doctrine-composed organism — STE ternary latents, LATE layers only (8-11), LR 1e-4 cap, ABSOLUTE-anchor tripwire, fp32-vs-fp64 update-absorption instrument riding along. Pre/post MPS gates make it a clean paired delta.

- `ternary(w)`
- `class TLin` (forward)

### scratch/tier_escalate.py
3-rung escalation policy (pre-reg 2026-07-29 night: cell 1). matryoshka_d56_3tier.pt: per gate row eighth -> half -> dense, escalate on oracle-fail. TierP inlined (module-level-script scar). Desk, MPS.

- `shift_perm(n, nb, sh, dev)`
- `class TierP` (forward)
- `try_row(lv, i, nb)`

### scratch/tier_retry.py
Tier-retry controller (pre-reg 2026-07-29: attention-core Leg 0). d56 matryoshka pair: attempt each gate row on the CHEAP tier (commutant projection, 1/8 gate params); on failure retry the same row on the DENSE tier. Oracle-fail = the free difficulty signal. Reports retry solves, the overlap census, and effective gate-params per row. Desk only, MPS.

- `shift_perm(n, sh, dev)`
- `class TierP` (project, forward)
- `try_row(lv, i, tier_on)` — One gate row under the given tier; same seeds as gate_eval.

### scratch/train_fp64.py
fp64 end-to-end birth (the rounding-loss-veil A/B, banked 2026-07-17): all weights/activations/optimizer double precision on CPU. One variable vs seedvar-1 (fp32, same seed/diet). If the gate moves >=3, matmul/update rounding at fp32 costs capability at birth — the veil is real. If flat, fp32 birth arithmetic is above the noise floor and precision stays an ONLINE-only knob.


### scratch/traj_accept.py
TRAJ acceptance driver (spec 2026-08-06-lab-traj-session.md, tiers 2 / 3a / 3b): runs the FROZEN scratch drivers with ONLY the instrument swapped for llmopt.lab.traj.patch_moe_router — corpus, chat template, oracle path, and every row-write remain the frozen code by import, so any byte difference is the unified patch's.


### scratch/umoe_conserve.py
UMOE-1 (pre-reg 2026-07-30): micro-MoE conservation 3-arm. First house MoE births. d64 h8 L8, FFN -> 4 experts (SwiGLU ffn_e=128) + top-1 switch router per block; gen-4 diet, 3 epochs, seed 1, all arms one device (3080).

- `class MoEFFN` (forward)
- `rope(q, k, pos0=0)`
- `moe_forward(self, x, mask, past=None)` — Block.forward twin with the FFN swapped for self.moe.
- `build()`
- `probes(model, enc, dev)` — corr / MI / meter on the trained model.
- `main()`

### scratch/v4flash_anatomy.py
V4-Flash offline anatomy: streamed experts -> instruments -> lake.

- `main() -> None`

### scratch/v4flash_census.py
CENSUS (unregistered, free): what is actually IN DeepSeek-V4-Flash's 48 shards? Headers only -- ~8 MB of range reads, no weights.

- `is_routed(name)` — Routed expert weight or scale -- NOT the always-on shared expert.
- `main()`

### scratch/v4flash_f1b.py
F1b (PRE-REG V4-F1): boot the VENDOR's model.py over the kernel twin — truncated architecture, RANDOM weights, no downloads beyond the sha-pinned vendor source (checkpoints/v4flash_vendor/, fetched from the HF repo; model.py sha c0c19e6c9fa439ba matches the rung-D pin).

- `load_vendor_model_module()`
- `boot_args(mod)`
- `init_random(model, n_routed, gen)` — The vendor model has no init (inference-only, torch.empty).
- `run(dev, mod, args, tokens)`
- `main()`

### scratch/v4flash_f1c.py
F1c (PRE-REG V4-F1): REAL weights into the vendor model — embed + layers 0-2 (all hash-routed, so expert demand is EXACT from tid2eid), prefill + one decode step on cpu and mps.

- `_get(url, lo=None, hi=None)`
- `index_map()` — tensor name -> (shard file, [lo, hi]) from per-shard headers of
- `fetch(man, name)` — Sha-side-cached byte-range fetch with the length assert.
- `tensor(man, name)`
- `demanded_experts(man, ids)` — Exact per-layer expert demand for hash layers, from tid2eid.
- `load_real(model, man, demand)` — Load embed + layers 0-2 dense + demanded experts; zero the head.
- `run(dev, mod, args, man, demand, ids)`
- `main()`

### scratch/v4flash_f1d.py
F1d (PRE-REG V4-F1): DeepSeek-V4-Flash GENERATES TOKENS ON THE MAC — full 43-layer dense path resident, K experts/layer subset-resident, greedy decode, tok/s and RSS measured, text logged verbatim.

- `manifest()`
- `tensor(man, name)`
- `rss_gb()`
- `class ExpertProvider` (load, ensure_hash)
- `choose_residents(man)` — Score layers, two rules:
- `load_dense(model, man, dev)` — Everything except routed experts, straight onto dev.
- `install_batched_moe(mod)` — F1e arm 5 (RIDER 3): batch the <= 6 hit experts of a SINGLE-TOKEN
- `check_batched_equiv(model, dev)` — Equivalence bar (RIDER 3): patched vs vendor MoE on a real
- `install_recall(model, mod, orig_bias)` — F2a READOUT 2: per score-layer per token, log the UNMASKED top-6
- `main()`

### scratch/v4flash_header.py
RUNG -1 (spec 2026-08-02-v4flash-lossless-recode): read a DeepSeek-V4-Flash safetensors HEADER by HTTP byte-range and report the tensor inventory — names, dtypes, shapes, and the implied fp4 scale granularity. Costs well under 1 MB; downloads no weights.

- `_get(url, lo=None, hi=None)`
- `read_header(shard)`
- `group_of(n)` — Coarse tensor class, from the name alone.
- `main()`

### scratch/v4flash_router.py
RUNG R (pre-reg V4-RUNG-R + 2B-ROUTER): read DeepSeek-V4-Flash's MoE router for free — 2 MB of gate weights and 1 KB of bias, no inference.

- `bf16_to_f32(raw, shape)` — safetensors BF16 -> float32 (upper 16 bits of the f32 word).
- `read_router(shard, layer)`
- `null_cosines(n, d, rng, reps=1)` — Matched null: cosines among n random Gaussian vectors in R^d.
- `main()`

### scratch/v4flash_rung0.py
RUNGS 0/0b/1/2c/3 (pre-reg V4-RUNG-0/1): entropy and lossless rANS of DeepSeek-V4-Flash's shipped fp4 expert stream, from byte-range fetches only.

- `_get(url, lo=None, hi=None)`
- `header(shard)`
- `fetch(url, base, name, spec, cache)` — Byte-range fetch one tensor, sha-pinned on disk.
- `nibbles(raw)` — Unpack packed fp4 bytes to 16-symbol codes, low nibble first.
- `entropy(sym, k)` — Order-0 empirical entropy in bits, and the probability vector.
- `kl(p, q)` — KL(p || q) in bits, over the support of p.
- `main()`

### scratch/v4flash_rung2b.py
RUNG 2b (pre-reg V4-RUNG-2B): are DeepSeek-V4-Flash experts closer to each other UP TO A PERMUTATION than they are coordinate-wise?

- `load_expert(idx, hdr, url, base)` — Integer weights on the shared dyadic lattice, per projection.
- `hidden_cost(a, b)` — Squared-distance cost between hidden units of two experts.
- `permute(ref, perm)` — Apply a hidden-unit permutation to a reference expert.
- `resid_entropy(e, ref)` — Order-0 entropy in bits/param of the exact integer residual.
- `raw_entropy(e)`
- `main()`

### scratch/v4flash_rung2b_router.py
RUNG 2b-ROUTER (pre-reg V4-RUNG-R + 2B-ROUTER): the retest VERDICT V4-RUNG-2B could not do.

- `pairs_from_router()` — Top-NPAIR by raw gate-key cosine (the pre-registered rule), plus
- `measure(a, b, hdr, url, base, rng)` — Residual entropy of expert a against expert b, three alignments.
- `main()`

### scratch/v4flash_rungA.py
RUNG A (pre-reg V4-RUNG-A): a full DeepSeek-V4-Flash expert forward run ENTIRELY IN INTEGERS on the vendor's shipped fp4 codes, hash-locked across backends. Ported from the certified K3-D2 chain (scratch/k3_expert_demo.py:99-151); RECEIPT V4-RUNG-MINUS-1 established the two formats are byte-identical, so only constants and the activation change.

- `_get(url, lo=None, hi=None)`
- `header()`
- `cached(name, hdr=None, url=None, base=None)` — Blob bytes, sha-pinned. Byte-range fetches on a cold cache so the
- `decode(proj, hdr, url, base)` — Shipped bytes -> (codes2x [out, din] int64, exps [out, g] int64).
- `det_gemv(codes2x, exps, x, dev, chunk=512)` — Exact integer y = W @ x on the shipped codes. Per-group-32 int64
- `rdiv(v, d)` — Round-half-away-from-zero integer division (house convention).
- `to_scale_A(y, e)` — Requant det_gemv output (scale 2^(e-1), x already at A) to A.
- `main()`

### scratch/v4flash_rungd.py
RUNG D (pre-reg V4-RUNG-D + S0; rewritten after AMENDMENT RUNGD-0803): how much of DeepSeek-V4-Flash's routing survives deleting the shared router key direction, and under WHICH input model?

- `vendor_scoring()` — Confirm the score function, top-k, and the ABSENCE of group routing.
- `shared_directions(W)` — Four defensible definitions of "the" shared key direction.
- `score(X, W, bias)` — Vendor scores for selection: sqrt(softplus(X W^T)) + bias.
- `topk_sets(S, k)` — Top-k indices per row (sorted, so the sets are canonical).
- `agreement(X, W, Wd, bias, k)`
- `main()`

### scratch/v4flash_rungd2.py
RUNG D2 (pre-reg V4-RUNG-D2): measure <u,x> on DeepSeek-V4-Flash's REAL traffic by inverting its trained load-balancing bias. No forward pass.

- `input_norm(shard, layer)` — ||ffn_norm.weight|| -- the length the gate's input actually has.
- `perp_basis(u, n, rng, gain)` — n unit vectors orthogonal to u, shaped by the layer's channel gains.
- `imbalance(W, bias, X, topk)` — Coefficient of variation of expert load under top-k selection.
- `main()`

### scratch/v4flash_s0.py
RUNG S0 (pre-reg V4-RUNG-D + S0): is the entropy-coded form of a DeepSeek-V4-Flash expert EXECUTABLE, or only an archive?

- `vendor_shape()` — Route width, layer count and shared-expert bytes -- READ, not typed.
- `blob(name, nbytes=None)` — Cached bytes, cache-integrity checked, cold-fetching if absent.
- `nibbles(raw)`
- `bench(sym, nrep)` — Encode once, decode nrep times; return (bytes, best decode s, enc s).
- `main()`

### scratch/v4flash_twin.py
F1a (PRE-REG V4-F1): pure-torch twin of DeepSeek-V4-Flash's inference/kernel.py — the six tilelang kernels plus the Hadamard rotation — so the vendor's model.py runs unmodified on Mac CPU/MPS.

- `_pow2_ceil_log2(x)` — 2^ceil(log2(x)) exactly, via the vendor's own IEEE bit trick
- `_rne_to_grid(x, grid)` — Round to nearest grid value, ties to even CODE INDEX — which is
- `_quant_values(x, grid)` — |x| RNE'd onto grid, sign restored, fp32.
- `_f8_to_f32(a)` — fp8 e4m3 tensor -> fp32 values, device-pure (uint8 LUT gather).
- `_e8m0_to_f32(s)` — e8m0 byte -> 2^(b-127), bit-constructed (MPS-safe, no ldexp).
- `_f32_to_e8m0(s)` — Exact power-of-two fp32 -> e8m0 byte: the exponent field IS the
- `_scale_f32(s)` — Any scale tensor (f32 / e8m0) -> fp32, device-pure.
- `_unpack_fp4(b)` — [.., K//2] packed e2m1 (any 1-byte view) -> [.., K] fp32.
- `act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False)` — Vendor contract: per-(row, block) amax (floor 1e-4); scale
- `fp4_act_quant(x, block_size=32, inplace=False)` — Per-(row,32) amax (floor 6*2^-126); scale 2^ceil(log2(amax/6))
- `_deq_act(a, a_s, group=128)` — Quantized activations (fp8 on cpu, bf16 grid values on mps) +
- `fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32)` — C = A_fp8[M,K] @ B_fp8[N,K]^T; A per-1x128 scales, B per-128x128
- `fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32)` — C = A_fp8[M,K] @ B_fp4[N,K]^T; B packed 2 codes/byte along K
- `sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)` — q [b,s,h,d], kv [b,n,d] (K==V latent), idx -1 masked; the sink
- `hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-06)` — pre=sigmoid(m0*s0+b0)+eps; post=2*sigmoid(m1*s1+b1); comb =
- `hadamard_transform(x, scale=1.0)` — Sylvester-order Walsh-Hadamard along the last dim (power of 2).
- `install()` — Register the twin as `kernel` and `fast_hadamard_transform` so
- `_smallest_pow2_geq(v)`
- `_bars(dev)`

### scratch/verify_intbirth_prims.py
House-side acceptance of axiom's intbirth PRIMITIVE layer (relay 2026-08-01-3): rebuild the R2b training loop from intbirth.Block / AdamW / rdiv alone, house-authored composition, and check all 8 r2b_ref.json milestone digests + losses. This is also the shape the multi-block reference will take (dx0 chaining, one AdamW over the concatenated param list).


### scratch/vmasm.py
vm-asm closed system (code continent rung 1): straight-line mini-ISA over r0-r3, one-rule rewrite chains, EXACT symbolic oracle (programs are polynomial register maps; sympy decides equivalence). Emits diet + probe with standing doctrine: stable string seeds, determinable one-rule rows, every row oracle-verified before write. Usage: vmasm.py <n_train_rows> <out_prefix>

- `run(prog)` — Symbolic execution -> tuple of 4 polynomial maps.
- `show(prog)`
- `gen(rng, n)`
- `step(prog)` — One rule application, first match. Returns (nxt, rule) or None.
- `farm(n_rows, seed_base, exclude=None)`
- `parse(s)`

### scratch/vmasm_probe.py
vm-asm rung 1 probe: greedy emission on 401 held-out steps. Score: pred parses AND is symbolically equivalent to cur AND differs from cur (a valid productive rewrite — any equivalent answer accepted, string match never used). Exact-gold reported separately. Usage: vmasm_probe.py <ckpt>


### scratch/vrm_ab.py
Valuation-routed metabolism v0: committee-gated per-neuron plasticity, one-variable A/B (see RIFF-LEDGER 2026-07-21). Mask: per-layer FFN committee probe; per-family top-5% neurons = heavy -> LR x0.2, rest x1.5, field normalized to mean 1.0 (equal average LR vs uniform arm). Arms: uniform vs routed, same 8k rows, 1 epoch, honest 120-gate each. Baseline 19m_v21 = 64.

- `committee_masks()`
- `load_rows(n=8000)`
- `run_arm(name, masks, rows)`

### scratch/weight_fft_euler.py
Weight-FFT euler read (pre-reg 2026-07-26, RESULTS.md).

- `ks_uniform(theta)`
- `phase_stat(W)`
- `fft_stat(W)`

## llmopt/

### llmopt/__init__.py
llmopt — an oracle-verified mathematics and physics ML lab.


### llmopt/intmath.py
The certified integer core of the deterministic-birth program.

- `rdiv(x, d)` — Round-half-away integer division, exact + deterministic.
- `int_mm(a, w)` — [..., K] x [N, K] -> [..., N], exact int64 (order-free sum).
- `isqrt_newton(x, iters=40)` — Exact elementwise floor-sqrt for int64 tensors (Newton +
- `mean_square_q32(x, dim, q=Q, eps32=EPS32)` — Mean square in Q16-squared units without false int64 overflow.
- `rms_isqrt_q16(x, dim, q=Q, eps32=EPS32)` — Q16 magnitude for integer RMSNorm, exact floor square root.
- `build_silu_tables()` — Shipped-bytes doctrine: build ONCE on cpu, pin the shas.
- `build_exp_table()` — exp on [-TSE, 0] in Q units, values at Q scale.
- `table_sha(t)`
- `lut(t, xq, hi_pos)` — Table lookup with per-table saturation: beyond +TS the value
- `class IntAdamW` (step)

### llmopt/reproduce.py
One-command reproduction of pinned llmopt results.

- `_arm_env(arm: str, pin: dict) -> dict[str, str]` — Resolve the pin schema plus its declared gravmoe cell contract.
- `available() -> dict[str, dict]`
- `_runner_env(spec: dict, base: dict[str, str] | None=None) -> dict[str, str]` — Remove ambient experiment knobs before applying the pinned contract.
- `reproduce(name: str) -> int`
- `main(argv: list[str] | None=None) -> int`

### llmopt/runlog.py
llmopt.runlog — standard run logging with honest wallclock.

- `class ElapsedFormatter` (format)
- `get_logger(name='llmopt', level=None, stream=None, fmt=DEFAULT_FMT)` — Idempotent: repeated calls return the same configured logger.
- `timed(label, log=None, level=logging.INFO)` — Context manager: logs '<label> done in <t>s' on exit,

### llmopt/window_artifact.py
Validation and decoding for committed gravmoe training windows.

- `load_contiguous_windows(windows_path: Path, contract_path: Path, sequence_length: int) -> list[list[int]]` — Load ``tok[T] ++ tgt[T]`` records as contiguous ``T+1`` rows.

## llmopt/lab/

### llmopt/lab/__init__.py
llmopt.lab — permanent instruments, adopted from the scripts that proved them (spec 2026-08-05-llmopt-lab-extraction.md; CODEMAP is the move gate). Adoption law: function bodies are VERBATIM copies of their source scripts, guarded by source-identity + behavior tests (tests/test_lab_adoption.py); the originating scripts stay frozen — they are the record booked verdicts cite. New code imports from here; existing scripts migrate only with a re-verified pass.


### llmopt/lab/anatomy.py
Weight-space anatomy: neurons as dots, reusable for ANY matrix.

- `_sha8(path: str) -> str`
- `_repo_head() -> str`
- `neuron_rows(ckpt: str, key_sub: str='gate.weight')` — All matrices of one family from a checkpoint, rows stacked:
- `project(W, method: str)` — (xs, ys, mag) for one of pca | sphere | polar.
- `rank_scale(mag)` — Magnitudes -> uniform [0,1] ranks. Row norms cluster tightly,
- `render_dot_views(W, out_stem: str, title: str, source_label: str, provenance: str, modes=('dark', 'light'), dpi: int=300) -> list[str]` — The dot-view triptych for ANY neuron matrix.
- `checkpoint_provenance(ckpt: str) -> str` — Footer text for a checkpoint source: basename + sha256[:8].

### llmopt/lab/catalog.py
catalog.py — model-checkpoint catalog rows (logs-doctrine EXHAUST).

- `read_arch(path: str)` — House-format arch from state-dict shapes; None otherwise.
- `parent_ids(name: str, siblings) -> list` — Filename-lineage parents (basenames) present in `siblings`.
- `scan_checkpoint(path: str, repo_root: str, cited_names, siblings=None, want_sha: bool=True, want_arch: bool=True) -> dict`

### llmopt/lab/config.py
lab.config — typed env-var config for arm drivers (spec 2026-08-05-llmopt-lab-extraction module 2). Kills the typo-takes-default class: 237 bare os.environ.get sites across 65+ scratch files meant a misspelled knob silently ran the WRONG EXPERIMENT with a clean exit. Here the contract is loud both ways:

- `class ConfigError`
- `_cast(name: str, raw: str, typ) -> object`
- `class LabConfig` (from_env, echo)

### llmopt/lab/figstyle.py
lab.figstyle — the house figure style: validated palette, vendored fonts, light and dark surfaces.

- `_register_fonts() -> bool` — Add the vendored fonts to matplotlib. Returns whether Inter is
- `color(entity: str, index: int=0, mode: str='light') -> str` — Color for a named entity, stable across every house figure.
- `sequential(n: int) -> list[str]` — n evenly spread steps of the single-hue sequential ramp.
- `rc(mode: str='light') -> dict` — House rcParams. Recessive chrome, thin marks, real typography.
- `figure(title: str='', subtitle: str='', mode: str='light', figsize: tuple[float, float]=(7.2, 4.0), **kw)` — A styled figure + axes with house title furniture.
- `footer(ax, text: str) -> None` — Provenance line under the plot: the sha, the verdict, the fence.
- `save(fig, name: str, outdir: Path | str | None=None, png: bool=True, svg: bool=True, dpi: int=220) -> list[Path]` — Write the figure. PNG for README/LinkedIn, SVG for the paper.

### llmopt/lab/figsvg.py
lab.figsvg — web-grade figures as hand-emitted SVG.

- `_esc(s) -> str`
- `_nice_ticks(lo: float, hi: float, target: int=5)` — Ticks a reader can hold in their head: steps of 1, 2, 2.5, or 5
- `_fmt(v: float, step: float) -> str`
- `load(name: str) -> dict`
- `_head(c, w, title, scope, pad_x, y)` — Title block. Returns the y cursor below it.
- `_fence(c, w, h, pad_x, text)` — The signature element: provenance as part of the figure, not a
- `gate_track(spec: dict, mode: str='light', width: int=880) -> str` — Fixed-denominator capability, drawn as filled rails.
- `curves(spec: dict, mode: str='light', width: int=880) -> str` — A measure over a shared x, with direct end labels.
- `ladder(spec: dict, mode: str='light', width: int=880) -> str` — One measure across an ordered axis, value printed at each point,
- `_svg(w, h, c, head, body, fence, pad_x) -> str`
- `composition(spec: dict, mode: str='light', width: int=880) -> str` — One whole, split into labelled parts — a single stacked rail.
- `render(name: str, mode: str='light', width: int=880) -> str`

### llmopt/lab/figures.py
lab.figures — matplotlib chart forms for ANALYSIS figures.

- `_both_modes(build, name: str, outdir=None) -> list[Path]` — Render light and dark from one description. Dark is drawn with
- `gate_bars(name: str, arms: dict, title: str='', subtitle: str='', source: str='', outdir=None)` — arms: label -> (solved, total). Percent bars with solved/total
- `curves(name: str, xs, series: dict, title: str='', subtitle: str='', xlabel: str='', ylabel: str='', source: str='', logx: bool=False, outdir=None, annotate_last: bool=True)` — series: label -> y values over shared xs. Direct end-labels, no
- `ladder(name: str, points: dict, title: str='', subtitle: str='', xlabel: str='', ylabel: str='', source: str='', reference: tuple | None=None, entity: str='series', fmt: str='{:.4f}', outdir=None)` — points: ordered x-label -> y. One series across an ordered axis,
- `scatter(name: str, series: dict, title: str='', subtitle: str='', xlabel: str='', ylabel: str='', source: str='', identity: bool=False, outdir=None)` — series: label -> (xs, ys). Every pair of colors is compared in a
- `stat(name: str, value: str, label: str, detail: str='', source: str='', entity: str='series', outdir=None)` — A hero number: the finding IS the value, so no chart is drawn.

### llmopt/lab/gate.py
lab.gate — the standard 120 gate, ADOPTED VERBATIM from scripts/step_grpo_micro.py (2026-08-11; that file stays frozen — it is the 91-reference hub backing every gate number in RESULTS). Function bodies below are character-identical to the source; guarded by tests/test_lab_adoption.py (source-identity). Fix a bug here and there in the SAME commit, or the guard fails.

- `class GateSpec`
- `apply_spec(spec: GateSpec) -> None` — Point the module constants at a lineage. Single-threaded use.
- `sample_wave_lp(model, tok, prompt_ids, seeds, dev, max_new=120)` — KV-cached (2026-07-22): token-identical to the eager
- `gate_eval(model, tok, dev, n=None)` — Honest chain gate. n<GATE_N = cheap proxy tier (same seeds,
- `gate_checkpoint(ckpt: str, d: int, layers: int, ffn: int, heads: int, label: str, device: str | None=None, spec: GateSpec | None=None)` — The scratch/gate_ckpt.py behavior as a callable: load a house

### llmopt/lab/gen.py
lab.gen — fork-isolated problem generation, ADOPTED VERBATIM from scripts/bench_step_tokens.py (2026-08-06; that file stays frozen — it backs the step-token race verdicts). The function body is character-identical to the source; guarded by tests/test_lab_adoption.py. Fix a bug here and there in the SAME commit, or the guard fails.

- `_gen_isolated(level: int, seed: int, wall: int=45)`

### llmopt/lab/hash.py
lab.hash — ONE digest semantics for the lab package (grok-seat cross-check adoption, 2026-08-11: three helpers with three semantics landed in one night — catalog 8MiB-chunk file sha, merge 1MiB-chunk file sha, runfiles short git sha WITHOUT a cwd anchor, which reports whatever repo the CALLER happens to be standing in).

- `sha256_file(path: str | Path, chunk: int=CHUNK) -> str` — Streaming file sha256, hex. Chunk size never changes the digest.
- `git_sha(short: bool=False) -> str` — HEAD of THIS repo (anchored to this file, never the caller's

### llmopt/lab/jsonl.py
lab.jsonl — one jsonl read/write semantics (grok-seat cross-check adoption, 2026-08-11: 40+ hand-rolled open/loads loops across scripts/ and scratch/, each with its own blank-line, encoding, and partial-write behavior).

- `read_jsonl(path: str | Path)` — List of rows. Blank lines skipped; malformed rows raise with
- `write_jsonl(path: str | Path, rows) -> None` — Atomic full-file write (tmp+rename, same directory).
- `append_jsonl(path: str | Path, row) -> None` — One row, appended and flushed — the streaming shape for

### llmopt/lab/keepsets.py
lab.keepsets — keep-set / coalition algebra. CANONICAL BODY since 2026-08-12 (Phase 3 module 1); scratch/gt2_jaccard.py is a re-export shim over these symbols and keeps only its CLI. Originally adopted verbatim from that file 2026-08-06. Guarded by tests/test_lab_keepsets.py (shim identity + synthetic battery + full acceptance against the booked stats and the byte-frozen checkpoints/gt2_*_arm0_decode.json dumps).

- `_frac(frac=None)`
- `_flag(name, default, value=None)`
- `decode_counts(path, pred=lambda r: True, gate_only=None, drop_tail=None)` — DROP_TAIL=1 (default) drops the FIRST decode-phase row per
- `keep(counts, n=128, top_k=8, frac=None)`
- `jmean(ka, kb)`
- `coverage(demand, kp)` — Count-weighted fraction of `demand` routed inside keep-set kp.

### llmopt/lab/lake.py
Parquet lake over the lab's jsonl/file exhaust — QUERY layer, not a write format.

- `_write(table: pa.Table, lake_dir: Path, name: str) -> Path`
- `build_runs(jobs_dir: Path=Path('jobs'), lake_dir: Path=DEFAULT_LAKE_DIR) -> Path` — jobs/<id>.{cmd,rc,pid} + <id>.log mtime -> runs.parquet.
- `build_results(index_path: Path=Path('docs/results-index.jsonl'), lake_dir: Path=DEFAULT_LAKE_DIR) -> tuple[Path, Path]` — docs/results-index.jsonl -> results.parquet + result_edges.parquet.
- `build_models(catalog_path: Path=Path('data/catalog/models.jsonl'), lake_dir: Path=DEFAULT_LAKE_DIR) -> Path` — data/catalog/models.jsonl -> models.parquet; absent file => EMPTY table
- `build_gates(lake_dir: Path=DEFAULT_LAKE_DIR) -> Path` — Materialize an empty gates.parquet with the pinned schema (idempotent;
- `append_gate(row: dict, lake_dir: Path=DEFAULT_LAKE_DIR) -> Path` — Append one gate row. REFUSES (ValueError) rows missing/null in any of
- `append_weights(rows: list[dict], lake_dir: Path=DEFAULT_LAKE_DIR) -> Path` — Append shards.weigh() rows to weights.parquet. REFUSES rows
- `query(sql: str, lake_dir: Path=DEFAULT_LAKE_DIR)` — Run duckdb SQL over the lake. Every *.parquet under lake_dir is exposed

### llmopt/lab/merge.py
Merge API over house .pt state dicts — average / task_vector / shell_graft.

- `_sha256(path: str) -> str`
- `_git_sha() -> str`
- `_load(path: str)`
- `_check_out(out: str, *inputs: str) -> None`
- `_check_match(a: dict, b: dict, la: str, lb: str) -> None`
- `_row(op: str, out: str, inputs: list[str], alpha, arch=None, label=None) -> dict`
- `is_ternary_lattice(sd: dict, min_numel: int=16) -> bool` — True if any 2D weight looks absmean-lattice / ternary-quantized:
- `average(a: str, b: str, out: str, alpha: float=0.5, *, shared_lineage: bool=False, arch: dict | None=None, label: str | None=None) -> dict` — out = (1-alpha)*a + alpha*b. REFUSES unless the caller asserts
- `task_vector(base: str, a: str, b: str, out: str, alpha: float=1.0, *, arch: dict | None=None, label: str | None=None) -> dict` — out = base + alpha*((a-base) + (b-base)). PROBE-GRADE: no booked
- `shell_graft(small: str, large_arch: dict, out: str, *, seed: int=6, arch: dict | None=None, label: str | None=None) -> dict` — Grow `small` into large_arch's FFN shells function-preservingly.
- `gate_cmd(row: dict, device: str) -> str` — Return (never run) the shell command for the standing 120-problem

### llmopt/lab/oracle.py
lab.oracle — the boxed oracle, v3.2 lineage (spec 2026-08-05-llmopt-lab-extraction module 1). Parent side of the subprocess line-server in lab/oracle_worker.py; behavior ported line-for-line from scratch/moe_gt1_arm2.check_isolated (which stays frozen — booked verdicts cite it).

- `class CheckResult` (timed_out)
- `class Oracle` (check, close)

### llmopt/lab/oracle_worker.py
Standalone oracle worker for timeboxed p.check — ADOPTED from scratch/oracle_worker.py (MOE-GT-6 v3; that file stays frozen). main() is character-identical to the source (guarded by tests/test_lab_oracle.py); only the repo-root sys.path depth differs.

- `main()`

### llmopt/lab/runfiles.py
Run-artifact contract — the Spark-_SUCCESS pattern for the lab.

- `run_dir(name: str, root: str | Path='logs') -> Path` — Create (idempotently) and return logs/<name>/ AT NAME TIME.
- `_git_sha() -> str`
- `write_marker(dir_or_path: str | Path, kind: str, rc: int | str, wall_s: float | None=None, artifacts: list[str] | None=None, **extra) -> Path` — Write the run's single marker line, atomically.
- `read_marker(dir_or_path: str | Path) -> dict | None` — Return the marker dict, or None when absent/unparseable.
- `is_done(dir_or_path: str | Path) -> bool` — True iff a marker exists AND parses. Absence is 'never ran
- `rc_of(dir_or_path: str | Path) -> int | None` — The marker's rc as an int, or None for absent/non-integer
- `require_resume_marker(ckpt: str | Path) -> int` — REFUSE to proceed when a checkpoint exists without its .ep

### llmopt/lab/runlog.py
Per-step receipt writer — streaming jsonl rows for long runs.

- `_device() -> str` — Best-effort device string, torch-optional (tests must skip
- `class FallbackCounters` (bump)
- `class RunLog` (step, abort, close)

### llmopt/lab/shards.py
Streamed big-model weights -> instruments -> lake, as one call each.

- `dequant(packed, scale)` — MXFP4-pack -> (codes2x int64 [out,in], exps int64 [out,groups],
- `v4flash_manifest(cache: str=V4FLASH_CACHE) -> dict` — The cached safetensors index: tensor name ->
- `_cached_bin(cache: str, name: str) -> np.ndarray`
- `list_v4flash_experts(cache: str=V4FLASH_CACHE, proj: str='w1') -> list[tuple[int, int]]` — (layer, expert) pairs whose packed weight AND scale for `proj`
- `v4flash_expert(layer: int, expert: int, proj: str='w1', cache: str=V4FLASH_CACHE)` — One routed expert's projection, exactly dequantized to a
- `iter_v4flash_experts(sample: int | None=None, seed: int=0, proj: str='w1', cache: str=V4FLASH_CACHE)` — Yield ("L<l>E<e>", W fp32) one expert at a time — stream,
- `weigh(W, source: str, model: str='', proj: str='') -> dict` — Run the desk instruments on one weight matrix. Returns a flat

### llmopt/lab/traj.py
lab/traj — unified MoE router instrument (module 4; DESK TIER ONLY).

- `begin_prompt(state, prompt_id)` — The certified per-prompt driver resets (surface 5, A verbatim:
- `class patch_moe_router`

### llmopt/lab/verify.py
lab.verify — the fast wave-verifier, ADOPTED VERBATIM from scripts/bench_verify_fast.py (2026-08-06; that file stays frozen — it backs the parity bench and every verdict that cites it). Function bodies below are character-identical to the source; guarded by tests/test_lab_adoption.py (source-identity + behavior parity). Fix a bug here and there in the SAME commit, or the guard fails.

- `_wave_worker(prev_s: str, cands: list[str], q) -> None` — One fork verifies a whole wave; verdicts streamed per candidate
- `verify_wave(prev_s: str, cands: list[str], wall: int=20) -> dict[str, tuple[bool, bool]]` — Levers 1+2: cache, then one streamed fork for the misses.

## llmopt/train/

### llmopt/train/__init__.py
llmopt.train — closed-system births, controlled diets, and comparable interventions.

- `__getattr__(name: str)`
- `__dir__() -> list[str]`

### llmopt/train/complex_ffn.py
Complex-valued SwiGLU-style FFN (modReLU + genuine complex multiply), promoted from scratch/complex_model.py.

- `class ComplexFFN` (forward)

### llmopt/train/fused_ce.py
Fused (chunked) cross-entropy for MLX — the Liger-style trick.

- `naive_ce(hidden: mx.array, weight: mx.array, targets: mx.array) -> mx.array` — Reference: full logits, mean CE over non-ignored targets.
- `_make_fused(c: int)`
- `fused_ce(hidden: mx.array, weight: mx.array, targets: mx.array, chunk: int=1024) -> mx.array` — Chunked CE: same value/grads as naive_ce, O(chunk*V)-class peak.

### llmopt/train/hebbian_moe.py
Hebbian-coupled MoE birth with a merge-free dense endpoint.

- `class HebbianCoupler` (observe, maybe_relax)
- `merge_experts(expert_params: list[list[torch.Tensor]]) -> list[torch.Tensor]` — Ship-time collapse: average E experts into one weight list.

### llmopt/train/lora.py
LoRA family: low-rank adapters on frozen linears, plus DoRA.

- `class LoRALinear` (forward, merge)
- `class DoRALinear` (forward)
- `apply_lora(model, target_names, *, r=8, alpha=16.0, cls=LoRALinear)` — Wrap every nn.Linear whose qualified name contains any of
- `trainable_fraction(model) -> float`

### llmopt/train/mathnative.py
Math-native micro-model: tokenizer + architecture (spec 2026-07-15-mathnative-micromodel). From-scratch decoder trained exclusively on closed-system chains — no pretraining, no habits.

- `class MathTokenizer` (encode, decode)
- `build_model(vocab_size: int, d: int=384, layers: int=8, heads: int=6, ffn: int=1536, ctx: int=512)` — Standard decoder: RMSNorm, RoPE, SwiGLU, untied head. ~19M.

### llmopt/train/packing.py
Sequence packing: fill fixed-length training rows with multiple documents instead of padding each to max length.

- `pack_greedy(lengths: Sequence[int], capacity: int) -> list[list[int]]` — First-fit-decreasing: returns bins of sequence indices.
- `pack_batch(seqs: Sequence[Sequence[int]], capacity: int, pad_id: int=0)` — Pack token sequences into rows. Returns dict of tensors:

### llmopt/train/population.py
Population LoRA for MLX: K adapters, ONE frozen base, one forward.

- `class PopLoRALinear`
- `apply_population_lora(model, k: int, *, r: int=16, alpha: float=32.0, targets=TARGETS) -> int` — Freeze the model, wrap every matching nn.Linear. Returns count.
- `population_loss(hidden: mx.array, head_weight: mx.array, targets: mx.array, k: int, chunk: int=1024) -> mx.array` — Sum of per-adapter mean CEs. hidden: (K*B, T, D) or (K*N, D);
- `adapter_state(model, i: int) -> dict` — Extract adapter i's {name.a, name.b} for saving/merging —

### llmopt/train/preference.py
Preference-optimization losses: DPO, IPO, KTO, ORPO, SimPO, GRPO.

- `dpo_loss(pc, pr, rc, rr, *, beta: float=0.1)`
- `ipo_loss(pc, pr, rc, rr, *, beta: float=0.1)`
- `kto_loss(p, r, desirable, *, beta: float=0.1, kl_baseline=0.0)` — p, r: [n] policy/ref logprobs; desirable: [n] bool.
- `orpo_loss(pc, pr, chosen_ce, *, lam: float=0.5)` — chosen_ce: mean CE on chosen tokens (the SFT term). pc/pr are
- `simpo_loss(pc_norm, pr_norm, *, beta: float=2.0, gamma: float=1.0)` — pc_norm/pr_norm: length-normalized (per-token mean) logprobs.
- `grpo_advantages(rewards, group_ids)` — Z-score rewards within each prompt group (GRPO's critic-free
- `grpo_loss(logp_new, logp_old, advantages, *, clip: float=0.2, dual_clip: float=3.0)` — PPO-clip objective with group-relative advantages, per sequence.

### llmopt/train/ref_logprobs.py
Batch precompute and disk-cache reference (teacher) logprobs.

- `class RefLogprobs` (perplexity)
- `_cache_key(model_name: str, token_ids: Sequence[Sequence[int]], top_k: int) -> str`
- `precompute_ref_logprobs(model, token_ids: Sequence[Sequence[int]], *, model_name: str='', top_k: int=128, batch_size: int=8, cache_dir: str | Path | None=None, device: str | None=None) -> list[RefLogprobs]` — Run the reference model over all sequences, return per-sequence RefLogprobs.
- `kl_vs_ref(ref: RefLogprobs, new_logprobs) -> float` — Mean per-token KL(ref || new) estimated over ref's top-k support.
- `_save(path: Path, results: list[RefLogprobs], meta: dict) -> None`
- `_load(path: Path, n_seqs: int) -> list[RefLogprobs]`

### llmopt/train/task_vector.py
Task vectors from LoRA adapters: skill = weight delta, applied by arithmetic.

- `load_adapter(path: str | Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]` — Read the {module_path}.a / {module_path}.b flat dict saved by
- `apply_task_vector(model, adapter: dict, scale: float, *, r: int=16, alpha: float=32.0)` — Merge scale·(alpha/r)·B@A into each named Linear's weight.

## llmopt/search/

### llmopt/search/__init__.py
llmopt.search — symbolic derivation search and verified circuit reduction.

- `__getattr__(name: str)`
- `__dir__() -> list[str]`

### llmopt/search/axiom_oracle.py
Axiom oracle adapter — Phase A of the axiom backend (docs/superpowers/specs/2026-07-18-axiom-backend.md).

- `class AxiomOracle` (equivalent, stats)

### llmopt/search/axiom_slots.py
External-slot callbacks for axiom's hybrid engine config.

- `_heurisch_worker(node_s: str, q) -> None`
- `heurisch(node_sstr: str) -> list[str]`
- `_equiv_worker(lhs: str, rhs: str, q) -> None`
- `equivalence(lhs_sstr: str, rhs_sstr: str) -> str`

### llmopt/search/derivation.py
Derivation search: Stockfish-for-math foundations (roadmap #1).

- `class State` (key)
- `is_solved(state: State) -> bool`
- `hce(state: State) -> float` — Hand-crafted evaluation, v0. Lower is better.
- `_euler_rewrite(e: sp.Expr) -> sp.Expr` — The ceiling-mover (Artin's complex-numbers thread): rewrite trig
- `_subs_eval(e: sp.Expr) -> sp.Expr` — Back-substitute solved Subs carriers (from i_usub) — a visible ply.
- `_is_zero(d: sp.Expr) -> bool` — Bounded zero-test for edge verification. simplify() can burn
- `verify_edge(parent: sp.Expr, child: sp.Expr) -> bool` — Oracle check: a legal move preserves the value. Integral edges
- `class _RuleTimeout`
- `_timeboxed(fn, *args, default)` — Run fn under a RULE_WALL timer, returning default on timeout or
- `successors(state: State, *, use_macros: bool=False, verify_p: float=1.0, only_rules: 'set[str] | None'=None, move_filter: 'Callable[[str], bool] | None'=None) -> Iterator[tuple[str, State]]` — Legal, non-identity, sympy-verified successor states. Rule moves
- `class SearchResult`
- `replay_verify(root: sp.Expr, history: tuple[str, ...]) -> bool` — Fully re-verify a winning path edge by edge (verify_p=1).
- `beam_search(expr: sp.Expr, *, width: int=8, max_plies: int=12, max_nodes: int | None=None, use_macros: bool=False, trace: list[State] | None=None, eval_fn: Callable[[State], float]=hce, proposer: Callable[[State, list[tuple[str, State]]], list[tuple[str, State]]] | None=None, propose_k: int | Callable[..., int] | None=None, verify_p: float=1.0, state_filter: 'Callable[[State], bool] | None'=None, select_fn: 'Callable[[list[State], int], list[State]] | None'=None, expand_rules: 'Callable[[State], set[str] | None] | None'=None, ply_hook: 'Callable[[int, list[State], int], bool] | None'=None) -> SearchResult` — Minimize hce over the rewrite tree. Returns the best solved

### llmopt/search/engine.py
The measured-best engines, as one import (integration of the 2026-07-06..08 racing results — see docs/RESULTS.md).

- `class SyndromePolicy` (load, proposer)
- `class MarkovPrior` (load, from_rows, proposer)
- `solve(expr: sp.Expr, *, budget: int=200, prior: MarkovPrior | None=None, llm_score_fn: Callable | None=None, use_macros: bool=True, magic: bool=True, ply_hook: Callable | None=None) -> SearchResult` — Solve with the measured-best configuration.

### llmopt/search/features.py
Structural features for the NNUE eval (spec: 2026-07-07-nnue-eval-design.md). Cheap, deterministic, pure — the NNUE lesson is cheap features + tiny net. State.plies is deliberately absent: probes restart fresh, so history cannot affect solvability and would leak the training label.

- `_depth(e: sp.Basic) -> int`
- `featurize(expr: sp.Expr) -> list[float]`

### llmopt/search/magic.py
The magic detector (RESULTS: 55v54 + replication, 71 certified cuts at int L4): Liouville/Risch as integration's Gottesman-Knill. sympy's risch_integrate PROVES integrands non-elementary in ~10ms on our death-state shapes; a state carrying a certified non-elementary Integral node is dead WITHIN THE ENGINE'S OPERATOR CLOSURE (no rule merges integral nodes, so split non-elementary siblings can never recombine — the mathematical loophole is closed by the move set). Pruning it is a theorem per cut: provably zero false positives.

- `_risch_dead(integrand: sp.Expr) -> bool`
- `is_dead(state: State) -> bool` — True iff the state contains a certified non-elementary

### llmopt/search/parallel.py
Problem-level parallelism for CPU benches (spec: 2026-07-07-engine-optimizations-design.md, O3).

- `default_jobs() -> int`
- `pmap(worker: Callable[[T], R], items: Sequence[T], jobs: int | None=None) -> list[R]` — Order-preserving parallel map. jobs=1 is a true serial bypass

### llmopt/search/proposer.py
Move proposer: a policy model in front of the classical searcher (spec: 2026-07-07-move-proposer-design.md). The searcher enumerates LEGAL moves; the model only ranks them — rank-not-generate keeps legality by construction. Ranking = likelihood of each numbered choice's answer tokens under the fine-tuned model.

- `build_prompt(state_str: str, labels: list[str]) -> str`
- `make_proposer(score_fn: ScoreFn)` — Wrap a scoring function into the beam_search proposer callable.
- `make_scoring_proposer(score_fn: ScoreFn)` — Like make_proposer, but returns (ranked_children, scores_desc)
- `entropy_k(k_min: int=1, k_max: int=6, temperature: float=1.0)` — Confidence-gated branching: peaked ranking -> deep (k_min);
- `hf_score_fn(model, tok, device: str) -> ScoreFn` — Score each candidate as the mean logprob of its answer tokens

### llmopt/search/rules.py
Primitive differentiation rewrite rules (HCE rung 1, spec 2026-07-06-hce-rung1-primitive-moves-design.md).

- `_unpack(node: sp.Derivative) -> tuple[sp.Expr, sp.Symbol] | None` — (f, x) for first-order single-variable Derivatives, else None.
- `d_const(node: sp.Derivative) -> list[sp.Expr]`
- `d_x(node: sp.Derivative) -> list[sp.Expr]`
- `d_sum(node: sp.Derivative) -> list[sp.Expr]`
- `d_product(node: sp.Derivative) -> list[sp.Expr]`
- `d_power(node: sp.Derivative) -> list[sp.Expr]`
- `d_chain_table(node: sp.Derivative) -> list[sp.Expr]`
- `d_quotient(node: sp.Derivative) -> list[sp.Expr]` — MACRO: textbook quotient rule. Redundant with d_product+d_power;
- `d_const_factor(node: sp.Derivative) -> list[sp.Expr]` — MACRO, data-certified: d_product -> d_const carries 14.8% of
- `_unpack_int(node: sp.Integral) -> tuple[sp.Expr, sp.Symbol] | None` — (f, x) for single-variable indefinite Integrals, else None.
- `i_const(node: sp.Integral) -> list[sp.Expr]`
- `i_inverse_trig(node: sp.Integral) -> list[sp.Expr]` — Inverse-trig antiderivatives (L5 autopsy 2026-07-09: the
- `_trace(msg: str) -> None`
- `i_sqrt_basis(node: sp.Integral) -> list[sp.Expr]` — sqrt-of-poly ansatz (L5 autopsy: root family 14/94 solved —
- `i_log_power(node: sp.Integral) -> list[sp.Expr]` — x**n * log(k*x)**m closed form (2026-07-09 frontier-gap autopsy:
- `i_transcend_div(node: sp.Integral) -> list[sp.Expr]` — Generator-shape splitter (2026-07-09 frontier-gap autopsy:
- `i_heurisch(node: sp.Integral) -> list[sp.Expr]` — sympy's integrator as a gated LEAF CLOSER (2026-07-11, Artin's
- `i_power(node: sp.Integral) -> list[sp.Expr]`
- `i_sum(node: sp.Integral) -> list[sp.Expr]`
- `i_const_factor(node: sp.Integral) -> list[sp.Expr]`
- `i_table(node: sp.Integral) -> list[sp.Expr]`
- `_usub_candidates(f: sp.Expr, x: sp.Symbol) -> list[sp.Expr]`
- `i_usub(node: sp.Integral) -> list[sp.Expr]` — u-substitution: if f == h(g)·g', rewrite to Subs(∫h(u)du, u, g).
- `i_parts(node: sp.Integral) -> list[sp.Expr]` — Integration by parts, stepwise: ∫u dv = u·∫dv − ∫(∫dv)·u'.
- `_unpack_lim(node: sp.Limit)` — (f, x, a) for finite two-sided-representable limits, else None.
- `l_direct(node: sp.Limit) -> list[sp.Expr]` — Continuity move: substitute when the value is finite/defined.
- `l_factor_cancel(node: sp.Limit) -> list[sp.Expr]` — 0/0 rational forms: cancel the common factor, emit a new Limit.
- `l_hopital(node: sp.Limit) -> list[sp.Expr]` — L'Hopital on 0/0: Limit(f/g) -> Limit(f'/g') with UNEVALUATED
- `i_apart(node: sp.Integral) -> list[sp.Expr]` — Partial fractions (ceiling-mover #2): rational integrands split
- `_linear_coeff(e: sp.Expr, x: sp.Symbol) -> sp.Expr | None` — Slope of e if e is linear in x (slope x-free), else None.
- `i_cyclic(node: sp.Integral) -> list[sp.Expr]` — Table macro (ceiling-mover #3): exp(ax+d)*sin/cos(bx+c) closed
- `i_unprod(node: sp.Integral) -> list[sp.Expr]` — Reverse product rule (ceiling-mover #4): sum integrands of the
- `i_ansatz_exp(node: sp.Integral) -> list[sp.Expr]` — Polynomial ansatz for P(x)·exp(w(x)) (ceiling-mover #4b): the
- `i_linear_basis(node: sp.Integral) -> list[sp.Expr]` — Bidirectional search v0, collapsed into linear algebra: d/dx is

### llmopt/search/zx_engine.py
T-count engine rung 1 (spec: 2026-07-08-tcount-engine-design.md): the derivation-engine chassis pointed at ZX diagrams.

- `tcount(g) -> int`
- `class ZXState` (key)
- `_phases_ok(g) -> bool`
- `moves(state: ZXState, max_per_rule: int=8)` — (label, child) pairs. Each child is an independent graph copy.
- `_phase_teleport_macro(g) -> None` — Rung-5 winner as a macro move: teleport_reduce moves phases
- `macro_moves(state: ZXState)` — Whole-graph macro moves (the algebra-moves analog): pyzx's
- `zx_eval(state: ZXState) -> tuple`
- `best_first_zx(g0, budget: int=300, max_per_rule: int=8, edge_cap_factor: float=3.0)` — Minimize T-count by best-first over ZX rewrites. Returns the
- `verify_equal(c_or_g1, g2, qubits: int) -> bool` — Boundary oracle: exact tensor equality for small circuits.

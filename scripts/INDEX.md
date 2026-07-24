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

### scripts/consolidate_mathnative.py
Self-distillation consolidation (post-climb strategy item B).

- `main(src: str, out: str, lr: float, cap: int, d: int, layers: int, ffn: int, heads: int, seed: int) -> None`

### scripts/control_round.py
Control round: retrain on the EXACT rounds-2/3 diet, gate it.

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
House chart helpers (figs/ instrumentation). One style, two forms:

- `_color(name: str, i: int) -> str`
- `_save(fig, name: str, png: bool=False) -> Path`
- `_style(ax, title: str, ylabel: str)`
- `grouped_bars(name: str, bins: list[str], series: dict[str, list[tuple[int, int]]], title: str='', png: bool=False) -> Path` — series: label -> [(solved, total) per bin]. Percent bars,
- `lines(name: str, xs: list, series: dict[str, list[float]], title: str='', xlabel: str='', ylabel: str='', png: bool=False) -> Path` — series: label -> y values over shared xs. Direct end-labels,

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

### scripts/gen_regret_labels.py
Regret/corrective labels (DAgger-style, Artin's 'make it regret the wrong node' — hindsight credit assignment made mechanical).

- `_syndromes(expr)`
- `_worker(job, q)` — Policy-guided search over one problem; every VISITED state gets
- `main(n_per: int, workers: int, out: Path, levels: list[int] | None=None, seed_base: int=980000) -> None`

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

### scripts/plot_neurons.py
Neuron-geometry plots for the micro-model program (docs/assets).

- `torch_svd_top2(X)`
- `neuron_matrix(ckpt: str, key_sub: str)`
- `project(W, method: str)`
- `scatter(ax, xs, ys, mag, title, cmap)`
- `main() -> None`

### scripts/probe_depth.py
Depth anatomy: WHERE in the stack does the rewrite decision form?

- `main(ckpt: str, d: int, layers: int, ffn: int, heads: int, n: int) -> None`

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

### scratch/build_merged_diet.py
Build data/merged_diet.jsonl (schedule-law queue item 1): gen-6 cumulative corpus (v22 + l8 + gen4 sidecar) + the L9a shard, with L1-L3 rationed to 45% (the gen-7 lesson). Stable string seed.


### scratch/chain_carry.py
CHAIN-CARRY ABLATION (Artin's carry hypothesis, spec'd 2026-07-21): same content, format ablated, equal TOKEN budget, both arms from scratch (d384/8L/3ep). Arm 'chains' = cur->nxt pairs as-is. Arm 'oneshot' = reconstructed root->final-answer rows (chains followed by nxt->cur linkage), upsampled to equal tokens. Gate both. If chains >> oneshot, capability numbers carry a format dividend. Usage: chain_carry.py <chains|oneshot>


### scratch/confluence.py
Metabolic-vs-champion confluence: where did 471 signed rows land? Per-matrix ||dW||, effective rank of delta, top-layer localization, ternary flip census (would the 1.58-bit deployment even change?).

- `ternary(w)`

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

### scratch/exchange_test.py
THE EXCHANGE TEST (pre-registered 2026-07-23): train the v4 organism on axiom's engine-farmed chains at OUR stuck states, re-probe the SAME fixed seeds (55_000_000, cuda — device law), must beat 2/12. v4 measured self-practice at +1/12 (no gradient at true walls); the exchange supplies exactly the missing gradient. 10/12 walls have chains; ceiling = 12/12, bar = >=3/12, headline read = how many of the 10 taught walls flip.

- `ternary(w)`
- `class TLin` (forward)
- `try_state(cur0, seed0, plies=8)`
- `probe(tag)`

### scratch/gate_batched.py
Batched gate v2 (2026-07-21): batch ACROSS problems, 8 seeds each — one forward serves K*8 rows instead of 8. Right-padded buffer + attn_mask (model supports it); per-row write positions keep RoPE phases identical to the unbatched path. NOTE: float reduction order changes => near-ties may resolve differently => this is a NEW GATE LINEAGE (re-baseline models of record once). Usage: gate_batched.py <ckpt> <d> <layers> <ffn> <heads> <label> [K]

- `batched_wave(model, tok, prompts, seed_lists, dev, max_new=120)` — prompts: list of K token-lists; seed_lists: K lists of B seeds.
- `gate_eval_batched(model, tok, dev, K=12)`

### scratch/gate_ckpt.py
*(no docstring)*


### scratch/gate_rarity.py
Rarity-stratified gate (schedule-law queue item 2): capability as a curve over expression rarity, not a scalar. Rarity = skeleton frequency — integer constants normalized to '#', skeleton counted in the corpus cur-set. Probes drawn WITHOUT exclude-filtering (the full spectrum is the point); bins: common / mid / rare / unseen-skeleton. Usage: gate_rarity.py <ckpt> <d> <layers> <ffn> <heads> <label>

- `skeleton(e: str) -> str`
- `binof(n: int) -> str`

### scratch/grpo_shaped.py
Potential-shaped GRPO on the gen-6 champion (2026-07-21, Artin GO — 'ahead of metabolic v3'). The b-lever: reward bandwidth. r = verified * (1 + LAM * tanh((Phi(cur)-Phi(next))/SCALE)), Phi = -(count_ops + 40*n_Integral). Unverified stays 0 (oracle floor intact; Ng-shaping preserves optimal policy). Monkeypatches G.collect's r_of via a wrapped collect; everything else (driver, gates, rollback) is the production harness. Pre-registered against the plateau: solves flat by cycle 4 in every unshaped run — shaped must beat +2 solves over 12 cycles or the b-lever nulls.

- `phi(s)`
- `shaped_collect(model, tok, dev, n_groups, seed0)`

### scratch/holdout_gate.py
FROZEN HOLDOUT battery (2026-07-21): virgin band 88M, same L3-L7 x 24 structure as the production gate, run ONLY at promotions. Includes a corpus-overlap audit (contamination doctrine: verify the band is virgin, don't assume). Usage: holdout_gate.py <ckpt> <d> <layers> <ffn> <heads> <label>


### scratch/holdout_v2.py
Holdout v2: exclude-guarded (the doctrine I broke in v1 — 281 collisions caught by the audit). Probes drawn from band 88M but each slot advances its seed until the expr is NOT in the corpus cur-set. Usage: holdout_v2.py <ckpt> <d> <layers> <ffn> <heads> <label>


### scratch/kv_equiv.py
KV-cache sampler + equivalence oracle (house rule: token- identical to eager full-recompute, or it doesn't ship).

- `sample_wave_lp_kv(model, tok, prompt_ids, seeds, dev, max_new=120)`

### scratch/l9_probe.py
L9 probe: 24 fresh L9a problems (band 90M — disjoint from the farm's 72/73M and roots_c1), gate_eval-style rollout, 12 plies. Usage: l9_probe.py <ckpt> <d> <layers> <ffn> <heads> <label>


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

### scratch/phys_probe.py
Physics rung 1 probe: greedy emission on held-out phys steps (seeds 17-19), sympy-equivalence in t, fork-isolated. No math gate — the physics expert is vocab-41, a separate model class by design. Usage: phys_probe.py <ckpt>

- `_equiv(q, pred, gold)`
- `equiv(pred, gold, deadline=10)`

### scratch/practice_mine.py
PRACTICE MODE, model-side (the mirror of axiom's arg-10): duo-wave rollouts that (1) BANK verified steps from ALL attempts — solved or not (the solved-only leak fix, Artin) — tagging rows by outcome so the gen-8 A/B can split them; (2) LOG stuck states — the exact cur where every unsolved attempt died — to a worklist in axiom's format ({id, level, root, from, why, plies}), ready for the stuck-state exchange AND as maximum-surprise metabolic v4 food.

- `skeleton(e)`
- `binof(n)`

### scratch/series_probe.py
Series rung 1 probe: greedy next-partial-sum emission on the 142 held-out steps (seeds 17-19), scored by sympy polynomial equivalence in fork-isolated workers (the solve_isolated doctrine). Also runs the standard 120 gate for the paired regression read vs seedvar-1 (65). Usage: series_probe.py <ckpt>

- `_equiv(q, pred, gold)`
- `equiv(pred, gold, deadline=10)`

### scratch/synonym_test.py
Synonym gauge test: TWO label tokens per family on the frozen 19M readout (vocab 40 -> 55: <name> + 7x2 synonyms). Train rows pick either synonym 50/50. Gauge-law prediction: both fire near-equal off the same concept. Reports family-accuracy + per-synonym share.

- `label_of(s)`
- `_one(args)`
- `gen(n, band, exclude=None)`
- `encode(e)`

### scratch/ternary_session2.py
Ternary compounding session #2 (Mac, MPS lineage, paired gates): the doctrine-composed organism — STE ternary latents, LATE layers only (8-11), LR 1e-4 cap, ABSOLUTE-anchor tripwire, fp32-vs-fp64 update-absorption instrument riding along. Pre/post MPS gates make it a clean paired delta.

- `ternary(w)`
- `class TLin` (forward)

### scratch/train_fp64.py
fp64 end-to-end birth (the rounding-loss-veil A/B, banked 2026-07-17): all weights/activations/optimizer double precision on CPU. One variable vs seedvar-1 (fp32, same seed/diet). If the gate moves >=3, matmul/update rounding at fp32 costs capability at birth — the veil is real. If flat, fp32 birth arithmetic is above the noise floor and precision stays an ONLINE-only knob.


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

## llmopt/

### llmopt/__init__.py
llmopt: inference + training optimization library.


## llmopt/train/

### llmopt/train/__init__.py
*(no docstring)*


### llmopt/train/fused_ce.py
Fused (chunked) cross-entropy for MLX — the Liger-style trick.

- `naive_ce(hidden: mx.array, weight: mx.array, targets: mx.array) -> mx.array` — Reference: full logits, mean CE over non-ignored targets.
- `_make_fused(c: int)`
- `fused_ce(hidden: mx.array, weight: mx.array, targets: mx.array, chunk: int=1024) -> mx.array` — Chunked CE: same value/grads as naive_ce, O(chunk*V)-class peak.

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
*(no docstring)*


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

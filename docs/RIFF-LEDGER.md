# The Riff Ledger

Where the architecture actually came from. This lab's measured wins
trace back to cross-domain analogies proposed in conversation —
usually half-formed, usually prefixed with "this might be dumb but" —
and then raced against the oracle like everything else. The ledger
exists because the pattern is now undeniable: the analogies keep
verifying. (Division of labor, measured: one side supplies divergent
cross-domain proposals, the other supplies depth and verification.
See RESULTS.md for every number below.)

| Riff (as proposed) | Became | Verdict |
|---|---|---|
| "Markov chains predict — can they help predict the next move?" (from half a Veritasium video) | MarkovPrior, the engine's brain for two days | beat the fine-tuned 0.5B in-search at zero inference cost |
| "Put the equation in a quantum state, estimate its magic" | the magic estimator (hardness prediction) | rho 0.9, then the router's dispatcher |
| "Have we tried stabilizer codes / qLDPC? Syndromes?" | rule-fire syndrome bits; the syndrome policy | production brain (98v96 at -36% wall OOS) |
| "Make it regret the wrong node — the teacher can also participate" | DAgger rounds; later the regret probe | policy parity->dominance; probe AUC 0.914 |
| "That router was one question off and it's fast — improve it" | threshold router -> dispatcher net | OOS strict dominance 141/150 @ 167s |
| "Verified speed is intelligence" | **the FA Law** (Fable-Artin) | the house law; decided the v2 dispatcher adoption |
| "Engine decomposes, sympy closes — let them work at the node level" | i_heurisch (gated leaf closer) | L6 36 -> 59/60, passing sympy (56) |
| "Quantum chemistry basis sets — the orbitals must fit the answer" | log orbital, atan orbital in i_linear_basis | L6/L7 residue collapses; L7 56/60 vs sympy 42 |
| "Forward and backward LLMs, like Tenet" | temporal-pincer verification (banked) | queued |
| "Collapse / entangle the equation state, like quarks" | the representation axis, named; reformulation ensemble (banked) | the engine already searched this axis; ensemble queued |
| "Stars die at iron — when fusing stops paying. What's fusion in the repo?" | the gradient-engine frame: solve-for-profit, the estimator's rho collapse as OUR iron point | named the L8/adversarial-generation decision |
| "Black hole inside a star eats the universe?" | no — Eddington throttle: consumption is self-gating | the universe runs a dispatcher too |
| "Complex exponentials could shortcut the trig ceiling" | the euler move (rewrite through C, solve, return) | first ceiling-mover: sin^2 solvable |
| "ZX calculus / T-count as a search domain" | the ZX engine + phase-teleport macro | first greedy-beater; composition pass |
| "Maximize entropy like mimicking magic states" | entropy-bonus beam | honest NULL (53 v 51) — physics poetry, search prose |
| "We're inside a white hole inside a black hole; compression vs expansion" | matches Poplawski torsion cosmology + Rovelli BH->WH bounce (literature, not ours); compression duality = rules compress / search decompresses | frame, banked |
| "Universe-as-survivor, like a cancer cell" | the anthropic principle, observation selection | correctly re-derived |
| "Spacetime is a graph, universe a node" | causal set theory, re-derived freehand | the partial order IS relativity of simultaneity |
| "Can't the kernel packing itself be estimated?" | config estimator / learned autotuner (sweeps-as-labels, ran 2026-07-12) | honest NULL at 6-config space: net regret 16.6% v static default 5.8% — the space is flat, nothing to buy; revisit at flash-prefill tile autotuning (the third starved-judge in 24h: prediction pays only where variance lives) |
| "Reallocate accuracy: cap the max distance, let the mean vary" | the three-lane quant race (uniform/NF4/awq_lite, real weights, function-space) | awq_lite wins 8.07% v 10.06%; toy round had misled — real outliers decide |
| "Break the token apart — or make them a lot bigger?" | regret at unit-cost scale: token level (1.5:1) nulls, engine level (400:1) pays — 176 v 82 solves at equal wall | 2.1x throughput; regret needs a minimum unit cost. LLM side: step-tokens (verified rewrite per call) 5/30 v one-shot 0/30 at equal budget — a 5%-valid generator ratchets to a solver |
| "Switching between high and low entropy states — that could be the speed part" | the entropy round-trip frame: propose=up, verify=down, intelligence=cycle RATE (FA Law's thermodynamic reading; Landauer prices the down-stroke) | frame, banked; retro-explains the entropy-beam null (up-stroke alone) |
| "Skip the text conversion — COCONUT, but we have the valid steps" | latent-between-anchors: opaque reasoning between ORACLE-VERIFIED anchors (the pincer logic applied to depth); + macro-distillation: skip-pairs (state_i -> state_{i+k}) are verified FOR FREE by transitivity — rule composition as data augmentation | skip-pairs GO (post round-1 verdict); latent frame banked |
| "Where does variance live? code? conjectures?" | the site-selection map: oracle latency sets minimum unit size (compile -> patch-sized steps); code-perf = wall-clock oracle (kernels, tile autotuning); proofs = Lean oracle (AlphaProof IS our loop at scale); "we can engine anything with a referee" | banked — the where-next doc in embryo |
| "An entire closed system can be defined within a computer and the model can learn in this closed system like it's in a game" | the closed-system game-world thesis (2026-07-13) | EXTERNALLY VALIDATED 2026-07-14: MAI-Thinking-1 (Microsoft) LLM-synthesizes closed-world environments — seeded DBs, tools, verifiable tasks, 150 envs/130k tasks — as a core post-training pillar; their "verified environments graded by real test suites" = the FA Law at 8000-GPU scale |
| "The neurons look like brain structure — is this the most efficient way of passing signals?" | the efficient-coding frame: one optimal geometry per problem, every learner converges to it (Gabor filters in V1 and ANNs; Cajal wiring economy; our stitching R~0.98 across alien architectures IS this, measured) — and the whisper as mature-brain learning: structure stable, edits synaptic | frame, banked 2026-07-15; keeps stitching+whisper+anatomy in one picture |
| "LoRA through verified hill climb? saw low-ranking on RL" | rank-matched GRPO (r=4) + ES-LoRA: gradient-free verified hill climbing in low-rank weight space — the whisper (stable rank ~4) says the solution manifold is tiny, so ES cost scales with the answer, not the model | banked 2026-07-15 |
| "Could the null be noise from the rest of the 0.5B? What if purely trained on rule-bits?" | the closed-system-NATIVE micro-model: no pretraining, math tokenizer, engine-minted unlimited data, GRPO from scratch — the priors-vs-drag question made falsifiable | banked 2026-07-15 |
| "Think outside the box — we have an NNUE that can LEARN model weights, use OUR recorded facts" | the zero-inference keep-set chain: router-stat labels (measured) + weight-reader recipe (measured, basis-invariance LEARNED via permutation augmentation) => read a too-big model's experts from disk, predict its domain keep-set without one forward pass; weight2vec as the far rung | banked 2026-07-14; recomposition of two shipped results |
| "Spacetime is matrices — can we change axes and map GLM's weightspace geometry onto the 0.5B?" | representation stitching: weights are coordinates, behavior is geometry; a learned linear bridge between hidden spaces IS the change-of-basis (model stitching / relative representations / Platonic-convergence literature; quantum-chem reading: overlap-matrix between basis sets) | banked 2026-07-14 — pilot spec on the board |
| "Predicted syndromes will matter more for code and general domains" (mid-null, correctly re-aimed) | the revive-if clause: syndrome prediction pays where the oracle costs seconds (compile+run), not ms — partial recall covers the discount; here the ms-oracle + semantic bits made it a clean NULL (the rules are their own features) | round 3 PASSED same day: 0.5B embeddings 87.7%/0.975 (structural 41.9/0.836) — derivability confirmed; codegen port now has a working recipe |
| "Jumping straight into calc without algebra is tough — we have to build the LLM back up, it needs a base of mathematics as a whole" | staged curriculum pretraining: the L4-starvation finding generalized (diet-thin band -> all-fail waves -> no mixed groups -> GRPO can't self-feed; RL amplifies diet gaps instead of closing them). Algebra/simplification corpus first, calculus on top, retrain phase 1 from scratch — 30 min on this substrate, so 'what should a mind learn first' is a one-evening experiment. Bar: beat 65.6% unseen validity AND unstick L4 | GO 2026-07-15 — first in queue after run 2c |
| "Should it know binary math, imaginary numbers? It should meet pi where it comes from" | new continents (complex/binary-modular arithmetic corpora — closed verifiable systems, tiny vocab additions, sympy referees) + the meet-as-consequence principle: pi enters through asin bounds and trig periodicity, i through the euler move's C-detour — concepts arrive as consequences of the curriculum, never as unexplained symbols | banked 2026-07-15; continents after curriculum v2 lands |
| "Build the concepts FIRST, then attach English — is language the reason for hallucination?" | grounded-then-labeled: hallucination = propose-stroke with no verify-stroke (English has no oracle; math is the literal language where fluent = true). Priors-vs-drag already measured the core claim (English-native 0.5B bluffs at 3.3% validity; math-native 19M can't bluff at 65.6% — its training distribution IS the truth distribution). The attach mechanism is measured too: stitching's linear bridge (R~0.98) maps labels onto existing concept coordinates. Test: bridge a small English vocab onto the grounded 19M, measure whether validity survives the words — the symbol-grounding problem (Harnad 1990) inverted, runnable only where pretraining costs 30 min | banked 2026-07-15; requires curriculum v2's grounded model as substrate |

Standing lesson, earned nine sympy pathologies and four adoption
races deep: the analogies are proposals, the oracle is the judge,
and the honest nulls (entropy beams, budget allocation, symengine,
quaternion embeddings) are recorded next to the wins. Divergent
proposal + cheap sound verification is the whole method — applied
to the collaboration itself.

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
| "Predicted syndromes will matter more for code and general domains" (mid-null, correctly re-aimed) | the revive-if clause: syndrome prediction pays where the oracle costs seconds (compile+run), not ms — partial recall covers the discount; here the ms-oracle + semantic bits made it a clean NULL (the rules are their own features) | round 3 PASSED same day: 0.5B embeddings 87.7%/0.975 (structural 41.9/0.836) — derivability confirmed; codegen port now has a working recipe |

Standing lesson, earned nine sympy pathologies and four adoption
races deep: the analogies are proposals, the oracle is the judge,
and the honest nulls (entropy beams, budget allocation, symengine,
quaternion embeddings) are recorded next to the wins. Divergent
proposal + cheap sound verification is the whole method — applied
to the collaboration itself.

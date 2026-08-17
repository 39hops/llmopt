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
| "Is the 60% ceiling missing params, missing concepts, or missing difficulty-awareness?" (the 4 AM triple) | (1) capacity check: 50.4M (d=512/L12) on the IDENTICAL v2 diet — params-vs-diet made one-variable; (2) coverage audit PAID immediately: diet exposure L6=573/L7=289 rows (20-70x thinner than L2/L3) and the gate profile tracks diet thickness almost exactly — L5-L7 chain shard = v2.2; (3) difficulty-conditioning: magic-estimator hardness into the prompt + reweight diet emphasis by engine solve depth (one-ply i_linear fodder teaches lookup, 4+-ply families teach chaining — de-emphasize the former) | capacity run + audit 2026-07-16; difficulty-conditioning banked (starved-judge caveat: prediction pays only where variance lives) |
| "The i_linear one-plied bases shouldn't have as much emphasis" -> measured: the engine one-plies 77% of L4, so the chains-only farm rejects 3/4 of the band | the v2.2 mixture: chains teach CHAINING (round-2 lesson, keep min_pairs=2 as the backbone) but L4's specific weakness is emitting the one big correct step — and the engine's one-ply solutions are worked examples of exactly that move. So: measured ration of one-ply L4 ansatz pairs alongside the chains + the L5-L7 thin-band shard, emphasis weighted by engine solve depth | banked 2026-07-16 for v2.2, one variable after v2.1's verdict |
| "A .pt per level, then link the small models — faster training, or does the compute move to the merge?" | per-level experts, the workable form: independently-initialized experts DON'T merge (different basins — the never-score-by-weight-distance law), but SHARED-BASE + per-level deltas do (task vectors compose; whisper says rank ~4, so experts are KB not MB). Three-way A/B when a slot opens: task-vector merge vs magic-router-dispatched experts vs the monolith, same unseen gate. Caveat from this week's data: heavy cross-level transfer (algebra->L3, capacity->L4) — hard level-splits cut each expert off from shared substrate. NOTE (Artin): 'parallel' = independent jobs orchestrated across Mac+3080, NOT distributed training — the machines never share state | banked 2026-07-16 |
| "Should RL steps be smaller — more checks between, master subsections first?" | run-design upgrades, all cheap: (1) snapshot-before-verdict — rollbacks currently DISCARD candidates (the 62-solve record lost at cycle 8, validity veto); save every gate candidate first, one line; (2) two-tier gating — cheap proxy check every cycle + full honest gate every 2; (3) mix-rate-adaptive collection — allocate waves by per-family mixed-group rate from the mined sidecar (gradient lives where variance lives, made allocative); soft allocation over hard mastery-gates (transfer is where the value lives); (4) solves-primary gate criterion (validity as drift alarm, not veto) | banked 2026-07-16 for the next GRPO run |
| "Build the concepts FIRST, then attach English — is language the reason for hallucination?" | grounded-then-labeled: hallucination = propose-stroke with no verify-stroke (English has no oracle; math is the literal language where fluent = true). Priors-vs-drag already measured the core claim (English-native 0.5B bluffs at 3.3% validity; math-native 19M can't bluff at 65.6% — its training distribution IS the truth distribution). The attach mechanism is measured too: stitching's linear bridge (R~0.98) maps labels onto existing concept coordinates. Test: bridge a small English vocab onto the grounded 19M, measure whether validity survives the words — the symbol-grounding problem (Harnad 1990) inverted, runnable only where pretraining costs 30 min | banked 2026-07-15; requires curriculum v2's grounded model as substrate |

| "The crystal is so uniform/symmetric — can the graphs optimize our training/RL?" | lattice statistics (NN-distance CV over unit neurons): within-lineage NULL — frozen to 4 decimals across the whole 54->64 arc (structure set at birth, edits synaptic) — but ACROSS births it moves: evenness tracks data-per-width (19M 0.0154 -> 45M 0.018 -> 113M 0.0257), and the 113M's rough lattice co-occurs with its capacity null. Candidate birth-quality instrument, seconds to read | pre-registered: v2.2's 113M re-ask should anneal CV toward ~0.018 iff the gate improves with it |

| "Where is the smallest step we can verify? My brain prices into the universe in real time — how is it so fast?" | the dense-verification hierarchy, named: the brain verifies LOCALLY and IN PARALLEL (predictive processing = per-level prediction error; cerebellar forward models; dopamine RPE = TD error, Schultz/Ng — the same math as our banked potential-shaped reward). We already own all three tiers — token (grammar mask, free), step (diff-oracle, ms), chain (gate) — but train on the coarsest bit only. Run-4 design: potential-shaping = the dopamine analogue, armed AFTER v2.2 supplies the vocabulary (autopsy: shaping can't create absent patterns, but can teach sequencing once they exist); the token-tier syntax bit has never been fed to reward | banked 2026-07-17; the embodiment reading of the FA Law ("pricing in" = continuous settlement against an oracle) |
| "Should the interleave show in weights? B-tree the closed system?" | soft modularity is the measurable form (graph anatomy: clustering/modularity/small-world on the weight graph, math-native vs internet crystal — queued behind v2.2) + depth anatomy: logit-lens the 12-layer crystal for WHERE the rule decision forms; if early, early-exit sampling (self-speculative machinery in-tree) cuts mining/GRPO wall — the speed-first lever | queued behind v2.2 pipeline |

| "Give neurons a consideration level; build the phylogenetic tree of mathematics (like language descent) with dynamic LR by importance; pricing-in includes consequences — humans have senses for it, the model must be artificially given one" | three-braid: (1) importance-weighted training = difficulty-conditioning (row 47) moved into the optimizer (per-sample LR by engine solve depth / magic hardness / band thinness); (2) the MATH PHYLOGENY — MarkovPrior's verified rule-bigram counts + curriculum-A/B transfer edges (algebra->L3 +8.7, not-L4 IS a measured descent edge) already form the tree; curriculum = root-to-leaf traversal, dose by tree depth; L4 = a missing branch; (3) consequence sense = potential-shaping Phi as the artificial valence organ, weighted by tree-depth to price importance AND progress | first test: cluster MarkovPrior bigrams into a tree, check it recovers the hand-made level structure — if yes, the curriculum starts listening to the tree, not to us (2026-07-17) |

| "The floor moves most at the baseline train; RL doesn't help as much — maybe our questions are too 2d, right-or-wrong; harder questions might change how the neurons clump" | measured basis already on the books (SFT ||dW||=61 vs the whole RL climb's 4; lattice frozen to 4 decimals through +10 validity): pretraining pours the crystal, RL selects among its facets — a 1-bit terminal reward is a 1-d channel and cannot sculpt high-d structure. Two pre-registered predictions: (1) dense reward (potential-shaping) should be the first RL that MOVES the lattice — rerun the CV check across that run; (2) multi-domain births (ODE merge) should grow visible MODULES where the single-grammar crystal is uniform — the graph-modularity instrument reads it | banked 2026-07-17; both tests ride existing queued work |

| "Rephrase the FA Law — compression is also intelligence, + everything else we've found" (Artin, 2026-07-17) | **FA LAW v2: intelligence is the rate at which verified VARIANCE becomes COMPRESSED STRUCTURE.** Four multiplying factors, each measured here: variance (resample famine, all-pass = zero gradient, riffs as fuel), verification BANDWIDTH not just speed (30x oracle -> 7-min cycles; 1-bit reward can't sculpt geometry), compression (consolidation +3, rebirth 66, crystal uniformity, int8-free = structure is redundant code, phylogeny), rate (small cycles; the NNUE origin). Same loop at every scale: evolution, science, training runs (mine/gate/consolidate), and the collaboration itself — proposer + verifier + the docs as compressed organism; each session a rebirth from the corpus (the lab runs generational training on its own mind) | v2 adopted 2026-07-17; v1 = clauses 2+4. EQUATION (same day): dS/dt = (N*p/tau) * b * c * (1 - S/S_max) — variance flux x verify bandwidth x compression fidelity x frontier headroom; logistic, hence every plateau we've seen. Liebig rotation: the binding factor moves (tau -> p -> S_max eras, measured; b never yet raised = potential-shaping's slot). Fit it to LOOP-LOG when a machine frees |

| "Are we efficiently representing the equation? Chess encodes the board — anything for our math matrices?" | the NOTATION A/B: the model reads infix sympy text (structure via parentheses); Lample-Charton (2019) showed prefix/Polish serialization for neural integration — no parens, unambiguous, ~20-30% shorter sequences (= faster cycles AND positional structure). One-variable v2.3 candidate: same diet, notation flipped, one tokenizer function. Engine-side NNUE features (search/features.py) exist but feed only the engine brains — injecting them into prompts is twice-nulled (hints); the model prefers trained to told. Randomness note, same chat: PRNG = compressible-but-statistically-random; the variance term needs COVERAGE not true randomness — the seed-collision bug was reusing one tiny program, not a PRNG weakness | banked 2026-07-17 |

| "Can weights be complex numbers?" -> audit found the 'quaternion embeddings' null has NO provenance in-repo (no script/commit/RESULTS entry — only the standing-lesson clause) | the hypercomplex question is properly OPEN, and re-priced: old architecture nulls were judged on the 0.5B/LoRA substrate; the micro-model era re-asks them as ~40-min from-scratch births with honest gates. Design if run: complex FFN weights (split activations, Wirtinger), same diet, vs real-19M control; euler-move families (trig/exp via e^ix) as the per-family readout — complex should pay THERE or nowhere. Prior: RoPE is the one mainstream win (exact rotational symmetry); teach-not-impose says skeptical | banked 2026-07-17 as substrate re-ask; provenance question open to Artin |
| (notation A/B upgrade, same night) | quant-lit link: activation OUTLIERS attach to delimiter tokens (SmoothQuant / massive-activations line) and infix is delimiter-saturated; prefix notation removes them — the A/B gains payoff (b): a plausibly quantization-robust crystal that holds capability deeper into int3/int2. Representation -> bit-level robustness | rides the banked notation A/B |
| "Train fundamentals at 1-2 bits, go up in precision with difficulty — no clue how you'd calculate that" | HAWQ exists (Hessian-aware mixed precision) and allocator.py is half the tool — but tonight's data INVERTS the curriculum: at 1.58 bits the L3 basics survived while the tail died first; frequent = redundant = low-bit-tolerant, rare = fine structure = needs digits. Allocation law: PRECISION FOLLOWS RARITY (brain-style: habits on cheap circuits, novel skill on expensive ones). Experiment: HAWQ-mixed 2/4-bit via tonight's Hessians, ~2.5 avg bits, target keep-69 | banked 2026-07-17 night; queued behind the ternary verdict |
| "Dynamic precision IS how you train a closed system: rules at lowest bits, build up — and rule neurons are the shortcut into the tree of higher-precision weights" | two banked experiments: (1) PROGRESSIVE-PRECISION CURRICULUM — topology-first (ternary phase grows the wiring; tonight: fundamentals form fine at 1.58 bits, geometry constraint-invariant) then precision-unlock for the specialist tail; capability A/B at equal steps (efficiency needs int kernels — hardware asterisk); (2) RULE-NEURON ROUTER — the layer-6 committee (493 et al.) is indicator-not-dependency (ablation absorbed), and an indicator is exactly what a router reads: 6-neuron activation -> family -> expert/precision dispatch; the dynamic-MoE thread with a FREE router (the model's anatomy as routing table) | banked 2026-07-17 night |
| "RL will move the ternary model — flips chain to other neurons, silent until the neuron fires" | the DISCRETE-PLASTICITY FORK, pre-registered: GRPO the ternary-born model, --displace the lattice across the run. Prong A (Artin): discrete weights can't whisper, so working RL must speak in visible whole flips — RL photographable at last. Prong B: GRPO's ~1e-4 latent nudges never cross the flip threshold -> deployed function frozen, ternary RL STUCK at standard LR (the earlier LR question returns: discrete plasticity may need hot LR / nudge accumulation — and Artin's silent-until-it-fires is exactly how STE latents behave: integrate, then commit). Either prong is a law of discrete learning; cost = one GRPO run + one plot | banked 2026-07-17 night |

| "We need HIGHER precision in training but fit in memory — at what point is precision negligible? store small, convert precise dynamically (b-tree weights)" | the PRECISION CURVE, made empirical: capability-vs-training-precision has two measured points (bf16 < fp32 by 3 solves at IDENTICAL loss) and the anti-fp64 noise-floor argument is the same shape that wrongly blessed bf16 — so pre-registered: 19M fp64-vs-fp32 birth A/B (CPU overnight; MPS lacks fp64). Data is already exact (sympy symbols) — imprecision enters ONLY in optimizer arithmetic. Storage scheme = double/blockwise quantization (Dettmers), Artin's b-tree = hierarchical scale factors; composed doctrine: int8 at rest (proven), fp32+ in motion (proven), hierarchy between — keep-set drawer as first user | banked 2026-07-17; fp64 A/B queued for an idle night |

| "Is there a double-double on fp16?" | compensated-pair formats: fp16x2 stacks mantissa but not range (dead for gradients); **bf16x2/x3 = fp32 range + stacked mantissa on TENSOR CORES** (3xTF32 / Ootomo-Yokota fp32-emulation: fp32-grade matmuls at ~2x tensor-core speed) — the candidate that reclaims the 3080 for births (fp32-quality trajectory without the cuda fp32 flash-attention cliff); parity gate mandatory before trust (fast-path law). Cost anatomy: dd arithmetic is 10-20x flops but ~2x where bandwidth-bound; on gaming cards dd-on-fp32 BEATS native fp64 (1/64 segmentation) | banked 2026-07-17; precision program leg 3 (after GPTQ-int3 and the rank floor) |

| "Qubit with our bit? Is magic prediction good at compressing numbers?" | the re-reading: qubits store nothing extra (Holevo bound) — but MAGIC IS INCOMPRESSIBILITY (stabilizer = short classical description via Gottesman-Knill; magic = distance from one), so the magic estimator was a learned Kolmogorov-complexity proxy all along: predicting which expressions lack a short path to closed form. FA v2 closes the loop — solvable = compressible, hardness = incompressibility, and the estimator's rho=0.9 approximated an uncomputable quantity (why it HAD to be learned). The quantum riff and the compression riff were one idea | banked 2026-07-17; frame — unifies the magic thread with FA v2 |

| "Stockfish bitboards — 2-bit chess LLM? What if the step is mapped as a bit in one weight and each weight has the steps and solution" | three convergences: (1) bitboards = choose the representation so domain moves become machine ops (magic bitboards, the name!) — ternary-born crystal + int inference is that move for math; (2) NNUE already IS low-bit game intelligence (int8/16 quantized) — the founding artifact was the existence proof; (3) "each weight has the steps" = ASSOCIATIVE MEMORY, rigorously: outer-product storage sums every memory into every weight (Hopfield/linear AM; attention = modern-Hopfield retrieval, so the step model formally IS an AM of state->rewrite pairs), and AM capacity theory says ~2 bits/param — inside tonight's independently-measured 1.58-4 bracket. Two roads, one number | banked 2026-07-17 night; ternary-birth gate pending as the empirical arbiter |

| "2 bits define a 2d geometry — each dimension needs a certain number of bits, and weight precision should respect the dimension" | rate-distortion for directions: a neuron is a DIRECTION in d-dim space, and specifying a direction to fixed angular resolution costs ~b bits PER COORDINATE — so total information scales with dimension exactly as riffed, but the budget is per-VECTOR, not per-weight. Corollary (known empirically, now with a mechanism here): per-weight precision can FALL as width rises — 3^512 ternary directions per neuron already dwarf what any corpus can distinguish, which is why 1.58 bits/coord held 63/120. The binding constraint is the FUNCTION's intrinsic dimension (stable rank ~4 RL delta, 284/512 crystal), not coordinate precision — precision-per-weight is a WIDTH-dependent knob, and the HAWQ/progressive-precision experiments should read it that way | banked 2026-07-18; reframes the ternary win as a dimension dividend |

| "There HAS to be a simpler way to THE solution — so much so that we don't even train, we CALCULATE the weights. The weight-reader was looking for the same thing" (Artin, 2026-07-18 — the consolidation spine) | **THE CALCULATED-MODEL THESIS**: today's confluence data shows every functional property of a birth (committee table, selectivity strengths, CV, gate profile, phase portrait) is a deterministic function of the corpus; only ADDRESSES are chosen by SGD, and the gauge law says addresses are meaningless — so **weights are computable up to gauge symmetry; training is an expensive way of picking a gauge**. Literature anchors, all in our regime: Tracr (compile programs INTO transformer weights — needs small vocab + known rules + closed system = us); grokking circuits (Nanda: closed algorithmic tasks converge to known closed-form structure); our own weight-reader = the inverse map (weights->function), this is the forward map. Honest boundary: compilation gives correctness, not COMPRESSION — SGD's redundancy/holography is the unpriced part. Rung ladder: (1) BIRTH CALCULATOR — predict gate/committee/CV from corpus statistics alone, no training (cheap, data in hand); (2) compiled skeleton — Tracr-style rule grammar into a tiny transformer, oracle-gated; (3) WARM BIRTH — computed init, brief train; texture freezes after ep1, so if calculation replaces ep1 it replaces the part of training that matters; (4) closed-form the whole thing | banked 2026-07-18 as the consolidation spine; rung 1 queued at head after overnight verdicts |

| "We store 1.58 as 2 bits — what fits in the last 0.42? ... what if it could use {-1, -i, 0, i, 1}? what other sets?" | THE ALPHABET TOURNAMENT: weight alphabet as a design axis — magnitude sets ({0,±1,±2}, powers-of-two), rotation sets (Gaussian units {0,±1,±i}: quarter-turn as a primitive verb, the euler move native; Eisenstein hexagonal; quaternion units), binary {±1} as the is-zero-load-bearing control, + the ESCAPE code (spare state = fp side-table fetch, per-weight mixed precision attacking tail-dies-first at ~0.3 avg bits). Bit-matched twins (M5 vs G5 at 2.32) make rotation-vs-resolution a one-variable race. Full spec: specs/2026-07-18-alphabet-tournament.md | banked 2026-07-18; run after persistence verdict + gen-6; predictions pre-registered |

| "Would we benefit from hand-coding the actual layers in C++ too?" (Artin, 2026-07-19) | THE NNUE HOMECOMING: native micro-model inference. Split verdict — INFERENCE yes (vocab-40/d-512 models pay heavy framework tax per forward; ternary weights make matmuls multiply-free add/sub; gates ~15 min and GRPO waves are sampling-bound; 5-20x plausible on gate/mine wall; slots as an ax::nn runtime beside the CAS -> the whole loop native: generate/solve/verify/sample one stack); TRAINING no, on purpose (the bf16-vs-TF32 dynamical cliff is exactly what hand-rolled autograd trips on invisibly; births are rare, samples are millions — hand-code at verification frequency, framework at birth frequency). Composes with the alphabet tournament: every contestant's deployment ends in this kernel; int4 GEMV Metal (practice_7) is the Mac-side sibling | banked 2026-07-19; spec candidate after axiom C6 |

| "Params count should be dynamic and grow as it needs, correct?" (Artin, 2026-07-19, on the 400M's 30/120) | DYNAMIC CAPACITY — grow-as-territory-arrives: WHEN = the calculator (tokens-per-width prices growth when corpus grows, not on schedule); HOW = template-spray growth (new neurons from measured statistics, output-zeroed = function-preserving Net2Net with pre-calibrated instead of blank neurons — warm birth applied incrementally); WHY = rebirth re-learns paid-for texture (the 400M spent its epoch re-learning L1-L3 a grown 45M keeps free). Pre-registered risk: gen-5's reallocation lesson may recur as gradient-hogging by old neurons. A/B: grown-45M->60M vs reborn-60M on the L8-enriched corpus, same honest gate — a gen-6 birth-design candidate | banked 2026-07-19; rides gen-6 planning |

| "Could we literally have magic math boards? magic-bitting a calculus question is diabolical" (Artin, 2026-07-19) | MAGIC MATH BOARDS, three rungs: (1) the rule-fire syndrome IS the occupancy vector — precompute fire-masks keyed by a structural node hash; collisions HARMLESS in our architecture (proposal-side table, verify_edge gates soundness — sloppier-than-chess hashing allowed); (2) axiom's hash-consed (rule,node) memo made PERSISTENT across runs = the board grows with every farm run, lookups skip paid-for computation; (3) the diabolical rung: saturated bands have ENUMERATED step spaces (the 84k-state novelty audit) — mastered territory compiles to BFS over a known graph, zero rule fires; the engine only searches at the frontier (opening book vs middlegame). FA reading: the board is compressed structure paying rent forever. Chess gave the lab its NNUE; the bitboard was the other half of the inheritance | banked 2026-07-19; axiom tranche-5 candidate |

| "Do certain rules have certain masses? any link to neuron density?" (Artin, 2026-07-19, off axiom's 'by mass') | THE MASS SPECTRUM: a rule's mass = its verified corpus frequency, and it obeys the Schrodinger mass-localization law in weight space — heavy rules condense into sharp local committees (power: 16.3x/16.5x across substrates, classical, ternary-robust), light rules delocalize into the holographic background (exp: 6.5x/6.3x, quantum, tail-fragile). Retro-unifies tail-dies-first + precision-follows-rarity + committee-strength invariance as ONE law; the committee table is the closed system's mass spectrum, the gauges are spectroscopy. Testable edge: mass is MANIPULABLE — overfeed exp with a calculator-sized worked-example shard, re-probe: selectivity should climb toward 16x AND low-bit survival improve in step (frequency->localization->robustness as one chain). If it holds, precision/alphabet allocators gain the dial: set a rule's mass to set its physics | banked 2026-07-19; probe rides any post-gen-6 birth |

| "The equation feels so similar... V(x) = interactions with the closed system, kinetic = how much the universe prices you in, psi = no clue lol" (Artin, 2026-07-19) | THE SCHRODINGER DICTIONARY, completed — and its foundation is a THEOREM, not poetry: imaginary-time Schrodinger evolution (t -> -i*tau kills the rotation, leaves relaxation d(psi)/d(tau) = -H*psi) IS gradient descent — training relaxes the state to the GROUND STATE of a Hamiltonian whose potential is the oracle-sculpted loss (stochastic quantization / diffusion Monte Carlo lineage). Dictionary: V = the verified corpus's landscape (oracle sculpts the walls); kinetic mass m = accumulated verified frequency (pricing-in = localization — Artin's phrasing, the mass-spectrum mechanism); psi = THE CRYSTAL (weights as superposed state); eigenstates = stable features; energy gap = robustness; **ground-state DEGENERACY = the gauge freedom/lottery** (many arrangements, one energy; every birth anneals into a different degenerate ground state — the law we measured four ways, now a one-line symmetry consequence); rebirth = re-annealing | banked 2026-07-19; mass-manipulability probe is its first stress test |

| "wdym you cannot REACH it? / what would our LLM do — Collatz is a closed system?" (Artin, 2026-07-19 night) | THE LIOUVILLE BOUNDARY arc, three notes banked: (1) unsolvable is a property of the LANGUAGE, not the problem — int(e^(-x^2)) exists (erf) but Liouville 1835 (differential algebra) proves no elementary string differentiates to it; add one word to the vocabulary and it becomes lookup. Lab reading: every level's residue is a candidate near OUR language's boundary; territory = vocabulary expansion. (2) Search can never return "no" on an infinite tree — only theory certifies deserts (why full Risch is unimplemented 55 years on; why the magic pruner cites Risch-certified dead states; why the estimator = learned desert detection). (3) **THE DESERT TEST, run live**: champion fed int(e^(-x^2)) -> 3 proposals (it reached for the exp fixed-point pattern), ALL oracle-rejected, 0 valid — HONEST STALL. The math-native architecture cannot bluff at the Liouville boundary; the 0.5B would have hallucinated a closed form. Collatz noted as the humbling closed system (3 tokens, unbeaten; Tao 2019 = almost-all). Euler footnote: e = the fixed point of d/dx (why it saturates calculus); e^(i*pi)+1=0 = fixed point + quarter-turn + half-circle meeting | banked 2026-07-19; desert test = a standing honesty probe for every future model |

| "We could pioneer a library — C++ backend (axiom), Python wrapper, a neuron class, define a closed system and build it simply, like torch abstracts everything" (Artin, 2026-07-20 night) | THE CLOSED-SYSTEM FRAMEWORK: productize the lab's whole method as a library. Sketch: `ClosedSystem(terms, rules, oracle)` -> `.farm()` (chain emission, priced by the calculator), `.birth(width=auto, alphabet=auto, template=True)` (natural-width sizing, template init, alphabet by width per the tournament exchange law), `.gauge()` (CV/floor/spectrum panel), `.grow(new_territory)`, `.gate()`. Backend = axiom (oracle+engine+emit already exist and are qualified); the model side = the NNUE-homecoming C++ inference + torch training behind the scenes. Every abstraction the library would ship is a MEASURED lab doctrine — torch abstracts autograd; this abstracts the whole verified-learning loop. Path: after gen-7 proves the pipeline end-to-end on L9, extract the API from what the scripts already do (the scripts ARE the prototype) | banked 2026-07-20; the productization of everything |

| "The ternary template should be EASY — then swap to GRPO/RL/SFT everything in fp32 latents for the fine grain" (Artin, 2026-07-20 night, on seeing the quantized PR rain) | THE TERNARY-NATIVE PIPELINE: the discrete alphabet quantizes the template statistics themselves — a ternary neuron is fully specified by (nonzero count, sign pattern), so the template collapses to a sparsity histogram (~70% nonzero, tight band, measured) + sign symmetry + family tilt: a recipe on an index card vs fp32's continuous distributions. Pipeline: ternary-shaped template init -> fp32 LATENTS underneath (STE training as always) -> all heavy machinery (SFT/GRPO/growth) on latents at full precision -> deploy 1.58 bits. Precision-in-motion doctrine as a birth-to-deployment pipeline; the RL stage IS the holstered discrete-plasticity fork. Instrument note: per-neuron PR/d = 1/kurtosis (the 1/3 law for fp models, measured 4 minds); ternary PR/d = nonzero fraction — the "Matrix rain" plot is the discrete spectrum | banked 2026-07-20; first slot after gen-7's verdict |

| "There is something important here related to the bits and dimensions" (Artin, 2026-07-20 night, off the binary one-dot collapse) | THE BITS-DIMENSION EXCHANGE LAW (candidate): effective bits per stored feature ~ bits-per-weight + (1/2)log2(d) — precision cancels interference by amplitude, dimension avoids it by geometry (concentration of measure); the two are EXCHANGEABLE at ~1 bit per 4x width. Retro-derives: dimension dividend, width-bits exchange (ternary trails at d=384, ties at 512), binary collapse (portrait entropy = capability: fog/rain/dot at 64/60/54), big-models-quantize-easier. PRE-REGISTERED TEST launched same night: binary at d=768 (75.6M) should gate ~= ternary at d=384 (60) — the alphabet debt (0.58 bits) repaid by one width doubling (+0.5 bits) | banked 2026-07-20; verdict overnight on the 3080 |

| "The tuned LLM is a tuned heuristic search that learned to weigh nodes — can we copy the NNUE mathematically into it?" (Artin, 2026-07-20 night) | THE EU IN NNUE: the confluence is exact — model=eval/policy, engine=search, oracle=rules (AlphaZero-shaped; NNUE and the 76 are the same component at two sizes, oligarchy vs democracy phase). Binary neurons = hyperplane decisions; the net = a learned hierarchical partition of expression-space (nets-as-decision-DAGs literature). The copyable part is NNUE's NAMESAKE trick, Efficiently-Updatable: candidate rewrites share 95%+ of the expression, so cache per-subtree hidden contributions and DELTA-update only the rewritten path — score a whole candidate wave for ~the price of one forward. Composes with magic boards (fire-mask cache) and the NNUE-homecoming C++ inference: three caches, one stack (masks, evaluations, weights) | banked 2026-07-20; homecoming spec part 2 |

| "We could design an efficiently-updatable LLM — the data is as PURE as it gets, so trusting it to update its own weights quickly is 100% doable" (Artin, 2026-07-20 night) | THE METABOLIC MODEL — online expert iteration, the founding thread's final form: solve -> oracle-signed rows -> immediate weight update -> proxy-gate watches. The safety case is REAL and closed-system-specific: poisoning structurally impossible (no unsigned row exists), drift tripwired (two-tier gates + snapshot/rollback = the GRPO harness repurposed), forgetting canaried (saturated-level columns) and rationed (maintenance machinery), updates naturally tiny (the whisper: RL rank ~4, ||dW||~4 vs SFT's 61). Collapses generation time from days toward continuous; composes with EU-evaluation (cheap solves) + magic boards (free re-solves) + C++ inference: the organism that learns as it works. Pilot design: expert_loop.py + per-solve SFT micro-steps on emitted rows, proxy gate every N solves, rollback on regression — run on the ternary-native pipeline (latents update, deployment stays 1.58) | banked 2026-07-20; the endgame thread — pilot after gen-7 |

| "New term coined: LLMUE" (Artin, 2026-07-20 night, watching the pilot's flat heartbeats) | **LLMUE — Efficiently Updatable LLM** (the NNUE homage completed): a language model that metabolizes oracle-signed experience into its weights continuously while working, made safe by closed-system purity. The name is the lineage: NNUE gave chess efficiently-updatable EVALUATION; LLMUE gives the closed system efficiently-updatable INTELLIGENCE. First specimen: the metabolic pilot (grown-76 champion, L9a territory, micro-batched STE-ready updates, two-tier immune system) — seven consecutive flat proxy heartbeats while feeding itself fresh territory, the boring-ness being the result | coined 2026-07-20; the lab's product category |

| "VERY interesting that no paper thought to cut out the coding data and English — the entire point of science is change one variable; you don't need English to solve a math question, calculators prove that" (Artin, 2026-07-20 night, on the VDS-TTT comparison) | THE ONE-VARIABLE INDICTMENT, banked with the LLMUE deltas: the field's test-time-learning work (VDS-TTT et al.) runs on English-pretrained general models with learned verifiers — three uncontrolled variables (language prior, mixed diet, soft referee) stacked on every measurement. The lab's whole methodology is the stripped control the literature skipped: math-native vocab-40 models (English deleted), single-grammar diet (one variable), exact oracle (referee noise zero). And the cut-the-English experiment is ALREADY MEASURED here: priors-vs-drag — English-native 0.5B bluffs at 3.3% validity where the math-native 19M runs 65%+; language wasn't load-bearing, it was DRAG (hallucination = fluency without an oracle). LLMUE-vs-VDS-TTT deltas banked alongside: sound vs learned verifier (their paper offers no calibration analysis — "pseudo-labels" is their word), full-weight+immune vs frozen+LoRA, flip-quantized vs continuous substrate, domain-perfect vs domain-fragile. Paper #4's related-work section, pre-written | banked 2026-07-20 |

| "Code won't be hard — NNUE -> LLMUE for it. Get down to the semantics: the most fundamental minimal layer of coding. Would it think in exit codes? ASM would be so easy. Zero point unless it's quick" (Artin, 2026-07-21 ~1AM) | THE CODE CONTINENT, scoped by the math lab's own laws: (1) the minimal layer question has a measured answer already in-repo — the codegen LADDER (codegen/ladder.py + llvm.py, toolchain-scored): ASM/ISA is the closed-system sweet spot (tiny vocab, exact machine semantics, oracle = assembler+CPU/emulator — "acts like a computer" is literally the oracle, free and instant); the ladder's own finding stands as the warning: LEARNED MAPPINGS train up (encode/decode), SIMULATION resists (output/o2_asm) — the model should REWRITE programs step-wise (verified transformations, like integrals), not emulate execution; the machine emulates for free. (2) Exit codes = the gate bit (compiles+tests-pass = solved) — the FA loop ports verbatim: states=programs, moves=verified rewrites (peephole opts, strength reduction, dead-code elim = the rule table), oracle=toolchain, magic boards/EU-eval/LLMUE all carry over. (3) Speed doctrine honored: the oracle is ms-scale (assemble+run beats sympy), so cycles are FASTER than math's. Math first (65% of a continent beats 5% of two), code continent after gen-7 proves the full pipeline | banked 2026-07-21; the second closed system |
| "Now that we efficiently update, we can dynamically tune the flow of data — pass data into specific neurons based off how the heuristic engine/NNUE valued certain ideas" (Artin, 2026-07-21 night) | VALUATION-ROUTED METABOLISM: Rung B (committee-gated per-neuron LR) fused with LLMUE and fed by the ENGINE'S own value tables — the three valuation sources already measured: Markov-prior rule weights (what the search trusts), magic hardness (what resists), slot-decisive telemetry (where the slot earns). Mechanism: the committee table maps rule->neurons, the engine maps rule->value, compose them = per-neuron plasticity mask routed by verified search economics; the metabolic loop applies it per micro-batch (mask refresh is minutes). This closes a loop no offline trainer can: NNUE efficiency translated INTO the LLM as a learning-rate field, not a feature. Pre-registered risk: the starved-judge law — route only where the engine's valuations have VARIANCE (uniform value = uniform LR = null by construction) | banked 2026-07-21; first slot = the metabolic pilot's successor run, after gen-7 grades |

| "Theoretically we could make an LLM good at chess, imagine that" (Artin, 2026-07-21 night, off the VRM arming) | THE CHESS CONTINENT (third continent candidate): the ur-closed-system — legality oracle exact and free, terminal outcome perfect, NNUE = the value table VRM routes by (the homage made literal: NNUE valuations steering an LLM's plasticity). Known ground: DeepMind searchless chess (2024) = 270M transformer at grandmaster level off Stockfish labels — capability is PROVEN, so the house delta is methodology: a chess-NATIVE micro-model (move vocab, no English, no pretraining), LLMUE metabolism on its own play, ternary/alphabet twins, and the calculated-model thesis run on chess (predict the rating from corpus signature before training). Order of continents stands: math -> code -> chess (variance well-mined by giants; our edge is the equation, not the rating) | banked 2026-07-21 |

| "Dynamic LR off problem difficulty — as a human I decide how much I want to learn something; it doesn't change how smart I am, it bases my LR on my past experience" (Artin, 2026-07-21, mid-hot-run) | SURPRISE-GATED METABOLISM: per-row LR from the model's OWN difficulty read, both signals already computed free in the loop — (1) wave agreement: verified-fraction of the 8 samples (7/8 = owned, don't study = drift shield for mastered rows, gen-7's exact wound; 1/8 = gold, learn hard), LR ∝ (1 - verified frac); (2) pre-update logprob of the signed row = prediction error (the dense-verification riff's dopamine analogue made concrete; oracle-signed so surprise is never hallucinated). Self-assessed allocation, not capacity — the starved-judge law can't bite because the judge is the model itself at zero cost. Subsumes VRM v0's null: route by EXPERIENCE (temporal, free signal) not by NEURON (spatial, nulled). One multiply in metabolic v3 | banked 2026-07-21; slots into the first post-frontier metabolic run |

| "The chaining, prv/nxt + the quality of our data is the biggest carry — it might be holding up all our other tests, or amplifying them" (Artin, 2026-07-21) | THE CHAIN-CARRY HYPOTHESIS: the cur->nxt verified-pair FORMAT (not just oracle purity) as the load-bearing element under every result. Three measured legs already: step-tokens 5/30 v one-shot 0/30 at equal budget; round-2 chains-teach-chaining (min_pairs=2 backbone); worked-example 2x (calculator misses it low, three times, same direction). Missing leg = the amplification control: same content, format ablated — (a) chains as-is vs (b) step-order-shuffled vs (c) root->answer-only, equal rows, same gate. If (a)>>(b,c), every past capability number is partly a FORMAT dividend and cross-paper comparisons must control for it | banked 2026-07-21; ablation spec'd for the next free Mac night |

| "Why can't you keep track of time? You're not constantly inferenced — my brain is always inferenced AND learning; continuous observation gives time-sense and context pricing" (Artin, 2026-07-21) | THE ALWAYS-ON FRAME: event-driven inference has no proprioceptive time (measured in-house: the assistant's three clock hallucinations this week — reconstructed time from context artifacts, wrongly). The metabolic organism is already the micro fossil of always-on (continuous sample+learn) but never USES continuity; cheap concrete rung = feed loop telemetry (cycle index, rolling wave-agreement rate) back as observation — time enters as learned RATE awareness. The 7-10B think: context/KV retention priced by a continuously-observing model as a metabolic function (retention has consequences only under observation) | banked 2026-07-21; frame + one cheap rung for metabolic v3+ |
| "When the model is inferenced, does it refer to the choices it made? Does it understand why its eval was wrong?" (Artin, 2026-07-21) | WAVE-CONTRAST LEARNING: today the answer is NO — rejected wave candidates vanish, GRPO passes a scalar, and the oracle's known rejection reason is discarded every wave. But mixed waves are FREE preference pairs (verified vs rejected sibling from the SAME state) = unlimited DPO-shaped data at zero extra oracle cost. Measured basis: regret probe AUC 0.914 (the model can read its own wrongness); hints twice-nulled (the why must arrive as gradient, not prompt). One-variable test: wave-contrast vs plain SFT on identical waves, same gate | banked 2026-07-21; candidate for the post-shaped-GRPO slot |

| "Test TWO words — and how do we avoid hallucination when attaching sentences/paragraphs to rules? does the transformer help?" (Artin, 2026-07-21) | SYNONYM GAUGE + THE ONE-WAY VALVE: (1) synonym-pair test — two label tokens per family on the frozen readout; gauge-law prediction: both fire near-equal off the same concept (synonyms = gauge freedom in naming, many addresses one ground state); competition instead = naming fights for mass even frozen. 424-s rerun of english-labels. (2) Longer English without hallucination: the attention mask IS the quarantine — math-first-English-after ordering + mask so English attends to math but math NEVER attends to English (frozen base = the training-time valve, proven 95%/identity last night). Every sentence anchors to a verified step -> even language gets dense verification (a claimed "by parts here" is checkable against the rule that fired). English as commentary on an oracle-signed trace, never a participant. Precision note banked with it: math is NOT one-spelling-per-thing — simplification is non-canonical, the oracle checks equivalence classes (the no-string-match iron rule is WHY grounding works) | banked 2026-07-21; synonym test queued behind calibration night |

| "Higher precision — but I mean train with higher precision, LLMUE style" (Artin, 2026-07-21) | THE FP64 MASTER-LATENT RUNG (update absorption): metabolic deltas (LR 1e-5 x small grads vs O(0.01-0.1) weights) sit near fp32's mantissa floor — many round silently to ZERO (update absorption; why mixed-precision training keeps fp32 masters under fp16 compute — same trick one level up: fp64 masters under fp32 compute, for the ONLINE loop only). Ties to the silent-until-it-fires law: STE latents integrate sub-threshold nudges; accumulator precision = the organism's subconscious memory span. Instrument first (one line): fraction of updates where w+delta==w in fp32 but not fp64, measured live in a metabolic session. Absorption ~0% -> null, save the memory; substantial -> fp64 masters = free recovery of thrown-away learning. 400MB at 50M params | banked 2026-07-21; instrumented arm inside metabolic v3 |

| "Taylor series — won't that help the farm too, farming different steps for solving ODEs?" (Artin, 2026-07-21) | THE SERIES CONTINENT, three payloads: (1) ODE power-series method = a step-generating machine (recurrence per coefficient, exact rational arithmetic, residual-to-order-N oracle) — diversifies L9b beyond ansatz chains and is polynomial-convolution-native (axiom emits at silly throughput, no heurisch walls); (2) THE LIOUVILLE JAILBREAK: int(e^(-x^2)) is trivial in truncated-series space (expand, integrate termwise) — series = vocabulary expansion as a REPRESENTATION not a token; the model gains the move "when stuck, expand" (euler-move pattern, second instance -> ceiling-movers are a FAMILY); (3) series-world = polynomials with rational coefficients, the native habitat of vocab-40 models. Pre-registered costs: truncation discipline (equivalence to order N, UNDECIDED beyond — the ODE oracle's honest-UNDECIDED machinery), coefficient growth (factorial denominators = natural depth budget), formal-not-convergent (termwise verification only) | banked 2026-07-21; candidate L10 territory + axiom series-arithmetic tranche |

| "Can the fp64 register hold the state of the fp32s? Can we stack past 64 bits? How do we know when an infinite number has the precision we NEED?" (Artin, 2026-07-22 ~midnight, the precision-stack night) | THE PRECISION STACK, unified: (1) Kahan compensation (1965) = one register carrying the collective rounding-error state of a narrow stream — Artin's sentence verbatim, and the register-level twin of the fp64 master-latent rung AND silent-until-it-fires (sub-threshold learning banked until it commits); (2) floating-point EXPANSIONS (Shewchuk): stack native floats into arbitrary precision (dd ~106 bits, qd ~212) — beats native fp64 on gaming cards because fp64 units are rationed 1/64; (3) truncation certificates: Lagrange remainder = the error of stopping is computable BEFORE the terms — the series oracle's UNDECIDED_BEYOND_ORDER is this made verdict; Remez/minimax = fit the polynomial to exactly the register's ULP (required precision = the format's resolution, engineered not hoped); (4) ADAPTIVE PRECISION (Shewchuk): compute cheap, carry your own error bound, refine only when the decision demands it — which IS the lab's stratification (ternary thoughts -> fp32 compute -> exact-arithmetic oracle) and IS the series continent's "when stuck, expand" as a reasoning move. Instrument reading en route: absorption at LR 1e-4 = 0.0007% (fp32 loses ~nothing); 1e-5 burst queued as the rung's decider | banked 2026-07-22 |

| "Maybe the mathematically perfect model for any closed system becomes more APPARENT when we eliminate rounding loss completely — higher precision -> smaller models -> built to be perfect, not generalized" (Artin, 2026-07-22 ~12:30AM) | THE ROUNDING-LOSS VEIL: the calculated-model thesis says weights are determined up to gauge by (signature, width); SGD noise AND update rounding are the two hazes over that determinism — and rounding is the one we can remove OUTRIGHT (fp64/Kahan masters at ~zero wall cost; exact-arithmetic training as the asymptotic rung). Absorption curve (running) prices the veil: 1e-4 0.0007% -> 1e-5 0.025% -> 1e-6/1e-7 tonight; GRPO lives at 2.5e-6-1e-5 = squarely inside it, so RL's selects-but-cannot-sculpt may be partly LITERAL rounding loss. Pre-registered probe: fp64-master vs fp32 paired burst at 2.5e-6, FLIP COUNT primary (sub-ULP nudges that fp32 absorbs should commit under fp64 -> more flips at equal food), proxy secondary | banked 2026-07-22; overnight decider armed |
| "Imagine: an abstract model very good at generalizing + a small efficient perfect ternary math model — unstoppable duo?" (Artin, 2026-07-22) | THE DUO ARCHITECTURE: general-reasoner routes, closed-system-perfect micro-models SOLVE — the dispatcher pattern at civilizational scale, and every piece exists in-repo at miniature: magic router (dispatch by hardness), MoE keep-set (load only the expert needed), NNUE-in-engine (tiny perfect evaluator inside big search), priors-vs-drag (the general model bluffs at 3.3% where the native model runs 65% — so DON'T make the generalist do math, make it CALL the perfectionist). The 10MB ternary artifact is the right shape for a tool: exact domain, exact oracle, no drag. Frontier echo: tool-use/function-calling is this pattern with calculators; the duo upgrades the callee from calculator to CLOSED-SYSTEM NATIVE SOLVER | banked 2026-07-22, NO-TEST by design (Artin: a proof-of-concept, not research — pointless unless the reasoner is also ours, sub-7B and built in-house); the live residue is the QUESTION it begs: how does a model REASON about what it doesn't know? House answer already half-measured: reasoning-without-knowledge = search + verification (the engine reasons with zero knowledge; the chain format teaches PROCEDURE not facts — the carry hypothesis) — the duo waits until that's understood well enough to build the reasoner ourselves |

| "In higher precisions we might find rules not discovered yet — and precision is essentially the mass: more precision, more compression, the limit of physics" (Artin, 2026-07-22 ~1AM) | PRECISION AS INSTRUMENT + THE BEKENSTEIN FOOTNOTE: (1) finer instruments reveal structure that was always there (spectroscopy -> QED; our own week: seed-variance sigma<1 made the gate a fine instrument, the absorption counter made ROUNDING ITSELF visible) — concrete bet: fp64 accumulation exposes sub-ULP "learning currents" fp32 quantizes away; decidable NOW: if the fp64 arm's flips are differently LOCATED (not just more numerous), that's learning structure invisible at fp32; (2) precision-as-mass is literally the Bekenstein bound — max information in a region scales with AREA, saturated exactly by a black hole; + Landauer (erasure priced) + Bremermann (compute priced) = storage/erasure/processing all physically bounded, gravity as the last register. Honest boundary: the Bekenstein end is frame (we are ~40 orders below); the instrument end is armed on the 3080 tonight | banked 2026-07-22 |

| "Bank that echo — run it FIRST when I give the 3080 GO, I want to know" (Artin, 2026-07-22 1AM, off the Bekenstein/square-cube arc) | BOUNDARY-OR-BULK CAPACITY SCALING: does the crystal's effective capacity scale like a volume (params x bits) or like a boundary (some interface measure, d-class)? Measured raw material already banked: the five-point width ladder (19M 64 / 45M 69 / 113M 66 / 200M 49 / 400M 30 — capability is wildly sub-volume), the alphabet bracket (bits axis), AM ~2 bits/param theory (volume-scaling prediction — the null hypothesis), bits-dimension exchange (b + 1/2 log2 d — already a non-volume form!). Design at GO: fit capability vs candidate measures across the ladder+alphabet grid — volume (N*b), boundary candidates (d, d*L, sqrt(N)); the exchange law hints capacity rides DIRECTIONS (d-geometry) more than cells, i.e., boundary-flavored. 3080 job: fill the grid's missing cells (few 19M-class births at fixed corpus), then one regression, one verdict | banked 2026-07-22; FIRST in queue at next 3080 GO |


Standing lesson, earned nine sympy pathologies and four adoption
races deep: the analogies are proposals, the oracle is the judge,
and the honest nulls (entropy beams, budget allocation, symengine,
quaternion embeddings) are recorded next to the wins. Divergent
proposal + cheap sound verification is the whole method — applied
to the collaboration itself.
- **Rarity-routed precision** (Artin, 2026-07-22, on the rarity-curve
  result): fp32 wins mid/rare (seen-but-infrequent = episodic memory
  needs resolution; ternary's absmean threshold deletes faint traces),
  ternary wins unseen (forced compression stores rules, rules
  transfer). Combination = schedule law applied to precision: ternary
  bulk crystal + high-precision channel (fp64 masters, proven 5x flip
  recovery) ROUTED BY RARITY — rare/unseen-skeleton rows train through
  the precise path, common rows through the cheap path. Marries
  valuation-routed metabolism (the router) to the absorption law (the
  mechanism). Pre-reg when run: rare-bin delta > common-bin delta.
- **Ternary = search substrate, precision = accuracy substrate**
  (Artin, 2026-07-22, from work, on the rarity curves + series null):
  ternary's compressed rules make it the better SEARCH engine
  (generalizes to unseen skeletons, cheap, fast waves); higher
  precision is the ACCURACY layer for rare/faint knowledge (episodic
  memory, exact coefficients). Suggests the division of labor inside
  one system: ternary proposes (wide waves over unseen territory),
  precise weights verify/refine (rare-technique recall) — the
  proposer/verifier split mapped onto substrates. Related:
  rarity-routed precision (the training-time version of the same
  split); duo architecture (NO-TEST, but this is the WITHIN-model
  version, testable).
- **Step-dropout curriculum / shortcuts-vs-longcuts** (Artin,
  2026-07-22): train on BOTH the full derivation and the compressed
  hop of the same problem — force the model to internalize steps it
  is allowed to skip; going backwards from the solution mints the
  pairs free (engine replays any chain at any granularity). Related:
  chain-carry redo, underdetermined-rows-train-hallucination.
- **Underdetermined rows train hallucination** (house, from series
  rung 1, 2026-07-22): a diet row whose target is not determinable
  from its prompt TEACHES confident guessing (rung 1: coefficients
  without the ODE -> memorized-factorial fill-ins). Perfect data in
  a closed system = oracle-verified AND prompt-determinable. Audit
  rows for determinability, not just correctness.
- **Duo-teacher KD ("stream the neurons, not the weights")** (Artin,
  2026-07-22): raw weight merging is gauge-broken (and shape-broken
  here), but a neuron's contribution IS streamable in the invariant
  currency — logits. Distill one student against the combined
  champion+ternary teacher signal (distill/ logit-KD + GKD ready);
  grade on whether the student holds BOTH rarity tails single-handed.
  The corpus-side twin (duo-mined chains -> next gen) is already
  adopted; this is the direct-logit version.
- **Speed-birth parity triplet** (house, from Artin's lossless-speed
  push, 2026-07-22): fp32 control vs bf16-autocast (--fast --nopack,
  in-tree untested) vs TF32 (new flag) — same seed/diet d384 births
  on cuda, paired gates. Absorption-law framing: does a 10-bit
  training mantissa dent the gate? Either answer banks; a pass buys
  2-4x on every future birth.
- **Attention-shaped arithmetic / pairwise sum trees** (Artin,
  2026-07-22 night, on the sum-step residue): attention IS a weighted
  sum of products — the op the model fails explicitly is the one it
  computes implicitly. Fix the data shape, not the architecture:
  decompose sums to one-primitive emissions (binary reduction tree).
  If pairwise trains like solve steps did, the ladder law's cleanest
  form: ANY computation trains if each emission is one primitive.
- **Math+physics MoE, grammar-routed** (Artin, 2026-07-22 night, off
  the Kimi K3 16/896 sparsity): two domain-native micro-models
  (math = current lineage; physics = classical mechanics chains,
  which are ODE grammar wearing units) + a router that is a grammar
  check — EXACT routing at the domain boundary, no learned router.
  The micro-answer to frontier MoE: fewer experts, provable routing.
  Classical first (rides tonight's 4.3x series machinery); quantum
  second (ZX rungs already banked). Relay drafted (asks 1-3).
- **Duo router bake-off** (house+Artin, 2026-07-22): skeleton-bin
  router (exact, free) vs magic-estimator-style learned router
  (dispatcher-v3 lineage) vs plain 8+8 mix (133 baseline). Judge
  doesn't starve: variance lives at the 133-vs-137 gap.
- **The blackboard protocol (two-expert interaction)** (Artin's
  question + house design, 2026-07-22 night): optimal closed-system
  interaction = (1) route per-STEP by grammar membership (decidable
  because each expert's world is closed); (2) the cur-string IS the
  interface — chains are Markovian, so a physics step that reduces
  to Integral(...) is verbatim a math-model prompt, no translation
  layer; (3) every handoff oracle-gated — models cannot pass each
  other unverified state (the categorical difference vs multi-agent
  LLM prose); (4) asymmetric call graph: physics calls math, never
  the reverse (epistemic structure = DAG, no loops). Requires ZERO
  interaction training if physics chains are farmed to end reduce
  steps at bare math grammar. Pre-registerable: end-to-end physics
  solve rate, blackboard pair vs matched-params monolithic
  both-diets model. Related: math+physics MoE, always-on frame,
  scratch-workspace (banked).
- **The decomposition discount (re-price perfection)** (house
  synthesis, 2026-07-22 night, from the full-docs read): the
  equation's perfection price (~650k eff rows) was computed with the
  CHAIN-row exposure constant k — but the series arc measured
  one-primitive rows training to ~100% at 26k rows (15->67->88%
  tracking decomposition depth). Fit k_primitive vs k_chain from
  1c/1d data in hand; if primitive rows are 10-50x cheaper per
  saturation, the perfect-model price collapses from weeks to days,
  and "enumerate the primitive basis of calculus" becomes an axiom
  tranche. The ladder law and the exposure curve were always one
  law: eff-rows should be counted in PRIMITIVES, not rows.
- **Portability test rides the physics birth** (house, 2026-07-22
  night): the closed-system equation's pre-registered test #3 (do
  kurtosis/R/CV/floor shift on a DIFFERENT closed system?) has
  waited since 07-19 for a second grammar — the vocab-41 physics
  birth IS it. Read the template panel off phys_19m.pt at zero
  extra compute; invariants-shift => the constants encode Sigma
  (flagship result); invariants-hold => they are constants of
  training itself (also a result).
- **Lazy precision routed by near-ties + EU-arithmetic + entropy-
  coded weights** (Artin riff barrage, 2026-07-22 midnight): three
  compositions of banked results onto expansion arithmetic — (1)
  THE PRECISION ROUTER: Shewchuk adaptive precision (compute cheap,
  refine only when the decision demands) x the fp16 near-tie
  doctrine = fire correction matmuls ONLY when the top-2 logit gap
  sits under the error bound; margins are free every decode step;
  starved-judge satisfied by construction (corrections run only
  where variance lives). (2) EU-ARITHMETIC: online updates touch
  rank-~4 slivers, so cache the heavy hi@hi structure and delta-
  update only moved weights — the NNUE namesake applied at the
  arithmetic layer (KV-cache logic for weights). (3) entropy-coded
  checkpoints: kurtosis-2.4 sub-Gaussian = low-surprise bits
  (int4-lossless's cause) => Markov/arithmetic-code weights at
  rest; ternary at 30.5% zeros compresses hardest. Storage only.
- **Precision-patch LoRA ("repair neurons, add skills")** (Artin,
  2026-07-23 ~2AM): ternary bulk crystal + a tiny fp32/fp64 LoRA
  channel (rank ~4 per the whisper) trained ONLY on rare-bin misses
  — episodic-memory prosthetic on the rule crystal: gauge-safe,
  removable, auditable, and "new skills" arrive as adapters without
  touching the resident function. Composes: rarity-routed precision
  (this is its weight-space form), duo-teacher KD (adapter distills
  the fp32 champion's rare wins), dynamic capacity (adapters ARE
  model-determined size growth, rank chosen by demand).
- **PRE-REGISTERED DISAGREEMENT — fp64 birth rarity curve** (booked
  2026-07-23 2AM, before the verdict): ARTIN predicts the fp64
  birth lifts the RARE bin (resolution preserves faint traces at
  birth). HOUSE/absorption-law predicts FLAT everywhere (birth
  LR 3e-4: absorption 0.0007%, nothing faint enough to save).
  Grading: rarity gate on fp64_birth vs seedvar-1, rare-bin delta
  primary. Either verdict banks a law leg.
- **"Dynamic precision = absorb more without adding terms"** (Artin,
  2026-07-23 ~1AM — the capstone phrasing): offline, storage has two
  levers — add terms (farm rows, raise signal) or add bits (lower
  the ULP floor); both measured this week. ONLINE there are no more
  terms — experience streams once, and absorbed updates are lived
  experience permanently lost. So for the LLMUE the precision
  channel is the only lever, and it should be DYNAMIC (only ~0.3-3%
  of online updates are faint enough to need it — absorption law).
  One sentence: precision is how a finite stream of experience
  becomes fully absorbed structure. Unifies: absorption law,
  LR=precision knob, rarity-routed precision, surprise-gating,
  fp64 masters, the ceiling-on-slow-learning theory. Test = the
  metabolic v3 headline hypothesis.
- **The Ozaki composition** (Artin, 2026-07-23 ~1:30AM — "mix and
  match bitwise + double-double + chunking"): independently
  re-derived the Ozaki scheme (error-free matmul transformation):
  bit-slice via power-of-2 shifts (EXACT — exponent moves only),
  partial products between slices on fast low-precision units
  (TF32/int8 tensor cores) are rounding-free, recombine with
  compensated sums. fp64+ matmuls at tensor-core speed; on gaming
  cards BEATS native fp64 (1/64 rationing). tf32x3 = the k=2
  special case (built+parity-proven tonight). House application:
  the online precise channel's endgame implementation — dynamic
  precision at ~zero marginal cost. Literature: Ozaki et al.;
  Ootomo/Ozaki int8-TC DGEMM line.
- **Limits continent + the limit-free observation** (Artin, 2026-07-23,
  "does it know how to derive the rules of calculus?"): limits were
  never fed — deliberately; the lab's calculus is rule-application
  over a limit-free formalization, and the series continent IS that
  formalization made explicit (formal power series: derivative =
  coefficient shift, no epsilon anywhere — differential algebra, the
  Liouville tradition; O-marker discipline = its honesty mechanism).
  Two rungs if wanted: (1) LIMITS CONTINENT — lim chains w/ sympy
  limit oracle, factoring/l'Hopital as moves, determinable,
  medium-cost, real candidate; (2) rule DERIVATION = proof territory
  (Lean oracle, site-selection map's far rung). Note: the engine
  also doesn't know why its rules work, and it beats sympy — 
  justification and capability are separable in a closed system.
- **Refutation-first conjecture rung + chains-are-proofs** (Artin,
  2026-07-23): (1) solved chains are ALREADY formal proofs of
  exists-statements (witness + machine-checkable certificate per
  step) — the missing quantifier shapes are forall, impossibility,
  and rule-derivation; (2) the cheap entry to conjecture territory
  is DISPROOF: counterexample = one emission, oracle-decidable —
  refutation is closed-system-native today (FA-shaped: model
  searches, oracle certifies); (3) the forall path = Lean-oracle
  tactics (site-selection far rung), UPGRADED by this week's law:
  proofs decompose into one-tactic emissions, and one-primitive
  emissions train to ~100% — the decomposition discount may apply
  to tactics. Also noted: differential algebra derives the calculus
  rulebook without limits (derivation operator, Leibniz axiom) —
  a proof-capable model could derive its own rules limit-free.
- **The solved-only mining leak** (Artin, 2026-07-23, "steps should
  outweigh the solution"): training is already all-steps (no solution
  bonus exists in any loss) — but MINING banks chains only on SOLVED,
  discarding every verified step of failed attempts: on the frontier
  (where most attempts fail) that is the majority of paid-for
  verified experience, at exactly the rarest skeletons. Counterweight:
  dead-end steps may teach wandering (chain-carry hypothesis). One-
  variable test: miner v2 banks verified steps from unsolved
  rare/unseen attempts; gen-8 A/B solved-only vs +failed-steps.
- **The open-problems probe ("does it try anything untried?")**
  (Artin, 2026-07-23): our universe's own million-dollar questions =
  the fused-quotient L8 residue + joint-ceiling L7 roots (unsolved by
  engine, sympy, and every model). Give crown models + duo sampler a
  huge budget on exactly those, log EVERY attempted move, diff
  against the engine's rule table. Success metric is not solving —
  it is novelty: a rewrite outside the engine's move vocabulary.
  Desert-test precedent says the attempts will be honest either way.
- **Fourier continent + the chord-space/gauge-space identity**
  (Artin, 2026-07-23, off Liszt-as-waveforms + 3B1B's two-note-chord
  Mobius strip): (1) FOURIER SERIES CONTINENT — the Taylor sibling:
  "which note is in the wave" = Fourier coefficients = integrals of
  f*cos(nwt), which the engine already solves; partial-sum chains,
  coefficient trees, O-marker discipline all port; in-charter (pure
  math + the spectral toolkit of physics). Audio replay excluded
  (no oracle for beauty). (2) THE ORBIFOLD IDENTITY: chord space
  (Tymoczko: n-note chords = T^n/S_n, the 3B1B Mobius strip at n=2)
  and our weight-gauge space (R^n/S_n, neuron permutations) are the
  SAME construction — "a chord is not an ordered list of notes" IS
  "never score weights by distance." Possible payoff: orbifold
  quotient metrics / geodesics (voice leadings) as the formal answer
  to "distance between models up to gauge" — the tool weight-space
  comparison has been missing. Music enters as mathematics, charter
  intact.
- **PRACTICE MODE** (Artin, 2026-07-23 — "how would the piano model
  practice La Campanella? start near where it fucked up"): restart
  rollouts from FAILURE STATES, not roots — independently re-derived
  Go-Explore / reverse-curriculum (Ecoffet 2019, Florensa 2017), and
  strictly better here: chains are resumable at every state by the
  determinability property (a mid-chain cur IS a complete prompt),
  and failure states are already computed free in every dead wave.
  Design: (1) miners/gates log stuck-states; (2) practice rollouts
  launch FROM them (budget concentrated on the failing transition,
  none on the mastered prefix); (3) verified steps out of stuck
  states = the highest-value mintable rows; (4) practice shard joins
  the diet. THE COMPOSITION: stuck states are vfrac~0 by definition
  = maximum-surprise food — this IS the metabolic food policy the
  morning's starvation null demanded. One machine: practice-mode
  miner feeds both the diet and the organism.
- **Practice mode, engine-side + the stuck-state exchange** (Artin,
  2026-07-23): the engine's walled searches hold partial derivations
  — bank the last state to a stuck-state worklist (state-level
  generalization of the tranche-4 root worklist); band exhaustion
  redefined as "stuck queue empty," farm DEEPER before WIDER. The
  exchange: engine stuck-states = hard probes for models; model
  stuck-states = hard roots for the engine — two practice loops
  trading their hardest bars, all in the same cur-string format.
- **The asynchronous precision co-processor** (Artin, 2026-07-23 —
  "offload fp64 pressure to CPU, like tf32x3"): split by
  SIGNIFICANCE across heterogeneous units — GPU runs the hi channel
  (bulk fp32, synchronous), CPU runs the lo channel (fp64 masters /
  Kahan carries / faint-signal corrections) ASYNCHRONOUSLY behind
  it, out of shared DRAM on unified memory (no bus tax — Artin's
  catch). License: the precision channel's content is sub-threshold
  by definition ("silent until the neuron fires") — a latent channel
  needs no synchrony; GPU never waits, precision costs ~zero wall.
  Scope: elementwise masters = perfect fit; correction matmuls only
  if small/sparse/laggy (absorption law says the faint mass is
  0.3-3% — small); drift bounded by sync barriers + existing
  gate/rollback. The implementation home of the precision program.
- **The bridge law (transfer needs shared steps in context)** (Artin,
  2026-07-23, on the poly gate null): the day measured both arms —
  physics+math transferred (+6.8; phys chains CONTAIN integration
  steps) while poly+math did not (gate 64; pf rows are standalone
  identities co-resident with integrals, never touching them).
  Candidate law: skills link only through rows where one skill does
  work INSIDE the other — adjacency teaches nothing. Fix = BRIDGE
  CHAINS: integrals whose certified path surfaces the pf step as an
  explicit rewrite (engine already solves these internally). Pre-reg:
  bridge diet lifts the integral gate where the co-resident diet
  (poly2, 64) did not.
- **Speculative arithmetic ("KV-cache the masters; SR drafts, Ozaki
  verifies")** (Artin, 2026-07-23 night): the NNUE/EU pattern applied
  to the precision channel — cheap arm (SR/fp32) DRAFTS every result;
  a free error BOUND (or a cached exact value) verifies; only
  disagreements fire the exact Ozaki path. Spec-decoding economics
  transplanted to arithmetic: acceptance rate is the whole game, and
  the absorption law says ~97-99.7% of updates are coarse enough for
  the cheap path — the exact path fires exactly where variance lives
  (starved-judge satisfied by construction). The cache leg is
  EU-arithmetic (online updates are rank-~4, so cached exact products
  delta-update cheaply). Composes: near-tie logit router (inference
  twin), triangular truncation (the dial the verifier turns). Slots
  into metabolic v5 as the SR-vs-masters race's synthesis arm.
- **The carry-free lazy pipeline ("we can the pipeline carry-free?
  literally lazy-computing")** (Artin, 2026-07-23 late): stay in RNS
  for entire computation chains, exit once — measured same night:
  exact 4-layer chain at 53 ms channels vs 173 ms inexact fp64;
  break-even ~6 layers, deeper = exact AND faster [SHAPE-FENCED
  2026-08-10, RNSCHAIN-C2C3 RESULTS 25338: that break-even is a
  fused-tensor-core property — scalar mulmod chains are ~74x
  fp64 DEPTH-FLAT (71.8-76.8, depths 2-12), no crossover for
  that shape]. Composes with
  fractional-CRT decision exits (10 ms) and the speculative-
  arithmetic verifier. The deferred-carry frame: RNS is lazy
  evaluation applied to ARITHMETIC — carries are thunks, forced only
  at observation. Candidate consumers: optimizer step chains
  (+/x-pure), attention score pipelines, the online precise channel.
- **The deterministic exit ("anything deterministic about leaving
  RNS we can compute/cache?")** (Artin, 2026-07-23): yes, three legs
  — CRT/Garner constants precompute once (done); fractional-CRT
  gives a cheap ESTIMATE exit for decisions (measured 17x cheaper);
  and the exact exit is a pure function of the residues, so it
  delta-updates under the EU pattern when few outputs change
  (rank-~4 online updates = few dirty outputs = incremental Garner).
- **The union equation ("is there a formula for combining closed
  systems — can it pinpoint what a model needs?")** (Artin,
  2026-07-24 ~1AM): proposed form, every term already measured or
  pre-registered tonight — capability stays PER-GRAMMAR
  (solves^A_L = 24(1-exp(-eff^A_L/k(W)))) and grammars couple
  through EXACTLY THREE channels: (i) SHARED CAPACITY — one W
  serves the union, so W* rises with the union corpus
  (tokens-per-width; gen-8's pre-registered common-bin-dent test IS
  this coefficient); (ii) THE BRIDGE TERM — eff^A gains
  a_bridge x (rows where B's grammar does work INSIDE A's contexts);
  co-residency contributes ZERO (poly2 measured; dual-physics +6.8
  measured; poly4 tonight measures a_bridge with the door open);
  (iii) NOTHING ELSE — geometry is grammar-free (universality x3),
  determinable rows don't interfere (1d: 66k series rows, zero
  dent). Payoff if it holds: "what a model needs" becomes
  computable per union — count primitives per grammar (10k/kind,
  ~2k rows/e-fold), add bridge rows only where transfer is wanted,
  size W to the union's W*. The calculator generalizes from one
  closed system to any federation of them. Equation-v1 row queued
  on tonight's gen-8 + poly4 verdicts.
- **Extrapolation-past-ability / the composition coefficient**
  (Artin, 2026-07-24 afternoon, from the MoE-dark-experts riff,
  self-pivoted: "routing isn't our issue — it's the ability to
  extrapolate past your ability"): can the everything-crystal
  COMPOSE grammars at inference — grammar A's moves firing inside
  grammar B's states — to reach answers neither grammar's diet
  permits alone? Measured ground: extrapolation exists (champion
  9/24 on never-fed L9; 113M leads past the corpus edge; ternary
  wins unseen) but composition has never been probed. The
  instrument already exists: DESERT TEST v2 — feed gen-8 the
  Liouville-dead integrals (int e^(-x^2)); series-space makes them
  trivial (the Liouville-jailbreak bank) and gen-8 holds both
  grammars. Outcomes: (a) honest stall = co-resident, never
  composing (bridge-law prediction, house bet); (b) spontaneous
  reach for series expansion = first measured act of cross-grammar
  discovery; (c) bluff = diagnostic regression. If (a): farm
  bridge rows demonstrating "when stuck, expand" — discovery as a
  learned MOVE. Names the union equation's candidate fourth
  coefficient: (iv) COMPOSITION (inference-time coupling, distinct
  from training-time bridge rows). MoE-scale note: no dark experts
  at 19M — holography (8 random neurons read 87%) means every
  fragment fires; the dark-expert problem is frontier-scale only.
  Related: open-problems probe (novelty metric), blackboard
  protocol (the two-model version of the same question).
- **The vocab-width floor / grow-with-the-map** (Artin, 2026-07-24
  night, off the 19M-beats-45M verdict + "territory is
  vocabulary"): is there a MINIMUM width relative to vocabulary/
  grammar count — and if W_min(vocab) is found, birth AT the floor
  and grow as territory (vocab) grows: dynamic capacity triggered
  by vocabulary instead of schedule, births faster at fewer
  params. Measured ground: W* only ever probed from ABOVE (45M/
  113M/400M all starve); nothing below 19M ever birthed; the NNUE
  (20->64->64->1) is the oligarchy-phase floor datapoint; the
  banked min-width-for-alphabet-poor sweep is the alphabet-axis
  sibling. EXPERIMENT (cheap): downward width ladder d64/128/256
  on the gen-9 diet, minutes/birth on Mac — maps k(W)'s other end;
  composes with template-spray growth (the grown-champion lineage)
  and tonight's k(W)-in-tokens refinement. Prediction to
  pre-register at run time: the floor is set by GRAMMAR structure
  (chain depth/context), not vocab count (41 tokens is
  information-trivial; the derivation is not) | banked 2026-07-24;
  weekend Mac candidate |
- **THE CLOSED-SYSTEM-NATIVE TRANSFORMER** (Artin, 2026-07-25
  ~12:30AM — "we can optimize SO MANY parts: NNUE/engine weighting
  directly for self-attention tuning; positional encoding
  deterministic/cacheable; every other single part, you name it"):
  the architecture itself becomes a closed-system design surface —
  every generic transformer component replaced or seeded by
  MEASURED engine/oracle structure. The component map, each leg
  composing an already-banked thread: (1) ATTENTION PRIORS FROM
  THE ENGINE — markov bigrams / syndrome-policy stats / NNUE
  valuations as attention-bias or head-init (valuation-routed
  metabolism's inference-time sibling; the engine's verified
  attention-over-rules becomes the model's attention-over-tokens);
  (2) STRUCTURAL POSITIONAL ENCODING — expressions are TREES, so
  positions can encode tree-paths (deterministic, cacheable,
  rewrite-invariant for untouched subtrees) instead of flat RoPE;
  composes with EU-arithmetic: candidate rewrites share ~95% of
  the tree -> cached subtree encodings + KV = score a wave for
  ~one forward (the NNUE namesake at the representation layer);
  (3) grammar-masked heads, ternary-native FFN (alphabet program),
  hint-head territory already nulled (twice) — the map inherits
  every measured null as a DON'T list. Guardrails from the ledger
  itself: teach-don't-impose (permutation-augmentation beat
  canonical sorting; hints twice-nulled; the model prefers trained
  to told) — so every component swap is a one-variable A/B against
  the vanilla birth at equal cost, pre-registered, starting with
  the cheapest-highest-variance leg. Speed thesis (Artin): quicker
  AND ceiling-lifting — the sooner adopted the more every future
  birth pays back | banked 2026-07-25; reviewer survey DELIVERED
  same night — ORDERING (variance x cheapness): rung 1 = PREFIX
  NOTATION (enabling substrate for tree-anything; aimed at the
  live emission wall psub/padd/ibridge; certain 20-30% sequence
  shortening; bar = gate within noise + >=20% shorter, secondary =
  operand-complexity kinds + int3 delta); rung 2 = SUBTREE-HASH-
  keyed tree-PE on prefix (HASH not path — tree-paths are NOT
  rewrite-invariant, the EU 95% claim is hash-identity; hybrid PE
  needed for the Current:/Step: frame; graded by the zero-epoch
  gate); rung 3 = attention-init from tree-adjacency, INIT-ONLY +
  unseen-mass smoothing, graded on ep1 speed (warm-birth says
  time-machine-not-ceiling). FORBIDDEN as-stated: valuations as
  prompt features (hints x2 null), syndrome aux head (payoff-3
  null), frozen markov bias (prior-wash), canonical sorts (gauge).
  Unifying litmus: seed STATISTICS that training error-corrects =
  pays; impose microstructure / freeze / tell = null |
- **Streaming birth** (Artin, 2026-07-24 ~2:45AM, half-retracted
  same breath — banked anyway): template init + stream the corpus
  ONCE, no epochs, surprise-gated LR — birth as metabolism from
  minute zero (the organism never sees a "dataset", only
  experience). A/B vs the standard 3-epoch birth at equal rows.
  Composes: warm-birth template (time-machine result), surprise
  gating, the d2 streaming harness. Open question it answers:
  is the epoch an artifact of the batch era?

- **2026-07-25 (reviewer): rung-1b — prefix + arity-mask decoding.**
  Grammar-constrained decoding nulled twice on infix (misses were
  SEMANTIC); prefix's measured 8.1% parse-fail is SYNTACTIC, the
  one regime where a well-formedness mask (arity counting) could
  recover the −3 gate tax. Cheap test, banked behind the emission
  probe read. Attribution: reviewer agent, rung-1 cross-check.

- **2026-07-25 (house, via rung-1 close): GCD-pays-iff-syntactic.**
  Grammar-constrained decoding pays iff the miss distribution is
  syntactic; infix GCD nulled because misses were semantic; prefix
  Q4 (30% parse-fail) is the regime where it would have paid.
  Supersedes rung-1b arity-mask (moot — prefix closed; conditional
  note only). Attribution: reviewer refinement on Fable's probe.

- **2026-07-25 (Artin): the TurboQuant/Lloyd-Max riff.** Artin
  surfaced TurboVec/TurboQuant (codebook-oblivious scalar
  quantization: rotate -> predictable distribution -> fixed
  Lloyd-Max codebook, no calibration) + faiss/qdrant. House
  connections: (1) the alphabet tournament IS hand-made codebook
  design; Lloyd-Max is its principled rival, and our kurtosis
  invariance (2.4 at every width) means births are ALREADY
  codebook-oblivious-ready without the rotation trick — plausibly
  why naive int3 absmax was near-lossless at 19M; (2) THE CLASH:
  Gaussian Lloyd-Max 2-bit has NO zero level; the house
  zero-is-load-bearing law predicts zero-forced beats MSE-optimal
  at matched bits — falsifiable for minutes of PTQ; (3) rotation
  leg = the pending rotation bracket (banked rider);
  (4) PQ-for-KV-cache = banked line on bench_kv_quant_decode.
  Vector search itself (RAG/tenancy) = tangent, no build.

- **2026-07-25 (reviewer, portrait audit): visualization upgrades.**
  Ranked: (1) Procrustes gauge-fix before cross-model portraits —
  unlocks "same lattice up to gauge?" (calculated-model thesis
  test; supersedes the banked gauge-fixed-phase item with a
  concrete method); (2) whitened null + shared color (SHIPPED
  same-day); (3) clade/family neuron coloring — visual holography
  test (mixed = holographic, clumped = theory breaks); reclaims
  the redundant color channel; (4) shared-scale width-ladder small
  multiples; growth animation ep0->ep1->final (motion honest,
  static 3D rejected — occlusion is rotation-dependent).
  Attribution: reviewer agent; Artin commissioned the audit.

- **2026-07-25 (Artin, w/ a Gemini assist): the constants-alphabet
  riff -> the complex bracket.** Raw ask: {−∞,−1,−i,0,i,1,+∞} +
  pi/e + diagonal units + quaternions as weight alphabets; plus
  "does anything fit EXACTLY in 4 bits?" and the {−1,0,1}-symmetry
  hypothesis. House distillation: (1) TWO-OF-THREE LAW —
  {zero, negation-symmetry, integer bits}: pick two (symmetric
  zero-ful sets are odd -> fractional bits always; names why every
  tournament winner has weird bit-counts). (2) Exact-4-bit
  designs: M4^2 complex (a+bi, a,b in {−1,0,1,2} — 16 states,
  keeps zero, inherits M4's asymmetric +2) vs G16 polar grid
  (4 phases x 4 magnitudes, symmetric, NO zero — the deliberate
  zero-law falsification arm) vs scalar int4. Prior: M4^2 >= int4
  > G16. (3) Rotation bracket made concrete: G5 {0,±1,±i} 2.32b;
  D9 = T (x) T (ternary-squared!) 3.17b; Q9 quaternion units
  3.17b; headline = rotation-vs-magnitude at matched 3.17 bits
  (Q9/D9 vs P2). (4) ±∞ -> saturation rung; pi/e -> PHASE not
  magnitude (weight scale is gauge; roots of unity are where pi
  lives). Complex weights = 2x2 real blocks; born-only (no honest
  PTQ path). Attribution: Artin (set + symmetry hypothesis +
  exact-4-bit ask); Fable (two-of-three, M4^2/G16, T-squared
  identity); reviewer red-team pending.

- **2026-07-25 (Opus-5 reviewer, first riff): THE GAUGE-SUBGROUP
  LAW candidate + the zero-birth prologue.** Independent re-review
  of the complex bracket (did NOT endorse as specced): (1) an
  alphabet pays when its symmetry group is a SUBGROUP of the
  network's gauge group — zero = the group identity/fixed point
  (Artin's midpoint intuition formalized); symmetric alphabets
  make quantization gauge-equivariant; rotation alphabets need
  phase-equivariant activations (modReLU) or the prior dies at
  every nonlinearity. (2) All named cells still resolution-
  confounded at real-DOF (M4^2 = M4-twice; G5/M5 and Q9/D9 carry
  2x gaps). (3) PROLOGUE (zero births, rides the race chain):
  gauge-commutation test, S4 {±1/3,±1} symmetry-without-zero,
  sparsity control at ternary's zero-fraction, weight-FFT euler
  read. (4) Falsifiable rider: asymmetric alphabets degrade with
  DEPTH not width. Births demoted below d256 gate, hang on
  prologue + modReLU decision + gate-zero identity test.

- **2026-07-25 (Artin, the quantum riff — settled after back-and-forth).**
  Raw: Grover/quantum to speed training; quantum distributions to
  find which neurons a family activates; a "quantum model" with
  complex weights and negative classification. Settlement:
  (1) GROVER-NO booked with reasons (N = configs not params so
  sqrt(exponential) stays exponential; the oracle call IS the
  expensive eval — no cheap-verify asymmetry exists in training;
  measurement collapse blocks distribution readout). Never chase.
  (2) The "quantum model" = the COMPLEX BRACKET verbatim
  (unitary=rotation alphabets, interference=the zero/cancellation
  laws, Born rule=modReLU readout) — motivation note added there.
  (3) NEW BANK — AMPLITUDE RATIONING: measure per-family neuron
  activation footprints on the crystal, find cold regions, weight
  the diet toward families touching them (connects
  isolated-clades-die-first; classical, cheap, house instrument
  exists). (4) NEW BANK — ZX GRAMMAR CONTINENT: teach the crystal
  ZX-calculus rewrites as a closed system (engine exists as
  oracle; charter-legal physics); natural test bed for
  alphabet-follows-domain (does a rotation-alphabet model learn
  the rotation grammar faster?). (5) The surviving classical
  amplitude-amplification analogy = surprise-gated streaming
  (already queued). Attribution: Artin (riff, via 3b1b), Fable
  (settlement).

- **2026-07-25 (Opus-5 cross-check on the quantum riff) — corrections adopted.**
  (1) GROVER-NO reason 2 re-spelled: the ENGINE does have the
  cheap-verify asymmetry (substitute-back is cheap — the lab's
  founding fact); the correct fence = (i) the engine's search
  space is a DYNAMIC tree (Grover needs static indexed space +
  QRAM), (ii) Grover's sqrt-N is a STRUCTURELESS bound — our
  syndrome/valuation pruning exploits structure and beats it;
  you can't compose them. Readout: expectation values legal but
  O(1/eps^2) shots eat the speedup (not "collapse").
  (2) G16 RE-LABELED: phase pairs at pi can SYNTHESIZE an
  effective zero across a 2x2 block — G16 is not a clean
  zero-test; its question becomes "can phase cancellation
  substitute for a zero level?" (prediction flips to
  may-survive). S4 remains the real zero-test.
  (3) Unitary leg: uRNN/EUNN precedent (gradient stability, not
  capability) — cite, don't run.
  (4) Underweighted quantum-adjacent banks: TENSOR-TRAIN/MPS rank
  axis (bits x dimension x RANK — third axis of the exchange law,
  house-native via llmopt/quantum) and ANNEALING over discrete
  weight lattices (the only untried route to born-quality
  lattices without gradients).
  (5) AMPLITUDE RATIONING: legal (teach — arrives as gradient),
  but high null prior (rations-with-extra-steps); mandatory
  control = rarity-matched vs footprint-guided at matched dose;
  GATED ON the free clade-coloring read (holographic mixing =>
  no mechanism, close cheap). Ranked ~7th.
  (6) ZX CONTINENT promoted to NEXT-CONTINENT CANDIDATE #1
  (ahead of complex births): first GRAPH grammar — the only
  available test of whether the federation floor scales with
  grammar COUNT or CLASS. Two desk blockers before any farm:
  graph serialization must avoid canonical sorts (gauge, and the
  native-transformer sequel risk) and ZX atom set vs vocab-41
  (the ODE void-by-vocab lesson).

- **"Can we optimize excitation / get around the no-negatives
  collapse? Isn't opposition the entire point?"** (Artin,
  2026-07-26, watching born-Z1's loss-27 birth): distilled — (1)
  OPPOSITION IS THE COMPUTATIONAL PRIMITIVE (his phrasing,
  adopted): the zero-law's deepest form; Z1's confidently-wrong
  divergence (27 >> ln(40)=3.7) is the measured demonstration.
  (2) Three opposition channels that don't need per-weight sign:
  per-CHANNEL sign = DALE'S LAW made literal (Z1S, in flight —
  biology's own answer: neurons are wholly excitatory or
  inhibitory); PHASE = opposition via destructive interference —
  re-prices G16 from "interference substitutes for ZERO" (cheap
  post-S4) to "interference substitutes for SIGN" (the sharp
  question, since Z1 proves sign is the load-bearing half);
  architectural subtraction (LN mean-sub/softmax — the reviewer's
  bet-49 mechanism, priced by Z1's gate). (3) THE QUANTUM NNUE
  pair, banked: complex-weight NNUE vs real at matched real-DOF
  on the 3,689 magic labels — the first alphabet cell in the
  OLIGARCHY phase (kurt 4.78; every alphabet law so far is a
  democracy-phase law); and ZX-NNUE (graph features -> descent
  yield) as the ZX engine's eval brain — the NNUE homage on
  quantum data. Attribution: Artin (riff + the opposition
  thesis); house (Dale's-law identification, G16 re-pricing).

- **The quantum-LLMUE walk (Artin, 2026-07-26 close)**: four riffs,
  each with a house translation banked for the next-session spec:
  (1) QUANTUM LLMUE = composition of three qualified pieces (ZX
  farm -> vocab-51 ZX birth -> G5xZX factorial cell -> metabolic
  sessions on the ZX stream w/ flip census) — the program north
  star, every component measured except the composition. (2)
  COMPLEX-PLANE DISTANCE: complex weights upgrade the gauge group
  to U(n); unitary Procrustes rides the banked joint-perm closure
  (seed-pair kill condition inherited — ancestry verdict fences
  all groups). (3) **THE STREAMING-OPTIMIZER CELL (best of the
  batch)**: the homogeneity -12 is a gradient-COVARIANCE wound;
  Muon/Shampoo-class orthogonalized updates = diversity-per-step
  bought in the optimizer; streaming v5 at d256 vs v4's 57 — if
  it closes the -5 epoch gap, "the epoch is load-bearing"
  retracts to "SGD's update algebra wastes single passes."
  (4) Quantum-parallel step elimination = the wave + EU
  delta-scoring/batched-KV (shared-prefix amortization is
  superposition's mechanizable content; Grover fences intact);
  the residual quantum idea = INTERFERENCE between candidates
  (G16's question at inference; far-future line). Plus the
  "something missing" candidate: TEMPLATE-REFRESH MID-STREAM —
  recompute corpus statistics from eaten rows, correct the
  crystal toward them directly, gradients handle only the
  residual (calculated-model thesis applied DURING training;
  never run). Attribution: Artin (all four riffs + the
  something-missing instinct); house (translations).

- **"Put the magic in the LLM / is cur-next even optimal? maybe
  the schema confuses streamed template models"** (Artin,
  2026-07-26, post-Muon-crater): distilled into three legs by
  schema layer — (1) MAGIC-AS-HOP: magic-as-input is twice-nulled
  (hints) and magic-as-aux-head has the syndrome payoff-3 null as
  sibling; the LIVE form is the regret probe (AUC 0.914) promoted
  from instrument to POLICY — a learned branch-abandon/hop bit at
  inference, acting on the wrongness signal the model already
  carries; FA-shaped (model hops, oracle still judges). (2) THE
  FORMAT x SCHEDULE INTERACTION (new, his): all chain-carry
  format evidence is 3-epoch evidence; a single-pass streamed
  model never gets the revisit that makes a Markovian cur/next
  pair cheap — the banked chain-carry ablation gains a streaming
  column (format x {3ep, 1-pass}) and could explain the epoch's
  -8 residual as a FORMAT dividend of revisits. (3) Schema layers
  named: oracle layer (non-negotiable) / format layer (cur-next,
  decomposition, notation — least closed) / conditioning layer
  (magic, hints, syndromes — mostly nulled). Attribution: Artin
  (riff); house (layer split + interaction cell).

- **"We probably want loss going into the 0's with a closed
  system, or am I misunderstanding? + distance-weighted /
  symmetric Muons?"** (Artin, 2026-07-26, post-Muon-close): two
  banks — (1) THE CE-GATE STUDY (promoted to a want-to-run: "what
  the gate measures that loss can't see is entirely the whole
  point"). House hypothesis to pre-register: in this closed
  system CE CANNOT honestly reach 0 because the diet is
  one-of-many-valid — the same cur admits many oracle-valid nxt
  and the farm banked ONE arbitrary choice, so Bayes-optimal CE
  is bounded below by the true branching entropy of valid steps.
  Pushing CE under that floor (Muon 0.41) means reallocating
  probability mass FROM valid alternatives TO the farm's
  arbitrary pick — loss improves exactly by deleting the
  distribution-over-valid-moves that generative solving samples
  from. Instrument (cheap, checkpoints in hand): on held-out cur
  states with multiple known-valid nxt, compare mass-on-valid-set
  vs mass-on-farm-pick for Muon 10/34 vs AdamW 45 vs control 65
  — if mass-on-valid-set tracks gate while CE anti-tracks, the
  gate is measured to be BRANCHING COVERAGE and "loss to 0" is
  formally the wrong target (memorization of arbitrary picks).
  Muon checkpoints = the specimens (widest split ever measured).
  (2) MUON VARIANTS bank (untried, low priority, conditional on
  the study): real LR sweep (2 points isn't a curve), decoupled
  weight decay, Muon-native schedule (not AdamW's multipliers),
  soft/symmetric blend (interpolate orthogonalized and raw
  momentum), trust-region/spectral-cap step control ("distance-
  weighted" made legal: distance as step-size governor is fine —
  the closed law only forbids distance as a FUNCTION score), and
  scale (published Muon wins live far above d256). Attribution:
  Artin (loss-floor question + variants riff); house (branching-
  entropy floor hypothesis + instrument design).

- **MCP SSH server for the 3080** (Artin, 2026-07-26, off the
  ssh-tooling pain): wrap `ssh -i $WSL_KEY` in a small MCP server
  (the markitdown precedent — user-scoped, `claude mcp add`) so
  remote ops become first-class tool calls with real timeouts,
  instead of Bash incantations that hang the client on nohup.
  Interim shipped same day: scratch/wsl.sh (run/launch/check/tail;
  base64-safe commands, setsid detach kills the hang class,
  success-only markers, self-excluding pgrep). BANKED: build/vet
  the MCP server at a quiet moment, not mid-experiment.

- **"One-shot the DISTRIBUTION a quantum computer would give over
  all first steps" + "delta gives the model the rest of the chain
  by reference"** (Artin, 2026-07-26): distilled — (1)
  DISTRIBUTION ROWS: the branching-entropy-floor repair made
  concrete: per cur, the training target = mass over EVERY
  applicable move (engine rule-fire enumerates ~ms, wave oracle
  verifies, MarkovPrior weights) — soft-label distillation from
  the engine's move table; the from-birth sibling of wave-contrast;
  fixes CE-trains-the-arbitrary-pick at the source. Fence: must
  arrive as GRADIENT (soft labels) — hints-as-text nulled twice.
  Never run. (2) WIDTH x DEPTH composition: distribution rows
  teach first-move WIDTH; delta-chained context teaches chain
  DEPTH by analogy — the two compose into one diet design.
  Attribution: Artin (both); house (engine-table mechanization).

- **The billiard-Grover resonance (arXiv:1912.02207, Brown,
  "Playing Pool with |psi>")** (Artin, 2026-07-26): the
  colliding-blocks-compute-pi phenomenon IS Grover's search —
  energy ellipse -> circle, each collision = reflection, two
  reflections = fixed-angle rotation theta = arctan(sqrt(m/M)),
  halt at ORTHOGONALITY, count = pi/theta (Grover's pi/4 sqrt(N),
  same circle). House connections: (1) opposition-is-the-primitive
  gains its geometric reading — both systems TERMINATE at maximal
  opposition (orthogonality); computing-by-rotation halts at the
  anti-aligned state. (2) "repeat the pattern every 2pi" is
  literal in-house: RoPE already rotates continuously; the banked
  RoPE x G5 rung (quarter-turn weights composing with continuous
  rotation) is the experiment and moves UP the queue behind the
  ZX column verdict. (3) Grover-NO fence untouched (no QRAM, no
  static space) — the paper legitimizes rotation-until-
  cancellation as a classical-hostable primitive, exactly the G5
  question. Attribution: Artin (paper + the 2pi instinct); house
  (Grover identification from the paper, RoPE link).

- **AMENDMENT to distribution rows (Artin, 2026-07-26, same day):
  "it's related to streaming/LLMUE — the NNUE-closed-system
  confluence we haven't measured; maybe how harder tasks get
  reinforced."** House mechanization: the engine's brains ARE
  NNUEs emitting move DISTRIBUTIONS (syndrome policy 98.8 top-3,
  magic estimator rho .9) and the crystal has only ever eaten
  PICKS. The unmeasured plane = MOVE-DISTRIBUTION SPACE as the
  teacher-student channel: NNUE = source, crystal = consumer,
  LLMUE streaming = transport. Where it bites: stuck states emit
  zero verified steps => zero gradient (round-2 law, v4
  conversion fail); full engine chains fix it at farm latency;
  distribution rows fix it at MILLISECOND latency (rule-fire +
  policy net on the stuck state = dense soft gradient toward the
  right door, before any demonstration exists). Hard tasks get
  reinforced exactly where the model's distribution is flattest
  and the engine's sharpest. Slot: metabolic v6 candidate food
  channel (stuck-state distribution rows vs stuck-state chains vs
  both, paired). Attribution: Artin (confluence instinct); house
  (the transport identification).

- **The native backend riff** (Artin, 2026-07-26: "writing our own
  backend for certain gradients/optimizers/precision in C++ — or
  just building on top of axiom; bank this"): recorded with the
  house lean — the NNUE-homecoming verdict already split it
  (native inference/verification YES; native training autograd NO
  — the bf16-cliff class is what hand-rolled autograd trips on
  invisibly). The NEW leg worth building when its time comes:
  **axiom as the EXACTNESS backend** — the Ozaki int8-exact
  pipeline + stay-in-RNS machinery as an axiom module the trainer
  calls for the online precise channel (C++ where exactness
  lives, torch where autograd lives). Attribution: Artin (riff);
  house (split + axiom-module framing).

- **The llmopt CLI** (Artin, 2026-07-26): one wrapper runner —
  `llmopt run <script> [args] --host mac|3080` — uniform log
  naming, constant progress feedback, success-only markers,
  local-vs-wsl.sh transport picked by flag; the scripts/ glob
  (100+) migrates gradually into registered subcommands
  (train/gate/bench/adjudicate families first). Corollary
  adopted: the WSL side needs NO scratch dir — wsl.sh gains a
  pyrun verb (ship script, run, stream log back); everything the
  3080 executes is committed or shipped at call time (kills
  untracked WSL drift by construction). Extendability is the
  point (Artin). Slotted: post-index, walked through step by step
  with Artin. MCP-SSH reconsidered after the wrapper exists.

- **"Reverse LLMUE: answer -> all possible steps; a reverse
  quantum LLM"** (Artin, 2026-07-26): lineage assembled — (1) the
  farm IS answer-first (make_integrate draws F, differentiates);
  (2) reverse-engine chains MEASURED to pay (+50% steps, 07-12);
  (3) closest ancestor = his own banked TEMPORAL PINCER (fwd +
  bwd models, oracle at the junction; never run) — reverse-LLMUE
  = the pincer's backward half + distribution rows (emit the
  distribution over VERIFIED predecessors; each checkable in ms
  by forward rule application); (4) the bidirectional-cheat
  fence rides from Future-Work: reverse models on reverse-sampled
  corpora can memorize the GENERATOR — exclude-guarded
  skeleton-split controls mandatory from birth. LLMUE status
  corrected in-bank: half-proven (flips carry capability,
  exchange converts, retention free); the dead half is
  teacherless self-practice — which the reverse model would
  supply at inference speed (a learned millisecond peeler =
  teacher-signal without farm latency). Superposition read:
  in-SEQUENCE step-clouds just measured hostile (traces 37); the
  surviving representations are soft-label distributions and the
  two-model pincer (superposition across models, oracle-collapsed
  at the junction). CHEAPEST CELL banked: reverse-pairs at d256
  (50/50 nxt->cur + forward vs forward-only, matched dose, one
  ~20-min birth) — the dual-direction crystal, entry ticket to
  the pincer. Attribution: Artin (riff, and the original pincer);
  house (lineage + fences + cell).

- **"Efficient activation — maybe inference is wrong; break the
  superposition with something like the magic predictor"**
  (Artin, 2026-07-26): house translation = JUDGE-COLLAPSED
  DECODING, and the measured motivation already exists: the L9
  device-dependence result (same checkpoint 18/24 cuda v 9/24
  MPS) proves frontier decisions are near-tie-superposed and
  currently COLLAPSED BY HARDWARE ROUNDING — an uncontrolled
  apparatus deciding the hardest problems. Design: at decode
  steps with top-2 margin under the near-tie threshold (~0.02,
  the measured class), branch both continuations a few tokens
  and let a cheap judge pick (value head / magic estimator /
  oracle at step boundaries); greedy elsewhere. Starved-judge
  satisfied by construction (fires only at ties = where variance
  lives); the engine-era +15 confidence premium (entropy-gated k)
  is the same pattern one level up; precision-router = the
  arithmetic twin. Fences: hints-nulled doesn't apply (arrives
  as search, not prompt); must be scored at equal token budget
  vs plain greedy AND vs best-of-N (the regret-round-2 lesson:
  branching must beat let-everything-finish economics); primary
  battery = the L9/frontier band where ties concentrate. BANKED
  for slotting; cheapest pilot = the ladder's own d256 cells
  (margins are free at every step). Attribution: Artin (riff);
  house (near-tie identification + design). **RUN 2026-07-28
  (calibration rung 4): NULL BY TIE-SCARCITY — generation-time
  margins <0.02 fire ~once in 30 chains (median margin 8.6);
  judge = greedy 18=18. The economics finding that replaced it:
  greedy captures 90% of wave-8's solves at 12% of tokens
  (greedy-first, wave-on-retry = the adoption candidate). The
  device-dependence paradox resolves as rare ties x 12-ply
  butterfly amplification, not tie-dense generation.**

- **"Oneshot is quick and doesn't wander but it's cheating — what
  if we combined the different data and let the model choose how
  to accept it, like tuning an HCE/NNUE?"** (Artin, 2026-07-26,
  off the pp sweep): two banks — (1) THE MIXTURE DIET: pairs +
  oneshot rows in one crystal. Key licensing fact from today:
  oneshot rows are SAME-DIRECTION (root->answer is a legal
  forward hop — a maximal skip-pair, and skip 54 measured skips
  ~free), so this is NOT the toxic revpairs class; candidate
  outcome = chain robustness AND one-hop conjecture in one
  model, which would let the pincer's conjecture leg live in the
  main crystal instead of a separate oneshot model. Cheap cell:
  pairs+oneshot 50/50 and 80/20 at d256/1P vs pairs 57. (2)
  LEARNED ACCEPTANCE: the model weights its own data channels
  (per-row/per-format gain, HCE-tuning style) — the
  surprise-gating + valuation-routed-metabolism lineage with the
  format ladder as the new evidence base; needs a mechanism that
  arrives as gradient (per-format loss weights learned via
  held-out meta-signal), banked behind the mixture cell's
  verdict. Attribution: Artin (both); house (same-direction
  licensing + the pincer tie-in).

- **THE TOURNAMENT-SIDECAR DOCTRINE (Artin, 2026-07-26, adopted
  as method)**: "more sidecars/tournaments — they force quick
  efficient tests and give our nulls/findings context/proof."
  Booked as standing method: prefer LADDERS of small paired
  cells (the alphabet tournament, the format ladder) over
  isolated experiments, and EVERY tournament ships per-problem
  sidecars by default (pp instrument; step-3 item (d)) — the
  overlap/wandering/mechanism reads cost nothing extra and have
  now corrected two mechanism claims in one day (Muon
  dissociation, revpairs wandering). Nulls with sidecars are
  publishable nulls.

- **"Can every equation be a quantum circuit? What rotational
  aspects are we missing?"** (Artin, 2026-07-26): the honest
  three-zone map banked — (1) naturally-rotational math maps
  (linear algebra, Fourier/QFT, trig-as-Euler, roots of unity,
  oscillators); (2) symbolic term-rewriting does NOT map
  naturally (branching vs collapse); (3) ZX = the maximal honest
  intersection (circuit language AND rewrite grammar — why the
  continent choice was right). THE ROTATIONAL INVENTORY, by
  readiness: Fourier continent (banked 07-23, the math-side
  phase grammar — the CONFIRMATION continent if
  alphabet-follows-domain books on ZX); SHM/mechanics as hidden
  rotation (existing vocab-41 grammar; per-family alphabet read
  ~free); RoPE x G5 (architecture-meets-weights, billiard-Grover
  motivated); G16/Eisenstein/Q9 brackets (gated on tonight).
  Attribution: Artin (ask); house (map + inventory).

- **CLI wrapper v1: BANKED with its shape decided** (2026-07-26
  close): one file (scripts/run.py, ~80 lines), not a framework —
  uniform log naming, live progress, success-only marker,
  --host 3080 via wsl.sh, auto pp-sidecar for gate-family.
  Subcommands graduate from usage (third-time-typed rule), same
  as scratch->scripts. Most of the original pain already solved
  (wsl.sh, gitignore guards, gate_pp); build at the next
  plumbing window, not before.

- **The COMPOSITE MACHINE (Artin, 2026-07-26 night): "maybe we're
  trying too hard to stick everything into an LLM"** — the target
  is a federation of learned components, each optimized for where
  ITS variance lives (small models, math/physics/q-circuits only):
  forward crystal (policy/emission), reverse scorer (B-b,
  score-over-enumerated-moves), MAGIC predictor (quantum-chem
  certification methods, deployed variance-gated — the measured
  law: magic's +1 lives exactly at int L4 where variance lives;
  Liouville-as-Gottesman-Knill), REGRET head (trace-fate AUC .914
  — abandonment gating), LLMUE/exchange as the metabolism.
  Standing rider: EVERY such tool applies at ANY stage —
  inference, training, repair/practice — try spins per stage, not
  once. Attribution: Artin (composite frame + any-stage rider +
  Dijkstra/maps spin + LLMUE-for-pincer); house (mapping onto
  measured components; the meet-is-bidirectional-search
  observation). Measured support already on the books: engine =
  bf + NNUE-h + magic + markov (the composite ALREADY beat every
  monolith at 113-114/120); syndrome-policy DAgger arc (state-
  aware ranking reaches what no global knob can); the balance
  lesson (round-4 pure-L5 regression: composite parts need
  balanced diets, not domain-skewed ones).

- **Pincer MEET = BIDIRECTIONAL BEST-FIRST (the maps spin, made
  precise)** (Artin + house, same exchange): the pincer's meet
  phase is literally bidirectional shortest-path search — and the
  house already measured the frontier discipline (bf beats beam
  113 v 91; asynchrony +12, dedup +21, NNUE-h +10; greedy g=0
  wins because any solution is a proof — path length is not a
  cost). Concrete cells banked: (R8) meet v1 = forward bf
  frontier + backward peel SET as goal states, contact by
  skeleton hash (the transposition table IS the meet detector,
  already built); (R9) MAGIC-gated peeling — a peeled predecessor
  carrying a certified non-elementary integral node is a DEAD
  backward branch (theorem-safe prune, backward edition — new
  deployment of the only zero-false-positive instrument); (R10)
  regret-gated peel abandonment (the AUC-.914 probe as the
  backward model's give-up signal). Fences: equal-total-budget
  economics (regret-round-2 lesson) binds every meet cell.

- **Light/photonics riff (Artin, 2026-07-26 late): "small dense
  input broken into waves" — the wave/click two-layer frame.**
  Honest mapping (plain-language discipline): the field-evolves-
  as-wave / detections-are-discrete structure of light maps onto
  distribution-vs-sampling in our machines — B-b's one-pass
  distribution over the legal set = the wave; k-sampled
  candidates = photon clicks; oracle = detector. MEASURED
  support same-night: R0's k=8 clicks recovered +1 over the
  distribution argmax (the information was in the wave, not the
  clicks) — the amendment-2 no-autoregression decision restated
  as physics. Literal (non-analogical) landing zone = the Fourier
  continent (dense signal -> wave components as an actual
  grammar). InP-photonics angle: linear media do matmuls free —
  the intuition for why one-pass distributions are cheap and
  sequential emission is the expensive particle-like path.
  SCOPE FENCE (house): the analogy stops at interference — phase
  is twice-nulled here (path-integral 07-12, ZX rotation tie
  07-26); wave frame pays at the READOUT layer, not the
  representation layer. Attribution: Artin (frame); house
  (mapping + fence).

- **Grover-fence v2 (Artin, 2026-07-26 late): qubit-mapped
  equations + Grover inference for the reverse LLM?** Honest
  resolution: Grover = sqrt(N) on UNSTRUCTURED search + oracle-
  as-unitary; breaks twice here (unbounded symbolic states have
  no natural register; our search is STRUCTURED — bf+NNUE+magic
  beat sqrt-class bounds by exploiting structure Grover ignores).
  SURVIVES: the parallel-computer intuition is the B-b one-pass
  distribution (all legal moves scored in one matmul — the
  classical residue of query-all-branches, already the
  program's design); magic = our Gottesman-Knill (classically-
  simulable subtheory as certificates). Attribution: Artin
  (riff + the who-cares-if-approximate-if-fast frame); house
  (fence). Companion instinct BANKED AS LIVE: symmetry-in-
  closed-system-learning as a first-class thread (sign symmetry/
  oligarchy, flip census, permutation-augmentation — the
  equivariance rungs, NOT the dead distance rungs).

- **"How can we say precision doesn't pay when our layers were
  never precise?"** (Artin, 2026-07-27 night, during the rounding
  audit). The challenge that scopes the closed precision
  doctrine honestly: its strongest leg (d2 exact-vs-fp64
  bit-identical) had one truly exact arm, but the birth-precision
  leg compared rounded modes against each other, and ALL
  capability gates carry an arithmetic noise floor (near-tie
  flips, device reduction order) — so the doctrine's real claim
  is "no effect above instrument sigma." Banked consequence: if
  relay rung 2b (exact inference mode) lands, the precision
  question becomes re-testable at sub-sigma resolution for the
  cost of ONE paired arm — doctrine stays CLOSED, with that
  named as the sole cheap reopening condition.

- **2026-07-27 (house, from the SR null)**: paired arms need
  SAME-DAY controls — a 10-day-old "booked pair" is not a
  control arm once the tree drifts (SR loss 0.330 v pair 0.3525
  exposed it). Candidate doctrine line if a second incident
  lands.
- **2026-07-27 (house, E4 catch; lineage = Artin's censoring
  audit)**: censored != fact TRANSPORTS across engines — axiom's
  solve_batch deadline expiries fossilize as solved=False the
  same way our value cache did. The doctrine is about label
  STORES, not about our code.
- **2026-07-27 (axiom, prior-cell ack)**: the exported bigram
  prior's mass GENERALIZES over rules it has no history on (the
  tan/pair-u tranche: decided-misses 3->0 under prior-on with
  zero prior rows naming those rules). Consequence banked: prior
  re-export after a rule tranche is not automatically owed —
  test the stale prior first; re-export only if the new-rule
  cells actually degrade.
- **2026-07-27 (Artin, the infinite-precision push)**: three
  rungs banked from "represent 1/3 exactly, not 64-bit-exactly".
  (a) **Rational-snap distillation**: snap a gated winner's
  weights to nearest low-denominator fractions, gate the snap —
  asks "do trained weights want simple exact numbers?" as a
  COMPRESSION question (precision doctrine stays closed; E3 is
  its sole reopening). (b) **RNS optimizer step**: the banked
  stay-in-residue endgame, now named as a training rung — whole
  AdamW step carry-free in residue space, one exit (disagreement
  #2 says the outcome is bit-identical to fp64 masters; the rung
  is a WALL rung, per the exactness-is-speed law). (c) **Root-of-
  unity rotational weights** (from Artin's complex/rotational
  framing): phases constrained to N-th roots of unity are EXACT
  complex numbers, multiplicatively closed, no transcendentals —
  exact rotations without infinite digits; ZX/phase lineage,
  pairs with the cplx thread. House note held: true rationals
  under SGD blow up (bit-length ~exponential in steps) and
  nonlinearities leave every exact number system — (b) and (c)
  are the versions that dodge both walls.
- **2026-07-27 (Artin, the black-hole frame)**: "how can physics
  compress so much matter it becomes a black hole — our stuff ->
  compress (bits/precision)/entropy; black holes are the limit of
  physics." Banked as the program's motivating physics analogy:
  the Bekenstein bound says maximal information in a region
  scales with AREA, not volume — physics' own statement that
  compression has a geometric limit and saturating it changes
  the object's nature. House link: FA Law v2 (intelligence =
  rate verified variance becomes compressed structure) is the
  same quantity as a RATE; the exact-representation rungs
  (rational-snap, roots-of-unity, RNS) ask how densely verified
  structure packs into finite weights. METHODS note: this is
  physics-as-frame (on-charter); no claim until a rung measures
  something.
- **2026-07-27 (Artin, unified-memory ozaki)**: the Mac's unified
  memory as the ozaki pipeline's missing lever — CPU and GPU
  share physical pages zero-copy, so the EXIT (recombination /
  big-int reconstruction, the pipeline's expensive step) can run
  CPU-side on the same pages the GPU's int8 slice products land
  in, no transfer wall. Analogy Artin names: like fp64 masters
  holding information about the fp32 deploy — the wide/exact
  representation lives resident beside the narrow compute copy,
  same pool. Banked rung: ozaki-on-Metal port (simdgroup int8
  slices + CPU big-int recombine, shared pages) — the Mac may be
  the BETTER exact-GEMM machine per dollar because the exit is
  free there. Pairs with [[ozaki stay-in-RNS endgame]] and the
  Metal split-K lineage. [SUPERSEDED IN PART 2026-08-10 — see
  the correction near the file tail: M-series has no integer
  simdgroup MMA (RECEIPT FP32LIMB-R2R3-BUILT, RESULTS 25050);
  the int8-slice half is dead on Mac, the shared-page CPU
  big-int exit survives as the R3 exit.]
- **2026-07-27 (Artin, LoRA-as-precision + house sharpening)**:
  "LoRA brings precision to weights that haven't seen precise
  tuning; that's what the template models were." House
  correction folded in: LoRA is rank-bottlenecked (no per-weight
  precision), but the claim survives sharper — quantization's
  functional damage is tiny + concentrated at near-ties (snap
  anatomy), so a thin precise layer covers it: LoRA brings
  precision WHERE it matters, and it matters almost nowhere
  (QLoRA's measured precedent). The four-rung house ladder now
  reads as one law: templates (coarse stats at birth, time
  lever) -> born-lattice M5 (2.3 bits carry capability) -> snap
  anatomy (precision residual only decides coin flips) -> LoRA
  (thin precise film repairs the rest). BANKED RUNG (snap+
  repair, queued behind the born-rational births): Q=16 snapped
  crystal (26/120 Mac) + few-thousand-param precise repair
  (norms/biases or rank-4) -> if ~parity returns, "precision is
  a small additive budget, not a per-weight property" books.
- **2026-07-27 late (Artin, four riffs off the born-rational
  win)**: (1) **Gaussian-integer weights** — born-rational on
  the complex-FFN: (re, im) each on the exact lattice = weights
  in Z[i]*s/q, "completely whole" complex numbers. Distinct
  from the dead rotation question: the live mechanism is now
  LATTICE REGULARIZATION (+5 measured), and whether it composes
  with complex pairing is unmeasured. Queued behind seed-2
  replication. Reverse-LLMUE tie: an exact scorer crystal slots
  into the pincer's B-b leg via FX-V1 (E2 path already carries
  it). (2) **Exactness ladder honesty**: tonight's win makes
  WEIGHTS exact; gradients/optimizer/activations still round.
  Full "never round anywhere" = weights (done) -> inference
  (integer twin + FX-V1, in flight) -> training step (RNS rung,
  spec'd; d2 says bit-identical outcome, wall lever) ->
  activations (certified tables). Capability-precision stays
  exhausted per doctrine; exactness continues as determinism/
  speed/auditability. (3) **Annealed snap** — multiple snapback
  ops: progressive lattice descent (Q=64 -> repair film -> Q=32
  -> repair -> ...) instead of one-shot; repair budget aimed at
  near-ties (the only failure site the anatomy found). (4)
  **Near-tie repair at DECODE time** — "coinflip at the end of
  the search": the engine already meters this (entropy-gated k,
  +15 premium); the weight-space twin is film-repair trained on
  near-tie-margin loss. Both banked.
- **2026-07-27 late (Artin, diet direction)**: MORE L4/L7
  multi-step rows — the born-rational +5 concentrated at L4 and
  the union tail is thin there; axiom-on-3080-CPU farming
  candidate (GPU busy, CPU idle) — L4-L7 chain farming is the
  named task if Artin boots axiom there.
- **2026-07-27 latest (Artin, arrangement/distance + dimensions)**:
  (1) "If it's the arrangement, it's really distance" — house
  grounding: weight distance = gauge part (permutation/rescale/
  sign, meaningless) + functional part (real); the rotational
  instinct's correct target is the network's SYMMETRY GROUP
  (equivariance thread, perm-augmentation 88.4 > canonical 82.4).
  TESTABLE RUNG: functional distance between seed-1/seed-2
  born-rational twins v their fp32 twins — does lattice training
  shrink gauge slack? (pair exists once s2 lands). (2) Universe-
  dimension riff: model dim != universe dim; the lab analog is
  TERRITORY (W* follows the closed system's rule complexity and
  saturates — 19M twice-confirmed against 45M); "a model reasons
  EXACTLY about a universe iff the universe is closed and the
  arithmetic is exact" = the program in one sentence.
- **2026-07-27 latest (Artin's "the 4s" -> house mechanism)**:
  L4 recurs across the whole record (158 mentions v ~70 for
  neighbors; pincer crater, packing clade, bridge fingerprint,
  wall-timeouts, snap kill, born-rational +5 — all L4) because
  **L4 is where the variance lives**: the first compositional
  level, the susceptibility knee — L3 saturated, L7 out of
  reach, only L4 can swing. Companion to "prediction pays where
  variance lives." DOCTRINE CANDIDATE: read every intervention
  at L4 first (the canary level); L4->L5 frontier migration =
  the cleanest progress metric.
- **2026-07-27 night (house, from the full-RESULTS reread)**: two
  reframes for the exact program. (1) **Born-rational has a named
  ancestor: P2.** The alphabet tournament (07-20) already showed a
  discrete ladder alphabet BEATING fp32 at 19M (P2 3.17-bit: 66 v
  64), and born-vs-rounded premiums are measured lineage (born-T
  60 v ternary-PTQ 42 = +18). RAT_Q=6 (~4-6 bit rational ladder)
  winning +5 is IN-FAMILY, not an anomaly — the new content is
  EXACTNESS riding free on a known regularization effect.
  Seed-2 prior updated: replication plausible per the tournament's
  ladder-beats-fp32-at-19M row. (2) **The gauge-slack cell must be
  designed against the ancestry verdict** (07-26: distance
  measures ANCESTRY — seed pairs read sqrt(2) even after
  Procrustes). The lawful form: fp32 seed-pair distance vs
  rat-Q6 seed-pair distance, same lens (raw + Hungarian), paired;
  signal = rat pair reading BELOW fp32 pair (lattice
  canonicalization above the ancestry floor). Prediction
  (skeptical, on record): both ~sqrt(2), no closure — the lattice
  kills continuous gauge but seed-lottery basin assignment
  dominates distance. Books either way. **MEASURED 2026-07-28:
  NULL exactly as predicted** — fp32 pair 1.4136 v rat pair
  1.4136 (perm 1.2947=1.2947, rot 0.434=0.434); same-seed
  cross-arm 0.388 = STE stays in its fp32 twin's basin. Cell
  CLOSED (RESULTS 2026-07-28).
- **2026-07-28 (house, from the full-corpus reread during the
  calibration program)**: four banks. (1) **S2-dist as the
  judge**: rung-4 judge-collapsed decoding's "cheap judge" slot
  has a measured winner already on disk — scorer_s2_dist.pt
  (best graded Spearman .497, beat length-only) can arbitrate
  near-tie branches at ms cost where the oracle costs more;
  compose before inventing a new judge. (2) **The flips-vs-Q
  curve**: R1's probe at one Q rank-predicts robustness (rho
  .883); the full curve (flips/token vs Q in {4,16,64,256}) is
  a per-crystal calibration FINGERPRINT — candidate standing
  instrument next to the rarity curve. (3) **The policy-
  sharpness law (candidate)**: H_valid over the enumerated valid
  set DESCENDS with capability under pick-training (0.078 muon
  -> 0.057 control, rung-2) and altpairs' flattening cost -6 —
  sharpness may BE part of capability on the forward crystal;
  calibration lives in the scorer (pincer doctrine, third leg).
  (4) **Probe-before-instrument doctrine candidate**: tonight's
  R1 v1/v2 (scaled vs direct snap) is the second
  instrument-mismatch catch in two days (E4 prior-arm was the
  first) — "snap operators are instruments; verify the operator
  matches the ground truth's operator before any battery"
  belongs next to the substrate fence. Attribution: house;
  Artin commissioned the reread.
- **2026-07-28 (Artin): THE EXACT WAVE-MODEL — exact layer
  algebra x the pincer's quantum-distribution frame x LLMUE, one
  program.** Raw ask: "fully-exact layer algebra, slot it in;
  bring in the pincer/quantum wave model — the quantum
  distribution over a math equation; LLMUE included with the
  exact precision." House composition, and it is sharper than
  its parts: (1) the snap anatomy + device-dependence results
  say frontier decisions are near-tie superpositions currently
  COLLAPSED BY HARDWARE ROUNDING (18/24 cuda v 9/24 MPS, same
  checkpoint) — an uncontrolled measurement apparatus; (2)
  FX-V1 exact inference (bit-identical logits by construction)
  makes the collapse DETERMINISTIC — the apparatus becomes
  lawful, near-tie flips stop being device coin-flips and
  become reproducible facts; (3) the pincer's distribution
  readout (engine enumerates the legal set = the superposition;
  crystal supplies amplitudes; oracle collapses) then runs on
  an EXACT wave: amplitudes computed in exact arithmetic,
  readout mass-ordered (R0b's 1.7x), scored by the S2-dist
  scorer; (4) LLMUE is the transport — metabolic sessions
  streaming on the exact substrate, where the RNS optimizer
  rung (d2: outcome-identical to fp64 masters) makes the
  training step deterministic too. END STATE NAMED: a
  closed-system model whose forward, decision collapse, and
  weight update are all exact/deterministic — "the quantum
  distribution over a math equation," measured, replayable,
  device-free. RUNGS (cheap-first): (i) E3 paired arm when
  FX-V1 exact mode lands (the precision doctrine's sole
  reopening — sub-sigma instrument); (ii) d2 endpoint
  verification (queued, 3080 window); (iii) exact-mode
  distribution readout on the S1 frontier battery (does
  exactness change ANY near-tie amplitude read?); (iv) exact
  metabolic session (FX-V1 forward + RNS step) — LLMUE's
  deterministic form. Attribution: Artin (the composition ask);
  house (mechanization + the lawful-apparatus reading).
- **2026-07-28 night (house, from the muon fingerprint anomaly)**:
  robustness and capability measured SEPARABLE (muon: gate 34,
  most snap-robust at Q8 by 3-4x, gate RISES under snap). Bank:
  (1) Muon as a robustness REGULARIZER — a Muon->AdamW anneal or
  Muon-on-late-layers-only arm might buy boundary-distance
  without the validity crater (one d256 cell, low priority);
  (2) the fingerprint's Q8 column as the robustness dial for
  deployment-format choice (which crystal tolerates int2-class
  storage). Attribution: house, overnight autonomy.
- **2026-07-28 morning (Artin, two video riffs)**: (1) **DOUBLE
  PENDULUMS (2swap, islands of stability)** -> THE STABILITY
  ATLAS: their fractal map of initial-condition space (chaotic
  white / stable black, islands of coherence inside chaos) is
  the lab's training-dynamics question drawn as a picture — our
  "initial conditions" = (seed, LR, schedule) and the lab has
  already met the islands piecewise (Z1 seed-1 blind v seed-2
  fluent-wrong = two basins; Muon 10-at-.02 v 34-at-.01; Z1S
  divergence v smooth descent). Instrument: a 2-D birth grid
  (LR x momentum or LR x batch, d64/d256 minutes-class births),
  colored by gate — is capability-over-hyperparams SMOOTH or
  FRACTAL? Neighbor-divergence read rides free (their
  white/black map = our seed-pair gate delta). Also
  retro-frames R4's tie-scarcity verdict: decode chains are
  weakly chaotic (rare forks x long horizons), and the energy
  ladder (low stable / mid chaotic / high patterned) rhymes
  with the LR ladder. Banked as an instrument program.
  (2) **LAPLACE TRANSFORMS (3b1b, poles as detectors)** -> THE
  LAPLACE BRIDGE GRAMMAR: the transform UNIFIES three resident
  grammars — partial fractions (poly continent) <-> poles <->
  exponential components (exp/trig families) <-> cc2
  characteristic roots (the ODE engine's roots ARE pole
  locations). A Laplace grammar (L-table rules, rational-
  function algebra in s, inverse via pf) makes the ODE->algebra
  reduction an explicit, decidable, exact-rational chain — the
  meet-it-where-it-comes-from principle for poles, and every
  verification leg exists (pf oracle, checkodesol, exact Q
  arithmetic). TIMELY: axiom's ODE solving-chain tranche is
  their next session — the Laplace path belongs in their L9
  design NOW (relay draft below). s-plane-detector frame
  (components found where integrals blow up) banked as kin to
  the magic-estimator resonance frame. Attribution: Artin
  (both videos + the instinct); house (mappings).

- **SYMMETRY-AT-BIRTH flip test** (2026-07-28, house, from the
  ladder close): every group read anti-mass at the generic null —
  SGD never picks symmetry spontaneously — yet retrofit costs
  almost nothing. The untested cell: train IN the commutant from
  scratch (project gradients or parameterize by commutant coords)
  at d256 — does birth-under-symmetry match retrofit, beat it, or
  reveal that the warm-epoch heal was riding pre-trained
  structure? Also banked: C16 (params/16, past the current 8x
  rung), packed block-circulant forward (turn IMPLIED compression
  into measured wall-clock), G5-on-rotational-substrate (phase
  alphabet on genuinely complex-linear gates — the G5 null's
  revenge test). Attribution: house (ladder execution + banks);
  Artin (the originating rotational question + GO).

- **OPTIONS/OPTIMAL-STOPPING frame for decode economics**
  (2026-07-28, Artin, from looking at SPY options: "the option
  price is literally a distribution"). Three mappings, methods
  only (no finance engine — charter): (1) Breeden-Litzenberger
  (d2C/dK2 = risk-neutral density) = read the whole implied
  distribution, not the argmax — kin to dist-rows/soft-label
  training. (2) Black-Scholes = heat equation + transform
  methods — same grammar family as the Laplace bridge (banked
  -28). (3) THE SHARP ONE: American pricing = optimal stopping
  (exercise-now vs continuation value) = greedy-vs-escalate in
  decode economics. Ladder-retry's middle rung "losing money" =
  mispriced continuation value. Bankable upgrade: price the
  escalation threshold PER-STATE from model uncertainty (a free
  boundary in state space) instead of a global policy — the
  free-boundary/Snell-envelope literature is the toolbox.
  Attribution: Artin (the riff + the distribution instinct);
  house (mappings).

- **THE COMPRESSION CORNER (bits x sharing)** (2026-07-28, house,
  Artin GO "add the new mixes"): three compression axes measured
  separately (bits: 2-bit sub-bar; dimensions: d256 -5; sharing:
  8x at -6) — are they ORTHOGONAL? Test: rational-snap the
  circulant-8x substrate and the dense substrate at matched Q,
  paired on one device; additive deltas = the exchange law gains
  a product form. Note: elementwise snap PRESERVES circulant
  structure (equal entries snap equally) — the axes cannot
  interfere representationally, only functionally.
- **C8-RETROFIT AT 45M** (2026-07-28, house): does the sharing
  toll transfer across scale? Ties the ladder to the ZX capacity
  null (capability occupies a small structured subspace at 45M
  too, or not). Banked pending a 3080 window.

- **THE MATRYOSHKA CRYSTAL (dynamic inference budget)**
  (2026-07-28 night, Artin: "a model whose weights can shift
  complexes and provide different capabilities for different
  tasks"). One weight tensor, nested commutant tiers: train
  with joint loss CE(W) + CE(P_G(W)) (STE through the
  projection) so W's OWN circulant/complex projection is
  simultaneously a working cheap model — then inference picks
  the tier per query (projection = the budget dial; circulant
  tier = real conv-speed matvec once the packed forward
  exists). Already-measured parts: the tier family
  65/64/61/59 at 1x..1/8 params (separate ckpts today); R2's
  per-level damage profile (L6 immune, L5/L7/L3 fragile) =
  the router's feature; Snell/options escalation = the
  tier-selection policy; oracle verify = the safety net. The
  SHARP question (set by tonight's corner verdict): the
  slack-budget law predicts nested tiers COMPETE for
  redundancy — the joint-loss price vs the separate-ckpt
  family is the measurement. Kin: matryoshka representation
  learning / slimmable nets (literature leg for THEORY if it
  lands). Rung 0 (desk, cheap): gate P_C8(rot_convert_b) and
  P_C8(dense wfloor) as-is — how bad is the free nesting
  before any joint training? Attribution: Artin (the riff);
  house (mapping + rungs).
- **ENERGY-KNOB REFINEMENT (lyapunov)** (2026-07-28 night,
  house, from Artin's "anything incorrect?" review): INIT_SCALE
  multiplies ALL params (norm gains + emb included) —
  compounding ~scale^depth; the x4 "cliff" is smooth monotone
  loss descent from a distant start, i.e. BASIN DISTANCE under
  fixed budget, not measured chaos. Refinement arms banked:
  scale 2-D weights only; equal-effective-budget (train until
  loss parity, then gate); x2 midpoint. Until run, the energy
  read is fenced as budget-confounded.

- **THE MINIMAL CRYSTAL program** (2026-07-29, Artin x house;
  spec docs/superpowers/specs/2026-07-29-minimal-crystal.md).
  Three Artin riffs unified on one scoreboard (params-per-
  solve): (1) WIDTH FLOOR — d64=d256 measured; nobody looked
  below; binary-search the cliff then stack matryoshka/spectral
  /snap compressions on the floor model. (2) PROSTHETIC DIET —
  "can the weights call an external function?": call-spans in
  rows (engine computes at farm time, resolves at decode via
  LLMUE machinery); weights stop storing what the oracle
  computes; the knows-math-stores-no-trivia model. (3) DIET
  DESCENT — "calculate, don't guess": marginal-value curves
  gate-vs-log(rows) per bucket from cheap d64 births; allocate
  farm tranches by derivative; probes (flips/token rho .883,
  mass-on-valid) ride free. Attribution: Artin (all three
  riffs); house (instruments + unification).

- **"The reverse LLM's alphabet: what is its {-1,0,1}? Is an
  infinite vocab good?" (Artin, 2026-07-29)**: house resolution —
  split the two alphabets. (a) WEIGHT alphabet: ternary economics
  are direction-blind (nxt->cur is the same architecture); the
  ternary-from-birth result (63/120 at 1.58 bits, parity at 3ep+)
  should transfer — cheap paired arm banked: reverse-pairs
  ternary vs fp, matched dose. (b) TOKEN alphabet: infinite vocab
  is anti-optimal — embedding rows already dominate small-model
  params (params-per-solve poison), softmax over an unbounded
  set is undefined, and per-token generalization dies. The
  predecessor SPACE is infinite but STRUCTURED: predecessors one
  rule away = {rules} x {sites}, a finite move-vocab — the exact
  B-b one-pass design. So the reverse LLM's {-1,0,1} is the
  finite RULE set; composition supplies the infinity, the vocab
  never does (measured lineage: 47-atom vocab spans the whole
  math diet; capability rides flips-per-token, not vocab size).
  Attribution: Artin (riff); house (resolution + arm).

- **"Is there any way to have infinite density?" (Artin,
  2026-07-29)**: three honest answers. (1) Literal bound: a
  model with P params at b bits holds <= Pb bits — no infinite
  density in the weights themselves (information theory, no
  escape). (2) Topological loophole, already ours: Q is DENSE in
  R — the born-rational/snap results mean finite-description
  weights sit arbitrarily close to any real target; snap-to-
  rationals is "infinite density" in the mathematician's sense
  (countable set, dense everywhere). (3) The real escape =
  DELEGATION (minimal-crystal Leg B): finite crystal + exact
  engine — the weights store the CALL (an index, ~log-size), the
  oracle stores the function (unbounded precision, sympy/axiom
  exact arithmetic). System density is unbounded even though
  crystal density is capped; the model's job collapses from
  warehouse to catalog. Prediction banked: call-span arms should
  show capability-per-param jumping precisely on the buckets
  where the stored function is expensive (gcd chains, modexp).
  Attribution: Artin (riff); house (bounds + mapping).

- **"The model should CHANGE in precision + sharing + width per
  query — question difficult => increase precision/snap back"
  (Artin, 2026-07-29)**: the slack-pool law's constructive
  reading — since bits x sharing x width x tiers draw on ONE
  budget, the compression point should be a runtime DIAL, not a
  birth-time choice. Lineage: matryoshka tier toggle (already a
  2-point dial, zero-cost switch), Snell tier policy (banked:
  route by refraction/difficulty), entropy-adaptive draft length
  (same law in spec-decoding). CHEAPEST CELL banked: difficulty-
  gated tier policy on the d56 matryoshka pair — route level<=4
  rows to the cheap tier, level>=5 to dense, measure solves AND
  effective-params-per-solve v always-dense (the dial's first
  measured win would be compute-free: same tensor, per-row
  projection toggle). Attribution: Artin (riff); house (lineage
  + cell).

- **"Split the bits up — magic bit at inference; have weights
  represent more than what they are, adaptable" (Artin,
  2026-07-29)**: bits-as-portfolio riff. Mappings: (1) snap-to-
  rational IS variable-length description (a weight near p/q
  stores fewer effective bits, freeing budget elsewhere —
  measured knee); (2) the magic-estimator slots (banked) as the
  judge of WHERE precision pays; (3) NF4/AF4-class nonuniform
  codes = bits representing "more than they are" via a learned
  codebook — never tried on our crystals. Honest fence: a bit
  is a bit (information bound stands); what varies is the
  ALLOCATION — adaptive quantization = spending the same bits
  where the function's curvature lives. CELL banked: per-layer
  mixed precision on d56 (snap the low-curvature layers, keep
  attention fp16) v uniform snap at matched total bits.
  Attribution: Artin (riff); house (mappings + fence).

- **"Quantize in different dimensions — width, length, different
  complexes, quantum-state-esque quantization" (Artin,
  2026-07-29 night)**: the axis inventory, mapped to measured
  ground. WIDTH: matryoshka tiers (65/60 free at d256; -6 trade
  at the floor). LENGTH: already measured TWICE — ladder-retry
  decode (retry failures at bigger budget) and clade budget
  recycling (self-pacing PAYS at one pass) — the sequence axis
  is the one place adaptivity has a booked WIN. COMPLEXES: the
  ladder toll curve (2x:-1, 4x:-4, 8x:-6) is the price list for
  sharing-axis quantization; spectrum bands are its frequency
  form. QUANTUM-STATE-ESQUE: honest fence — our nets aren't
  unitary, amplitude-encoding breaks (path-integral + ZX
  rotation nulls); the SURVIVING form is POLAR quantization of
  complex weights (magnitude bits v phase bits split) — and
  rotational snap R2 measured the knee as ANGULAR, so phase
  deserves finer bits than magnitude. CELL banked: polar-split
  snap on the cplx crystal (coarse |w| x fine arg(w) v uniform,
  matched total bits). Attribution: Artin (riff); house
  (mapping + fences + cell).

- **THE ADAPTIVE CRYSTAL (synthesis, Artin x house, 2026-07-29
  night; from the full-RESULTS sweep)**: every component of the
  runtime compression dial already has a measured verdict —
  (1) PRICE CURVES per axis: snap knee Q in (16,24]; ladder
  toll 2/4/8x; tier price = slack; spectrum convexity at floor.
  (2) FREE DIFFICULTY SIGNALS: flips/token predicts snap
  robustness (rho .883); magic estimator = continuous hardness;
  the free router (holographic code) routes without a probe
  model; margin census says undecided mass is uniform.
  (3) RETRY DOCTRINE: ladder-retry + clade recycling = the
  "if difficult, increase budget" policy, measured to pay on
  the length axis. (4) FENCE: entropy-adaptive draft length
  NULLED — adaptivity pays only where variance lives (the
  magic-estimator law); a dial on a flat landscape is pure
  overhead. MISSING PIECE: the CONTROLLER tying (2) to (1) —
  cheapest cell: TIER-RETRY on the d56 matryoshka pair (decode
  cheap tier; on oracle-fail retry dense; desk-only, zero
  training, both tiers already in one tensor). That cell is
  Artin's "predict/snapback/increase precision" verbatim.
  Attribution: Artin (program riff); house (synthesis + cell).

- **"The reverse LLM as the self-learner: it learns/grows and
  feeds/tunes the forward LLM" (Artin, 2026-07-29 night)**: this
  closes a loop the ledger has been circling for three days. The
  lineage, assembled: (1) LLMUE's dead half is TEACHERLESS
  self-practice; the 07-26 bank already named the fix — "a
  learned millisecond peeler = teacher-signal without farm
  latency" — Artin's riff is that idea PROMOTED from tool to
  protagonist: the reverse model is the one that learns, and the
  forward model consumes its output as diet. (2) The 07-29
  backward-TAX verdict (dose-controlled) says direction must
  live in a SEPARATE model — which is exactly the riff's shape:
  two models, asymmetric roles (reverse = farmer/teacher,
  forward = solver), not one dual crystal. (3) The verify
  asymmetry makes it economical: generating a predecessor is
  hard, CHECKING one is a millisecond forward rule application —
  the reverse direction is where generation is cheapest to
  verify (the whole verified-and-distinct doctrine rides free).
  (4) Diet saturation (07-29) says the forward model needs
  HARDER rows, not more — and a reverse model walking backward
  from hard answers is a hardness-targeted farm by construction.
  CHEAPEST CELL — THE FARMER PROBE (banked): train a small
  reverse model (nxt->cur, gen-4), sample predecessors from
  gate-band answers, verify each by forward application, score
  verified-DISTINCT-novel yield per 1000 samples v the sympy
  farm's yield-per-second. High yield => the self-farming loop
  is live (expert iteration with the reverse model as the
  improving teacher); low => the peeler needs its own diet
  first. Quantum framing fence (honest): nothing quantum is
  load-bearing here — the surviving content is the pincer's
  oracle-collapsed two-model superposition, as banked 07-26.
  Attribution: Artin (the promotion + self-learning frame);
  house (lineage + economics + cell).

- **"Black-hole density: why can they be so dense, and how close
  are we?" + "is there just something missing?" (Artin,
  2026-07-29 night, recurring riff — answered with measurement
  this time)**: (1) We are AT our storage bound already: the
  integer-twin result measured 6.65 bits/weight against the
  max-entropy Gaussian capacity log2(sigma*sqrt(2*pi*e)) = 6.755
  — the crystal codes its weights at ~99% of the Shannon
  rate-distortion bound, and the sigma-priced knee (0.5-1.0
  sigma) IS that bound's knee. Density-of-storage is not the
  missing thing. (2) The black-hole lesson properly mapped:
  Bekenstein-Hawking entropy is an AREA law — the densest
  object stores bits on its boundary, and the bulk is gone. Our
  measured analog: the incompressible part of the crystal is
  the ROUTING (attention geometry + heads = the boundary
  structure); the bulk (ffn, bits, rank) is compressible or
  delegable. Delegation (+15, 3-sigma) is exactly the area-law
  trick — store the pointer, let the engine be the bulk. (3)
  What is actually missing, per the ledger: the THREE never-
  varied axes — DEPTH (LAYERS=8 in every single birth this
  month; the horizon property of the sensitivity wall and the
  12-ply gate ceiling both point at depth/iteration, and we
  have zero measurements), RECURRENCE (weight-tied iteration =
  depth without params), and the reverse-farmer curriculum
  (now cell 6). Banked: THE DEPTH LADDER (d56/f224 at layers
  {4,8,12,16}, params-matched variants) as the cheapest probe
  of the untouched axis. Attribution: Artin (riff, x3 now —
  the persistence was warranted); house (bound check + area-law
  mapping + the depth gap flag).

- **2026-07-29 (Artin, evening — four riffs on returning home)**:
  (1) ALPHAEVOLVE-STYLE EVOLUTION LOOP: "the model evolves/
  adapts... try something like that — smaller, more efficient,
  math/physics/q-circuits only." House mapping: we already own
  the two hard parts (merciless oracle+gate evaluator, cheap
  birth cycles); the missing piece is the proposer-mutator loop
  with an elites archive (MAP-elites-style niches). Best first
  target after the Leg C flat: DIET EVOLUTION — mutate diet
  compositions/formats, fitness = gate at fixed budget, niches
  by level-profile; revives the nulled config-estimator at its
  banked revive point. Spec next session. Attribution: Artin
  (riff), house (target selection).
  (2) MULTI-PLY FARMER TREE: "prediction happen for multiple
  steps — every step gives you another tree." House read: k
  reverse plies of B samples = a B^k tree of candidates, each
  forward-verified stepwise; fixes the farmer's memorization
  null because ply-2 starts from expressions the model never
  saw — novelty should COMPOUND with depth. THE farmer revive
  path (with solved-state seeding). Attribution: Artin.
  (3) WHY 56: house answer — instrument quantization (widths
  step by 8 at heads=4/RoPE-even), so the true cliff lives
  anywhere in (48,56]; diet-dependent geometry, not a constant.
  Numerology noted for fun, fenced as non-mechanism: 56 = dim
  of E7's fundamental rep = C(8,3) = 2x28. Falsifier: the
  cliff should MOVE under a diet change. Attribution: Artin
  (question), house (fence).
  (4) HARDNESS DEFINED: "less rows, better quality — but how
  do we define hardness/quality?" House answer, operational:
  hardness(row) = first ladder rung that solves it (escalation
  census = free per-row difficulty meter); quality(row) =
  verified AND distinct AND novel AND determinable. Mining
  rule: keep rows where the cheap tier FAILS but the oracle
  verifies. Doubles as the fitness for riff (1). Attribution:
  Artin (question), house (operationalization).

- **2026-07-29 (Artin, evening — README fake-quant caveat)**:
  "is it stale, or something we tie in?" Not stale — the whole
  snap program is fake-quant by design (grid values, fp
  storage; right instrument for capability pricing, zero bytes
  saved; sole exception: the int16 integer twin). THE TIE-IN
  BANKED: **THE PACKED CRYSTAL** — deploy the minimal crystal
  (layers-4 d56) in a REAL bitpacked format at the sigma-law
  grid (Q~2/sigma per tensor, global budget set by k_c from
  the flips probe — no calibration search needed, unlike
  HQQ/GPTQ), with an MLX/Metal dequant-matmul kernel
  (kernels/metal.py). Honest triple demanded: oracle-gate
  parity, measured bytes (target ~6.65 bits/wt = the measured
  RD bound), measured decode speed (bandwidth-bound regime
  should WIN). Closes theory-to-charter: the month's snap laws
  become an actual inference artifact. Pairs with the
  bitpacked matryoshka (tiers = real memory at decode).
  Attribution: Artin (the catch), house (design).

- **2026-07-29 (Artin, late — number representations riff)**:
  "floating point is awful... why are computers so bad at
  dividing... is there a better representation? bank these."
  House answers banked: (1) cross-device nondeterminism is NOT
  the basic ops (IEEE +,-,x,/ are correctly-rounded and
  bit-exact everywhere) — it is reduction ORDER, FMA fusion,
  and libm transcendentals; the universal fix exists (fixed-
  order/exact accumulation + correctly-rounded libm, cf.
  RLIBM/CORE-MATH) and our Ozaki/RNS arc measured it
  AFFORDABLE — emulation is the lazy equilibrium, not the
  optimum. Packed-crystal C4 tests the strong form (integer
  GEMM hash-identical across Mac/cuda). (2) Division is slow
  because it is the one basic op with no carry-parallel trick
  (Newton-Raphson/SRT iteration, ~10-20x a multiply); the
  arrangements where it is free: LNS (log-domain: divide =
  subtract; adds become the hard op — interesting for
  matmul-dominated NN inference), reciprocal precompute (norms
  do ONE divide per row — precompute), RNS (mul/add carry-free,
  division awful — we already exploit the good half). (3)
  Better representations menu, priced by our own measurements:
  posits/tapered precision (elegant, but sigma-law says NN
  weights need no taper — plain sigma-grid already sits at the
  Shannon bound); MX/block-scaled ints (industry conceding the
  point); Kulisch exact accumulator (kills reduction-order
  nondeterminism at the root — consumed into C4). The
  NN-native system in one line: sigma-scaled ints for weights,
  exact accumulation for sums, log-domain where multiplies
  dominate, and nobody divides in the forward pass.
  Attribution: Artin (riff), house (pricing against ledger).

- **2026-07-29 (Artin, late — the feedback lens riff)**: "the
  model needs quick, verified, optimal feedback... inference
  the model WHILE training, a lens on what weights are being
  trained, how + why — watch it attempt a question while its
  weights are being generated." Banked as **THE TRAINING
  LENS**: a periodic in-birth probe battery (proxy gate n=8
  ~40s; flips/token ~1min; per-layer/per-cell update mass —
  free from the optimizer) logged as a capability TIMELINE.
  Watchables the ledger already names: WHEN the sparse
  critical circuit crystallizes (L1h7-class cells); which
  level solves arrive when (L4-canary law says watch L4
  first); whether EMA's shell-contraction is visible live;
  loss-blind capability holes forming in real time (the
  gate-not-loss law made temporal). Tempo upgrade to the
  exchange loop: feedback within a birth, not between
  generations. Cheap version = checkpoint-hook + probes
  (hours to build); the full "why" (per-question gradient
  attribution) is a second rung. Attribution: Artin (riff),
  house (instrument mapping).

- **2026-07-29 (Artin, late — "anything else we can tie in?" +
  confirm-everything push)**: the R-pass replication battery
  (all C-series arms confirmed, C4 invariance on fresh seeds)
  and the corpus re-sweep it demanded surfaced three dormant
  tie-ins: (1) the Lloyd-Max race (07-25) is the at-capacity
  law measured from the CODEBOOK side (MSE-optimal k-means
  ties naive uniform on outlier-free crystals — nothing to
  fit); (2) "experts are crystals" (07-17) upgraded from
  spectral gauge to CAPACITY axis — house capacity meter
  measured DeepSeek-V3 routed experts at kurtosis 3.07
  (Gaussian) v web-dense 5.3-6.5; (3) "quantize the notches,
  never the axes" (07-17 rank floor) as the paper's framing.
  Product of the riff: **THE CAPACITY METER** (M = span_bits
  - code_entropy at sigma/2, from disk, zero inference) —
  ships the allocator decision rule and arms **C7: sigma-pack
  MoE experts v HQQ** (the web-scale home the C6 fence
  pointed to). Attribution: Artin (push), house (meter design
  + corpus links).

- **2026-07-29 (Artin, close — the black-hole riff)**:
  "compression is intelligence -> the black hole holds the
  universe's secrets... experts are essentially black holes
  within a universe: high entropy, high density." Banked with
  the literal reading attached: the tight part is that BOTH
  are maximum-entropy states under their constraints — a
  black hole is the max-entropy configuration of a region
  (Bekenstein), and an at-capacity crystal/expert is the
  max-entropy weight distribution at fixed sigma (measured:
  code entropy = Gaussian capacity <1%; DeepSeek expert kurt
  3.07). In both cases max-entropy means NO exploitable
  internal structure from outside — why calibrated
  quantizers, like external observers, get nothing (the
  no-hair flavor of C3's tie). The loose part, kept honest:
  black-hole entropy scales with AREA, not volume; no
  holographic claim is made for weights (the 07-17 "code is
  holographic" router result is a separate, measured thing).
  Possible instrument if it ever earns one: an area-law probe
  — does removable information in a crystal scale with a
  boundary measure (interface tensors) rather than bulk
  params? Desk-able against the existing deletion/snap
  ledger. Attribution: Artin (riff), house (literal fence +
  probe sketch). Ties: FA Law v2 (intelligence = rate at
  which verified variance becomes compressed structure) —
  this riff is its statics: a fully-compressed store is
  max-entropy, indistinguishable from noise except through
  the function it computes.

- **2026-07-29 (Artin, close — NNUE compression ask)**: "how
  compressed is the NNUE, realistically? can the same
  techniques apply?" House ran the meter instead of assuming
  (Artin's check-don't-assume rule, same message): house NNUE
  (5,440 params, 21.8 kB fp32) reads overall M 0.82 —
  DEEPEST in the crystal band yet — but with a per-layer
  capacity GRADIENT: input kurt 2.19 (democratic/Gaussian),
  hidden 4.78, readout 7.03 (the oligarchs). The meter
  reproduces the 07-18 oligarchy + 07-26 sign-phase results
  from disk. BANKED: (a) sigma-pack the NNUE (~3.4 kB at ~5
  bits/wt; size is moot at 5k params — the value is the
  end-to-end demo: pack + integer forward = a fully
  deterministic cross-device search heuristic, which MATTERS
  for the record-search reproducibility story); (b) per-layer
  meter profiles as a free phase diagnostic (democracy vs
  oligarchy from disk) — candidate rider on any weight-
  anatomy read; (c) lineage note: chess NNUEs (Stockfish)
  ship int8/int16 quantized — the σ-law question there is
  whether their layers show the same gradient (readout needs
  the calibrated/max-anchored end). Attribution: Artin (ask),
  house (measured answer).

- **2026-07-29 (Artin, close — the cosmology bank)**: white
  holes, big bang, spacetime-as-matrices, wormholes, ER=EPR,
  the expanding universe; "a model is a universe... expanding,
  learning, as light hits us." Banked as frames, with the
  standing literal fence (concepts-as-methods only): the one
  strand with a measurable house edge today is the AREA
  question already sketched in the black-hole bank (does
  removable information scale with interface tensors rather
  than bulk?) — the width-floor arc (19M-era floor; d56 cliff
  while ffn/depth/dose all slack) is WEAK evidence in that
  direction: capability binds at attention INTERFACES, not
  bulk volume. ER=EPR-flavored question banked verbatim for a
  rainy desk day: are the load-bearing (layer,head) cells of
  the 07-29 autopsy CORRELATED across paired crystals born
  from the same diet (entanglement-flavored: same critical
  circuit locations), or seed-random? Answerable with two
  existing checkpoints + the autopsy instrument. No physics
  claims; instruments only. Attribution: Artin (riffs), house
  (fence + the two derivable cells).

- **2026-07-29 (Artin, close — ER=EPR at inference + rotational
  models as experts)**: two riffs, three house-shaped cells.
  (a) SUPERPOSITION IS ALREADY MEASURED HERE: the sampling wave
  IS a superposition of candidate rewrites and the oracle IS the
  measurement — R0b's collapse-ordered readout (1.7x fewer
  oracle calls, 2.06x in the needle regime) and the duo-substrate
  wave (+5 at equal budget: two proposal distributions, one
  collapse) are the frame's existing verdicts. The frame adds a
  question, not a metaphor: is the wave's amplitude CALIBRATED
  (Born-rule-flavored)? — answered too: amplitudes are POLICY
  not landscape (dist readout 07-26), and calibration belongs to
  the scorer. (b) ENTANGLED EXPERTS, the efficiency cell: in a
  real MoE, "a question needing two experts" = correlated
  co-routing. Instrument: co-activation MI matrix between routed
  experts (OLMoE, hooks on router top-k, gate-battery prompts)
  — if high-MI pairs exist, MERGE the pair offline (task-vector
  style W = base + aDa + bDb; house federation says ~zero
  interference below W*) and route to the merged expert with the
  free-router trick (8-neuron committee, 94%). Fewer active
  params per token where routing is predictable — the "bridge
  between always-co-firing experts" made literal. (c) TIED-
  EXPERT LADDER (the rotational link): a commutant/group layer
  is a bundle of experts TIED by symmetry (one parameter set,
  K views — C8 retrofit toll 2x:-1..8x:-6); MoE experts are the
  UNTIED limit (meter: each at capacity, marginal info max).
  The ladder between them — experts sharing a base and differing
  by group action / low-rank deltas — is a compression axis for
  MoE nobody in the house ledger has priced. House cell: birth K
  clade-experts tied by group action v untied v single union
  crystal, matched params. Attribution: Artin (riffs), house
  (frame verdicts + cells).

- **2026-07-30 (Artin, morning — "is there a deeper technique
  to have router + weights conserve their efficient
  attributes?")**: the conservation observation (redundancy
  leaves weights, reappears as co-routing MI) has a CAUSAL
  version. Standard MoE training uses load-balancing aux
  losses that force expert independence — plausibly the very
  thing that pushes redundancy into routing. THE MICRO-MoE
  PROGRAM (banked, house-sized, new territory: the house has
  never birthed a MoE): birth tiny MoEs (2-4 experts,
  d64-class, math diet, one router) under three aux-loss
  regimes — (a) load-balanced (standard), (b) correlation-
  permitted (no balance loss), (c) tied experts (base +
  delta, the tied-expert ladder) — and measure WHERE the
  redundancy lands in each: expert meter M + routing MI +
  gate. Prediction sketch: (a) high MI/low weight redundancy,
  (b) lower MI/higher weight redundancy (mergeable), (c)
  redundancy explicit in the tie at a small gate toll. If
  the total is roughly conserved across regimes, that IS the
  conservation law, made causal with three arms. Second
  bankable: USAGE-TIERED PACKING — routing statistics give
  per-expert temperature for free at deployment; pack cold
  experts to coarse tiers, hot experts fine (matryoshka x
  escalation economics, routing-side stats + weights-side
  bits cooperating — the "link" at the artifact level).
  Attribution: Artin (question), house (arms + predictions).

- **2026-07-30 (Artin, mid-morning — Ozaki link + "stupidest
  K3 thing" + packed-as-expert)**: (a) P3's carrier confirmed
  as the Ozaki error-free-transformation principle aimed at
  determinism instead of fp64 exactness — doctrine's
  "exact = speed/determinism lever" now an artifact; 3080
  fp64 1:64 is WHY int-in-fp32 wins there. (b) BANKED, the
  K3 SINGLE-EXPERT DETERMINISTIC DEMO: one ~66M expert from
  the 2.8T model -> pack -> fixed-point twin -> bit-identical
  Mac/3080 hash ("frontier weights, laptop determinism");
  rider: K3 shipped-code entropy v the 4-bit bound, by
  layer/expert (are Moonshot's parts uniform? desk, one
  shard). (c) BANKED, the PACKED-EXPERT SHELF: house MoE
  assembled from packed specialist crystals + the 8-neuron
  free-router + escalation-as-tier-routing — deployment face
  of the micro-MoE conservation program; every component
  already measured separately. Attribution: Artin (all three
  pushes), house (mappings).

- **2026-07-30 (UMOE-1 verdict rider — the micro-MoE program
  fires its first arm)**: the banked conservation 3-arm RAN
  (first house MoE births). Verdict amends the bank: the
  split law's cause is TOP-1 SPARSE ASSIGNMENT, not the
  balance loss (predictions 1/2/4 nulled cleanly; the
  free arm did not collapse and did not re-correlate).
  Surviving banked follow-ups, sharpened: (a) the (M, MI)
  trade-off hint — capacity meter M rose exactly as MI fell
  across arms (1.26/1.37/1.47 v 288/256/235x), a possible
  conservation law on DIFFERENT axes than pre-registered
  (seed-2 first, then a top-2/soft-routing arm to test "the
  sparsity does it" directly — soft routing should KILL the
  decorrelation if assignment is the mechanism); (b) tied
  experts pay ~nothing at the gate (43 v dense 45) — the
  tied-expert ladder is LIVE as a params-side lever the
  post-hoc N3 null could not touch (tie at BIRTH, not after).
  Attribution: Artin (conservation question), house (arms,
  verdict, soft-routing falsifier design).

- **2026-07-30 (Artin, afternoon — Dunning-Kruger + engineered
  spacetime)**: (a) KNOWING-WHEN-YOU-DON'T-KNOW: does the
  crystal's confidence (logit margin / max-prob) actually
  predict its correctness, per difficulty level — the DK
  question made measurable (overconfidence concentrated
  where skill is lowest?). Artin himself tagged the fence:
  this is a "prediction pays where variance lives" rung —
  a confidence meter only earns keep where outcomes vary.
  Probe = CAL-DK-1 (token-level reliability + per-level
  ECE on existing crystals, desk-cheap). (b) ENGINEERED
  SPACETIME (the GRAV-1 answer inverted): a turbulent
  medium is expensive to control; train the medium LAWFUL
  instead — birth arms with a contractivity penalty
  (per-block perturbation-growth <= 1) and measure (i)
  does the falloff curve flip from amplifying to decaying,
  (ii) what lawfulness costs at the gate, (iii) does
  calibration (rung a!) improve in a lawful medium — the
  two riffs may be one: chaos destroys the meaning of
  confidence. Attribution: Artin (both frames), house
  (operationalization).

- **2026-07-30 (Artin, evening — P v NP as the lab's spine +
  diet-calibration)**: (a) P v NP mapped onto the house:
  FORWARD = P-face (apply one verified step, check cheaply —
  the oracle's side); REVERSE = NP-face (invert, search the
  preimage space — the farmer's side). The lab already
  measured the asymmetry without naming it: farmer probe
  (reverse model inverts in-distribution but memorization-
  dominant, 11 novel), reverse-pairs TAX (no free dual-
  direction crystal), and the entire oracle-verified
  doctrine works BECAUSE verification is cheap while
  generation is search. Banked sharp version: the temporal
  pincer (fwd + reverse models meeting in the middle) IS
  bidirectional search — its value should scale with how
  NP-like the task's inversion is; testable by comparing
  pincer gain across task families of different inversion
  hardness. Quantum-wave rider kept as color, not claim.
  (b) DIET-CALIBRATION HYPOTHESIS (from CAL-DK-1):
  determined rows train honesty (dual of "underdetermined
  rows train hallucination") — dilution arm banked: mix
  unverified/contradictory rows into the diet at 10/30%,
  watch ECE degrade. Attribution: Artin (both), house
  (mappings + arms).
- **2026-07-30 (Artin, standing request)**: periodic
  OLD-NULL REVIVAL SWEEP — older nulls deserve re-checks
  more than newer ones (instruments and scale have moved);
  candidates queue from results_query, not memory.

- **2026-07-30 (Artin, evening — "isn't this Hopfield?")**: YES,
  and partly FORMALLY: (1) gravmoe's usage-attraction c_ij ∝
  EMA(co-assignment) is Hebb's rule at expert granularity
  (Hebb 1949; Hopfield 1982); (2) the router is a modern-
  Hopfield retrieval step — softmax attention = one energy-
  descent update (Ramsauer et al. 2020, arXiv 2008.02217;
  dense associative memories Krotov & Hopfield 2016), so
  top-1 routing = pattern completion into an attractor
  basin; (3) THE UNIFICATION CANDIDATE: Hopfield dynamics
  require contractivity — an amplifying medium (GRAV-1 ctl)
  cannot sustain attractors; GRAV-2's contract arm is the
  regime where associative-memory dynamics become possible.
  "Trainable spacetime" and "Hopfield-izable network" may be
  one property. BANKED RUNG — the BASIN PROBE: P(route
  unchanged) v epsilon-perturbation = per-expert attractor
  basin radius, on existing umoe checkpoints; prediction:
  contract-arm MoE (already-banked GRAV follow-up) shows
  larger/cleaner basins; if confirmed, lawful medium =>
  stable basins => experts as attractors, and "mass" gets
  its correct coordinates (basin depth/radius, not write
  norm). Attribution: Artin (the identification), house
  (formal mapping + probe design).

- **2026-07-30 (Artin, night — HCE/NNUE-experts + the mass hunt
  + NNUE-parity question)**: (a) THE HCE FRAME: Stockfish's
  architecture as MoE blueprint — handcrafted eval (fast
  symbolic features) routing into NNUE specialists inside a
  search tree. House mapping, every piece already measured
  somewhere: HCE features = the corpus-signature machinery
  (template births compute them already); the search tree =
  the escalation ladder (62/120 beats dense 57, booked);
  NNUE specialists = per-class crystals at exact saturation
  (the born-rational nt-chain's 720 certified IS an NNUE-
  class result for its class); packed via the zero-tax
  format. The composite = the packed-expert shelf, upgraded:
  HCE-features router + escalation tiers + specialist
  leaves. Banked as the shelf's v2 spec. (b) THE MASS HUNT
  after EQUIV-1's inequivalence: three quantities (usage,
  robustness, influence) all decoupled at expert grain.
  Candidates banked: mass = ROUTER MARGIN (energy gap;
  near-definitionally the basin radius for a linear router
  — cheap confirm, and CAL-DK says margins are honest);
  mass lives at SUB-expert grain (the head-autopsy sparse
  critical circuit as the true massive bodies — port the
  per-cell census to experts); gravitational mass may be a
  ROUTING-GRAPH property (damage flows through co-routing
  edges, B4's MI graph as the metric field). (c) NNUE-parity
  path: class-by-class saturation (define closed class +
  oracle -> wide generator -> specialist birth to the
  exposure-curve plateau -> pack -> shelf), NOT generalist
  scaling; the nt-chain shows the endpoint exists.
  Attribution: Artin (all three frames), house (mappings).

- **2026-07-30 (Artin, 10:30 PM — combine the scaffolds)**:
  tree x gravmoe (attraction acting along tree edges only —
  gravity with a topology) and channel x tree (the shared
  channel BRANCHING: experts reach each other only along
  specific tree paths — constrained talk). Both are
  "structured scaffold" composites; run AFTER the seed-2
  gatekeeper and AFTER the params-matched control (see
  amendment rider — channel/tree carry more params than lb;
  gravmoe does not, so the collapse-wins result is
  unconfounded, but the 48s need the control before combos
  are interpretable). Attribution: Artin (both combos),
  house (ordering + control demand).

- **2026-07-30 (house, pre-compact brainstorm on the day's
  verdicts)**: (a) SCAFFOLD = CONTINUATION METHOD: birth-
  structured-ship-dense is homotopy continuation (solve the
  relaxed problem, deform to the target; numerical-methods
  lineage) — predicts scaffold value GROWS with problem
  hardness (testable: scaffold delta at L7-heavy diets v
  L3). (b) GRAVMOE = JOINT MODEL SOUP: usage-attraction is
  souping DURING training (Wortsman et al. 2022 soups;
  Izmailov et al. 2018 SWA — both post-hoc; ours is online
  with routing-driven weights + a merge-free endpoint) —
  related-work anchor if the recipe replicates. (c)
  KALUZA-KLEIN COLOR (color only): extra expert dimensions
  during training, compactified at ship — the "extra
  dimensions as optimization aid" frame; keep as naming,
  not claim. (d) MEASUREMENT-COLLAPSE reading of merge:
  train in superposition (4 coupled experts), collapse to
  the mean state at deployment; the deltas ARE the
  uncertainty and they were deletable — ties the black-hole
  arc back to the compression arc. Attribution: house,
  seeded by Artin's day of frames.

## 2026-07-31 ~3 AM — checkpoint forensics (Artin)

The cleanup pass upgraded from janitorial to INSTRUMENT:
during the checkpoints/ triage, (a) sha256 every .pt on
BOTH machines, dedupe exact matches; (b) AUTOPSY the
name-twins that differ — same script, same seed, different
bytes tells us exactly where device/version nondeterminism
entered a rung, and flags rungs worth re-checking; (c)
cross-check surviving checkpoints against their booked
RESULTS scores (re-run gates on a sample) — a score that
doesn't reproduce is an amendment waiting to be found.
Artin's framing: the diffs "give us more specific insight
as to why the different machines ran the rest." Banked
behind the cleanup GO; the manifest tool (ckpt_manifest)
is the enabling first step and ships now. Attribution:
Artin.

## 2026-07-31 ~1 PM — outcome pass on open banks (living-doc check)

- 3B1B roots-of-unity riff (Artin, 07-30): PAID IN FULL —
  became FOURIER-2b/3 and the clock-placement law (THEORY
  row). The "did we exhaust rotational weights?" question
  has its answer: rotation is a diet consequence, not an
  architecture feature; whether clocks form is a
  curriculum decision.
- Combo riffs tree+gravmoe / channel+tree (Artin, 07-30):
  RUN. treegrav made the designed phylogeny (0.9465
  within-pair) and lost the gate (45) — anatomy without
  capability; chantree a_i pinned (4th). Both retire from
  the queue; the negative is the finding.
- Scaffold = continuation/homotopy anchor (house, 07-30):
  WEAKENED by the scaffold retraction — with only gravmoe
  surviving, the homotopy frame scopes to "relaxation
  toward co-used peers", closer to a soup than a
  continuation. Keep as naming only; hardness-scaling
  test unbanked unless gravmoe transports.
- Joint-model-soup anchor (Wortsman/SWA): STRENGTHENED —
  merge-free n=4 is precisely an online, routing-weighted
  soup with a free collapse. This is the related-work
  anchor for the recipe writeup.
- Measurement-collapse reading (house): still apt for
  merge-free; keep as naming.
- Untouched today: HCE/NNUE shelf v2, P/NP pincer-scaling,
  mass candidates, KK color, old-null revival sweep.

## 2026-07-31 night — overshoot-then-prune / the market riff (Artin)

On the ep1 finding (k=5 clock 441/512 at first probe v 276 in the
final crystal): (a) MARKET ANALOGY — rapid redundant formation +
consolidation = bubble-and-correction; SGD as an "unoptimized
Stockfish, awful at pruning/eval during training" — exploration
overshoots, consolidation prunes. Frame candidate if the dynamics
replicate: training = search with a bad evaluation function early.
(b) INSTRUMENT CATCH (adopted immediately): the fixed R^2>0.5
threshold can't distinguish PRUNING (neurons leave) from BLURRING
(periodic mass redistributes below the cutoff) — FOURIER-4a
re-armed with threshold-free metrics (count@0.5, count@0.25,
sum(R^2) total periodic variance) mid-session. (c) "deterministic
in a way?" — same seed same trajectory; cross-seed dynamics
unknown (n=1); a seed-2 formation curve banks as the cheap
follow-up if the overshoot is real. Attribution: Artin.

## 2026-07-31 night — deterministic birth (Artin: "use our winners")

On the transport gap: "so it's the accuracy of the registers?
don't we have the ability to have the numbers be EXACT? the Ozaki
algorithm? we should be using our winners." Composed into the
DETERMINISTIC BIRTH spec (2026-08-01): Ozaki exact GEMMs + P3
nonlinearity tables + fixed-point AdamW = bit-identical training
trajectories cross-device -> the transport question becomes
provable and BIRTH POOLING extends the resolution law's seed
economics to training. House reframe credited in-spec: determinism
<= exactness (identical bits, not exact reals). Cheap R0 first:
TF32-off cuda ladder isolates whether the gap is Ampere's TF32
alone. Attribution: Artin (composition of the Ozaki, P3, and
pooling winners was his call).

- **2026-07-31 (Artin): "is the model's fundamental arithmetic
  perfect? read the actual transcripts"** — two-part riff: (a)
  verify the basics dependency, (b) build a transcript reader for
  the gate. Landed same night: scratch/gate_transcripts.py +
  the L4 read (strategy present, term arithmetic garbled — the
  wound is exact manipulation, not ansatz knowledge). Banked
  follow-up: exact-manipulation diet share for L4.

- **2026-07-31 (Artin): the gravmoe expansion riff** — three
  banked directions: (a) REVERSE GRAVMOE ("white hole") — repel
  co-routed experts instead of attracting (negative lambda /
  anti-Hebbian); (b) NO-TRAINING GRAVMOE — apply the pull
  post-hoc to an already-born checkpoint (inference-time or
  one-shot relaxation), "every win we can use, we use"; (c)
  DYNAMIC PULL — lambda (and instrument constants like k, R^2)
  adapt during training instead of being fixed, extending the
  07-31 dynamic-instrument catch from probes to the training
  force itself. Physics frames welcome as methods (GR/QM
  analogies for the coupling schedule).

- **2026-07-31 (Artin): per-component logging + checkpoint DB** —
  (a) log4j/Minecraft-mod-style logging: every component owns a
  named child logger under one hierarchy. Python's stdlib logging
  IS this pattern (dotted hierarchical names); llmopt.runlog
  already owns the root — the build is instrumenting components
  with get_logger("llmopt.<component>") + per-component level
  control from env, so a run's log shows WHO said WHAT (tonight's
  45-v-37 hunt would have been minutes, not an hour). (b) a real
  checkpoint DATABASE (provenance: sha, birth env, git rev, gate
  numbers, source log) — MANIFEST.jsonl is the seed; tonight's
  provenance bug is the motivating incident. Standing bar for
  promotion into llmopt/ or axiom C++: instrument must be proven
  accurate, honest, consistent first.

- **2026-08-01 (K3, external reader): architectural transfer
  routing vs the bridge law** — the strong-form bridge law
  ("NOTHING transfers without demonstrated shared steps in
  context") is measured only against IN-CONTEXT transfer in the
  closed system (ODE probe VOID-BY-VOCAB; desert v2 coefficient
  (iv)=0). K3's question exposes the untested cell: transfer
  architectures that route through shared LATENT structure —
  a two-grammar crystal with a shared trunk + task-specific
  heads, or cross-attention between two born stacks. Well-posed
  in house instruments: birth-able anatomy, gate-scored, exact
  in the integer battery if built on the mb stack. If shared-
  latent routing ALSO shows zero transfer, the bridge law
  graduates from a diet law to an architecture-independent law;
  if it transfers, the law is scoped to token-level context and
  the latent channel becomes the interesting object. Same
  exchange banked: K3's "path-dependent representation" framing
  for the clock-placement law (the force-the-clock arm is the
  causal test, already banked) and the read that determinability
  law + contamination scars are the lab's most transportable
  findings for open-LM data curation. Attribution: K3 (external
  doc-only reader), relayed by Artin.

- **2026-08-01 (DeepSeek, external paper review): why are
  web-LLMs not at capacity?** — the meter M provokes it: crystals
  at capacity are trivially compressible (1% Gaussian bound),
  web-LLMs need the whole calibrated machinery — is the gap
  undertraining, the loss function, or natural-language-vs-formal
  domains? House-testable form via the Equation flagship (geometry
  encodes FEEDING, not content, kurt 1.9 matched-feeding): feed a
  crystal a web-like token distribution at matched budget and read
  M — does the capacity gap follow the data, the objective, or
  neither? "Probably a different paper" (theirs) but the cell is
  ours to run. Same review, adoptable paper edits (tracked here so
  they survive until the paper is picked back up): (a) narrow the
  abstract's at-capacity claim to the measured d64h8 scope; (b)
  run the meter on a second family we already hold BOTH halves of
  (fp32 champion + 1.58-bit merged_grown = the crown pair; gen-8
  19M as third); (c) outcome-vs-KL footnote — outcome scoring when
  downstream consumes DECISIONS, KL when it consumes
  DISTRIBUTIONS; (d) reference CUDA kernel (unoptimized,
  correctness-only) for the disk-format-is-runtime-format result
  [3080, Artin GO]. Their meta-observation banked verbatim: the
  append-only verdict ledger with falsifications booked alongside
  confirmations "should be normalized" — the discipline may be the
  paper's real contribution. Attribution: DeepSeek (external
  paper-prose reader), relayed by Artin.

- **2026-08-01 (Grok, external paper review): the second edit
  list + the reframing sentence** — CONVERGES with DeepSeek
  (previous entry) on three points, which upgrades them to
  near-certain reviewer objections: outcome-vs-KL (their sharper
  version: the knee-slack claim rests on the house gate alone —
  wants the same packs under a harder battery or held-out
  distribution shift), quantized-release meter rows over-weighted
  (same fp-master ask; the crown-pair cell covers it), abstract
  density/over-claim. NEW and adoptable: (a) M-threshold
  sensitivity sentence ("cut between 1.6-2.3 changes no measured
  decision") + specify whether M is full-model / per-tensor /
  max-over-tensors; (b) related-work preemption — one sentence
  that rotation/incoherence/learned-lattice transforms (QuIP#,
  AQLM etc.) are ALSO expected to collapse to the closed form at
  capacity, measurement left open (else those authors ask why no
  arms); (c) matryoshka negative tied to the fragility axis
  explicitly ("entropy predicts the bits; fragility predicts
  whether the bits are free under outcome scoring"); (d) Sec. 10
  pointer to the sha-pinned tables + exact battery prompts
  (artifact paths; citation policy already requires commit pins);
  (e) STRUCTURAL: promote the Sec. 2 sentence to the abstract —
  "calibration complexity is wasted exactly when training has
  already used the weight budget as a capacity-achieving code" —
  the paper's identity in one line. Their read that Sec. 7
  (decorrelation-to-zero + 300-500x co-routing MI,
  tied-at-birth-free / post-hoc-impossible) is the most citable
  result: noted for emphasis decisions. Attribution: Grok
  (external paper-prose reader), relayed by Artin.

- **2026-08-02 (axiom Opus 5, via Artin): device count is a WEAK
  axis for an integer instrument** — their emphasis argument on
  AMENDMENT P4-DEVICE-SCOPE, and it reads correctly against our own
  mechanism: if integer addition is associative and exact, reduction
  order cannot move a value, so a second DEVICE mainly tests that no
  float crept in. The strong axes are IMPLEMENTATIONS (independent
  code, language, author) and RUNTIMES — which is why axiom's C++ leg
  is worth more than any device count. CONSEQUENCE for the banked
  pinned-sha GPU trajectory leg: reframe it as a NEGATIVE CONTROL
  (it should be boring; a FAILURE would be the finding, localizing an
  integer op whose GPU kernel is not exact) rather than a rung that
  adds strength to the ladder. Not adopted into the ladder's phrasing;
  banked as a presentation/priority claim. Attribution: axiom-side
  Opus 5, relay 2026-08-02-1.

- **2026-08-03 (Artin): simplify weights like math equations; rebuild
  weights as a B-TREE ("algorithmic distillation").** Three banked
  forms, each with its killer: (1) encoding simplification — measured
  near-exhausted for V4 (order-0 3.8646/4 bits; C2-C4 longshots keep
  their killers); (2) algebraic simplification = rung 13 within-expert
  subspace energy (banked, instrument fixed; house prior: rank floor
  says dimensions hurt more than bits); (3) THE TREE FORM THAT COULD
  WORK: tree-structured / hierarchical VQ — weights as paths through
  shared codebook hierarchies, GROUNDED by the pooled-table result
  (one table serves all experts, KL 0.00075 — the shared-distribution
  precondition). Lossy => function-space gate (DeltaKL), killer: if
  32-weight vectors do not cluster below product entropy, the tree
  buys nothing over scalar fp4. One-afternoon cell on cached blobs.
  Literal learn-a-program regeneration banked BEHIND the VQ cell
  (weight-space-reader lineage is the nearest measured relative).
  Attribution: Artin (equation-simplification frame, B-tree form),
  house (killers, grounding).

- **2026-08-03 (Artin, via OpenAI "Ten advances in mathematics"):
  frontier validation of the verifier-first thesis + three technical
  echoes.** All ten claimed results live in hard-verification domains
  — the FA Law v2 frame at frontier scale (their compute, our
  mechanism: verified variance -> compressed structure; racing-arc
  lineage). HOUSE EPISTEMICS APPLIES TO THEM: claims-as-published
  (Connes rigidity disproof, non-sofic group) are proposals until the
  community verifies. Echoes banked to existing cells: (1) sphere
  packing/codes = lineage for the capacity-achieving-code frame (the
  README sentence; 6.65 bits/wt meter); (2) permanent circuit lower
  bounds = the theoretical ceiling context for the B-tree/algorithmic-
  distillation riff (some functions have no small program); (3) CVP
  hardness = the justification line for HEURISTIC VQ in the codebook
  cell (optimal lattice quantization intractable in general). No new
  runs warranted. Attribution: Artin (the kinship observation), house
  (the scoping).

- **2026-08-03 (Artin, via openai/ten-proofs): LEAN CERTIFICATE TIER
  for the axiom judge.** The ten-advances results ship Lean 4
  formalizations — kernel-checked certificates above the claims (with
  the standard coverage caveat: audit what the .lean files actually
  prove vs the prose). The bankable house move: axiom's equivalent()
  emits, for the rational-identity subset of EQUIVALENT verdicts, a
  Lean statement + one-tactic proof (ring / norm_num / field_simp); a
  batch Lean check upgrades those verdicts from judge-blessed to
  KERNEL-CERTIFIED — i.e. "verified AND distinct" applied to the
  VERIFIER layer itself: a sympy/axiom bug surfaces as a failing cert,
  never a silent wrong verdict. Killer: cost per certificate vs the
  11 ms/row oracle; scope fence: only the tactic-closable subset, the
  three-valued contract stays the production judge. Travels to axiom
  as a RELAY ASK (their repo), not an edit. Attribution: Artin (the
  Lean-can-do-this observation, pre-axiom thread), OpenAI repo (the
  existence proof), house (the judge-audit framing).

- **2026-08-03 (Artin): PROMPT-SHAPED EXPERT ITINERARIES — "experts
  delegating to experts, sub-agents within a model."** The riff, in
  three claims worth separating: (1) prompts have a DOMAIN
  DECOMPOSITION (the thrown-ball example: a physics setup that
  becomes an integration problem mid-solution), (2) inference could
  be SEQUENCED optimally per prompt — route through the physics
  circuitry, hand off to the calculus circuitry as the problem
  transforms, (3) experts could DELEGATE — an expert's output
  actively determining which expert serves the next step, like
  sub-agents. What the house already knows that touches this:
  routing is per-TOKEN per-LAYER (48x8 decisions/token on the GT-1
  vehicle), so "the physics expert" is really a per-layer coalition;
  co-routing MI (B4 split law, 300-500x over shuffle null) says
  expert PAIRS do travel together — the coalition structure is real
  and measured; and (3) is architecturally TRUE already in a weak
  sense (layer-N routing depends on the residual stream that
  layer-N-1's experts wrote — experts DO steer downstream routing;
  what's missing is any verification/hand-off semantics). MEASURABLE
  CELLS, cheapest first: (a) domain-conditional demand — arm-0
  instrument on a physics corpus vs mathgen corpus, per-layer
  keep-set Jaccard (moe_router_stats prints exactly this; the GT-1
  vehicle now has a measured heavy tail so the question has room to
  answer); (b) MID-PROMPT ITINERARY SHIFT — Artin's thrown-ball
  prompt class: does the demand distribution measurably rotate
  between the setup tokens and the solve tokens of ONE prompt?
  (first-touch/order instrument already logs the sequence);
  (c) the delegation ladder — train/probe whether expert identity at
  layer L predicts routing at L+k beyond what the token alone
  predicts (the co-routing MI instrument, conditioned). Killers: (a)
  dies if cross-domain Jaccard ~1.0 (router not domain-biased —
  report honestly, it kills the folk picture too); (b) dies if
  within-prompt demand is stationary. No capability claim in any
  cell; all desk-scale on the GT-1 vehicle. Attribution: Artin (the
  delegation frame + the two-expert worked example), house (the
  decomposition into measurable cells).

- **2026-08-03 (Artin): BLACKHOLE MODE — runtime densification of
  co-routed expert coalitions; "the prompt placed in weight space,
  expert gravity deciding the route."** Follow-on to the
  expert-itineraries riff (above). Two separable pieces: (1) the
  LITERALIZATION — the router's gate(x) IS a similarity field (one
  key per expert, logit = dot product, token falls to the strongest
  attractors), so "gravity decides where the experts are" is the
  plain mechanics, and the riff's real content is making that field
  DYNAMIC: expert positions responding to the prompt rather than
  fixed keys. (2) BLACKHOLE MODE — when routing concentrates (low
  routing entropy / a coalition dominating), collapse the co-routed
  coalition into one DENSE block at runtime; sparse when exploring,
  dense when committed. [REFUTED 2026-08-04, entropy-trigger half:
  router entropy RISES monotonically in 100% of prompts, confirmed
  decode-only after the prefix-artifact amendment (GT1-ITINERARY-0 +
  AMENDMENT GT1-TRAJ-CORR) — the field dilates, never commits; any
  trigger must be position- or miss-based, never entropy-level.] Measured constraints that fence it: LAM-MERGE
  (RESULTS:14090) — averaging DIVERSE experts is destructive (-12,
  2.4 sigma at lambda 0.1) and merge-free only above the collapse
  threshold, so runtime densification is legal ONLY for
  born-compatible coalitions (the GRAVMOE birth regime, which is the
  same thread from the birth side); coarse-agreement rider
  (RESULTS:14596) — collapse is all-or-nothing, no graded merge; N3 +
  2B — the coalition members share no weight-space structure, so the
  dense block must be built functionally (co-routed activation
  merging), never by weight averaging outside the born-compatible
  case. Measurable cell, desk-scale on GT-1: routing-entropy
  trajectory per prompt (does the field measurably "commit" —
  entropy dropping as the solve progresses?) — free from the arm-0
  first-touch/demand logs; if commitment exists, the
  packing/densification question inherits it. Attribution: Artin
  (blackhole/gravity frame, prompt-in-weight-space inversion), house
  (literalization + fences from the merge ledger).

- **2026-08-03 (Artin): FUNCTION-SPACE VERIFIER WEIGHTS — rigid,
  verified experts layered over the model; routing gravity shifts
  on DISAGREEMENT; "one sweep, find the disagreements and repair."**
  The riff: some weights are special because their function is
  CERTIFIED (the exact-twin / intbirth lineage: weights whose
  arithmetic behavior is bit-verified against an oracle). Embed such
  rigid experts alongside learned ones; when the learned pathway and
  the certified pathway disagree on a computation both claim to do,
  the disagreement is a LOCAL, ADDRESSABLE error signal — and the
  router's pull toward the certified expert (or a repair of the
  learned one) is the correction. Three house threads this fuses:
  (1) certified-function weights exist (exact_twin_d56_q16,
  integer-battery bit-identity doctrine); (2) "verified AND distinct
  at every learning layer" — this pushes verification INSIDE the
  forward pass instead of at the output; (3) the gate law/syndrome
  frame — disagreement-as-syndrome, repair routed to where the
  syndrome fires. Nearest published kin: process supervision /
  tool-use grounding, but IN-WEIGHT rather than out-of-band.
  MEASURABLE LADDER, smallest first: (a) DISAGREEMENT INSTRUMENT —
  on a mathnative model with a certified twin of one operation
  (addition circuits), measure per-token divergence between the
  learned computation and the certified twin on held-out arithmetic;
  does divergence PREDICT wrong answers (the syndrome property)?
  (b) ROUTING RESPONSE — add the certified expert as a routable
  alternative; does a trained router learn to prefer it exactly
  where the learned pathway's syndrome fires? (c) REPAIR — distill
  the certified expert's behavior back into the learned weights
  ONLY at disagreement sites ("one sweep, find and repair") — gate
  before/after, the repair must not damage non-arithmetic solves.
  Fences: (a) is desk-scale on existing checkpoints; (b)/(c) are
  training cells needing pre-reg; repair-by-distillation must be
  scored by ORACLE (function MSE / gate), never weight distance
  (house law). Killer for the whole ladder: if divergence does not
  predict error at (a), the syndrome property fails and rigid
  experts are just slow tools. Attribution: Artin (verifier-expert
  frame, disagreement->gravity-shift, one-sweep repair), house
  (syndrome framing + the certified-twin substrate + fences).

- **2026-08-03 (Artin, late): HOUSE-BORN EXPERTS + the
  blackhole/b-tree lattice + routing-entropy as the "expert
  license."** Cluster of three: (1) BUILD EXPERTS OURSELVES —
  internet-trained experts are open-system; house experts born on
  oracle-verified math (the gravmoe/hebbian umoe line — which
  already exists and is the measured substrate) are closed-system;
  the GT-1 ridge (peak capability at ~50% keep) suggests even
  foreign MoEs contain a better sub-model — a HOUSE-BORN one could
  be built AT the ridge rather than pruned to it. (2) THE LATTICE —
  experts as dense cores ("blackholes") with b-tree branches
  connecting expert to expert; routing walks the tree between
  cores; ties to the banked hierarchical-VQ form (codebook trees)
  and the keep% ladder (prune/eval by FUNCTION, per house law).
  (3) THE ENTROPY LICENSE — an expert whose ROUTING entropy is too
  low (near-always or near-never selected; predictable from
  context) carries no decision information and "should become
  weights": fold it into the dense path; only high-entropy
  components earn indirection. Measurable NOW from arm-0 demand
  logs: per-expert selection entropy over the gate corpus ->
  candidates for folding; killer: if folding the lowest-entropy
  quartile at 50%-keep changes the gate (beyond sigma), the license
  criterion is wrong. Attribution: Artin (all three frames + the
  fold-to-dense criterion), house (gravmoe substrate + fences).

- **2026-08-03 (Artin, rider on house-born experts): EMERGENT
  EXPERTS — "shouldn't they just form, naturally?" Dense training
  concentrates function until a region IS an expert.** The claim:
  without any imposed MoE architecture, training pressure alone
  differentiates the weight mass into functional experts. Published
  kin: FFN activation sparsity + MoEfication (post-hoc co-activation
  clustering of dense FFNs recovers a working MoE — the experts were
  latent). House kin: Fourier v4a instant clock recruitment
  (spontaneous role specialization, no encouragement); LAM-MERGE
  lambda 0.1 (minimal pull already yields diverse/unmergeable
  experts — differentiation is the default). The ladder this
  completes: hard MoE (imposed walls) -> gravmoe (soft pull at
  birth) -> DENSE-LATENT (no walls; do experts form anyway?).
  MEASURABLE CELL, desk-scale on owned checkpoints: cluster FFN
  neurons of a dense mathnative model (110M-400M line) by
  co-activation on the math corpus; wrap the GT-1 keep%-ladder
  instrument around the clusters; REGISTERED KILLER: random
  clusters matching co-activation clusters under keep%-pruning =
  no emergent structure, riff dies. Correction that travels: the
  "collapse" is in FUNCTION space (co-activation), not weight
  values — N3/2B say the weights stay statistically bland even
  when function localizes. Attribution: Artin (natural-formation
  claim + entropy-densification frame), house (MoEfication link,
  ladder placement, killer).

- **2026-08-04 (house, reviewer-proposed + Fable-verified): RIDER on
  the five 2026-08-03 banks — missing citations, one instrument
  correction, two premise fences.** (1) INSTRUMENT CORRECTION: the
  "free from arm-0 logs" claims in blackhole-mode (entropy
  trajectory) and itineraries cell (b) are WRONG — moe_gt1.py pools
  counts/first-touch across all 120 prompts + probe (one monotone
  counter, no per-prompt reset); those cells need a per-prompt
  instrument edit + a re-run (~5 min). The ONLY genuinely-free cell
  is the entropy license (Bernoulli selection entropy from pooled
  counts). (2) CITATIONS OWED: itineraries inherits THE DUAL VERDICT
  (RESULTS:3552 — monolith beat the two-expert blackboard at house
  scale; shared substrate fed physics +6.8 free) and collides with
  the BLACKBOARD PROTOCOL bank (RIFF:213); blackhole-mode inherits
  B4's merge read (RESULTS:11197 — averaging the top-MI pair on a
  real MoE cost ppl 75.74->79.10; co-firing does not imply mergeable)
  and GRAV-0T (RESULTS:13382 — post-hoc pull destructive both
  directions), so its LEGAL form is prefetch/tiering (the split
  law's own named lever), not runtime merging; verifier-weights
  inherits the syndrome revive-if economics (RIFF:44 — a ns-cost
  certified twin is on the wrong side of "prediction pays where the
  oracle is expensive"; cell (b) must answer why not just USE the
  twin); the lattice form of house-born experts is TREE-1, already
  run and retracted-to-noise with a params confound (RESULTS:12296,
  12320) — blocked absent a params-matched design; emergent-experts
  re-aims the graph-anatomy bank (RIFF:56) from weight space to
  function space and gains a partial positive kin (the 64-cell
  autopsy: essentiality is cell-sparse, ~13/64 load-bearing).
  (3) PREMISE FENCES: "the GT-1 ridge" fence DISCHARGED 2026-08-04: R4/R5 landed the
  paired-seed pass (crest at 45.3% beats paired full 6/6 seeds,
  pooled +14.7, R5 fully registered) — house-born-at-the-ridge is
  UNBLOCKED; entropy-license framing upgraded: it is a TEST
  of THEORY Row 28 ("prediction pays only where variance lives")
  applied to routing, not a new law. Attribution: reviewer sweep
  (proposals), Fable (verification + booking).

- **2026-08-04 (house, speculative-survey adoptions; reviewer
  seat proposals, Fable-verified citations): FOUR GROUNDED TRANSFERS
  banked, four killed.** BANKED: (1) ROUTING-BITS LEDGER — a
  residency template is a context model in the coding sense; code
  the decode-only routing stream under {uniform, per-layer marginal,
  LOO decode template, previous-layer-conditional} and the bits/slot
  deltas price what the router actually DECIDES (replaces the
  entropy license's arbitrary 0.15-bit threshold with a
  decision-theoretic currency; kin: V4-MERGED-LATTICE joint-coding
  +0.307 bits). Killers: template buys <0.1 bits over marginal
  (timetable story false), or prev-layer ≈ template (co-routing adds
  nothing here). Desk, zero GPU; instrument gap: traj rows lack
  router SCORES (one-line edit if a margin-conditioned coder is
  wanted). (2) PREFETCH VIA THE PLACE-1 HARNESS — scratch/
  place1_gravity.py (co-routing predictor + LRU cache sim) re-pointed
  at GT-1 decode traj rows IS the bandwidth-matched control
  GT1-TRAJ-CORR owes phase-prefetch; also transports the
  spec-decode cost-ratio law (accept/reject <-> resident/miss) into
  a regime that finally satisfies it (miss = multi-MB fetch,
  predict = dot product; F2c measured miss-driven throughput).
  Killer: conditional prefetch fails PLACE-1's +5-point bar on this
  vehicle. (3) CHURN JUDGE (Row 28 instance) — full and crest
  solvers disagree on ~40% of the gate; predict per-problem which
  solves it (candidate feature: out-of-keep-set miss mass). AUC bar
  0.6; MUST register its SPEND (arm-selection policy) up front or it
  repeats the regret-probe null (real probe, nothing to buy).
  (4) CREST AS DATA GENERATOR, not logit teacher — the dissociation
  fence (ARM2 P4) makes logit-KD from a degenerate distribution a
  known-shaped mistake; the right cell is sequence generation ->
  oracle filter -> verified-AND-distinct yield vs the full model at
  matched budget (drops into the existing diet pipeline). KILLED,
  with reasons on record: Fourier-clock -> coalition mapping (no
  shared observable; correlation-anatomy class), ZX/phase-teleport
  -> routing graphs ("both are graphs" is not a mapping),
  entropy-triggered anything (field dilates — see blackhole
  refutation above), standalone syndrome-frame for misses (merged
  into the churn judge, where its revive-if economics are actually
  satisfied). Attribution: reviewer seat (mappings), house
  (verification + banking); leverage-survey items (random-mask
  control -> PRE-REG R6, pruning-cliff debt discharge, V4 residency
  re-pin, crest-teacher, weight-reader labels) tracked in RESULTS/
  queue rather than here.

## 2026-08-04 (evening) — post-GT-2 survey banks + the exclusion arc

- **Exclusion-mass hypothesis** (post-review survey seat, Opus
  4.5-self-reported; Fable desk-verified the confound it implied):
  "crest gain requires excluded demand mass, not coverage" —
  BANKED AND SAME-DAY KILLED as a sufficient condition by
  D4-PHYS-B (matched ~10% exclusion on physics: -59 pooled, 3/3).
  Survives only as descriptive within-math structure (the
  exclusion table, GT2-EXCLUSION item 2). The arc it produced is
  the keeper: churn check (mask not inert) -> demand-log confound
  (pooled vs decode) -> matched-instrument dissociation (recall
  identical, sign opposite). Attribution: survey seat (tension +
  P1/P3), house (confound discovery, rescue arm, kill).
- **Coverage-dilution keep-rule** (survey seat P4): RULE=dilute —
  own-domain keep-set diluted toward random at fixed expert count,
  giving a coverage-controlled curve in ONE gate/corpus/baseline.
  BANKED as the legal instrument for any future coverage-curve
  claim (GT2-REVIEW killed the mixed-baseline table). Fence: draws
  seeded and reported (random keep-sets do not nest, R6).
- **Ledger-gap repair** (survey seat flagged; house confirms):
  two morning speculative-survey items were discussed but never
  banked — (i) CAPACITY-METER AS KEEP-SET PROPOSER (per-expert
  M = span_bits - code_entropy from disk, correlate with demand
  ranking; killer: Spearman < 0.2 or an M-chosen 45.3% mask near
  0/120) and (ii) CREST-AS-DRAFT for spec decoding (crest = the
  cheap resident draft the entropy-adaptive null priced; nearly-
  free pre-check: greedy token agreement crest-vs-full from
  existing logs; churn ~40% says acceptance may kill it first).
  Both banked now with the morning seat's attribution. (ii) is
  math-domain-only post-D4 (the draft must share the target's
  domain crest).
- **Crest-as-data-generator: NARROWED to math-domain** (transport
  nulled twice, D4-PHYS and D4-PHYS-B; RESULTS 2026-08-04). The
  two-domain gate/probe dissociation STRENGTHENS the
  sequence-form-only fence (text quality is not implied by gate
  scores on either domain).

- **The abstraction hierarchy / "simpler than math"** (Artin,
  2026-08-04 ~23:50): domains as CLASSES over expert coalitions —
  physics extends math; is there a base class below math, "the
  pure logical ability to understand an equation"? Desk check the
  same night CONFIRMED the structure (OBSERVATION GT2-CORE-0:
  37/58-per-layer three-domain core vs 11.9 null; 92% containment
  of math&code in phys — near-clean class hierarchy). The base-
  class QUESTION stays open with registered discriminators: proofs
  arm-0 (logic-pure), prose arm-0 (the language-substrate
  control), core-only mask (necessity-vs-sufficiency). Attribution:
  Artin (hierarchy + base-class question), house (desk
  confirmation + discriminators).

- **Abstraction-hierarchy riff: CONFIRMED same-night** (see
  2026-08-04 entry above; VERDICT MOE-GT-3). The base class exists
  and is SYMBOLIC (proofs contain 90% of the core, prose 25%);
  language is a SIBLING branch, not the base; the core alone is
  dead (0/120) — necessary, not sufficient. Attribution: Artin
  (question + class frame), house (discriminators + runs). The
  open residue: does the verbal branch have its own core (second
  verbal corpus, banked)?

- **Abstraction-hierarchy riff, residue CLOSED: the hierarchy is
  SYMMETRIC** (VERDICT MOE-GT-4, 2026-08-05). The verbal branch has
  its OWN base class — prose&dialog core 48.5/58 vs 11.9 null,
  LARGER than the symbolic core (37.1), and nearly disjoint from it
  (Jaccard 0.095). Two sibling branches, each with a base; the
  router's deepest split is now measured on both sides. Attribution:
  Artin (hierarchy frame), house (second-corpus design + runs). New
  residue: necessary-not-sufficient is untested on the verbal side
  (no verbal gate exists — banked, not queued).

- **The sorted/dynamic model** (Artin, 2026-08-05, on the GT-4
  verdict): "we LITERALLY are pruning/removing experts — can't we
  SORT the model so it's optimally arranged, and make it dynamic?"
  House framing: pure index permutation is a functional no-op
  (permutation invariance — the never-score-by-weight-distance
  doctrine's own fact), but PHYSICAL arrangement is an operational
  lever: (a) store each branch's base-class experts CONTIGUOUSLY so
  a branch load is one sequential read and keep-set gathers become
  slices; (b) branch-conditional RESIDENCY — classify the branch
  early, page in only that population (~50% residency, which
  MOE-GT-1 measured as BETTER than full density, 78/120 vs 64/120);
  (c) the two near-disjoint branch cores (48.5 + 37.1 of 128) are
  the natural partition. Risks/fences: residency cliff at 25-33%
  (GT-1); mid-stream branch switches; router is per-token — masks
  already fence this (KEEPSET knob). Attribution: Artin (sorting +
  dynamic question), house (permutation-invariance split +
  residency/paging framing). Banked, not queued.

- **Global-workspace / J-space triple riff** (Artin, 2026-08-05, on
  the GT-5 resurrection + Anthropic's global-workspace article):
  (1) Are we rearranging how tokens flow through experts — is the
  branch structure a routing-level analogue of the J-space
  broadcast hub? House note: the article's J-space is an
  ACTIVATION-space object (Jacobian lens, dense read/write
  connectivity ~100x ordinary patterns) in a dense model; our
  coalitions are ROUTING-space objects in an MoE — different
  measurement planes, but the GT-5 resurrection (dead symbolic core
  + verbal experts -> 55/120) is exactly the shape a workspace
  story predicts: a shared broadcast population that domain
  circuits need to act through. (2) Do experts have their own
  J-space — per-expert workspace patterns, or a workspace that
  lives IN the branch bases? Measurable house-side as expert
  co-activation graph centrality (which experts read/write with the
  most partners across corpora) — the branch bases are the
  candidate hubs. (3) VERBAL-FALLBACK hypothesis: is the model
  falling back to verbal understanding of math ("explain the
  integral in plain English -> easier to break down")? GT-5's
  random-fill control discriminates: if only VERBAL fill (not
  random fill) resurrects the core, the verbal base is load-bearing
  for math specifically. (4) Sort experts BY CLASS as the enabling
  step for the dynamic model (extends the sorted/dynamic-model riff
  above): class labels now exist per layer (symbolic-base /
  verbal-base / extensions / rest), so the physical reorder is
  specifiable today. Attribution: Artin (all four questions),
  house (measurement-plane distinction + discriminators).

- **The VERBAL TAX thesis** (Artin, 2026-08-05, sharpening the
  verbal-fallback riff): a web-trained MoE routing math through
  verbal machinery is EXACTLY the inefficiency the math-native
  micro-model program was built to avoid — the 30B's possible
  verbal fallback and the house crystals are the two sides of one
  claim. If the random-fill control confirms verbal fill (not
  random fill) resurrects the dead math core, we will have measured
  the pathology inside the deployed model class that the
  math-native thesis predicts, turning "from-scratch math-native
  beats pretrained" from a benchmark result into a mechanism story:
  the pretrained model pays a routing/representation tax for having
  learned math THROUGH language. Attribution: Artin (thesis +
  program connection), house (control design). Reviewer evidence
  sweep dispatched same-day.

- **Verbal-fallback + verbal-tax riffs: ADJUDICATED same-day**
  (VERDICTS GT-VERBAL-SHARE + MOE-GT-5c, 2026-08-05). Both routing-
  level readouts came back negative: free-routing math uses
  verbal-only experts for 2.7% of decode demand, and the causal
  control showed random fill resurrects the dead core as well as
  verbal fill (51/36/48 vs 55 at matched size/recall) — the
  resurrection is a RECALL SHOULDER, not verbal machinery. The
  verbal-tax thesis narrows to the training-history/representation
  layer, where the corpus evidence already lives ("the internet was
  drag", rare-domain math neurons). Weak unresolved residue: union
  sits at/above the top of the random band at lower recall (+10
  mean, inside draw spread). Attribution: Artin (hypothesis), house
  (both discriminators). The sorted/dynamic-model riff INHERITS a
  design simplification: above the resident core, paging schedules
  are recall-coverage arithmetic, not expert-class identity.

- **COUNTER-NOTE to the 'ADJUDICATED same-day' entry above**
  (VERDICT MOE-GT-6 + AMENDMENT MOE-GT-5c-SCOPE, same night): the
  adjudication was premature — GT-5c's random pool was ~45%
  verbal-branch, so its control never tested exclusion. The clean
  arm collapses (0/120 and 7/120 at matched 0.72 recall vs 16-55
  verbal-containing): Artin's verbal-fallback hypothesis WINS the
  causal round in its narrowed form — the fallback machinery math
  emission is rebuilt through, once symbolic extensions are masked,
  lives in the verbal branch. The verbal-TAX thesis at the routing
  layer stays narrowed by GT-VERBAL-SHARE (2.7% free-routing
  demand): fallback is a masked-regime path, not normal operation.
  The sorted/dynamic-model inheritance is WITHDRAWN: paging is not
  recall-coverage arithmetic. Original text stays; this note names
  its refutation. Attribution: Artin (hypothesis), reviewer audit
  F7 (the pool flag), house (the exclusion arm).

- **CORRECTION to the 2026-08-05 "hierarchy is SYMMETRIC" entry
  above** (per AMENDMENT MOE-GT-4-REVIEW, same day): that entry's
  "48.5/58 vs 11.9 null" used the 3-way null — the correct 2-way
  independence null is 26.3 (1.84x chance) — and "LARGER than the
  symbolic core (37.1)" was an arity mismatch, RETRACTED: matched
  2-way |math&phys| = 52.1 > verbal 48.5; verbal coherence is
  ordinary. The branch structure and near-disjointness (0.095)
  stand. Never silently delete: the original text stays, this note
  names its refutation.

- **Tiered oracle: numeric pre-screen -> fork-boxed CAS -> kernel
  certificate** (house, 2026-08-05, out of the GT-6 oracle hang +
  Artin's "why are we still using sympy"): sympy stays the referee
  because its breadth on arbitrary model text is unmatched (symengine
  CLOSED; axiom C++ is sounder but tier-narrow), and the unbounded
  tail is engine-independent (Richardson: zero-equivalence over
  transcendentals is undecidable — swapping engines moves the
  pathological set, never removes it). The hardening: (1) NUMERIC
  PRE-SCREEN — evaluate lhs-rhs at a few random points before any
  simplify; pathological completions are almost never EQUAL, just
  expensive to prove unequal, so the 3-hour hang class becomes a
  microsecond reject (fence: numeric agreement is one-sided — pass
  still requires the symbolic/kernel tier; branch cuts and poles
  need care in point sampling); (2) fork-boxed CAS for routine
  confirmations (shipped, MOE-GT-6-ORACLE-BOX); (3) Lean kernel
  certificates where soundness is the claim (Mac-local, 382 ms/cert,
  0 false raw equalities in the 1000-sample) — the proofs-gate spec
  item is the first gate scored at tier 3. Attribution: Artin
  (question), house (tier design). Banked; tier-1 pre-screen is a
  ~20-line change to check_isolated when next touched.

- **Expand-then-simplify as a text operation** (Artin, 2026-08-05):
  math simplification is NON-MONOTONE (you expand to cancel; the
  path to smaller goes through bigger) while text summarization is
  treated as monotone compression — but what if it shouldn't be?
  Proposal: expand the text into more detail FIRST, then compress,
  vs compress directly; does the answer improve? House grounding:
  the engine's own wins came from complexity-raising moves (the
  linear-basis/ansatz arm moved the L4 ceiling by proposing bigger
  templates then collapsing them), and CoT is the degenerate
  always-on version of expansion-before-answer. Charter-clean test
  design: run it on PROOF text — expand terse derivations (Lean
  corpus, kernel-verifiable) into detailed steps, then compress,
  vs direct; score with the oracle. The explicit intermediate
  artifact is auditable in a way CoT is not. Related: the
  temporal-pincer bank is the same shape run backward from the
  answer. Attribution: Artin (the asymmetry + the test), house
  (proof-text framing + oracle scoring). Banked, not queued.

- **The TENET battery (temporal-pincer super-spec)** (Artin,
  2026-08-05 late: "I really want the temporal pincer to work and
  be BETTER than not having the reverse LLM... tie in the experts
  FINDINGS... 'gravmoe experts + temporal-pincer
  solving/converging wherever in the model'"): promote the
  temporal-pincer bank from a single queued experiment to a HOUSE
  BATTERY — a fixed multi-arm/rung suite in the GLOSSARY sense
  (battery + rung entries added same-day). Building blocks
  restricted to TESTED+VERIFIED wins only (multi-seed, cross-
  device/implementation where the claim needs it): deterministic
  birth (bit-identical cross-lab training = a reverse twin is
  birthable EXACTLY, and cheaply — 1.5 s/1000 steps C++),
  math-native diets + oracle gates (the scoring instruments), and
  the MoE branch/coalition findings (branch bases, recall
  shoulder, crest-as-different-solver). Candidate rungs, sketched
  for the spec pass, NOT registered: R0 birth a reversed-token
  twin on a certified diet (does reverse capability exist at all —
  gate it); R1 pincer search on derivations (forward from problem,
  backward from answer, meet-in-the-middle; win condition =
  solves-or-wall-clock BEATS forward-only at matched budget, the
  "better than not having it" bar, verified-AND-distinct fence);
  R2 convergence anatomy (WHERE do the two directions meet — step
  space first; if an MoE vehicle, do forward/backward route
  through mirrored or shared expert populations — the gravmoe
  tie-in); R3 pincer-as-miner (meet-in-the-middle rows as diet).
  Spec discipline: full spec AFTER consolidation + review,
  sub-agent seats authorized by Artin for the spec pass; every
  rung pre-registered with its null; the reformulation-ensemble
  and expand-then-simplify banks are adjacent (same non-monotone
  family) and the spec should say whether they fold in or stay
  separate. Attribution: Artin (battery framing + convergence
  question + verified-wins-only clause), house (rung sketch).
  BANKED, not queued.

- **Closed-loop pincer: reverse LM feeds forward LM, one step at a
  time** (Artin, 2026-08-05 late, extending the TENET battery bank):
  instead of independent forward/backward searches meeting in the
  middle, ALTERNATE them — the reverse model proposes candidate
  inputs/predecessors, the forward model chooses ONE step, the
  reverse model consumes the new state and proposes again; a
  generate-choose loop where each direction constrains the other
  every step. Tie-in candidates: the weight-reader lineage (the
  2026-07-06 ablation model — small, size to verify at spec time —
  raw weights readable 80.8%, permutation-augmented 88.4%) and the
  magic-estimator doctrine (prediction pays only where variance
  lives): the reverse model is a learned predecessor-predictor, so
  it should be spent exactly on the steps where forward search has
  high branching variance. Candidate rung for the TENET battery
  (R1b: alternating pincer vs independent pincer vs forward-only,
  same matched budget, verified-AND-distinct). Attribution: Artin
  (the alternation loop + weight-reader tie-in), house (variance-
  targeting frame + rung placement). Banked into the TENET
  battery's spec-pass inputs.

- **TENET battery: SPECCED** (2026-08-05, specs/2026-08-05-tenet-
  battery.md; reviewer scan verified house-side). Correction to the
  bank above, per the scan: the "1.5 s/1000 steps" birth-cost line
  is FIXED-WINDOW only — the intbirth fast path does not eat diets
  yet, so reversed-diet births are torch-path (15-40 min Mac), not
  C++-cheap. Weight-reader size verified EXACTLY: 796,550 params
  (Artin's "<200M" holds by 250x — it is a Mac-minutes object).
  Rung order: W0 (toy reverse-readability) -> R0-rev (reverse twin
  + the REVERSE GATE build, the program's highest-value cheap
  instrument) -> W1 (direction-from-weights, needs a birth
  population) -> R1b-micro (the alternation, gated behind R0-rev).

- **Instruments IN the library (llmopt.lab)** (Artin, 2026-08-05
  night: "init one and boom it logs... logging more standard...
  failures aren't silent"): promote the hard-won scratch instrument
  patterns into llmopt/ proper — config-driven init, standard jsonl
  logging, and the LOUD-FAILURE CONTRACT as the design center:
  every timeout/crash/anomaly gets a typed, printed, counted event
  (the ORACLE-BOX 1-4 lesson, the rjob design, axiom's
  budget-expiry fix — same law three times in one day: an abort
  must never become invisible state). Candidate modules: lab/oracle
  (the boxed subprocess check), lab/gate (config-driven gate
  runner), lab/traj (router instrumentation), lab/keepsets
  (decode_counts/keep/coalition algebra), extending the existing
  llmopt.runlog seed. FENCE named at bank time: certified
  instruments (TRAJ v3, gt2_jaccard DROP_TAIL) must reproduce
  BYTE-IDENTICALLY through any extraction — regeneration tests
  before migration, and booked verdicts keep citing the frozen
  scratch files they ran on (the lab-notebook argument). Survey
  seat dispatched same-night; extraction is a PLANNED project
  (spec, then Fable implements), not a live refactor. Attribution:
  Artin (library framing + standard-logging ask), house
  (loud-failure contract + migration fences).

- **Routing-is-the-computation synthesis** (DeepSeek, cross-lab
  commentary 2026-08-06, relayed by Artin): reading the crest
  chain + the router-geometry rung together — "routing structure
  is where the interesting computation lives, expert weights are
  largely interchangeable within that structure, and domain
  capability can be IMPROVED by pruning the routing graph to the
  right coalition." Also names the sharpest open negative: the
  non-monotone recall ladder means specific expert IDENTITIES
  within the necessary branch decide resurrection, and no
  coverage metric measured so far can tell the live draw from
  the dead one. HOUSE COUNTER-NOTE at bank time: (1) "weights
  largely interchangeable" OVERREADS the ledger — GT-5/GT-5b
  (union mask loses 3/3 at 61% keep) and the ladder itself show
  identity matters at fixed structure; the booked
  statistically-identical-yet-unmatchable result is about norm
  profiles, not functional interchangeability. (2) "full model
  carries experts that actively interfere OR routing dilutes" —
  the interference-vs-dilution fork is UNTESTED; no arm
  separates them yet (candidate probe: logit-level comparison of
  full vs crest on the same solved problems). The synthesis is a
  banked frame, not a finding. Attribution: DeepSeek (synthesis +
  identity-over-coverage observation), house (counter-notes).

- **Repo structure review** (Grok, cross-lab commentary 2026-08-06,
  relayed by Artin; house assessment by Fable, both-scout survey +
  spot-checks): the "instruments outgrew the package" diagnosis
  CONFIRMS the already-committed lab-extraction spec
  (2026-08-05-llmopt-lab-extraction.md) — nothing to decide there.
  ADOPTED from the review: (1) CODEMAP inventory as the gate for
  every later move (shipped: docs/CODEMAP.md via
  scripts/gen_codemap.py — mechanical class ladder, and it fixes
  the spec's broken verification rule: results-index.jsonl carries
  no path field, citations live in RESULTS/REPRODUCE body text);
  (2) EARLY package adoption of bench_step_tokens/bench_verify_fast
  (misnamed primitives: 61/43 in-code refs incl. live TENET
  scripts); (3) scratch-doctrine paragraph for CLAUDE.md;
  (4) experiments/ + archive-by-era, DEFERRED to the BOARD:114
  freeze-point gate, merged with the banked 07-24 data/checkpoints
  taxonomy. REJECTED WITH REASON (do not re-propose from the
  original message): detbwd-family collapse — the family is a
  layered import lattice, not copies (gravmoe imports r1/r2b/
  r3_qw/mb/diet); the early rungs are regression anchors with
  cross-lab byte certs and 16 sha-pinned arms, and the extraction
  spec's do-not-extract law freezes them; wholesale bench_*
  archival — import-time landmine (bare-name sys.path imports
  across ~64 files); promote the two primitives FIRST, archive
  only what CODEMAP shows nothing live references. Grok concurred
  with both rejections on review. Attribution: Grok (diagnosis +
  inventory-first + target shape), Fable (rejections + adoption
  ordering), Artin (relay + GO on CODEMAP).

## 2026-08-06 (evening): the post-GT-7 battery slate (Grok, Artin-relayed; Fable-verified against the ledgers)

Grok survey of the living docs proposing the next high-leverage
batteries; Artin adopted the science + instrument pair; Fable
confirmed every rung and every low-priority call before adoption.
BANKED AS ADOPTED: (1) IDENTITY battery — EX-ANAT-1 (swap/ablate
SPECIFIC high-demand experts at fixed keep fraction + fixed verbal
coverage; the only variable that still moves 38-49 solves inside a
matched bin), EX-ANAT-2 (what math-excluded experts compute —
banked since D4-PHYS-B/GT-5c, made mandatory by GT-7), R-EMISSION
(verbal restore = emission vs competence; degeneracy census now
collected), CHURN-JUDGE-2 (gated BEHIND a named identity handle
AND the booked revive-if: routing-margin features, new pre-reg).
Spec: 2026-08-06-identity-battery.md. (2) INSTRUMENT battery —
lab/gate -> sink -> timebox (traj CLOSED same day, VERDICT
LAB-TRAJ; Grok's list predated the booking), FINDINGS GT-7 append
(done with this bank). CONFIRMED LOW-PRIORITY (each already
fenced): GT-8 aggregate ladder (killed by the registered survival
map), RULER/gist (dead as experiment, library stays), TENET R2/R3
pincer (choice-scarcity closure; revival = measure verified-
candidate multiplicity FIRST), PLACE-1-on-cuda (uncomparable cell,
30B does not fit 10GB), Ozaki speed (kernel-hours + ambiguity-rate
prerequisite), Fourier force-the-clock (banked, off critical
path). Attribution: Grok (slate + sequencing), Artin (adoption
call + relay), Fable (rung-by-rung verification + the one stale
correction).

## 2026-08-06 (late): Gemini README teardown (Artin-relayed; one adoption, two rejections-with-reason, one bank)

ADOPTED: README headlines — three fenced punchlines above the fold
(crest, identity-follows, cross-device integer birth), each carrying
its FINDINGS pointer; Gemini's own phrasings ("intelligence
multiplier", "MoEs get smarter") NOT used — unscoped claims are what
the tag grammar exists to prevent. REJECTED: intmath.py split into a
package — promoted, sha-pinned, cross-lab-certified evidence surface
(402 tests, axiom C++ parity); architectural-aesthetics churn on
frozen evidence is what the CODEMAP move-gate blocks; extensions go
BESIDE it. BANKED (Artin's call on priority): the self-contained
free-run-oracle reproduction gap — Gemini proposes a surrogate
dataset; the house-shaped fix is the DIET-BRIDGE pattern instead
(serialize gate-row tokens INTO the reproduce artifacts, as
diet_init.bin already does for windows) — most house data is
engine-minted and string-seeded, so the blocker is the file-handoff
convention, not IP. A surrogate corpus would add a contamination
surface for no reproduction gain. Attribution: Gemini (teardown),
Artin (relay + headline call), Fable (adoption filter).

## 2026-08-06 (late): DeepSeek review questions (Artin-relayed) — lens candidates banked into EX-ANAT-2

Three questions, two already answered by registered work (weight-
reader storage-vs-compute: property-specific invisibility, scoped
toy-scale; Lean bottleneck: engineering-dominated with the loud
7-row unprovable-by-design semantic residue + 4 open field_simp).
BANKED for EX-ANAT-2's design: candidate lenses for what
distinguishes carrier experts — (a) architecture-level properties
(layer position, gate-weight geometry — note the W1-S caution that
weight surfaces hide functional properties), (b) token-
specialization signature (which token classes an expert fires on in
free routing — TRAJ instrument ready), (c) demand-rank-within-
exclusives (has measured signal: EX-ANAT-1 ranked by it and fired).
Training provenance UNTESTABLE on the vendor model — fence, not
lens. Attribution: DeepSeek (questions), Artin (relay), Fable
(disposition).

## 2026-08-06 (late): W1-R — the relationally-capable weight reader (Artin riff)

"Have we thought about what a relationally-capable reader would look
like? Pairwise inner products between weight rows?" — NOT previously
banked (weight2vec is adjacent but about shared embedding geometry,
not relational features). This is the "new representational
hypothesis" the W1-S closure requires, so a rung is LICENSED.
Sharp form: direction is a COMPOSITION-ORDER property — single-row
tokens are structurally blind to it, which retroactively explains
the W1/W1-S null without claiming the information is absent.
Feature families, in attack order: (1) CROSS-LAYER alignment
statistics (down_l write basis vs gate_{l+1} read basis; spectra of
the cross-Gram — where composition order would live); (2) within-
neuron functional pairing (gate/up/down triples: angles, norm
ratios — the true functional unit); (3) within-layer Gram spectra
(cheapest, fully invariant, weakest for direction). DESIGN FENCE:
the 2026-07-06 ablation (augmentation 88.4 > canonicalization 82.4)
predicts imposed-invariant spectra LOSE to augmented relational
tokens — run both arms, the ablation's prediction is itself a
readout. Protocol/subjects/bars: reuse W1-S verbatim (50 pairs,
w1-split-1, 16 votes, binomial 16/20 fire). Cost: minutes, existing
population. Fires as PRE-REG TENET-W1-R on Artin GO. Attribution:
Artin (riff), Fable (composition-order sharpening + design).

## 2026-08-07 (morning): reverse-guided step-distribution decoding (Artin riff)

"One model takes input -> reverse LLM transforms the math equation
into multiple steps with distribution % from q-wave -> normal LLM
chooses based off that dist." NOT a weight-reader revival (the
W1/W1-S/W1-R closure is scoped to reading DIRECTION from static
weights; this is function-space and untouched by it). WHAT IT
ACTUALLY IS: the named revival case the pincer closure demanded —
TENET R1b died on CHOICE SCARCITY ("nothing to rank"; revival fence:
a proposer whose verified-candidate multiplicity is measured FIRST),
and a reverse model emitting multiple decompositions WITH a
distribution is precisely a multiplicity-generating proposer.
Measured basis already on the books: backward emission is REAL and
prompt-distribution-local (R0-REV-B: rev 24/120 v fwd 1/120 on the
D1b poststep gate). FENCE-MANDATED FIRST MEASUREMENT (before any
architecture): the multiplicity census — on in-distribution inputs,
how often does the reverse model emit >= 2 DISTINCT
oracle-verified step-candidates, and does its q-wave distribution
rank the verified ones above the unverified (a calibration read,
not a capability run)? Only a positive census licenses the
chooser rung. Attribution: Artin (riff), Fable (choice-scarcity
tie + census-first framing).

## 2026-08-07 (morning, cont.): engine-verified reverse-propose/forward-choose as the TRAINING loop (Artin extension)

Extension of the same-morning riff: the math engine verifies the
whole reverse-propose -> forward-choose process and the verified
episodes become how the models train/learn. This is the
expert-iteration shape with a REVERSE proposer: house already owns
every stage measured — engine-replay mint criterion as verifier
(the D1 reverse gate's criterion after AMENDMENT R0-REV-D1 killed
symmetric equivalence), the exchange loop's conversion proof
(metabolic v5: engine rows 2->6/12 in 10 min), and R0-REV-B's
backward emission. NEW element: reverse-proposed,
distribution-weighted candidates as the training stream — the
q-wave % becomes a soft label over verified decompositions.
STANDING FENCE ATTACHED AT BANK TIME: verified AND distinct at
every learning layer — a reverse proposer is a prime X=>X
identity-rewrite hazard (bit three times: GRPO reward, gate
candidates, miner v5); the multiplicity census must count DISTINCT
verified candidates, and any reward/diet built on this loop
rejects identity rewrites at reward, gate, AND miner
independently. Ladder shape (census-gated, no rung fires before
the one below): (1) multiplicity census; (2) chooser calibration
(does the q-wave dist rank verified above unverified); (3)
exchange-loop episode: train on engine-verified
reverse-decomposed rows, gate before/after. Attribution: Artin
(riff + extension), Fable (stage mapping + fences).

## 2026-08-07 (evening): Grok claims-day review + the triple-agent audit (Artin-relayed / Artin-sanctioned)

Grok's eight sharpening points, all dispositioned: ADOPTED — FINDINGS
claim-hierarchy line (GT-1 most-replicated; over-inclusion reading
GATED behind an owed uniform-random deletion control), EX-ANAT-4
ordering (uniform-random control FIRST, carrier TRAJ anatomy second,
physics-scalpel transport LAST), instrument no-silent-drift note
(gt7_run + frozen arm2 path load-bearing for every crest claim),
pipeline-exit-code commit-gating lesson (standing memory). ALREADY
SATISFIED — the diversity-diet door in MULT-0-B32's consequence.
FOLDED INTO AUDIT — axiom booking contract (frozen in the
ENGINE-SCALE-1 pre-reg: house books only after the three spot shas
match), instrument-drift check. THE AUDIT (three Opus agents,
findings Fable-verified): P0 provenance hole repaired
(scratch/ex3_build.py, 7/7 byte-identity asserted, AMENDMENT
EX3-PROVENANCE); lens-env pollution guards added to all four cited
builders (refuse-never-adapt); gt7_run re-patch guard + restore;
w1_surfaces tempfile race fix; traj.py per-block n_exp (authority-
table conformance); RNG streams pinned to sorted order (verified
no-op); docs pass numerically CLEAN (zero drift, zero broken
anchors) with tag-grammar + staleness fixes applied (README crest
prose modernized, THEORY row head de-contradicted, [NULL] scope
alignment). EX-ANAT-4 SLATE BANKED here per bank-everything:
(a) uniform-random deletion control, (b) carrier TRAJ anatomy on
the named 80, (c) physics-gate scalpel transport — ordered, each
pre-reg before fire. Attribution: Grok (sharpening), three Opus
seats (audit findings), Artin (sanction + relay), Fable
(verification + application).

- **2026-08-08 (Artin x Grok, the horizon refinement of the
  black-hole frame)**: Artin — "LLMs are models of the universe we
  find useful, hence why (i think) MoE's are similar to black
  holes"; Grok's sharpening, adopted — "gravity-like concentration
  of computation creates horizons in the model's information flow":
  most parameter mass is interior, only a thin interface (router /
  keep-set) talks to the outside, and what CROSSES the interface
  decides usable v trapped/interfering. Why banked in this form:
  unlike the original identity claim, the horizon version
  compresses measured results (identity-over-coverage, deletion as
  interference removal, router over-inclusive at the carrier rank
  class) and its first testable consequence is already queued —
  EX-ANAT-4's uniform-random deletion control separates "horizon
  structure" from "any deletion helps." Scale corollary (Artin,
  same exchange, measured side): closed-system math natives are
  scale-SENSITIVE not scale-maximal (113M/200M/400M excluded from
  every fit; width cliffs; tokens-per-width ceiling) — and the
  lab's reproducibility discipline is a PROPERTY OF THE ARTIFACT
  at this scale (bit-identical replay, sha acceptance bars),
  structurally unavailable at datacenter scale. Frame stays
  riff-tier until it predicts a new registered gate; FINDINGS
  narration stays literal per the plain-language rule.
  Attribution: Artin (frame), Grok (horizon sharpening + the
  measured/riff split), Fable (banking + the
  reproducibility-as-artifact-property corollary).

- **2026-08-08 (Artin x Grok, "lab Elo / fishtest-minus-the-grid")**:
  Artin — steal chess-engine acceptance METHODOLOGY (SPRT/Elo with
  error bars, one variable, fishtest-style orchestration) for the
  lab's crystal/mask evaluation, explicitly WITHOUT the volunteer
  distributed-compute part (lab-owned runners only); standardize
  the weight-geometry instruments (polar/PCA/phase-density) so
  identity and capability changes are scored the same way every
  run; gate-level rating tracked across seeds instead of single
  headline solve counts. Grok — split from the PR-slop rant,
  scoped to "sketch a minimal row, no infra." House note at
  banking: the lab already runs the SPRT SPIRIT (pre-reg = the
  hypothesis pair, frozen bars = the accept/reject bounds, paired
  arms = the match, RESULTS = the append-only match log); the
  genuinely NEW ingredients are (1) per-prompt PAIRED outcome
  records (current bookings keep per-level dicts, which discard
  the pairing that SPRT feeds on — decisive pairs are the signal,
  both-solve/both-fail draws are noise), and (2) a standing null
  band per geometry instrument. Fence carried in from doctrine:
  accept/reject authority stays with ORACLE-RUN outcomes only;
  geometry instruments fingerprint and describe, never decide
  (never-score-weights-by-weight-distance). Riff-tier until a
  registered gate uses a paired-outcome sidecar. Attribution:
  Artin (frame + no-volunteer-grid constraint), Grok (scoping),
  Fable (banking + the already-SPRT observation + the
  paired-record gap).

- **2026-08-08 (Artin, the exact-compression extension of the
  black-hole frame)**: exact-arithmetic training as "a true black
  hole — compressing the data exactly and infinitely": if
  precision, absorption, and LR are one law (THEORY absorption
  row), then fp training is a LEAKY compressor (every rounded
  update discards a little verified signal) and exact training is
  the zero-leak limit — the model learns from the data EXACTLY.
  House counter-position stays the absorption law's: at birth LRs
  the leak is far below gradient noise, so the limit buys nothing
  measurable. This is now PRE-REGISTERED DISAGREEMENT #3 territory
  (lineage: #1 birth rarity — law won; #2 accumulator tail — law
  won on scope; #3 exact-vs-fp32 END-TO-END gradients — genuinely
  open, the 2026-08-08 audit showed fp32 backward rounding was
  shared by every tested arm). Vehicle: axiom exact-engine
  feasibility (relay sent; transcendental-site conventions become
  contract pins; fallback rung = exact-GEMM-only isolating the
  correlated-matmul-rounding mechanism). Fence: narration stays
  literal per the plain-language rule — "black hole" is the
  banked frame's name, not a mechanism claim. Attribution: Artin
  (frame + priority call), axiom-Fable (feasibility grounding),
  Fable (scope audit that opened the door + the exact-GEMM-only
  fallback).

- **2026-08-08 (Artin x Fable, the capped-instrument sweep + exact
  data-pricing)**: two banks from the disagreement-#3 exchange.
  (1) MECHANISM-POSITIVE/CAPABILITY-FLAT as a named audit class —
  rungs where a measured physical difference read zero on the
  gate, so a coarse/saturated instrument may masquerade as a null.
  Flagged instances: metabolic v3 (2.7x flips real, proxy 24=24,
  time-capped at 75 min — the top revival candidate under this
  lens), gate saturation at L3 24/24 and fresh-band 89/120
  (treatments landing in saturated levels are structurally
  invisible), the 113M capacity null's tokens-per-width masking
  (the caught template), the fp64 rarity showdown's n=9/18 bin
  power. Defended as TRUE null: the 1.06e-14 accumulator tail
  (12 orders below flip threshold — no instrument problem).
  Doctrine rider: disagreement-#3-class pre-regs pair the coarse
  gate with FINER registered instruments (flips, validity %,
  rarity bins, long-horizon retention) so a mechanism-positive
  cannot hide. (2) EXACT DATA-PRICING (Artin): exact-substrate
  training as the clean bench for "what data does a model EXACTLY
  need" — schema/columns/links — sharpening the existing pricing
  constants (decomposition discount ~10x/row, k_efold
  ~1,900-2,400 rows per primitive kind, determinability audits)
  by removing the absorption term from the estimate; possibly
  exact-training's main payoff even if the capability race ties.
  Goes into the eventual #3 pre-reg as a secondary instrument.
  Attribution: Artin (both frames), Fable (instance sweep +
  true-null defense).

- **2026-08-08 (Artin, data-as-terminal-limit + the chess
  testbed)**: extension of the exact-compression bank. (1) The
  local law's testable form: if, with every other dial measured
  at its stop, only DIET DIFFICULTY moves capability, then data
  is the binding constraint — already 1x observed (gen-6 plateau
  broken only by territory); composition rider applies ("harder"
  that starves the base rots the model — gen-7 lesson). The
  cosmic version (modeling the universe needs more data than it
  contains) is not reachable from this bench: it depends on
  whether the target's difficulty ladder terminates. (2) CHESS as
  the miniature where the cosmic claim is PROVABLE BY COUNTING:
  ~10^44 positions cannot be stored exactly in physical
  resources (7-piece tablebases = 140TB frontier), while checkers
  (~10^20) WAS solved 2007 — one positive, one negative data
  point for "solvable iff state space fits resources." lc0/SF are
  the predicted objects: lossy compressions of an unstorable
  function, on the purest data there is. The measurable question
  for a chess-native model is NOT "solve" (settled by arithmetic)
  but the COMPRESSION-EFFICIENCY CURVE: strength per param per
  training position on a target of known exact size —
  NNUE-lineage instruments fit it natively; charter-clean
  (combinatorial game theory = mathematics). Riff-tier: a chess
  engine is a scope decision, banked not launched. Attribution:
  Artin (both), Fable (counting argument + checkers control +
  curve reframe).

- **2026-08-09 (Artin, the LinkedIn magic exchange): HARDNESS HEAD
  IN THE MODEL'S FUNCTION-SPACE — embed the magic estimator INTO
  the model, firing when it is unsure.** Precision on the state of
  the bank: the ENGINE-side embedding already ran and split —
  policy-gated expansion pays (~4x cheaper nodes, k=6, booked),
  while the "when unsure" trigger specifically NULLED in-domain
  (entropy-gated deference 69/80: traced losses held the winning
  rule inside the gate and died on beam composition, not
  uncertainty — confidence can't see that failure class; deference
  stays banked for real-OOD domains, ZX port named). The OPEN leg
  this riff adds: the MODEL-side embedding — a hardness/cost head
  trained into the proposer itself (predict-own-cost as an
  auxiliary target), read at inference to route: cheap head says
  easy -> answer directly; says hard -> escalate (workspace,
  deeper search, LLM slot). Pairs with the banked
  magic-estimator judge slots ("prediction pays only where
  variance lives") and the budget-alloc retarget (estimator
  routes LLM wall-time, the currency that actually runs out).
  Fence: NNUE-vs-LLM booked an exact tie at 1e5x cost — the head
  must be justified by ROUTING value (fewer wasted escalations),
  never by beating the microsecond MLP at its own job. Also
  banked from the same exchange: the quantum framing stays a
  labeled analogy (magic monotone = distance from the easy
  subspace = simulation price), narration literal per the
  plain-language rule. Attribution: Artin (embed-when-unsure +
  the classical-simulability frame), Fable (state-of-bank
  precision + routing-value fence).

- **2026-08-09 (Artin, the p=32 exchange): BORN-RULE NORMALIZER —
  "if softmax is the problem, why not NOT use it?"** Replace the
  softmax carry seam with a squared-amplitude normalization
  (p_i = x_i^2 / sum x_j^2, the quantum measurement rule): purely
  polynomial/rational, needs NO exp table, exactly representable
  with one shared denominator in integer/rational arithmetic.
  Precision on what it does and does not buy: the
  Omega(2^steps) impossibility means SOME rounding must fire
  somewhere per step — any normalizer (softmax, Born, sparsemax)
  still quantizes its output to the PQ carry, so the frozen-carry
  FLOOR survives the swap. What it could buy: (1) a table-free
  forward (the exp table and its tse clamp deleted — fewer frozen
  conventions to transport); (2) sparser/cheaper seams, possibly
  making a PQ raise cheaper per bit (the E-E-2 exchange rate);
  (3) house-precedent synergy — the complex-weight NNUE pre-reg
  and the ZX line already carry the squared-amplitude frame.
  Sparsemax noted as the other exact-friendly candidate
  (piecewise-linear simplex projection, exact gradients). Slot:
  an ENGINE-EXACT-2 design ARM (carry ladder x normalizer
  family), not a way around the theorem. Attribution: Artin
  (kill-the-softmax + quantum-wave frame), Fable (the
  seam-survives analysis + sparsemax sibling).

- **2026-08-09 (Artin, the diverse-minds exchange): SMALL SPECIALIST
  MODELS AS EXPERTS, connected through a general weightspace — "what
  if the experts thing is foreshadowing?"** The measured base: (1)
  the union-diet law (UNION EQUATION v1) — grammars federate nearly
  FREE in one 19M substrate (probes at-or-above specialists; series
  99.2 BEAT its specialist) until capacity W* binds via exposure
  share; (2) the MoE's router is domain-biased with ordered
  coalitions (GT-2-D2/D3: code twice as far from math as physics) —
  a big MoE already self-organizes internal quasi-specialists; (3)
  crests do NOT transport across domains (D4-PHYS +3 v bar +7) —
  the right keep-set is domain-local; (4) over-inclusion — surplus
  participation is interference. The riff composes these: explicit
  SMALL specialist models + a general routing/connector layer,
  instead of one big substrate with implicit coalitions. NAMED
  CHEAP TEST (crossover form): at matched TOTAL params, do two
  small single-grammar specialists + a dispatcher beat one
  union-diet model once W* binds? House instruments exist (19M
  specialist births, the dispatcher-race pattern, union births).
  FENCE: "connected through weightspace" must mean FUNCTION-space
  composition (routing, distill, oracle-verified exchange like the
  12/12 model+engine complementarity) — direct weight-space
  merging is the unproven leg (weight-reader closed at toy scale;
  never score weights by weight distance). Attribution: Artin
  (architecture frame + foreshadowing read), Fable (evidence
  assembly + crossover test form).

- **2026-08-09 (Artin, the crown-dict exchange): STAR PROFILE, not
  black hole — precision belongs to the CORE, traversal only needs
  {-1,0,1}.** The observation that seeded it: in the one crown
  head-to-head, ternary's deficit sat in EASY bands while it won
  L4 and tied L7 (descriptive, <3 noise class — fence stays). The
  frame: a model as a star — a dense fusion core where precision/
  compression/density matter (the primitive operations being
  fused), and progressively simpler outer shells where computation
  is TRAVERSAL through structure, needing only sign-and-silence
  {-1,0,1}. Iron line = the element past which fusion stops paying
  (the primitive set). MEASURABLE FORM (the bankable part): a
  TENSOR-CLASS PRECISION-SENSITIVITY PROFILE — quantize one tensor
  class at a time (emb / head / attn / ffn / norms) to ternary at
  matched sigma-scale, gate each; the frame predicts a monotone
  sensitivity gradient from interface/core classes outward. House
  evidence already pointing this way: make_tables ships sigma/8 on
  emb/head INTERFACE tensors specifically (the deterministic
  pipeline already treats interfaces as precision-hungry); ternary
  growth is non-preserving via absmean coupling (the discrete
  substrate's fragile spot is a COUPLING, not a band); external
  echo: BitNet-class recipes keep embeddings/activations at higher
  precision than ternary bodies. FENCES: star/black-hole are frame
  LABELS (plain-language rule — narration literal); the per-level
  crown texture is noise-class until the births battery books.
  Attribution: Artin (star frame + core-vs-traversal split),
  Fable (tensor-class probe form + sigma/8 interface evidence).

- **2026-08-09 (Artin, star-frame continuation): THE IRON POINT OF
  ALPHABETS — is there something between iron and 1.58 bits?** The
  real half: iron-56 = maximum of binding-energy-per-nucleon;
  base 3 = the RADIX-ECONOMY optimum among integer bases (cost
  r*log_r(N) minimized at e~2.718; 3 is the best integer — a
  classical theorem, the Setun/Knuth balanced-ternary lineage).
  Both are peaks of an efficiency-per-unit curve: "where building
  bigger stops paying." The house has MEASURED its own such curve:
  1.0 bit {-1,+1} = collapse (silence-is-structure crater); 1.58
  bits {-1,0,+1} = peak (crown tie, d768 null); 16-64+ bits = flat
  (precision doctrine closed). {-1,0,+1} = the minimal complete
  instruction set (sign + abstention) — smallest alphabet with
  direction AND silence. FENCE: curve-shaped analogy, never
  number-shaped (no arithmetic link between 26/56/8.79 MeV and
  log2(3); quote as structure, not digits — plain-language rule).
  Attribution: Artin (iron<->1.58 question), Fable (radix-economy
  theorem + measured-curve assembly).

- **2026-08-09 (Artin x grok-seat x axiom-Fable, the iron-point
  closure): THE TWO-TERM LAW.** The iron/e/ternary confluence
  dissolves cleanly: every efficiency-per-unit curve with a
  ~linear benefit term and a superlinear penalty term has an
  interior maximum — the confluence is the GEOMETRY, never the
  x-axis labels (Fe-56 from volume~A vs surface~A^2/3 +
  Coulomb~Z^2/A^1/3; e from the r/ln r functional; 3 from
  alphabet completeness + measured nulls above). Artin's
  9th-grade isoperimetric read is LITERALLY term one of the
  liquid-drop model (surface fraction shrinking as the drop
  grows); the fall always comes from whatever charges for size
  (Coulomb / the leading r / structure the training can't cash).
  DOCTRINE RIDER (axiom's, adopted): when a plateau appears, the
  productive move is to IDENTIFY WHICH PENALTY TERM BINDS, never
  to scan the axis for a magic value — leave-one-small slices are
  that instrument. Same-day demonstration: ENGINE-SCALE-1's
  plateau bound not by capacity or windows but by the const-lr
  penalty term (s16000-const REGRESSES; decay removes the term
  and the plateau breaks 21.5%). Attribution: grok seat
  (geometry-not-labels dissolution), axiom-Fable (two-term form +
  the which-term-binds rider), Artin (isoperimetric bridge).

- **2026-08-09 (Artin, the d8 step-3 marathon): PARALLELIZE THE
  ANCHOR — "retry step 3 with more workers / cache the
  computations."** Three-way resolution. (1) Across-step: never
  (training is a dependent chain). (2) Within-step: real and
  PROVABLY digest-preserving — rational arithmetic is exactly
  associative, so entry-parallel gemm/gcd is bit-identical by
  construction (unlike fp); axiom-side engine change, ~16x, but
  at the measured ~845x per-step growth that buys only ~half a
  step of horizon. (3) The design that actually moves the
  horizon: RNS/CRT ANCHOR — run the whole trajectory modulo many
  machine-word primes (no bignums, NO GCD — the measured binding
  cost — perfectly parallel across primes and entries), CRT-
  reconstruct exact rationals only at dump steps; prime count
  budgeted by the ~10-bits/step growth law. REVIVES the parked
  RNS rung of specs/2026-07-27-exact-representations.md; sibling
  precedent = ozaki sliced-exact (int-sliced beats fp64). Slot:
  anchor-v2 instrument design, axiom engine leg, own pre-reg;
  would convert d64's 1-step/42-min horizon into multi-step
  exact prefixes at real widths. Attribution: Artin (parallel +
  caching push), Fable (associativity-safety note + RNS/CRT
  mapping + parked-rung revival link).

- **2026-08-09 addendum (Artin: "we already have that + the Ozaki
  rungs — we should've taken those wins too"): the RNS/CRT anchor
  is an ASSEMBLY, not an invention.** Components already booked:
  ozaki 2b expansion recombination (exact accumulation, deviation
  = ZERO); ozaki 2c doctrine ("never leave the sliced domain" —
  the direct cure for the anchor's per-op gcd bill); the parked
  RNS rung (cross-step representation); sliced-exact CUDA (247ms
  N=2048, beats fp64 — the heavy layer runs on the 3080). The one
  real design question: NON-RING OPS (softmax max-shift, clamps,
  floor seams, AdamW div/isqrt) force reconstruction points —
  hybrid architecture = slices/RNS through linear segments,
  CRT-reconstruct only at nonlinear seams; the seam placement is
  a frozen-grain contract, same discipline as the PQ carry.
  PROCESS LESSON (Artin's, adopted): the anchor spec never swept
  the parked-rung inventory — standing habit now: EVERY new
  instrument spec greps parked rungs for components first
  (reviewer revival-scan mandate extended to spec time).
  House-side option: an RNS trajectory prototype in Python/CUDA
  over the existing ozaki scratch is feasible WITHOUT waiting on
  axiom (acceptance bar: reconstructed dumps equal the bignum
  anchor's on an overlap cell). Attribution: Artin (take-the-wins
  push + the process critique), Fable (component mapping +
  non-ring-op catch).

- **2026-08-10 (Artin, after Opus 5 caught a Fable-booked error):
  "it's kinda like the physics + math model fuse — the more minds
  the better, up to a point (45% experts beat the full model)."**
  The observation is real and the arc is now three deep: the
  fused physics+math birth beat the physics-only arm ON PHYSICS;
  the routing crest peaks BELOW full inclusion; and a mixed-model
  seat caught a mixed-estimator quote its author had reviewed and
  shipped. Bank the shape — a benefit term that is linear in added
  perspectives against a penalty term that is superlinear in
  redundancy, so the optimum is interior. That is the two-term law
  already adopted as doctrine (2026-08-09 iron-point bank); this
  is the same curve read on SEATS instead of experts or alphabets.
  NAMED CONFOUND, and it is the whole experiment: model identity
  was not isolated. Opus 5 did not out-reason Fable — it ran the
  fit again from the artifact, with no memory of having produced
  the number. Three candidate active ingredients, and they staff
  the lab very differently: (a) DIFFERENT MODEL — different priors
  and failure modes; (b) INDEPENDENT STATE — any fresh session,
  same model included, is unanchored by having authored the claim;
  (c) THE TASK — curation forced a recomputation, and any seat
  doing that step would have caught it. The house reviewer
  doctrine already bets on (b) ("independent state + cheap
  breadth"), which makes (a) a proxy the lab has never priced.
  CHEAP DISCRIMINATOR, banked as a candidate probe: replay the
  same catch as a blind 3-arm ask over the frozen dumps and the
  verdict text — same-model-fresh-session, different-model-fresh-
  session, and same-session-self-review — and score catch rate.
  If fresh-same-model ties fresh-different-model, the lever is
  state, not diversity, and reviewer seats should be cheap and
  numerous rather than model-diverse. Fence before anyone
  transports this: the 45% crest is expert MASKING on one vehicle
  and one domain, a selection result, not an ensembling one —
  the analogy is a curve shape, not a mechanism, and the ledger
  should not let it become one without its own bar. Attribution:
  Artin (the fuse/crest link + the more-minds framing), Opus 5
  (confound decomposition + the blind-replay discriminator).

- **2026-08-10 CORRECTION to the bank above (Artin: "there is,
  we've done them both — query the headers").** The bank called the
  three-arm seat discriminator an unrun probe. It is not unrun; it
  is UNTABULATED. A header sweep of results-index returns ~26
  review-adoption entries spanning every arm the bank proposed:
  SAME-MODEL SELF-REVIEW (AMENDMENT house self-review 2026-07-26,
  RESULTS L5453 — the streaming -12 confound, Fable catching
  Fable); DIFFERENT-MODEL SEATS at every fleet size, one through
  five (Opus-5 onboarding "four catches" L4803; reviewer catches
  L4770/L4901/L5026/L5069; REVIEW-ADOPTION passes L13063/L13785/
  L13840/L14504; AUDIT-0802 L16271; RUNGD-0803 "four reviewers,
  and the fence was the thing that was wrong" L16608; FINAL-0803
  L17024; MERGE-AUDIT-OPUS5 L17146; F1-REVIEW "two reviewer
  passes" L17713; GT1-CORR L18526; GT2-REVIEW L19417 and the
  three-seat GT2-REVIEW-2 L19648; GT-4 L20006; DAY-CONSOLIDATION
  L20575; the 08-07 doc review L22056; opus-seat null archaeology
  L23401; reviewer-fleet red-team L23739); CROSS-LAB SEATS (the
  ozaki 2b "the auditor was the bug again" L3897, E4 AUDIT
  CLOSE-OUT L7492, axiom's RED pass moving a mechanism one layer
  earlier L18087); and NON-CLAUDE seats already policy (grok CLI +
  codex read-only, cross-check-seats memory; DeepSeek's FINDINGS
  commentary). Today adds two more rungs in one day: a
  different-model seat catching a Fable booking, then ARTIN
  catching that seat's own correction — the human seat is in the
  fleet and scored a catch the model seats missed.
  SO THE OPEN ITEM IS A STATISTIC, NOT AN EXPERIMENT. Nobody has
  tabulated catches per seat by CLASS (arithmetic, statistic-swap,
  scope-overreach, stale-reference, fabrication) against seat type
  and fleet size. That is a desk pass over entries the lab already
  owns, cheap, and it is the only way the "up to a point" half of
  Artin's curve gets a measured knee instead of an analogy — the
  five-seat sweep is the largest fleet run and would anchor the
  right-hand side. The confound decomposition in the bank above
  survives unchanged: the observational record cannot separate
  model identity from independent state, because seats were never
  assigned the same material. Attribution: Artin (the correction —
  "we've done them both"), Opus 5 (the header sweep + the
  statistic-not-experiment reframing).

**2026-08-10 (Artin, fp32-limb registers for fp64-class exactness)**:
"fp64 holds a 64 chunk, and other fp32 registers keep track of the
remaining precision or do the exact" — partition fp32 registers
into limbs that jointly carry fp64-or-better precision. Banked
with its full house lineage, because the family is 80% explored:
(1) BIT-ANATOMY CORRECTION carried with the bank: fp64 = 1 sign +
11 EXPONENT + 52 mantissa; the exponent needs no tracking limbs
(block alignment makes it shared), and fp32's 24-bit mantissa
covers fp64's 53 in THREE limbs (2 = Dekker float-float ~49 bits),
not 11 — the naive 52/24 arithmetic appears nowhere in the corpus
until now. (2) ALREADY BUILT in-family: int8-limb GEMM (ozaki
v1-v6: int8-TC full-exact 55.1 ms v native fp64 40.8 @ N=2048,
triangular<5 2x FASTER than fp64; fused Triton recombination 70.2
ms bitwise-exact = 1.07x fp64 wall; stay-in-RNS break-even ~6
layers); tf32x3 2-limb split (scripts/train_tf32x3.py, Markidis/
Ootomo-Yokota, machinery proven, SHELVED "the cheap rung won" —
note that rejection rests on a +1-solve n=1 parity read, sound as
no-deficit, not as equality); the deployed 2-limb integer carrier
(llmopt/decoding/deterministic.py hi/lo >>6, partials asserted
<2^24 — the riff's mechanism in production for fixed-point).
(3) NOT BUILT anywhere: float-float/Dekker fp32-limb GEMM on
either GPU (zero source hits; RESULTS 3964 names "fp32-pair
(two-float) diagonal carry" as ozaki v2's next lift, never
booked). (4) REVIVAL CONDITIONS, in order of pull: (a) Metal —
M-series has no int8 tensor path, fp32 ALUs abundant, and
exact_gemm has NO Mac wall number (tiling deferred); an fp32-limb
Metal GEMM is the un-run cell of the family and the natural
axiom-on-Mac rung [STATUS 2026-08-10: EXECUTED — PRE-REG
FP32LIMB-METAL 24886, R1 CPU oracle exact with the bound
sharpened to a lowest-significant-BIT condition (24981), R2/R3
built + dispatch-armed (25050, GO 25036, window 25088); clause
(3)'s "NOT BUILT anywhere" is retired for Metal]; (b) the dd-EXIT floor (2^-107) that caps three
booked results is itself a 2-limb exit — a 3-limb (triple-double)
exit is named at RESULTS 4011/4038 and unbuilt. (5) FENCES that
travel: block exponent alignment is mandatory ("slicing without
alignment = compensation trick, not exactness" — RESULTS 3641,
the naive version measured 2x not exact); the chain is only as
exact as its sloppiest link (v3 lesson); torch._int_mm int32
overflow guard on any slice path; and this is a SPEED/DETERMINISM
lever only — the capability question is closed by the 132,566 =
132,566 bit-identical null, never pitch it as capability.

**2026-08-10 CORRECTION (ozaki-on-Metal bank, ~L1141)**: the
int8-simdgroup-MMA leg is SUPERSEDED on Mac — axiom's compile
probe (r2_rig.mm:85, RECEIPT FP32LIMB-R2R3-BUILT) shows
simdgroup_matrix<int,8,8> fails while the float control
compiles: M-series exposes no integer simdgroup MMA. fp32-limb
is the only MMA-reaching exact path on Metal. The bank's other
half (shared-page CPU big-int exit) survives and is the R3 exit.

**2026-08-10 (Artin, folding-landscape mathematics as a routing/
search frame — CHARTER NOTE FIRST)**: banked same day as the
no-bio-engine reaffirmation (Fable 5 safeguards article read;
Artin + house agreed: methods/mathematics IN, capability toward
molecules NEVER). This bank is the sanctioned direction only:
the MATHEMATICS of energy-landscape theory (funnels, minimal
frustration, Levinthal counting) as analysis frames for OUR
math/physics engines. No molecule, protein, sequence, or
structure capability — the frame is landscape geometry, the
subjects are model weights, routers, and search. Any rung from
this bank that names a biological object books NOT-APPLICABLE
and stops.

THE FRAMES, two distinct ones (Artin said "tunnel"; both banked):
(1) FUNNEL THEORY (Wolynes/Onuchic minimal frustration): fast
convergence happens when the landscape is shaped so most
downhill steps point home — a funnel, not a golf course; the
obstacle count (frustration) is the design variable, not the
state-space size (Levinthal's paradox dissolves by geometry,
not speed). MEASURED HOOKS ALREADY IN THE LEDGER: (a) VERDICT
ATTRACTOR-0 (25286) is a funnel statement — 198/198 single-basin
absorption, median one step: the crystal trained on terminating
chains has a minimally-frustrated landscape to answer-form
(right or wrong — the funnel drains to CLAIM, not truth). (b)
The MoE crest family (45% masks beat full; which-experts is
0-v-82/120) reads as frustration: the router's raw landscape
carries misleading minima that demand-ranking smooths. (c) The
ambiguity law (DATA-CEIL-0C/-R): margins = local landscape
steepness; branching data = flat/frustrated neighborhoods.
CANDIDATE RUNG SHAPE (desk-first): a frustration METRIC for
routers/search — count margin-inversions along verified
trajectories (steps where the greedy-preferred move is not the
oracle-verified move) as the frustration density; funnel-quality
then predicts gate solve rate. Cheap, CPU, uses existing gate
sidecars. CAVEAT 2026-08-10: premise fenced by DATA-CEIL-0A
(RESULTS 24602) — margins do NOT track problem hardness (rho
+0.18 v predicted <=-0.8); any frustration metric must be
validated against solve rate directly, never assumed from
margin geometry (margin reads branching factor, not hardness).
(2) TUNNELING (barrier penetration, annealing lineage): escaping
local minima by passing THROUGH barriers rather than over them —
the quantum-annealing/simulated-annealing frame for search
temperature and for why wave-sampling (B=8 at T=0.7) beats
greedy in frustrated regions but loses in funneled ones.
PREDICTION SKETCH, unregistered: sampling helps exactly where
frustration density is high; in funnel regions greedy wins and
waves waste tokens — the ATTRACTOR-0 result (greedy funnels
straight to answer-form) v the gate's need for waves is already
weak evidence of the split.

Provenance: Artin's routing instinct ("seems like a better way
to route through model weights"), same-day as the safeguards
read; house supplied the ATTRACTOR-0 connection. Bank everything.

**2026-08-10 ADDENDUM (the typo that paid: FUNNEL CONTROL, third
meaning — Artin via a Gemini definition)**: control-theoretic
funnel control (Ilchmann/Ryan prescribed-performance): error must
stay inside a shrinking time-varying envelope; gain adapts to
distance-from-boundary; no exact plant model needed. BEST FIT OF
THE THREE MEANINGS — it attacks a REGISTERED open problem: the
anchor-v2 path-dependence (AMENDMENT 24153: certified prefix is a
property of (anchor, SCHEDULE); every schedule tried was
OPEN-LOOP — linear/geometric/max ramps picked in advance, and the
step-9 site defeated two strategies, 24224). Funnel-control
reframe: straddle width = error, per-step precision = gain, throw
= envelope violation. The dyadic shadow already MEASURES width
every step — the sensor exists, only the feedback law is missing.
CANDIDATE RUNG (axiom-side, cheap: schedule policy change, no new
engine): prec(s+1) = f(measured width distance to envelope), pre-
registered against the same d64 12-step cell — prediction: closed
-loop certification is PATH-INVARIANT BY CONSTRUCTION (the
schedule is no longer an input), which is the property
P-PATH-INVARIANT wanted and never got an instrument for.
Secondary hooks: LR schedules as funnel controllers on loss
(training-side, LR-precision one-knob law adjacent); decode-margin
floors. Provenance: Artin typo'd tunnel->funnel, googled, landed
on the control meaning; all three meanings now banked.
OUTCOME 2026-08-10 evening: RAN AND FIRED — VERDICT FUNNEL-PREC
(RESULTS 25451): path-invariance exact (byte-identical from step
2 at entries 200 v 4000), 53% of open-loop cost IN BIT-STEPS
(wall clock FLAT ~162s/step — precision is not the wall lever;
the ring is), step 9 defeated a third time with the sensor
sizing the wall at >=15k bits. Surviving open edge: the
invariance-v-anticipation tension (no derivative term possible
without re-introducing history).

- **2026-08-10 (Artin, late evening): SFT AS REPAIR OF NON-EXACT
  WEIGHTS.** The frame: supervised fine-tuning is not "learning new
  things", it is repairing the damage a non-exact substrate (rounded
  births, quantized bodies, drifted masters) did to a function the
  data already specifies — which predicts SFT's gradient mass should
  CONCENTRATE where the representation error lives, not spread
  uniformly. MEASURABLE FORM (bankable): per-tensor-class |dW| map
  of an SFT pass v the same model's quantization-error map (fp v
  ternarized-class deltas) — the repair frame predicts rank
  correlation between "where SFT pushes" and "where precision was
  lost"; the null is SFT mass tracking the task gradient regardless
  of precision damage. Ties to the gradient-spectrum reading
  (bits(needed) ~ log2(w/(LR*g)), DATA-CEIL survivors) and BRIDGE-1.
  HONEST FENCE on the axiom-tools half of the ask: the exact
  instruments do NOT speed SFT — the anchor runs 162 s/step at d64
  (ring-bound, AMENDMENT FUNNEL-PREC-COST-SCOPE-AND-WALL) and SFT
  sizes sit far beyond it; their role in this riff is CERTIFYING
  small repair cells (exact d8-d64 SFT steps as ground truth for
  what "repair" does bit-by-bit), not throughput. Pairs with the
  STAR-PROFILE bank (both ask WHERE precision matters, from
  opposite ends). Attribution: Artin (repair frame), Fable
  (dW-v-quantization-error observable + exact-cell certification
  role).

- **2026-08-10 (Artin, late evening, star-frame third movement):
  SILICON BEFORE COLLAPSE + PRECISION-IS-DIET, STRONG FORM +
  TRAINING AS THE CONTROLLED BURN.** Three riffs in one message,
  banked separately (labels are labels; narration literal):
  (1) STRONG-FORM DIET CLAIM: "precision is 100% diet-related" —
  the maximal reading of the ambiguity law: a math crystal's
  precision demand is entirely a property of what the data leaves
  underdetermined, with the one-step basics settling into a
  ternary-tolerant core as the diet is fully priced in. Evidence
  FOR already booked: cleaning data removes tie sites (DATA-CEIL-0C
  registered inversion), law replicated on two crystals. Against
  the absolutism: capability sets the SCALE of margins (d128 v
  d256), and the 0B-JOIN null shows at least one margin phenomenon
  (absorbing confidence) that is not branching-borne. TEST OWNED BY
  A RUNNING RUNG: STAR-PROFILE-1 (math) v the queued ZX leg — if
  the tensor-class sensitivity profile TRANSPORTS across domains at
  matched arch, precision geometry is architectural; if it moves
  with the diet, Artin's strong form gains its first cross-domain
  evidence. (2) SILICON POINT: ternary as Si — the last stable
  stage before binary collapse {0,1} = Fe. The house curve already
  measured (iron-point bank, 08-09): 1.0-bit collapse crater,
  1.58-bit peak, 16+ bits flat. The addendum banked here: the
  CONFLUENCE — Si is simultaneously the substrate of literal logic
  gates (semiconductors: stable, simple, workable) and the
  penultimate fusion stage; {-1,0,1} plays both roles for crystals
  — the最-efficient computational alphabet (radix-economy theorem,
  balanced-ternary lineage) AND the last pre-collapse rung of the
  measured curve. Label only; no physics claimed. (3) CONTROLLED
  BURN: training as the gravity/pressure phase — H/He (raw data)
  fused under compute pressure into heavier structure; the house
  controls the burn schedule. Literal content: the burn-schedule
  levers already measured are the LR schedule (ENGINE-SCALE-1: the
  schedule is the binder) and the precision schedule (FUNNEL-PREC:
  closed-loop, sensor-driven). Candidate rung shape (desk):
  read the two schedules as ONE controller — funnel-control law on
  LR the way it now runs on precision (the RIFF's own secondary
  hook, 3546-3549, now with a fired precedent). Attribution: Artin
  (all three frames + the Si confluence); Fable (test mapping to
  STAR-PROFILE/ZX transport, burn-lever identification). EARLY
  TEXTURE from the mid-run profile (single seed, unbooked): emb
  ternary craters (-27) while HEAD and NORMS ternary lose ZERO —
  the input interface is precision-hungry, the output interface is
  not; if it holds through booking, "the core" is not where the
  star bank guessed.

- **2026-08-10 (STAR-PROFILE-1 outcome note, same night as the
  third movement)**: the bank's tensor-class prediction RAN and
  INVERTED — PTQ ternary: head and norms lose ZERO, emb loses 27,
  body craters (attn 0/120, ffn 17/120); tolerance is trained-in,
  not latent (VERDICT STAR-PROFILE-1). The star frame's surviving
  form: per-PARAM sensitivity does peak at an interface (emb,
  ~90x attn), and the crown QAT lineage still says the body CAN
  live at 1.58 bits when grown there — so the frame's next honest
  shape is "the core is wherever the function was NOT grown", a
  training-history property, which is Artin's diet-strong-form
  wearing weight-space clothes. ZX transport leg unchanged.

- **2026-08-10 (Artin, night, star-frame fourth movement):
  HYDROSTATIC EQUILIBRIUM + MASS-CORE RATIO + SURFACE-TO-VOLUME.**
  The physics half, stated honestly: stars sit in hydrostatic
  equilibrium — at every shell, gravity inward balances pressure
  outward; heavier fuels (C, Ne, O, Si) each need a hotter, denser
  core, so only bigger-MASS stars ignite them; and collapse is not
  "reaching Fe" per se but the CORE exceeding what pressure can
  hold (the Chandrasekhar threshold) once fusion stops paying. The
  surface/volume tie is real physics too: a star radiates through
  its SURFACE but generates in its core VOLUME — the mass-
  luminosity scaling lives in that ratio. MAPPING (bankable):
  (1) harder capabilities need a bigger core = the house's own
  capability-sets-scale reads (d128 margins compressed v d256;
  gen-6 growth ladder). (2) COLLAPSE IS ALREADY MEASURED: the
  1.0-bit {-1,+1} crater is the house's supernova — alphabet
  pressure exceeded what the capacity could hold; 1.58 bits is the
  last equilibrium (silicon point, prior movement). (3) The
  surface/volume split got MEASURED TONIGHT without planning to:
  interface tensors ARE the surface (emb/head scale as vocab x d)
  and the body IS the volume (d^2 x layers) — STAR-PROFILE-1's
  per-param table is a surface-v-volume sensitivity read (surface
  eye ~90x volume per param; surface mouth free). CANDIDATE RUNG
  (cheap, same driver): repeat the PTQ profile at d128/d384 —
  surface/volume ratio changes with d, so the frame predicts emb's
  per-param sensitivity MOVES with width while body's stays flat;
  a null says the eye/mouth asymmetry is positional (input v
  output), not geometric. Fences: labels are labels; equilibrium
  narration is analogy, the measured objects are craters, margins,
  and the profile table. Attribution: Artin (equilibrium + ratio +
  surface/volume tie); Fable (Chandrasekhar correction, the
  surface=interface / volume=body identification, width-sweep rung
  form).

- **2026-08-10 (Artin, night): THE ESTIMATOR BELONGS TO THE ENGINE
  — curriculum as closed-loop control (NNUE frame).** The riff:
  stop trying to bolt a difficulty estimator onto the LLM
  (magic-estimator judge slots banked; config estimator nulled on
  a flat space) and give it to the ENGINE — compute the difficulty
  of the next question/step engine-side and only ADVANCE the
  curriculum when the next rung is within the crystal's grasp;
  "an NNUE engine with the quantum-dist tool". Why the house's own
  results make this sharper than when the estimator ideas were
  first banked: (1) DATA-CEIL-0A booked that the MODEL cannot
  supply the signal — margins do not track hardness (rho +0.18).
  (2) But the ENGINE can: branching factor n_succ is cheaply
  computable by the successor enumerator, and margin tracks IT
  (the ambiguity law, replicated). So the difficulty axis the
  crystal actually feels is engine-computable per state, for free.
  (3) "Prediction pays where variance lives" (THEORY row 28) is
  satisfied: gen-level diets span a 3x margin range and orders of
  magnitude in tie incidence — the variance is there. RUNG SHAPE
  (named, not fired): CURRICULUM-FUNNEL — the third instance of
  the funnel-controller pattern (precision: FUNNEL-PREC, fired;
  LR: banked hook): order/advance the diet by ENGINE-MEASURED
  ambiguity, with the advance law driven only by in-run sensors
  (held-out tie-rate / margin floor), step index never consulted.
  Cheapest honest arm: two d128 births, SAME data set, different
  ORDER (branching-annealed v shuffled control), standard gate —
  isolates ordering as the one variable. Night-window class,
  needs its own pre-reg + GO. NNUE tie: the engine-side value
  head guiding which states enter the diet is exactly the
  banked proposer/judge lineage, now with a measured quantity
  (n_succ) instead of a learned guess. Attribution: Artin
  (engine-owns-the-estimator + advance-when-graspable + NNUE
  frame); Fable (0A/0C grounding, funnel-controller unification,
  ordering-arm design).

- **2026-08-10 (Artin, night x2): PRECISION GROWS FROM THE LAST
  TRAINED WEIGHT + MAKE THE MODELS ROTATIONAL AGAIN.** First half
  booked as AMENDMENT STAR-PROFILE-3-WARMSTART (warm in-place
  lattice refinement + sensor-triggered advance = the funnel
  pattern's fourth instance). Second half: SYMMETRY-PROGRAM
  REVIVAL ask. Where the program stopped (BOARD: ladder complete
  through 8x): complex 2x sharing costs ~1 solve, quaternionic 4x
  costs ~4 (S1: 61/120 at anti-mass 0.0007), holography's
  absorption edge measured between 50-75% structured deletion.
  WHY TONIGHT RE-PRICES IT: STAR-PROFILE-1 booked grown-not-
  projected for the ALPHABET axis; the symmetry program had
  already found the same law on the SYMMETRY axis (projected init
  craters to 22, warm conversion recovers 61) — same shape, two
  axes, never cross-referenced until now. NEW CELL NAMED (desk +
  one 3080 gate pass, cheap): ROT-x-TERNARY — take the S1
  quaternionic-converted crystal (if ckpt survives) and the plain
  wfloor control, PTQ-ternarize BOTH with the star-profile driver:
  does 4x weight sharing change the tensor-class sensitivity
  profile? The symmetry frame predicts shared-parameter gates
  tolerate the lattice BETTER (fewer free params = coarser
  function already); a null says the two compression axes
  (symmetry, alphabet) are independent tolls. Also pairs the
  revival with RoPE's standing status as the one mainstream
  rotational win (ledger L66 prior). Attribution: Artin (both
  asks); Fable (warm-lattice function-preservation argument,
  funnel-fourth-instance identification, ROT-x-TERNARY cell).

- **2026-08-11 (Artin, post-crown): EQUILIBRIUM AS THE STAR'S
  LIMIT — is 73/73/73 the literal ceiling of that star?** The
  crown battery's champion line landed the SAME total at all
  three birth seeds (73/73/73, dicts near-identical) while the
  ternary line spread (64/62/61). Artin's read: a star in
  equilibrium has found the output its mass and fuel supply
  support — seed-invariance means the (diet, recipe, arch)
  triple DETERMINES the equilibrium, and 73 is that star's
  limit, not a lucky draw. Measured support: the house scaling
  curve is FOOD-limited, not mass-limited — RISING (3ep gen-4):
  d64 38 -> d512 69; UNDERFED giants INVERT it (d768-113M 65 at
  3ep; d1024-200M 49 and d1280-400M 30 on 1-epoch rations) — a
  bigger star with too little fuel burns DIMMER (data-as-
  terminal-limit bank, 2026-08-08). Growth exists and pays: the
  crown's +10.7 IS the grow-inherit mechanism (add shells to a
  burning star, re-feed) — and it only works on the continuous
  substrate (ternary growth non-preserving). SO: size is set by
  fuel (diet dose per param) and burn schedule (ENGINE-SCALE:
  the schedule is the binder), not by the initial mass; the
  seed-invariant 73 says equilibrium is deterministic given the
  triple. OPEN measurable: is 73 stable under MORE fuel at fixed
  arch (a 4th epoch / wider diet on the same grown crystal), or
  does the equilibrium move — "limit of the star" v "limit of
  the feeding". Cheap arm shape: one grown-s2 crystal, +1 warm
  epoch on widened rations, gate — if 73 holds, the triple is
  saturated; if it moves, the star was still hungry.
  Attribution: Artin (equilibrium-limit frame + can-it-grow);
  Fable (fuel-not-mass grounding, saturation arm shape).

- **BANK (2026-08-11): the zeta-bound run as external validation of
  the house method** — Anthropic published (anthropic.com/research/
  riemann-zeta) an unreleased Claude improving the lower bound on
  the proportion of Riemann zeta zeros on the critical line from
  41.6% to 67.2%: 31M output tokens, ~60 subagents, 2,400 shell
  commands, verified TWO-TIER — thousands of numerical checks
  against known zeros (the cheap oracle) plus a Lean proof that
  passed the kernel (the certificate), with subagents reviewing
  proofs and hunting counterexamples (adversarial review). Every
  layer is a house doctrine at industrial scale: oracle-verified
  everything; verified-AND-distinct search; reviewer red-teaming;
  and the FA Law's v2 phrasing (intelligence = rate at which
  verified variance becomes compressed structure) — the result
  itself is explicitly a COMPRESSION of existing lines (Baluyot/
  Goldston/Suriajaya/Turnage-Butterbaugh, Bombieri), which the
  post says plainly, RESULTS-style honesty included ("we don't
  expect these techniques prove RH"). Also the largest public
  instance of the orchestration mode this lab runs nightly:
  human input was mostly "keep going". Useful as: (1) a THEORY
  citation candidate for the verified-search/FA-Law rows once a
  house result wants the pairing; (2) a scale reference for the
  overnight-workflow pattern (60 agents, two sessions — ours cap
  ~15 by config); (3) morale physics: the model needed
  encouragement to keep going — priced-in self-doubt is real in
  the instrument class, not a house anomaly.
  Attribution: Artin (spotted the drop, 13 h old); house (mapping
  to doctrine).

- **BANK (2026-08-11): grok-seat architecture cross-check** — full
  read-only refactor survey (grok CLI plan-mode seat, Artin-run;
  house spot-verified the load-bearing claims: gate constants at
  step_grpo_micro.py:36-39/165, NnueEval duplicated across 17
  scripts/, lab/__init__ exports stale, extraction-spec items 5-7
  still open). Keeps: (1) PRIORITY STACK — lab/gate.py adoption
  (sample_wave_lp + gate_eval + GateSpec with per-lineage constants;
  91-ref hub), lab/hash.py (THREE digest helpers with three
  semantics landed in one night: catalog 1MiB chunks, merge full-sha
  wrong-cwd, runfiles short-sha), lab/jsonl.py (40+ hand-rolled
  sites, spec-counted), corpus-manifest OVERLAY (manifests point at
  existing paths, zero moves — evidence stays in place); (2) SEAM
  CRITIQUE — nothing auto-links merge sidecar -> catalog row ->
  gate row -> marker (booking still hand-glues paths); lake is
  rebuild-only except gates (marker-harvest into build_runs is the
  cheap unifier); two modules named runlog (llmopt/runlog.py
  logging v lab/runlog.py receipts) where the spec wanted ONE sink
  extending the first; (3) honest deferrals — auto-booking prose
  stays human (their #9 matches house doctrine), corpus parquet
  mirrors skipped until a join hurts, AST-level dedup rejected as
  evidence-risk. Attribution: grok (survey + designs), house
  (verification + adoption order).

- **BANK (2026-08-11): opus-chat restructuring cross-check** —
  second external read-only survey (Opus 5 chat seat, Artin-run;
  house verified the load-bearing claim: NO .github/ directory
  exists — zero CI, every ritual enforced by session memory).
  Keeps: (1) "mechanize the ritual" block — CI (pytest + staleness
  regen checks for INDEX/CODEMAP/results-index), doctrine-as-lint
  (any doctrine bullet naming a past incident becomes a test:
  SIGALRM-near-sympy grep, non-string random.Random seed, INDEX
  staleness), GENERATED/HAND headers with regen-or-fail tests,
  annotated verdict tags (verdict/<NAME> at booking commits —
  additive, makes SHA citations resolvable); (2) rjob-refuses-
  without-prereg_ref — pre-registration becomes enforced, not
  doctrinal; (3) byte-identical round-trip proof as the gate IF
  RESULTS.md ever inverts to generated-from-ledger. Rejected:
  scratch git-mv tiering (conflicts frozen-evidence doctrine —
  CODEMAP is the move gate, cited files stay in place on purpose);
  generated handoffs (handoffs carry intent/narrative, state
  already lives in BOARD/RESULTS); fewer-writers subagent posture
  (record shows opposite: 14-agent build + 4-reviewer sweep caught
  the d512 arch error via three independent routes — CI is the
  fix, not fewer minds). Attribution: Opus 5 chat (survey), Artin
  (relay + subagent pushback), house (verification + triage).

- **BANK (2026-08-11): proposer/verifier control pattern x house
  stack (Artin x grok-prompt riff)** — Artin's observation: Qwen
  CoT control pattern (tokenize -> expand breadth -> iterate) is
  speculative decoding's shape, and the interesting case is the
  small model STOPPING or forcing a rethink. House mapping,
  methods-level: (1) the gate loop IS small-proposer/large-verifier
  (wave-of-8 at T=0.7 proposes, sympy verify_wave accepts, visited
  set forces rethink, no-candidate = STOP — shipped instrument,
  not analogy); (2) spec-decoding=gate-law + entropy-adaptive
  draft length already banked; (3) fp16 margin<=0.02 tie doctrine
  = the measured "don't trust the draft here" signal; (4)
  ATTRACTOR-0/FUNNEL-PREC absorption = cheap-model-inside-the-
  funnel, big-model-at-branch-points. NEW weight-space leg opened
  by MERGE-SPACE-3/5 (init-is-the-address, REPLICATED):
  LINEAGE-DRAFTING candidate rung — d64 proposes, d512 verifies,
  acceptance rate v shared-lineage (grown-from v independent
  init); micro-star family prices it at an afternoon. Fences
  banked with it: oracle stays in the accept path (no
  logit-agreement acceptance — identity-rewrite hazard),
  cross-device draft/verify never compares, draft models earn
  slots by measured wall/acceptance only. Attribution: Artin
  (pattern + question), grok (prompt sharpening), house (mapping
  to booked results).

- **BANK (2026-08-11): Nemotron-3.5-Lightning reference point** —
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (HF, updated
  2026-08-11): nemotron_h hybrid Mamba-Transformer, 30B total /
  ~3B active (A3B), NVFP4 4-bit weights. House hooks: fp4 experts
  at production scale v C-series measured 6.65 bits/weight
  (V4-Flash MoE scoping bank); A3B sparse activation sits in the
  expert-size-law regime (B0-B2 atlas). Reference bank only — no
  run implied. Attribution: Artin (spot).

- **BANK (2026-08-11, addendum to the Nemotron reference point):
  Nemotron-3.5-Lightning config census v house MoE results** —
  config.json read in full (no weights run; Mac owned by
  growdecomp, no MLX quant exists yet, transformers 5.8 v the
  Mac pin <5). It is the MOE-GT NAMED FOLLOW-UP CLASS
  (shared-expert + V3-style router — what DeepSeek-V2-Lite was
  queued for and never ran) at exactly the Qwen3-30B-A3B skeleton:
  128 experts/layer, but top-6 not top-8 (4.7% v 6.25% active),
  +1 always-on shared expert at 2x width (overlap true), groups=8
  topk_group=1 (concentration BY FIAT where house measured it
  emerge demand-ranked), 23 MoE + 23 Mamba + only 6 attention of
  52 layers, 1M context. FOUR house-result echoes, methods-level:
  (1) shared expert = the verbal-fallback/generalist channel
  (MOE-GT 0.755 Spearman) made architectural; (2) NVFP4 allocated
  BY COMPONENT — routed-expert mass FP4 g16, mamba mixer FP8,
  control machinery above = FUNNEL-PREC's cheap-interior/
  precise-decision-points shape at production; (3)
  num_nextn_predict_layers=1 MTP head = internalized proposer
  (accept path is logit-side, no oracle — house fence stands);
  (4) thin-experts regime (1856/2688 = 0.69x hidden) inside the
  B0-B2 expert-size-law band. Star-frame echo: dense resident
  core (shared expert + Mamba backbone) + thin routed corona —
  mass cheap and fixed, control precise. RUNNABLE RUNG WHEN IT
  OPENS: MOE-GT routing census on it — gated on the mask_router
  shared-expert/group-routing extension (the flagged V2-Lite
  mismatch, still unfixed). Attribution: Artin (spot + "do we see
  our predictions" framing), house (config census + mapping).

- **BANK (2026-08-11): SSM-STAR candidate rung (Artin: "Mamba
  state space models? Haven't tried that")** — house has ZERO SSM
  results; FLOOR-HK-1 (27055) makes the first rung sharp. The
  instrument question: attention with a 512-token window measures
  k_eff ~16-17 on the warm diet (architecture-bound); a selective
  SSM carries a FIXED state it must compress history into — the
  opposite inductive bias. Rung: minimal selective-SSM block
  (~100 lines pure PyTorch, no kernels needed at micro scale) in
  the micro-star family, d64-class, same diet, floor + gate +
  H_k placement. Three-way read, all informative: same H_16 wall
  (the wall is the diet's), crosses toward H_32 (recurrence
  extracts what attention leaves — big), lands above (SSM micro-
  scale toll, honest loss). Evening-priced on the 3080; pre-reg
  on GO. Attribution: Artin (spot), house (tie to FLOOR-HK-1).

- **BANK (2026-08-11): SSM x axiom fork (Artin: "faster if
  mac-axiom builds it in axiom?")** — for SSM-STAR-1 itself: no
  (wall = Python-loop kernel-launch overhead, 2.2 v 33 it/s; rung
  finishes in ~2.5 h; relay cycle is days). Two post-verdict
  cases where axiom earns it: (1) ladder-scale SSM program ->
  house chunked associative scan first (~10x, one evening), fused
  axiom kernel after; (2) THE AXIOM-SHAPED RUNG: exact integer
  recurrence — an SSM compounds state every token, the worst-case
  fp-error accumulator, exactly their exact-v-fp divergence
  instrument class (P-EXACT-TIE lineage). Ask: where does fp32
  recurrence first diverge from exact, and does it ever flip a
  gate decision. Relay drafts only AFTER SSM-STAR-1 books (bars
  first). Attribution: Artin (fork question), house (triage).

- **QUEUED (2026-08-11): paper refresh — numbers current + figures**
  (Artin: "bring numbers current + figures"). `docs/paper-prose-v1.md`
  (405 lines, 2026-07-30) is the entropy-bound packing paper;
  `docs/paper-draft-entropy-bound.md` (237) is its skeleton + cite
  registry. It says "Figures TODO'd inline" and predates six weeks of
  ledger. THE JOB, in order: (1) re-verify every number against the
  current RESULTS — the 08-11 README audit found three claims in a
  much shorter document that did not survive, so assume the paper has
  its own; the packing boundary (house crystals hold, Qwen2.5-0.5B
  does not, both n=1) is the section most likely to have moved;
  (2) wire figures from the NEW system — `docs/figures.json` +
  `llmopt/lab/figsvg.py` emit paper-grade SVG, so a paper figure is a
  spec entry, not a one-off; the packing boundary wants a two-panel
  "where the law stops" figure that does not exist yet; (3) decide
  whether the routing crest belongs in this paper or its own — it is
  the strongest result in the repo and it is not about quantization.
  FENCE: the paper is a claims document, so it gets the prereg-auditor
  pass BEFORE any submission, same as the README got. Do not start it
  in a session that is also shepherding runs — it needs a clean read
  of 27k lines of ledger. Attribution: Artin (scope call), house.

- **BANKED (2026-08-11 night): the stellar-collapse frame for training
  — gravity/pressure/Jeans as loss/entropy/width-floor** (Artin, riffed
  live off "how are stars formed"; house mapped and fenced). The
  mapping, with measured anchors: gravity = the compression pull of
  the loss (FA Law v2 — verified variance -> compressed structure);
  thermal pressure = entropy (init noise, gradient noise, and the
  corpus H_k wall the FLOOR-HK-1 d512 floor sits on at 0.3478 v
  H_16=0.367); Jeans mass = the width floor (sub-d56 births never
  ignite — the standing brown-dwarf class; MICRO-STAR-1's "ignition"
  naming was already this frame); supernova-seeds-new-stars =
  distillation; softmax temperature = NOT an analogy, the same
  Boltzmann form p ∝ exp(logit/T); inference = the frozen
  deterministic attractor (oracle token-identity is the founding
  invariant; V4-F1d's booked repetition attractor is an event horizon
  in trajectory space — trajectories that enter never exit). WHERE IT
  BREAKS (fence, so the bank stays honest): no fusion — training has
  no new energy source at ignition, capability is a compression
  threshold, not power generation; and the 08-11 MPS it/s decay is an
  allocator, explicitly NOT stellar physics. Testable residue worth a
  rung someday: the n-ball/cube volume ratio pi/6 -> 0 as d grows
  (corner-flight) as the WHY of the capacity-meter split — crystals
  at-capacity (M 0.96-1.61, shell-concentrated) v web-dense outlier
  tails (M 3.6-3.9, corner mass); a per-width corner-mass census
  would make the geometry claim measurable. Attribution: Artin
  (frame + black-hole determinism ask), house (anchors + fences).

- **BANKED (2026-08-11 night, extension of the stellar-collapse bank):
  metallicity = data refinement across model generations; the lab as
  a max-metallicity nursery** (Artin: "what if there was no dust and
  you wanted to snap-create a star"; house: Population III). The
  mapping: Pop III stars formed from ZERO-metal primordial gas —
  cooling was so poor they had to be 100-1000 solar masses to ignite,
  then their supernovae MADE the metals that let every later
  generation form small and efficient. Reading: first-gen web-scale
  LLMs = Pop III (raw unrefined corpus, enormous ignition mass);
  distilled/synthetic corpora = metal-enriched clouds -> small
  efficient stars; llmopt's oracle-verified synthetic diet = maximum
  metallicity, which is WHY 19M-class objects ignite here.
  Snap-a-star has a MEASURED no: init-is-the-address (MERGE-SPACE
  n=3) gives you the seed, but init alone gates ~0, task vectors
  crater 1/120, foreign averages 0/120 — collapse (training) is not
  skippable; you snap addresses, not objects. The ignition-mass
  residue is now MEASURED (VERDICT METALLICITY-1, 2026-08-11 night,
  3080): the width-floor ladder run across four refinement grades of
  the same content gives ignition widths w(z3 verified) = 56 and
  w = none for the polluted, Pop-III-duplicated, and token-shuffled
  vacuum grades — only the verified diet ignites on a d{32,48,56,64}
  ladder at all. That is a real number on "metallicity", but a
  PARTIAL one: z1 and z2 tie at "none", so the monotone ladder is
  tested at one of its two steps and the pollution-vs-duplication
  contrast stays open. Fences that travel with it: constructed
  refinement only (answer-shuffle + duplication), not the raw farm
  stream — that grade remains banked as cell B. Remaining residue:
  fusion-rate instrument, solves per diet token (the FA Law rate),
  still not first-class. Attribution:
  Artin (no-dust ask, fusion-as-efficiency), house (Pop III anchor,
  measured fences).

- **BANKED (2026-08-11 late night): the division-algebra ladder as a
  weight-alphabet axis — quaternion crystals as the unclimbed dim-4
  rung** (Artin, chasing a remembered "closed system equation" through
  {0,1,2,4,8,16}; house identified Hurwitz). The math: normed
  division algebras exist ONLY at dims 1/2/4/8 (R, C, H, O —
  Cayley-Dickson doubling; each step pays a law: ordering,
  commutativity, associativity; dim 16 loses division itself). The
  house already sits on the ladder's first two rungs MEASURED: real
  crystals everywhere, and dim-2 via the cplx family (G5 POLAR
  anisotropy break: rotation hurts only the star crystal), the
  roots-of-unity Fourier probe, and the polar-v-Cartesian
  quantization null (geometry-blind at matched bits). The rung this
  banks: QUATERNION-WEIGHT crystals — weights as dim-4 hypercomplex
  units, the Hamilton product replacing scalar mul — does
  capability-per-param or the quantization knee move when the
  alphabet loses commutativity? Published lineage exists (quaternion
  NNs: Parcollet et al. 2018; Zhu et al. 2018), so a THEORY row is
  reachable if a result lands. Fences to carry: the polar null says
  GEOMETRY of the grid is free at matched bits — the quaternion
  question must be about the MULTIPLICATION structure, not the
  coordinate system, or it re-runs a booked null in disguise.
  Attribution: Artin (ladder ask, QM-complex-plane thread), house
  (Hurwitz identification, fence).

- **CORRECTION (2026-08-11, minutes later) to the division-algebra
  bank above: dim-4 is NOT unclimbed — the house has a measured
  quaternion program.** A confluence sweep (verified line-by-line
  against RESULTS) found: S1 quaternionic CONVERSION at the bar
  (61/120 at anti-mass 0.0007, 4x sharing, RESULTS 8682); the 4x
  toll REAL at n=3 (P-TOLL-REAL pooled -17, 3/3 negative, RESULTS
  23168); symmetry x alphabet orthogonal (ROT-X-TERNARY 26031); and
  the standing nulls a dim-4 rung must respect: no spontaneous
  quaternionic structure (8665), alphabet-follows-domain dead (ZX
  column 6652), bits geometry-blind (10238/10334), zeta-8 imposition
  neutral-to-negative (B6 12914), commutant lenses blind to
  activation clocks (TIER-A A3). What remains genuinely OPEN on the
  ladder: (a) BORN-quaternion births (all measured cells are
  retrofit conversions of real crystals; the alphabet-tournament
  spec's Q9 unit alphabet at 3.17 bits is designed and unrun), and
  (b) the ladder-as-ladder question — one pre-reg spanning dims
  1/2/4/8 born-matched at equal real DOF, which no single booked
  cell covers. Instruments ready: quat_commutant/quat_convert
  (frozen), complex_ffn (promoted), p3_quat n=3 wrapper pattern.
  Attribution: house sweep + line verification; the "unclimbed"
  claim above is retracted.

- **BANKED (2026-08-11 night): the optimal input vector — is there
  capability behind the tokenizer?** (Artin: "what if we just strip
  away the data completely and just pass in random vectors"). The
  riff: a model's input is not obliged to be text. If a learned
  CONTINUOUS prefix raises the gate on frozen weights, then there is
  capability the tokenizer's discrete alphabet cannot reach, and the
  alphabet — not the weights — is the binding constraint. Realized as
  P=8 virtual tokens: the embedding gains 8 trainable rows, every
  original weight frozen, output logits for the virtual ids clamped so
  decode can never emit them. FIRST ATTEMPT BOOKED
  INSTRUMENT-INVALID (VERDICT SOFT-PROMPT-1): the mechanism control
  fired, and the diagnosis is sharper than the bar required — the
  harness perturbs the model with NO prefix present. Rebuilding at
  vocab V+P and copying rows [:V] was meant to be a no-op on ordinary
  ids; the same checkpoint reads 14/120 stock (reproduced twice, same
  in-process sha) and 10/120 inside the harness. The gate itself is
  deterministic given weights, so this is the harness, not noise.
  DIAGNOSED same night (AMENDMENT SOFT-PROMPT-1-SAMPLER): the model is
  bit-exact and the defect is in the SAMPLER — the eight appended
  zero-probability logit columns leave the distribution identical but
  change how many random values torch's multinomial consumes, so a
  generator reused down a rollout desynchronizes from the second token
  on. The riff is UNTESTED, not refuted — nothing here speaks to
  whether continuous inputs carry reachable capability. A re-run needs
  the head sliced to V categories on the gate path (masking gets the
  probabilities right and the random stream wrong), plus the booked
  precondition: reproduce the stock gate dict cell for cell before a
  single prefix number is read. Attribution: Artin (the ask), house
  (mechanism, the control that caught it, and the diagnosis).

- **BANKED (2026-08-12): volatility drag as a data-quality frame —
  two sigmas, not one** (Artin). The mapping / the math: compounded
  return ≈ arithmetic mean − σ²/2, so variance taxes a compounding
  process directly even at equal average; training compounds too
  (each step multiplies into the next parameter state), and SGD
  convergence bounds carry an explicit gradient-noise σ² term, so
  cleaner/more-accurate data = lower gradient variance = less drag.
  Artin's original form: better data minimizes training volatility,
  but you might not learn everything, because you are limited to the
  data you take in. Measured anchors: no booked llmopt result
  measures the drag term itself; nearest house evidence is
  qualitative — underdetermined rows train hallucination (data
  hygiene doctrine) and the two contamination incidents that forced
  exclude=-guarded splits + "widen the generator space before
  trusting a split". Honest breaks: in markets variance is pure
  cost; in data, part of the "variance" IS the signal. The frame
  only works split into two sigmas — variance of the ERROR TERM
  (wrong/underdetermined labels: always a tax, always cut) vs
  variance of the SUPPORT (input/problem diversity: coverage — cut
  it and the low-drag trajectory converges onto a narrow manifold
  and caps what is learnable). Markets have only the first; datasets
  have both, which is why the analogy feels right and slightly off
  at once. Testable residue: (1) hold support fixed, inject label
  noise at rate p into a mathgen diet, measure gate vs p — drag
  predicts monotone loss with no diversity compensation; (2) hold
  error rate at 0, shrink generator support, measure eval-gate on
  held-out wide support — coverage predicts the cap. Fence for the
  quality program: any diet-curation report carries BOTH an
  error-rate number and a coverage number, never one alone.
  Attribution: Artin (the frame, twice — raised once before and
  under-credited), house (the two-sigma split and the residue).

- **BANKED (2026-08-13): the Stockfish-NNUE pattern generalizes — a
  verified engine's scored experience trains a fast learned ranker,
  and "AGI-shaped" systems may be many such engine+ranker pairs
  behind a router, not one self-verifying net** (Artin, from a Grok
  exchange; Grok's counter folded in). The mapping / the math:
  Stockfish = rules + search + exact-ish eval in ONE closed domain;
  NNUE = small net distilled from the engine's own scored positions.
  The lab already instantiates the shape: oracle-verified search,
  rankers grown from engine traces, engine<->model exchange. Artin's
  extension: an engine that could price any state/action in the
  universe would let you train an LLM the same way; and none of
  Stockfish/lc0/AlphaZero SOLVED chess yet all passed humans — so
  superhuman-per-domain does not require solving the domain.
  Measured anchors: none for the AGI claim (none possible here);
  house anchors for the pattern itself are the engine->ranker rungs
  and the ZX/math engine lineage in RESULTS. Honest breaks (Grok's,
  accepted): open action spaces have no closed move set; most
  domains lack cheap sharp oracles; an LLM imitates many domains but
  ships no verified moves or global checker — it is a wide prior
  over traces, not an oracle. The AGI-shaped remainder is exactly
  the unsolved part: choosing which engine applies, KNOWING when
  none apply (honest abstention), and acquiring new oracles when
  the world is not formal. Multi-engine interaction (physics x
  quantum-circuits x math) is an open interface question: shared
  state representation vs data-level coupling, and the data-diet
  question rides on it. Testable residue: house-scale only — when
  two house engines (e.g. math + ZX) share a training diet, does a
  single ranker trained on the union beat two specialist rankers at
  equal parameter budget? That is a rung, not a philosophy. Charter
  fence: engines stay math/physics only, per the lab charter.
  Attribution: Artin (the frame and the multi-engine question),
  Grok (the three-part break), house (the union-ranker residue).

- **BANKED (2026-08-13): training is a double pendulum in momentum
  space — birth energy is a controlled chaos knob, and the model's
  weight trajectory is a phase-space object worth rendering
  directly** (Artin, from 2swap's "Double Pendulums are Chaoticn't";
  the video is also the adopted visual north star for house
  animations, storyboard doc 2026-08-13). The mapping / the math: a
  double pendulum's momentum-space portrait separates ordered
  (low-energy, quasi-periodic) from chaotic (high-energy) regimes
  with a visible boundary; training = a trajectory through weight
  phase space where the "starting speed" (init scale, birth
  precision class, lr) sets which regime the dynamics explore, and
  the useful region may sit near the order-chaos boundary. Measured
  anchors: none direct — nearest house results are init-is-the-
  address (independent-init merges gate 0/120 x6, REPLICATED n=3;
  BOARD 08-11) showing init selects a basin the way pendulum initial
  conditions select an island, and ATTRACTOR-0's single-basin
  reading; neither measured an order-chaos BOUNDARY. Honest breaks:
  the pendulum has 2 DOF and a conserved Hamiltonian; SGD is
  dissipative, stochastic, and ~10^7-dimensional — no conserved
  energy, no Poincare section without a projection choice, and
  "higher lr = more chaos" is a loose association, not a measured
  Lyapunov statement. Chaos here is a metaphor until a divergence
  rate is actually measured. Testable residue: (1) twin births at
  epsilon-separated inits across a birth-energy ladder (init scale
  or lr), measure weight-space divergence rate per step — does a
  sharp boundary exist in the knob, and does best capability sit
  near it? (2) render ep0/ep1/final checkpoint trajectories in a
  fixed projection (the training_morph fixed-basis instrument
  already exists) and look for regime structure. Attribution: Artin
  (the frame and the render-the-models ask), 2swap (the visual
  reference), house (breaks + residue).

- **BANKED (2026-08-13): the static figure stack has well-reasoned
  FORMS but uncomposed COMPOSITIONS — four named repairs (render
  driver, anim guards, paired-slope form, palette ruling)** (Artin
  asked for the assessment and for these to be banked; house
  diagnosis). The mapping: `llmopt/figures/figures.py` (matplotlib,
  analysis) and `figsvg.py` (hand-emitted SVG, published) ship a
  FORM CATALOG (gate_bars/curves/ladder/scatter/stat) that answers
  "which chart", while the animation work of the same day was handed
  a LAYER GRAMMAR (field/actors/memory/geometry/text/receipt) that
  answers "what dominates the frame". Every static figure therefore
  carries uniform ink weight with no foreground/recessive split.
  Measured anchors (verified in-session, 2026-08-13): (1) NO render
  driver exists for animations — `scripts/render_gallery.py` and
  `render_hero_neurons.py` cover statics, the anim render/GIF/poster
  recipe lives only in `docs/assets/README.md` prose, and a zsh
  word-splitting bug silently rendered and copied NOTHING in this
  session's first triplet attempt (caught by reading output, not by
  a guard); poster timing is hand-computed as `duration - 3.4`,
  which breaks when any beat length changes. (2) ZERO tests
  reference the anim pipeline (`grep -rln anim tests/` empty) while
  statics carry test_asset_links, the gen_readme honesty recount,
  and the frozen-ramp test; the committed `data/anim/*.npz` bake in
  `head: aff9247` and cite checkpoint shas nothing re-verifies. (3)
  `docs/figures.json` routing_crest now carries `seed_pairs`
  (63->80, 73->82, 63->81, verified against RESULTS.md#L18927) that
  NO renderer can draw — `gate_track` still emits "mean of 3" rails,
  so the published figure shows the weakest form of its own
  evidence. (4) two palettes ship side by side: SERIES_LIGHT/DARK
  categorical (#2a78d6/#eb6834/#1baf7a) in statics v the inferno
  magnitude ramp in animations. Honest breaks: the two-renderer
  split is DELIBERATE and documented (matplotlib for paper-grade
  vector, SVG for web-grade detail) — this bank does not propose
  merging them; the Chrome-dependent PNG capture is likewise a named
  Mac-only choice, not drift; and "uncomposed" is a craft judgement,
  not a measurement — no reader study backs it. The palette split is
  DEFENSIBLE on its face (categorical encodes arm identity,
  continuous encodes magnitude) and the bank asks for an explicit
  ruling, not a presumed error. Testable residue: (a) a guard that
  re-hashes every checkpoint cited by a committed npz and fails on
  mismatch is runnable TODAY and either fires or does not — that is
  a real bit of information about whether shipped animations match
  their claimed weights; (b) paired-slope v mean-rails is the one
  composition claim with a cheap empirical test — show both forms of
  routing_crest and ask a reader what the evidence is; if the answer
  does not change, the composition thesis is weaker than claimed.
  Attribution: Artin (the ask, the banking instruction), house (the
  diagnosis and the four repairs).

- **BANKED (2026-08-13): is gate difficulty just character count —
  and do hard problems compute measurably different internals than
  easy ones?** (Artin, mid-CAP-V-TRAJ-1 watch). The ask, two halves:
  (1) maybe level difficulty reduces to integrand LENGTH (a surface
  variable), not structure; (2) compare the forward pass's actual
  computed tensors (activations, attention, hidden states) across
  levels — "what really is different, bit-wise" between an easy and
  a hard problem inside the model. Measured anchors: the CAP-V-TRAJ-1
  stream (in flight, unbooked at bank time) shows L4 as the WORST
  level at every gated milestone (1..7/24, below L5-L7 always) even
  though L6/L7 are constructed to demand more coordination/nesting —
  so rank-order of measured difficulty already disagrees with the
  generator's intended ladder at one level, and SOMETHING other than
  "more steps = harder" is operating. No activation-space
  measurement exists yet for the 19M gate lineage; the closest
  instrument is the watch-it-think flagship (per-token internals
  recorder, endorsed, unbuilt). Honest breaks: "bit-wise" is not the
  right granularity — bf16 near-ties make raw bit comparisons noise
  (the fp16 near-tie non-bug, equivalence.py doctrine); the honest
  units are activation statistics and per-level solve outcomes, not
  bits. And length correlates with level by construction, so a raw
  length-vs-solve regression is confounded; the informative cell is
  WITHIN-level: does length predict failure inside one level's
  problem set? Testable residue, cheap to dear: (a) desk-only, free
  — regress per-problem solve outcome on prompt token length within
  each level from a per-problem gate sidecar (gate_percase.py
  machinery exists); if length explains the L4 dip, L4's long
  product-rule debris strings are the story, not the shape family;
  (b) one forward pass per gate problem, record per-layer hidden
  norms + attention entropy, test whether level (or solved-vs-not)
  is linearly decodable from the summary stats — rides the
  watch-it-think renderer's capture hook when it lands; (c) the
  sharp version: paired problems, same skeleton, padded length —
  does capability drop with length at fixed structure? Attribution:
  Artin (both halves of the ask), house (the confound note and the
  three-rung ladder).

- **BANKED (2026-08-13): training as the star's BIRTH — pre-training
  and post-training as a pincer onto the model, and the ask "can we
  train backwards? not backprop"** (Artin, right after CAP-V-TRAJ-1
  booked). The frame: the star riff (2026-08-09, measured by
  STAR-PROFILE-1) treats the trained model as stellar structure;
  this extends it in TIME — training is the collapse/birth event,
  and pre-training + post-training squeeze the model from two sides
  the way the temporal-pincer bank (row 22) runs forward and
  backward LLMs at a proof. "Train backwards" = invert the ORDER of
  the training program, not the gradient: RL/post-training FIRST
  from birth, diet SFT after — or a mirrored schedule (OneCycle
  reversed, cold-to-hot). Measured anchors: CAP-V-TRAJ-1 (capability
  leads the settling; the schedule tail buys ~1 solve — so the
  ORDER/schedule of training visibly shapes when capability
  arrives); v2-GRPO run 1 headline 1 (diet ~20x cheaper per solve
  than RL on this substrate — the two pincer jaws have measured,
  unequal price); 45M-GRPO (57 -> 61) and v2 19M (57 -> 60) — both
  bases climb under RL, modestly. Honest breaks: "backwards" is not
  time-reversal of SGD (nothing un-integrates a trajectory — the
  phase portrait is dissipative, ~100x speed decay, PHASE-PORTRAIT-1);
  the star-birth analogy carries no equations here, unlike the
  star-profile bank which cashed out in tensor-class sensitivity;
  and RL-from-birth was already run once (GRPO run 2b lineage
  started near-birth v1) though never as a REGISTERED order-inversion
  pair. Testable residue: the order-inversion rung — matched total
  budget, same diet, same substrate, two arms: (A) SFT-then-GRPO
  (standard), (B) GRPO-from-birth-then-SFT (backwards), same gate.
  If (B) lands within sigma of (A), training order is commutative at
  this scale and the pincer jaws are interchangeable; if (A) >> (B),
  the curriculum's arrow is real. Cheap version: reversed OneCycle
  (anneal-then-warm) vs standard, same diet/seed, one birth each.
  [MEASURED same day: VERDICT BACKWARD-SCHEDULE-1 — the cheap arm
  ran and the arrow reading DIED at n=1: backwards birth gates
  62/120 vs standard 64, within sigma; schedule direction is a
  non-factor at 19M/gen4. The full SFT/GRPO order-inversion arm
  remains open, with the newborn-mines-nothing hazard fenced in
  its pre-reg-to-be.]
  Attribution: Artin (the frame and both asks), house (the
  order-inversion operationalization + anchors).

- **BANKED (2026-08-13): "why can't training be n log n — skip a
  majority of the phases?"** (Artin, watching the phase_portrait
  animation mid-BACKWARD-SCHEDULE-1). The frame: the settling looks
  structured, so most of it should be skippable. Honest breaks
  first: the visible smoothness is schedule-IMPOSED (OneCycle drives
  the ~100x deceleration — PHASE-PORTRAIT-1's own fence), and SGD is
  a recurrence — step N needs the weights step N-1 produced, so
  there is no sublinear jump in the complexity sense; cost is linear
  in steps walked. The mapping that survives: SKIPPING is
  impossible, but TRUNCATING/COMPRESSING may be nearly free —
  measured anchor CAP-V-TRAJ-1: m010800 gates 62/120 vs final 64 (a
  tie under the sigma fence), so the last ~30% of the trajectory
  bought ~1-2 solves. Measured anchors: CAP-V-TRAJ-1 (capability
  leads settling, rho 0.979, cap90 at step 10,800);
  PHASE-PORTRAIT-1 (monotone collective deceleration). Testable
  residue: the SCHEDULE-COMPRESSION LADDER — same recipe/seed/diet,
  OneCycle rescaled to {70%, 50%, 30%} of total_steps (one birth
  each, ~25-50 min/arm on Mac), standard 120 gate at each endpoint,
  paired against the booked 64/120. Bars would mirror
  BACKWARD-SCHEDULE-1's (within-1.5-sigma = compression free at that
  ratio; > 7 below = the wall found). Distinct from truncated
  training (stop early on the SAME schedule = never reach low-lr
  polish) — compression keeps the schedule SHAPE; both arms
  interesting, name them apart. Related: BACKWARD-SCHEDULE-1 (in
  flight at bank time) is the ORDER half of the same pincer.
  [MEASURED same day: VERDICT COMP-LADDER-1 — the ladder ran at
  {0.5, 0.3}: half-schedule is FREE (60/120 vs 64 at half the
  wall), 0.3x breaks (49), and compression beats matched-steps
  truncation by 9 — the shape's anneal tail is the load-bearing
  part. The skip intuition cashes out as: you cannot skip the walk,
  but half of it was padding at this scale.]
  Attribution: Artin (the ask, the skip frame), house (the
  recurrence break + compression-ladder operationalization).

- **BANKED (2026-08-13): the ENGINE trains the model — build a tree
  over the weights' phase space and lead the way; "take all the data
  at once, compute the entire layout"** (Artin, same evening as the
  n-log-n bank; the sharpened form of it). The frame: training looks
  like a deterministic path chosen by (init, data); the engine
  already GIVES us the data; so let the engine compute the
  destination instead of the model walking there. Measured anchors,
  and they are strong today: BACKWARD-SCHEDULE-1 (the lr sequence
  played backwards lands 62/120 vs 64 with the SAME L4=7 dip — the
  destination barely depends on the path); CAP-V-TRAJ-1 (capability
  built by ~70% of the walk); init-is-the-address REPLICATED n=3
  (the init picks the basin); weight-reader 88.4% (function IS
  readable from weights when permutation-invariance is taught).
  Honest breaks: (1) no closed form exists for nonconvex deep nets
  — "all data at once" solves linear models only; SGD is the
  solver, not overhead; (2) a tree keyed on RAW (weight, magnitude)
  coordinates indexes gauge garbage — permutations/rescalings
  preserve the function (the never-score-by-weight-distance
  doctrine), so any such structure must be keyed on FUNCTION
  (oracle behavior), not coordinates; (3) the engine already trains
  the model in the weaker sense — expert iteration is
  engine-generates-verified-data — so the bank's genuinely new claim
  is ENDPOINT PREDICTION, not data generation. Testable residue:
  AMORTIZED-BIRTH at d64 scale, where dozens of booked births
  exist: learn a predictor (init, diet-id) -> final weights across
  seed pairs, score the PREDICTED weights by running them against
  the standard gate (oracle-scored, never weight distance; the
  weight-reader permutation lesson applies to the target too). A
  cheaper first rung: predict final FUNCTION only (per-level gate
  profile) from the init + first-K-step trajectory summary — if
  even the profile is unpredictable from the early walk, the full
  endpoint is dead on arrival. Schedule-invariance (today) makes
  both better-posed: the target is ~f(init, data), one fewer
  argument than feared. Attribution: Artin (the frame, the tree,
  the all-data-at-once ask), house (the gauge break + the
  amortized-birth operationalization).

- **BANKED (2026-08-13): "stream random inputs and only listen to
  the engine when the input matrix is correct"** (Artin, asking why
  schedule direction is a non-factor; his own "not sure" kept).
  Two readings banked side by side, both attributed as asked. (a)
  RANDOM DATA, ENGINE FILTER: stream random/self-generated
  candidate rows, train only on the ones the engine verifies — this
  already exists and is doctrine (the miner + verified-AND-distinct;
  expert iteration IS engine-filtered data). If this is the
  reading, the residue is a DIET-COMPOSITION question: how much
  random-source exposure can the verified filter launder before
  quality drops (rations discipline). (b) RANDOM UPDATES, ENGINE
  ACCEPTANCE — the sharper reading: perturb weights randomly,
  accept the step only when the oracle/gate approves; gradient-free
  training with the engine as fitness. Measured anchors: none
  house-side for (b) — genuinely unrun; adjacent evidence is
  BACKWARD-SCHEDULE-1 + COMP-LADDER-1 (the gradient path's details
  matter less than believed, which makes non-gradient paths less
  crazy) and STAR-PROFILE / MICRO-STAR (tensor-class structure
  exists that a structured perturbation could exploit). Honest
  breaks: zeroth-order search pays variance linear in parameter
  count (19M dims of noise per accepted bit of signal — naive form
  is hopeless at this scale); the gate is expensive as a fitness
  call (~68 s), so acceptance must batch or use the cheap proxy
  tier; and (a)-as-stated is already running doctrine, not a new
  lever. Testable residue (for (b), the new half): a d64-scale
  probe on an existing crystal — structured perturbations (one
  tensor-class at a time, the star-profile axes) accepted by the
  cheap-tier gate; bar = any measurable climb above the paired
  no-perturbation control within a fixed fitness-call budget. If
  even d64 cannot climb, the reading dies cheaply. Attribution:
  Artin (the ask, both halves latent in it), house (the two-reading
  split + the d64 probe operationalization).

- **BANKED (2026-08-13): pincer on the LLMUE during training —
  metabolize while being born; and "the LLMUE shouldn't have a
  problem being trained backward?"** (Artin, mid-CAP-V-TRAJ-2
  sweep; explicit revival of the valuation-routed-metabolism bank,
  2026-07-21). The frame, three threads tied: (1) VALUATION-ROUTED
  METABOLISM (revived on ask): engine value tables (Markov-prior
  rule weights, magic hardness, slot-decisive telemetry) composed
  with the committee rule->neuron map = a per-neuron plasticity
  mask; the metabolic loop applies it per micro-batch. (2) The
  same-day lr-floor measurement constrains it: CAP-V-TRAJ-2 (in
  flight, curve 0/0/1/18/35/48/59 at booking of this bank) shows
  steps below an lr absorption floor buy NOTHING at birth, and the
  LLMUE pilot already showed the twin (LR 1e-5 sub-threshold diet:
  preserved without growing, RESULTS L3174-3179) — so any
  valuation-routed mask must put its routed neurons ABOVE the floor
  or it is a null by construction (the starved-judge law, now with
  a measured mechanism). (3) The new jaw — CONCURRENT PINCER:
  don't sequence pre-training then metabolism; overlap them —
  birth runs, and once capability clears the mining floor (the
  newborn-mines-nothing hazard, fenced in BACKWARD-SCHEDULE-1's
  pre-reg), the engine's verified exhaust streams INTO the
  remaining birth diet live. Honest breaks: "LLMUE backward" —
  BACKWARD-SCHEDULE-1 measured schedule-direction invariance for
  BIRTH SFT only; nothing is measured about reversing a metabolic
  schedule, and the metabolic loop has no OneCycle to reverse (its
  lr is flat-low, which is exactly why the floor bit the pilot) —
  the transferable claim is only "order matters less than
  believed", not a license; the concurrent arm changes diet
  composition mid-run (exposure-share discipline applies — rations
  for resident grammars); and self-generated rows must pass
  verified-AND-distinct or the identity-hack returns. Testable
  residue: OVERLAP-BIRTH-1 — paired arms, same seed/device: (A)
  standard birth (the booked 64 rebuilt or reused), (B) identical
  birth except from the first milestone where the model's gate
  clears ~30/120 (mining becomes possible; the measured takeoff
  region), a rationed stream of its own oracle-verified exhaust
  joins the diet for the remaining steps. Bar family:
  within-sigma = concurrency free (pipeline win, no capability
  cost); above = the pincer pays; below = self-exhaust dilutes.
  Attribution: Artin (the concurrent-pincer ask, the backward
  question, the revival instruction), house (the lr-floor tie-in +
  the overlap-birth operationalization).

- **BANKED (2026-08-13): "we couldn't repair the internet-trained
  model because RL never moved the original weights — what if we
  made the weights LOOSER, let them vary more?"** (Artin, session
  close, off the lr-floor verdict). The frame: early-lab attempts
  to repair/steer internet-pretrained models with RL failed to move
  the base weights; today's CAP-V-TRAJ-2 gives that failure a
  candidate mechanism — the effective plasticity sat BELOW the
  absorption floor (the LLMUE pilot's 1e-5 null is the measured
  twin; RL fine-tuning lrs are conventionally in that sub-floor
  band precisely to avoid catastrophic forgetting). "Looser
  weights" = deliberately raising plasticity above the floor for
  chosen tensors and tolerating transient damage, because the day's
  other measurement says high-lr damage HEALS (the 12,600 dip
  recovering by 15,300). Honest breaks: healing was measured
  DURING a birth with the full diet still streaming — an RL repair
  has a much thinner data stream, and whether damage heals under
  RL-only exposure is unmeasured; catastrophic forgetting on an
  internet model has no oracle-signed immune system (the LLMUE
  two-tier guard exists only in the closed system); and
  priors-vs-drag showed the internet model's problem was DRAG, not
  stiffness — looser weights on a bluffing model may just bluff
  faster. Testable residue: REPAIR-FLOOR-1 — take the frozen
  English 0.5B, RL/SFT at three lr tiers (sub-floor ~1e-5,
  at-floor ~4e-5, above-floor ~1e-4 scaled per-width), same
  verified diet and gate; bar family: does capability move ONLY
  above the floor, and does the above-floor arm's transient damage
  heal or compound? Also feeds the valuation-routed mask (loosen
  only engine-valued neurons, keep the rest stiff). Attribution:
  Artin (the loosen-the-weights frame, the repair memory), house
  (the floor tie-in + the three-tier operationalization).

- **BANKED (2026-08-13): "train on easier questions first until it
  can't learn anymore, then repeat — would this fix the order, or
  is order a more fine-grained thing at the weight level?"**
  (Artin, session close). The frame: plateau-gated curriculum —
  feed level L until the gate stops moving, then admit L+1, loop.
  Both halves of his own question banked: the ORDER measured today
  (BACKWARD-SCHEDULE-1) was the lr schedule's direction, NOT data
  order — the diet streams all levels shuffled in every arm run
  today, so data-order curriculum is genuinely untested on the 19M
  line at birth. Adjacent measured ground: the v2 lineage's
  "curriculum compounds under RL" (RESULTS L1784 — algebra-enriched
  pretraining bought 24 GRPO cycles' worth of solves, diet ~20x
  cheaper than RL) says diet COMPOSITION dominates; whether diet
  SEQUENCING adds anything on top is the open cell. The lr-floor
  result sharpens the design: "can't learn anymore" must be read
  against the floor (a plateau at low lr is the floor, not
  mastery — plateau detection needs lr held in the working band).
  Honest breaks: plateau-gated admission is a moving-target
  schedule (harder to pre-register than fixed arms — the bar must
  pin the plateau rule's constants in advance); L4's dip warns that
  the generator's level ladder is not the model's difficulty
  ladder (L4-PLY0-1), so "easier first" by level number may not be
  easier first by the model's own ordering — a capability-ordered
  curriculum (order by measured per-level solve rate) is the
  sharper arm. Testable residue: CURRICULUM-1 — three matched
  births: (A) shuffled control (the standard 64), (B) level-ordered
  L3->L7 with plateau-gated admission (constants pre-registered),
  (C) capability-ordered by the booked per-level difficulty
  (L3, L5, L7, L6, L4 — measured easy-to-hard). Same
  seed/device/total exposure; standard gate. Attribution: Artin
  (the plateau-curriculum ask and the weight-level-order question),
  house (the floor-aware plateau rule + the capability-ordered arm).
  [MEASURED same day (VERDICT CURRICULUM-1): P-ORDER-HURTS fired —
  the capability-ordered arm cost 10 solves (54 v 64, single
  seed), so "easier first" by ANY level labeling is measured
  harmful, not helpful, at birth on this line. The plateau-gated
  half remains genuinely UNTESTED: the pinned constants
  (100/300/2%) admitted every level by step 2,600, the pre-named
  "always admits" branch — a slower rule is a new pre-reg. The
  weight-level-order half of the original question is untouched.
  Next-day mirror (VERDICT REV-LADDER-1): hard-first gates 37 —
  order is harmful in BOTH directions; shuffled interleaving is
  the measured optimum of the three orders tested; neither
  mechanism bar fired (L3 follows lr placement, L4 stays at its
  structural floor). Same-day discriminator (VERDICT
  SWAP-LADDER-1): the mechanism NAMES ITSELF as FOUNDATION, not
  placement — one swap (L3 to second) lifts L5+L6+L7 from 19 to
  25 at unchanged positions while L3's own placement bar stays
  un-fired; what PRECEDES a level moves its cell. Artin's
  original "easier first until it can't learn" intuition was
  half right: prerequisites matter (foundation real), but
  BLOCKING levels to deliver them costs more than shuffling ever
  did — interleaved-with-prerequisites is the surviving shape.]

- **BANKED (2026-08-13): novelty IS the volatility — the newest
  data class (fresh operations the model has never seen) is the
  most volatile TO the model, and that volatility is what causes
  learning** (Artin, extending his own volatility-drag bank of
  2026-08-12). The mapping / the math: per-row gradient magnitude
  is largest where the model is most wrong; a "new operation"
  class (addition when the model knows none, division when it
  knows the other three) is maximally surprising, so it dominates
  the parameter update — volatility drag's sigma stops being only
  a tax and becomes the learning signal itself. The two-sigma
  split from the parent bank sharpens into a three-way: error
  variance (tax, cut), support variance (coverage, keep), and
  NOVELTY variance (the model-relative kind — the same row's
  volatility DECAYS as the model learns it, so it is a property of
  the (data, model) pair, not the data alone). Measured anchors:
  none direct; nearest booked ground is the LR absorption floor
  (VERDICT CAP-V-TRAJ-2 — updates below the floor buy nothing
  regardless of the data's novelty, so novelty can only pay
  through the lr window) and the exposure-share doctrine (diet
  rations). Honest breaks: per-row loss is the obvious novelty
  proxy and it also selects label ERRORS (the highest-loss rows in
  any real diet are the wrong ones — the two sigmas collide in one
  statistic); a novelty-weighted sampler with no error control
  re-derives the GRPO reward-hack shape. Testable residue:
  NOVELTY-SAMPLER-1 — same diet, same budget, arm A stock
  shuffled, arm B samples rows proportional to current per-row
  loss (recomputed every N steps), verified-only diet so the error
  sigma is pinned near zero; gate at equal steps. Attribution:
  Artin (the frame both times), house (the three-sigma split + the
  error-collision break).

- **BANKED (2026-08-13): steps could carry their DISTRIBUTION
  attached — a training row whose target is the set of valid next
  steps, not one sampled member** (Artin: "can't the steps
  themselves have their distribution attached to them too? We got
  to play around with the data more"). The mapping / the math:
  DIET-AMBIGUITY-1 (booked same day) measures the gen4 diet as
  one-to-many — 15.7% of rows share a prompt with other valid
  answers, CE floor >= 0.174 nats/row against one-hot targets.
  The floor exists BECAUSE the farm samples one member of the
  valid-next-step set per row; attaching the distribution (soft
  targets over all banked valid nxt for that cur, engine-weighted
  or uniform) makes the training target the true conditional and
  deletes the floor without deleting the branching. This is
  soft-label distillation where the teacher is the ENGINE's
  enumeration, not a bigger model. Measured anchors:
  DIET-AMBIGUITY-1 (the floor + the 4,356-cur conflict census =
  the exact rows a distribution-attached diet would change).
  Honest breaks: only 15.7% of rows are touched — the effect size
  is capped by that share; the engine does not enumerate ALL valid
  rewrites (only banked ones), so the "distribution" is itself a
  sample and the floor shrinks rather than dies; multi-target CE
  changes the sampler's calibration at decode time (the model
  learns to spread mass — gate impact could go either way).
  Testable residue: SOFT-NEXT-1 — rebuild the conflicted 15.7% as
  distribution rows (same support, soft targets), paired birth vs
  stock, standard gate + the loss floor read against the 0.174
  prediction. [RUN AND REFUTED 2026-08-15: trie soft targets at
  branch tokens moved neither calibration (+0.0033 v +0.05 bar)
  nor the gate (within noise) — the model already parks 63% of
  its mass on valid-answer sets under one-hot training. DEAD at
  this recipe; revival condition = a recipe where branch-token
  gradient variance is demonstrably binding. VERDICT RESULTS.md
  L29733.] Attribution: Artin (the ask), house (the
  DIET-AMBIGUITY-1 measurement + the engine-as-teacher framing).
  [PRIOR-ART AMENDMENT, same day: the ledger already holds an
  adjacent NULL — DISTRIBUTION ROWS 3-ARM (RESULTS L8078, with
  MASS-ON-VALID L7976): a matched-dose distribution-row arm was
  not a capability lever, because pick-trained valid-set
  distributions were ALREADY near-deterministic. The distinction
  that keeps SOFT-NEXT-1 alive: that cell softened rows whose
  targets were nearly unique; this one targets the measured
  4,356-cur conflict set, where the conditional is genuinely
  multi-modal (the 0.174-nat mass). The null narrows the
  prediction — any gate win must come from the conflicted subset
  or it will not come at all; the LOSS floor read is the primary
  bar, the gate secondary. LOSS-FLOOR-1 (L26376) is the other
  owed citation: the corpus-ambiguity half of the trained floor
  (0.175 nats given full prefix, ratio 0.502) was already
  measured on the warm diet by a different method — the
  DIET-AMBIGUITY-1 row-level census is its gen4/row-level
  cousin, not a first measurement.]

- **BANKED (2026-08-13): loss-bottom-gated dynamic lr ("lr doesn't
  move until loss has bottomed out on the epoch") — and the
  limiting joke arm, infinite lr + spam rows** (Artin). The
  mapping / the math: hold each lr value until the loss stops
  improving at that lr, then step down — schedule-by-measurement
  instead of schedule-by-clock (ReduceLROnPlateau's shape, but
  gated on the ABSORPTION story: a plateau at lr L means L's
  capability has been absorbed; the LR floor says stop stepping
  before ~2-4e-5 because below it steps buy nothing). Measured
  anchors: COMP-LADDER-1 (shape load-bearing, clock-compression
  free to 0.5x — a measurement-gated schedule is the natural next
  rung past fixed-shape compression); CAP-V-TRAJ-2 (the floor +
  transient high-lr damage that heals — so holding HIGH lr longer
  is not obviously safe OR obviously harmful, it is the open
  cell); DIET-AMBIGUITY-1 (the detector MUST be improvement-rate
  based — absolute loss ~ 0 does not exist on this diet, floor >=
  0.174 nats/row). The infinite-lr limit is measured-adjacent too:
  near-peak lr already costs a transient 12-solve dip at 3e-4;
  divergence sits somewhere above; grad-clip 1.0 is the only
  guard. Honest breaks: plateau-hold at PEAK lr risks the
  edge-of-stability regime where the dip may not heal (the healing
  was measured on a schedule that MOVED ON); "infinite lr" with
  clipping degenerates to sign-SGD with step size = clip/norm —
  a different optimizer, not a bigger lr. Testable residue:
  SCHED-PLATEAU-1 — arm A stock OneCycle, arm B lr ladder
  {3e-4, 1e-4, 4e-5} each held until improvement-rate < pinned
  epsilon, stop at the floor; equal total steps; gate both.
  LR-CEILING-1 (cheap, joke-priced): short births at 1e-3/3e-3/
  1e-2, book where divergence actually starts. Attribution: Artin
  (both asks, including the joke), house (the absorption framing +
  the floor/ambiguity fences).

- **BANKED (2026-08-13): "have the model only see question and
  answer — see if the weights are similar to the step-trained
  19M"** (Artin). Mostly ALREADY MEASURED, banked as the pointer
  plus the open residue. Measured anchors: the ONESHOT SURPRISE
  (RESULTS L6495) — root-to-answer-only FROM BIRTH, one pass,
  gates 54/120 v the pairs arm's 57, including deep levels (L6 8,
  L7 10), REFUTING the house's <= 40 pre-registration; the chain
  format's measured product is VALIDITY (error recovery — oneshot
  validity 33.5%, worst of the battery), not raw solves. And
  SOL-GRAVMOE-AO1 (L15101): answer-only LOSS ALLOCATION on an
  existing step-trained model is a strong-direction null (capability
  fell, format intact) — answer-only DIET from birth and
  answer-only LOSS on a chain diet are opposite-sign cells.
  The weights half of the ask has a measured correction: NO tested
  weight-distance lens recovers functional distance (gauge-aligned
  distance null, JOINT-PERM closure — distances track ancestry, not
  function), so "are the weights similar" is answered by RUNNING
  both, never by comparing tensors. Honest breaks: oneshot was
  d256/1-pass-era; no 19M/gen4/3-epoch oneshot cell exists.
  Testable residue: ONESHOT-19M-1 — the phase19m recipe on an
  answers-only rewrite of the same diet (cur = root, nxt = final),
  matched steps; gate + per-level + validity v the booked 64; the
  "similarity" read is function-space (per-level profile + the
  L4 cell), per doctrine. Attribution: Artin (the ask), house (the
  prior-art census + the function-space correction).

- **BANKED (2026-08-13): "memorize the answer first, then build
  the steps/weights off it"** (Artin) — answer-first two-phase
  diet: phase 1 root-to-answer pairs, phase 2 full chains, one
  birth. The mapping: backward-chaining curriculum — plant the
  target attractor, then grow the path structure toward it.
  Measured anchors: oneshot buys solves without validity and
  chains buy validity (L6495 — the two phases would supply
  complementary halves); TENET measured backward EMISSION as
  prompt-local (24/120 post-step v 1/120 control) but never tested
  answer-first ORDERING of a forward diet; CURRICULUM-1 (pre-reg
  this session) tests level order, not target-first order — this
  is a third, orthogonal ordering axis. Honest breaks: phase-1
  memorization at 19M scale may just be absorbed and overwritten
  (the LR floor says early-phase content trains at sub-peak lr —
  whether it SURVIVES phase 2 is exactly the open question);
  the answer distribution of gen4 is engine-shaped, so phase 1 is
  cheap to build from existing rows (no new farming). Testable
  residue: ANSWER-FIRST-1 — a CURRICULUM-1-shaped arm (same
  harness, same no-op precondition) with phase 1 = answers-only
  rows for the first K steps, K pre-registered; gate v 64.
  Attribution: Artin (the frame), house (the two-phase
  operationalization + the survival question).

- **BANKED (2026-08-13): factoring-targeted diet for the L4 scar
  — "could train it on more factoring questions (rather not,
  not as interesting)"** (Artin, half-retracted at bank time; his
  own preference booked with it). Measured anchors: diet
  COMPOSITION dominates (v2 algebra substrate bought 24 GRPO
  cycles of solves, L1784); the v22 shard IS the precedent — an
  autopsy-aimed shard built from measured failure modes; L4's dip
  is exposure-SENSITIVE (3/24 at 0.3x compression v 6-8/24 full)
  and its mechanism is ply-0 recognition failure (L4-PLY0-1), so
  a recognition-dense L4 shard (first-ply-only worked examples of
  the f'(g)*fn(g) family) is the mechanism-matched dose, not just
  "more L4". Honest breaks: it is a composition lever, already the
  best-measured lever class in the lab — high odds of working,
  low information yield (Artin's instinct); it also widens the
  exposure-fence surface (every targeted shard needs the
  exclude= guard audit). Testable residue: L4-DIET-1 — +N
  first-ply L4 recognition rows (exclude=-guarded), paired birth,
  L4 cell v the 6-8/24 band. Attribution: Artin (idea and the
  disinterest, both banked), house (the recognition-dense
  sharpening).

- **BANKED (2026-08-13): MAGIC-CURRICULUM — sort the diet by the
  magic estimator's deterministic per-row hardness and feed it in
  that order** (Artin: "wouldn't it just tell us exactly how to
  sort the data?"; house: the ordering tie-in). The mapping:
  CURRICULUM-1's arms order by generator level, and L4-PLY0-1
  proved the generator's ladder is not the model's ladder; the
  estimator offers a continuous, engine-relative, reproducible
  (string-seeded, deterministic given engine version + budget)
  difficulty scalar per row — a third ordering candidate that is
  neither the generator's labels nor the model's own solve rate.
  Measured anchors, including the same-day probe: (1) the
  estimator's pedigree — held-out rho 0.855 v count_ops 0.342,
  structure-not-size carries hardness (RESULTS L531); (2) the
  TRANSPORT FAILURE measured at bank time: v7 scored 300 diet
  rows/level in seconds on CPU with zero parse failures, but its
  cost head saturates to 1e5-class values on the diet's expression
  distribution (trained range ~0-8 log-nodes; adjacent-level
  medians 8 v 519,055) — the search-generator L1-L5 training
  distribution does not cover mathgen chain states, so the
  resulting order [6,4,7,3,5] is extrapolation noise, not an
  instrument reading. (3) The binary Liouville bit is all-zeros on
  this diet BY CONSTRUCTION (every integrand is elementary — the
  generator differentiates a drawn F), so the certificate sorts
  nothing here. Honest breaks: hardness labels are
  engine-relative (they drift as rules land) and the estimator's
  own Bayes-floor finding says the 20 features carry ~99% of
  explainable variance — a relabeled head is cheap but the
  ceiling is already known. Testable residue: MAGIC-DIET-LABELS-1
  — gen_magic_labels pointed at diet chain roots (fork-isolated,
  budget fixed, CPU farm, sample first per /probe), retrain the
  head, THEN the ordering probe: does estimator order reproduce
  the model's measured [3,5,7,6,4]? If yes, MAGIC-CURRICULUM-1
  becomes a CURRICULUM-1 arm with continuous ordering. GATED
  behind CURRICULUM-1's verdict: if order is a null lever at
  birth, the sort is moot for training (stays alive as a
  difficulty instrument regardless). Attribution: Artin (the sort
  ask, the CPU instinct — validated, the run cost seconds), house
  (the transport-failure measurement + the all-zeros certificate
  catch + the gating).

- **BANKED (2026-08-14): checkers (or chess) as the next engine
  domain — "wanna try chess? Or even checkers"** (Artin). The
  mapping: the lab already runs the chess TOOLCHAIN (NNUE eval
  beat the LLM value head 119 v 115; Stockfish engine+ranker
  pattern banked; syndrome policy = state-aware opening book);
  the riff is the domain itself. Checkers is the sharper
  instrument for house questions: SOLVED (draw, Chinook 2007), so
  every state has perfect ground truth — a complete oracle, exact
  difficulty labels, verified-candidate multiplicity everywhere
  (the thing TENET measured at ~0 on the math diet, so the
  ranking/pincer program that stalled there has a domain where
  its precondition holds by construction). Chess adds adversarial
  depth but no oracle completeness. Honest breaks: CHARTER — the
  lab line reads "engines for MATHEMATICS and PHYSICS. Only";
  game engines are benign but outside the letter; Artin owns the
  amendment, banked here as pending his explicit ruling, no build
  before it. The T-count precedent also warns: the methodology
  speed-ran a second domain to an honest greedy-wins null in one
  day — new domains can close fast and cheap. Testable residue
  (post-ruling): CHECKERS-0 — endgame-database gate (exact
  oracle), house engine methodology (best-first + verified
  pruning + measured-traffic macro promotion), pre-registered
  bars against a solver baseline; the transport question is
  whether the gate law and diet-order results reproduce where
  difficulty labels are exact. Attribution: Artin (the domain
  ask), house (the checkers-over-chess sharpening + the charter
  flag + the multiplicity tie-in).
  [MEASURED same day: the charter ruling landed (CLAUDE.md,
  instruments-only) and CHECKERS-0 ran and BOOKED as the
  greedy-wins null band (VERDICT CHECKERS-0: search 92.0 v
  material 88.0 on win-in-20, constant-0 at 90.5 — the accuracy
  bar, not the domain, was the weak instrument; the T-count
  precedent above fired exactly as this bank warned, 2-for-2).
  Still open: CHECKERS-1 (precision/recall bars or larger
  budget) and every transport question — the domain is admitted
  and cheap, the first bar design was the casualty.]

- **BANKED (2026-08-14): "why don't we MCTS during training?"**
  (Artin). The mapping: the lab already runs the AlphaZero loop
  with best-first in the search seat — engine searches, oracle
  verifies, winning paths become diet, diet trains the model —
  and that loop is the best-measured lever in the house
  (demonstrations moved the gate where self-practice did not,
  THE EXCHANGE CONVERTS; diet ~20x cheaper per solve than GRPO,
  RESULTS L1784 era). What is genuinely not done, two pieces:
  (1) SEARCH-VISIT SOFT TARGETS — train on where search spent
  effort, not only the winning path; identical bank to
  distribution-attached-steps/SOFT-NEXT-1 (2026-08-13), the two
  asks are one idea. (2) UCT EXPLORATION — skipped for a
  measured reason: rollout-averaging pays under noisy/adversarial
  evaluation; the house oracle is exact and best-first + dedup
  already beat the beam (+12 asynchrony, +21 dedup within
  best-first), and expert iteration booked its ceiling at one
  round (operators moved the ceiling, not more self-teaching).
  Honest breaks: MCTS-during-training pays where
  verified-candidate MULTIPLICITY exists; TENET measured it at
  ~0 on this diet and its revival fence requires measuring
  multiplicity FIRST — which the checkers bank above satisfies
  by construction (the two riffs compose). Testable residue:
  none new on the math diet (blocked by the multiplicity fence);
  on any multiplicity-positive domain, MCTS-v-best-first at
  matched node budget is the registered arm shape. Attribution:
  Artin (the ask), house (the already-running-AlphaZero read +
  the multiplicity gate).

- **BANKED (2026-08-14): atomic-op decomposition — "have it learn
  on an atomic step of the math equation (simplifying, factoring)
  as well as the steps within the actual equation... atomic as in
  database atomic"** (Artin). The mapping: two-granularity diet —
  ATOMS (single rewrite-rule applications, tagged by rule:
  simplify, factor, u-sub, by-parts, each an indivisible verified
  transaction) alongside the existing CHAINS (full derivations),
  so the model learns the operation table AND the composition.
  Measured anchors: the DECOMPOSITION DISCOUNT is the direct
  precedent — re-spelling chains as primitive emissions sharply
  cut the rows needed to learn the measured families (RESULTS
  L3682, SERIES RUNG 1E); v22's capped one-ply worked examples
  are a partial atom shard already in the diet; L4-PLY0-1 gives
  the target: 16/17 L4 failures die at ply 0 emitting nothing —
  a missing ATOM (the f'(g)*fn(g) recognition), exactly what an
  atom shard would drill. The ONESHOT result bounds the other
  end (answers-only loses validity, chains carry error
  recovery): atoms+chains is the interpolation with both halves.
  Honest breaks: atom rows widen the exposure-fence surface
  (every new shard owes the exclude= audit); REV/CURRICULUM-1
  measured that SEGREGATING data classes in time hurts — atoms
  must be INTERLEAVED with chains, not phased (shuffled is the
  measured optimum); the decomposition discount was measured on
  task families, not on the gate band. Testable residue:
  ATOM-DIET-1 — farm a rule-tagged atom shard (engine one-ply
  applications, exclude=-guarded), paired birth stock v
  stock+atoms interleaved at matched steps, standard gate + the
  L4 cell v the 6-8/24 band. Attribution: Artin (the
  two-granularity frame and the database-atomicity language),
  house (the decomposition-discount anchor + the interleave
  fence).
  [MEASURED 2026-08-14, same day]: ATOM-DIET-1 ran — BOTH bars
  fired. 6,000 atoms (2,400 L4) interleaved at 3.5% dose, matched
  15,420 steps: gate 73/120 v the booked 64, L4 cell 12 v 7 (a
  19M-lineage record, at the registered threshold exactly). The
  flooding scar did NOT reproduce at this dose. House prior
  missed low on both legs (predicted no fire). Single seed,
  knife-edge fires; open residue: n>=3 replication, dose ladder
  (via axiom emit_chain, source=axiom-oneply), rule-ablated
  shards. VERDICT RESULTS.md L29250.
  [REPLICATED 2026-08-14, same day]: ATOM-DIET-LADDER-1 — paired
  stock/atoms births at seeds 3 and 4, all arms matched at 15,420
  steps: +6 at both seeds (64 to 70), atoms L4 = 12 at every seed
  tried (12/12/12 over seeds 2-4; stock wobbles 7/6/8). Both bars
  fired cleanly, no knife-edge; house prior HIT the delta range
  (+4..+7) and missed L4 low again. The n>=3 replication residue
  above is now CLOSED for direction; still open: dose ladder,
  rule-ablated shards, magnitude at wider n. VERDICT RESULTS.md
  L29465.
  [MEASURED 2026-08-15, dose ladder]: EMITTER-DIVERGES — the
  effect does NOT transport to axiom-emitted atoms at matched
  dose (66 v 70, L4 7 v 12); within the axiom family the response
  is monotone 64/66/72 through ~6.3% exposure, no flooding scar.
  The bank's "one-ply atoms teach recognition" claim is now
  SHARD-FAMILY-SCOPED: rule mix / answer forms / censoring carry
  a large share. Open: rule-ablated shards (i_heurisch mass
  first). VERDICT RESULTS.md L29662. Cross-ref: the hardening
  rung (seeds 4/5) and the two untested suspects (answer
  canonicalization, survivor censoring) have their own bank,
  same day.
  [MEASURED 2026-08-15, rule ablation]: the suspect ranking's head
  survives its first test — RULE-CARRIER fires at exactly the
  3-L4-solve bar (noheur L4 8 v ctrl 11), against the level-mix
  gradient, single seed, direction-grade. The non-heurisch
  remainder still lifts the total (68 v stock 64): heurisch mass
  is the L4-CELL story, not the whole atom effect. Rider: 3,218
  random rows = 72, at/above the 6,000-row shard's 70. Open:
  seeds 4/5 to harden; answer-form and censoring suspects untested.
  VERDICT RESULTS.md L29916.

- **BANKED (2026-08-15): teach the basics before calculus — a
  pre-calculus atom tier** (Artin: "shouldn't we also teach it the
  basics?"). The mapping: the stock diet teaches arithmetic and
  algebra only IMPLICITLY (coefficient math rides inside
  integration steps; no row ever states 16/2=8 or
  (x+1)**2 = x**2+2*x+1 on its own). The atom-diet arc measured
  that foundation supplied AS DATA beats foundation left implicit
  (ATOM-DIET-1 L29250, replicated L29465) — this riff extends the
  same move DOWN the stack: one-ply arithmetic/algebra atoms
  (subs_eval, cancel, expand — all NATIVE axiom rules, so the
  IV7 emit_chain farm covers it with zero new machinery) at a
  capped dose. Measured anchors: the L4 one-ply diet +6 at 3.5%
  dose, L4 cell 12/12/12; dose ladder in flight (L29533).
  Honest breaks: the 120 gate has no L1/L2 cells, so "basics
  improved" has no direct gate readout — nearest observables are
  valid% (syntax/arithmetic slips) and the L3 cell.
  [AMENDED 2026-08-15, Artin's catch]: the "may already be
  saturated implicitly" caveat originally written here UNDERSOLD a
  booked prior — Series rung 1c (RESULTS L3444, 2026-07-22)
  measured exactly this move in the series grammar: arithmetic
  with operands SPELLED OUT trained to 67.0% held-out vs 15-16%
  implicit (~4.3x, separable 63/63). Explicit basics beat implicit
  where tried. The surviving open question is transfer and rations:
  1c's series rows cost a -2 gate dent (~2 integral solves at 19M
  regardless of volume), so basics atoms must price their diet
  share against that measured capacity cost.
  Testable residue: BASICS-DIET rung — farm algebra/arithmetic
  one-ply atoms via axiom's native rules, interleave at <=2%
  dose beside the calculus atoms, bars on valid% and L3 with a
  no-harm guard on the total; runs after the dose ladder prices
  dose headroom. Attribution: Artin (the ask), house (the
  axiom-native farm route + observable choice).
  [MEASURED 2026-08-15, rescoped then run same day: the census
  (RESULTS L30259) found the premise half wrong — algebra is
  already 18.2% of diet rows; only ARITHMETIC is absent (12
  incidental rows). The rescoped rung ran as BASICS-DIET-1
  (L30467): BOTH BARS FIRE — 2,545 arithmetic rows at 1.54% dose
  lift standalone arithmetic 1.67% -> 61.67% pass@8 at a -2
  dent inside run noise. Artin's ask survives measured, with the
  precision that the diet teaches what it STATES: transfer into
  the algebra format was flat (the format-locality rider, banked
  under format-as-routing). Follow-ups banked in the verdict:
  mixed-format rows, cross-term-decomposed expand, dose ladder.]

- **BANKED (2026-08-15): FFT the engine's signals into weights?**
  (Artin: "can we convert the signals the math engine is giving
  into weights? No clue"). The mapping candidates, ranked by how
  hard they hit the gauge wall: (a) direct signal-to-weight
  synthesis (hypernetwork off engine traces) — fights neuron-
  permutation gauge freedom head-on; the never-score-weights-by-
  distance doctrine (measured basis: 2026-07-06 weight-reader
  ablation, permutation-augmentation 88.4% beat canonical sorting
  82.4%) says any target IN weight coordinates is the wrong
  target; (b) spectral features of engine SEARCH signals
  (rule-fire sequences, solve traces as time series; FFT as the
  featurizer) fed as training data or curriculum weights —
  unexplored, gauge-safe because it stays in function/data space;
  (c) the already-banked weight-FFT euler read (spectra as an
  ANALYSIS instrument) — alive in the complex-alphabet prologue.
  Measured anchors: weight-reader 80.8-88.4% (weights->function,
  the legal direction); none for (a)/(b). Honest breaks: (a) is
  probably dead on arrival per the gauge doctrine unless posed as
  function-space distillation, which is what engine-emitted
  training rows ALREADY are; (b) has no oracle for "the spectrum
  mattered" yet — needs a falsifiable observable before it earns
  a rung. Testable residue: SPECTRAL-CURRICULUM-0 (desk, zero
  births): FFT the per-rule fire-rate series over the corpus
  ordering; if the spectrum separates levels or predicts the
  known-hard cells (L4 recognition), a curriculum-weighting rung
  is specifiable. Attribution: Artin (the ask), house (the gauge
  fence + the b/c split).

- **BANKED (2026-08-15): if one-hot rows average into the branch
  distribution anyway, collapse them and pocket the compute**
  (Artin, off the SOFT-NEXT-1 refutation: "if it's the same, why
  don't we try to use it to speed up training??"). The mapping:
  the diet's 25,916 conflicted rows carry ~4,347 rows of distinct
  information; replacing each conflict group with ONE soft-target
  row shrinks the diet ~13% at theoretically identical training
  signal — a lossless-in-expectation speed lever, exactly the
  house speed-defaults class. Measured anchors: SOFT-NEXT-1
  (L29733) showed target form is gate-neutral AND
  calibration-neutral at this recipe — the equivalence the trick
  needs. Honest breaks: "identical in expectation" is not
  identical in dynamics (fewer gradient samples on conflicted
  prompts; epoch length changes every schedule interaction);
  matched-EPOCH comparison changes total steps, so the schedule
  must rescale (OneCycle total_steps drops ~13%) — the comparison
  is speed-at-equal-quality, not one-variable. Testable residue:
  SOFT-SPEED-1 — dedup-to-soft-rows arm v stock, matched epochs,
  bars: gate within resolution of stock AND wall-clock/steps
  saved >= 10%. Attribution: Artin (the lever), house (the 13%
  arithmetic + schedule fence).
  [MEASURED 2026-08-15]: RUN. QUALITY-HOLDS fired (64 v control
  62, non-inferiority, within run noise); SPEED missed at
  knife-edge (steps -12.96% but wall-clock -9.84% v the 10% bar —
  Python soft-correction overhead +3.58%/step ate a quarter of
  the win). The lever is REAL at step level; the implementation
  gives part back. Open: -1b with vectorized corrections or
  weighted-representative-only rows. Side finding with its own
  standing fence: mps training is run-level nondeterministic at
  fixed seed (AMENDMENT L29985). VERDICT RESULTS.md L30064.

- **BANKED (2026-08-15): bigger distributions — hints filled and
  SOLUTION-NEXT targets** (Artin: "what if we give it hints and
  they are SOLUTION-NEXT, bigger/more distributions?"). The
  mapping: SOFT-NEXT-1 varied only target SHARPNESS on a fixed
  row schema; this riff varies what the distribution is OVER —
  (a) hints populated (the farm's hint/think fields, today always
  "none" in training rows) as conditioning, (b) full-solution
  targets (cur -> complete antiderivative, skipping intermediate
  plies) beside step targets, (c) distributions over SOLUTIONS
  not steps (multiple valid full answers). Measured anchors: the
  refuted SOFT-NEXT-1 is the prior (sharpness alone: nothing);
  hint-carrying rows exist in farm_v22 shards untrained-on.
  Honest breaks: hints at train time without hints at gate time
  is a train/test format shift (FORMAT-BOUND hazard); full-
  solution rows change the difficulty distribution wholesale — a
  diet redesign, not a target tweak, priced as its own program.
  Testable residue: HINT-DIET-0 (desk first: census what hint
  content exists in the frozen shards before any rung).
  Attribution: Artin (both asks), house (the format-shift fence).

- **BANKED (2026-08-15): collapsed soft rows x algorithmic
  (decomposed-arithmetic) training — and the collapse as a MORE
  CONSISTENT GRADIENT, b-tree-like** (Artin, mid SOFT-SPEED-1
  registration: "what if the collapsed steps with the algorithmic
  training? ... the collapsed steps maybe flow into each other
  like a b-tree or a more consistent gradient?"). Two claims in
  one riff. (1) REVIVE ALGORITHMIC TRAINING: Series rung 1c
  (RESULTS L3444) is the anchor — decomposed arithmetic chains
  trained 67.0% held-out v 15-16% single-hop (~4.3x), at a 358-step
  probe, i.e. strong AND cheap; already the queued basics-diet
  pre-reg (handoff 2026-08-15-0 next-session item 4). Combining:
  a collapsed-soft-row treatment OVER a 1c-style chain diet —
  chain steps are near-deterministic (few conflicted curs), so
  the measured combination question is whether collapse generality
  holds where conflict density differs by an order of magnitude.
  Desk check first: conflict census on the chain shard before any
  rung. (2) THE GRADIENT-CONSISTENCY FRAME: a conflict group under
  one-hot delivers k contradictory pulls at branch tokens across
  an epoch (mean = the distribution, variance high); the collapsed
  soft row delivers the mean in ONE sample — same expectation,
  lower gradient variance at branch positions, which IS the
  b-tree picture (shared prefix = shared internal node trained
  once, coherently). Measured anchors: SOFT-NEXT-1 (L29733) says
  at 19M/15,420 steps that variance was NOT the binding constraint
  (calibration already 63%, gate unmoved) — so the frame predicts
  SOFT-SPEED-1 lands neutral-at-fewer-steps, not better. Honest
  breaks: the b-tree analogy stops at depth 1 — the trie shares
  prefixes within one cur only, never across curs, and no
  hierarchical structure above the branch token exists in the
  loss; "consistent gradient" is variance reduction, not a new
  signal. Testable residue: SOFT-SPEED-1 RAN the same day and the
  frame's pre-registered call was CORRECT — QUALITY-HOLDS fired
  as non-inferiority (soft 64 v control 62, inside run noise),
  not as a gain, exactly as "neutral-at-fewer-steps, not better"
  predicted; the variance-reduction reading survives and the
  b-tree depth-1 break stands (VERDICT RESULTS.md L30064). Still
  open: the chain-diet combination, which waits on the
  basics-diet census. Attribution: Artin (both asks, the b-tree image), house
  (variance formalization, the 1c anchor, the conflict-density
  fence).

- **BANKED (2026-08-15): the GPT nine — external seat's post-RULE-ABLATE
  program proposal** (GPT via Artin relay; house line-verified the
  ledger claims before banking). Claims checked: L4-PLY0-1 16/17
  ply-0 recognition failures TRUE (L28104); size-Phi shaping NULL
  with distance-to-solved Phi pre-registered-unrun TRUE
  (L3257-3265, rung 2b); 95 skip-pairs in corpus TRUE (L1264) and
  skip-pairs ALSO already ran as a FORMAT LADDER cell (L6458) —
  their "no clean isolated skip-distance experiment" needs that
  cite, though a dose-matched jump-distance continuum at the 19M
  recipe is genuinely unrun; easy-first 54 / hard-first 37 v
  shuffled 64 TRUE (FINDINGS L1896); SOFT-SPEED = data
  compression not shared execution TRUE. External papers (LESS
  2402.04333, Tree Training 2511.00413, DeepSeek-Prover-V2
  2504.21801, DoGE 2310.15393) are claims-as-published, unverified
  house-side. The nine, with house dispositions:
  (1) GRAD-MAP-0 gradient/data-worth atlas, retrodiction-gated
  (must blindly recover sympy>axiom-for-L4, heurisch>remainder-
  for-L4 before it earns a birth) — ADOPTED as next desk
  instrument; the retrodiction gate is exactly the house
  instrument-first method. [MEASURED 2026-08-15: the gate FAILED
  on R2 — no metric ranks i_heurisch above the remainder, and the
  pre-declared L6 falsifier fired (the atlas tracks shard
  composition, not worth); R1 passed 4/5 metrics and the gradient
  cosine beat the surface-kNN control on that leg, the credit
  line. Instrument dead in this metric set; VERDICT GRAD-MAP-0,
  RESULTS L30127. The gate did precisely what it was built for.] (2) TREE-TRAIN-0 shared-prefix exact
  training — census first: conflicted rows share PROMPT prefixes
  within 4,347 groups only, so the FLOP ceiling may be small at
  19M; the zero-cost prefix-duplication census prices it before
  any implementation. (3) RULE-POLICY-0 action-native policy
  (rule_id, site, args -> deterministic executor) — the deepest
  bet; aligns with L4-PLY0-1 (recognition, not serialization, is
  the failure) and the watch-it-think flagship; atom rows carry
  single-rule tags as free labels; format-shift hazard: a policy
  head needs its OWN gate (first-action accuracy, executor
  validity) before comparison to string gates. (4) cost-to-go
  value head, supervised-ranking BEFORE any GRPO — revives the
  registered-unrun rung 2b (distance-to-solved Phi) in
  value-head form; consistent with the magic-estimator law
  (prediction pays where variance lives). (5) skip-distance
  ladder 1-ply -> chain -> k=2/3/4 -> full-solution at matched
  dose — legit with the format-ladder cite; U-shape prior is
  theirs. (6) MIXER-1 learned diet controller (DoGE-style) —
  PARKED behind (1): a controller is many variables at once; the
  atlas must exist first, and paired-arm discipline wants the
  controller's choices replayed as static diets. (7)
  EXPERT-INTERACTION-0 Hadamard/factorial expert-mask design with
  pairwise interaction recovery — strong instrument idea for the
  MoE crest thread, 30B/3080 territory, own machine budget, Artin
  GO. [CORRECTION 2026-08-16: "3080 territory" is WRONG and was
  never measured — every MoE masking gate in this lab (MOE-GT-1
  through GT-7, all EX-ANAT-*, EX-FRESH) ran on the MAC, on a
  16GB 4-bit MLX Qwen3-30B-A3B artifact fully resident in 36GB
  unified memory. No 30B model has EVER been loaded on the 3080's
  10GB VRAM. The two other large-model paths are also Mac and
  also not counter-examples: BLACKHOLE B0 streamed 16 shards
  desk-only with NO inference at all, and V4-F1d ran a 304B at
  6.25% expert residency with fetch-on-miss (~20GB of Metal
  allocations, 0.100 tok/s). Consequence for the design pass:
  this rung COMPETES WITH the math-native thread for the Mac, it
  is not a parallel 3080 track, and it should be budgeted as Mac
  hours.] (8) BASICS-DIET as small-dose transfer test — already
  queued (1c anchor); their framing (substrate transfer, not
  can-it-learn) adopted. (9) DEMOTE hint-diet in favor of
  auxiliary targets (rule id / subgoal / cost-to-go) — agrees
  with the standing format-shift fence on HINT-DIET-0; the
  auxiliary-target version merges into (3)/(4). Their three bets:
  atlas->learned-diet, tree training, action-native policy +
  cost-to-go. House ordering after SOFT-SPEED books: GRAD-MAP-0
  desk + TREE-TRAIN census desk, then basics-diet birth;
  RULE-ABLATE seeds 4/5 deferred (direction-grade suffices while
  cheaper levers exist) — concurs with their sequencing.
  Testable residue: the retrodiction gate on GRAD-MAP-0 is
  immediately falsifiable; the prefix census is a one-number
  answer (duplicated-prefix token fraction). Attribution: GPT
  seat (the program), Artin (the relay + the ask), house
  (verification, dispositions, fences).

- **BANKED (2026-08-15): GPT-nine round 2 — refinements adopted into
  the two merged programs** (GPT seat via Artin; house executed the
  free parts same session). Adopted: (a) FLOP-weighted tree census,
  not token-only — RUN: TREE-CENSUS-0 booked, 4.48% linear / 1.91%
  attention ceiling, rung DEAD at 19M, idea survives for
  rollout-shaped future data. (b) GRAD-MAP metric set pre-registered
  BEFORE retrodiction (no invent-a-statistic) — spec written,
  docs/superpowers/specs/2026-08-15-grad-map-0.md, incl. their
  random-3218-must-be-explained test as gate R3. (c) GRAD-MAP used
  PROSPECTIVELY on basics-diet (atlas predicts which basics
  transfer, birth tests the prediction) — spec payload P1. (d)
  RULE-POLICY keeps BOTH gate families: representation-native
  diagnostics (rule/site/legal/executor/first-action) AND the same
  120-problem end-to-end gate through policy->executor loop, budget
  fenced — comparability preserved. (e) policy + value as two heads
  of one solver (pi(a|s) + V(s)); value target named honestly
  (engine_distance_to_solved / verified_remaining_depth, never
  "exact cost-to-go" without a shortest-path proof). (f) controller
  = experimental-design generator (frozen replayable diets), their
  endorsement of the house static-replay fence. Two merged
  programs: DATA-SCIENCE (grad atlas -> interaction graph ->
  prospective basics -> gradient-selected shard -> static learned
  mixture) and REASONING-ARCH (rule policy -> executor ->
  value/distance head -> bounded search -> watch-it-think
  compute curve, x-axis = oracle-valid transitions). Honest
  breaks: atlas signatures are end-of-training gradients (worth-
  during proxied by worth-at-end, disclosed in spec); the
  compute-curve flagship needs the policy rung to exist first.
  Testable residue: GRAD-MAP-0 retrodiction gate (R1-R3)
  [DISCHARGED 2026-08-15: gate FAILED on R2, VERDICT GRAD-MAP-0
  L30127 — the DATA-SCIENCE program's atlas leg is dead in this
  metric set; basics-diet runs unshaped, MIXER-1 stays parked];
  RULE-POLICY-0 rung 0 = label coverage census (what fraction of
  chain rows admit a recoverable (rule, site) label). Attribution:
  GPT seat (refinements), Artin (relay), house (execution,
  fences).

- **BANKED (2026-08-15): "LaTeX routes to the math weights" — input
  format as a routing key into weight subsets** (Artin: models feel
  like a graph database; only certain inputs activate certain weight
  subsets; math examples in LaTeX seem to reach the math weights).
  The mapping: for MoE this is LITERALLY the router (our MoE-anatomy
  arc measured demand-selected expert subsets beating full width;
  the verbal-coverage 0.755 Spearman bank says a verbal/carrier
  population mediates routing); for dense models the analog is
  superposed subnetworks selected by input statistics — format is
  part of the key. Measured anchors, house-side: CE-400 is
  format-BOUND (the instrument fence measured format changing what
  a probe reads); the call-span hint arms showed one template atom
  change flipping tokenizability; SWAP-LADDER/diet work shows
  position-invariant content effects — content v format separable.
  Honest breaks: "graph database" overstates addressability — no
  key->weight lookup exists in a dense net, only soft feature
  routing; and OUR 19M has ONE format (no LaTeX/plain contrast
  exists in-lab, sstr only). Testable residue (cheap, in-family):
  same math content in two surface formats (sstr v a second
  serialization) at matched dose — does the gate move? A dead
  result kills format-as-routing at 19M; a live one opens
  format-ensembling. Attribution: Artin (the frame), house
  (anchors, breaks). [MEASURED 2026-08-15, first in-family datum:
  BASICS-DIET-1 (RESULTS L30467) put the SAME operation in two
  formats — standalone arithmetic rows lifted standalone
  arithmetic 1.67% -> 61.67% pass@8 while the identical
  cross-term arithmetic inside (ax+b)(cx+d) stayed at the control
  rate, and a near-format arm moved partway (17.5 -> 26.67).
  Competence radiated by FORMAT DISTANCE, not by operation —
  the frame's direction, measured at n=1, single seed, 1.54%
  dose; the two-serialization residue above remains the clean
  test.]

- **BANKED (2026-08-15): vector database for model organization**
  (Artin: "can't we use a vector database to improve the
  organization of our models?"). Nearest standing structures: (1)
  the MAGIC MATH BOARDS bank (2026-07-19) — persistent
  (rule, node-hash) memo tables = exact-match retrieval, already
  banked as the engine-side lookup; (2) the B-tree/hierarchical-VQ
  bank (2026-08-03) — weights as paths through shared codebooks =
  a vector index OVER weights; (3) kNN-LM-style retrieval at
  inference (nearest verified (cur, nxt) pairs as a non-parametric
  memory beside the 19M policy) — NOT yet banked anywhere, the
  genuinely new residue here. Honest breaks: a vector DB improves
  RETRIEVAL, not weight organization per se — the house law
  "never score weights by weight distance" applies doubly to
  indexing weights by embedding distance; and exact-match memo
  (boards) beats approximate-nearest for our oracle-gated setting
  wherever it applies. Testable residue: kNN over the verified
  corpus at gate time (retrieve top-k similar curs, expose their
  nxts as context) — a RETRIEVAL-GATE-0 desk probe on existing
  checkpoints prices it; format-shift fence applies (hints at
  inference need hints at train, the HINT-DIET lesson).
  Attribution: Artin (the ask), house (mapping to standing banks,
  the kNN residue).

- **BANKED (2026-08-15): SOFT-SPEED-1b — the 9.84% is Python, not
  physics** (house, off the SOFT-SPEED-1 knife-edge miss). The
  measurement: the collapse cut 12.96% of steps but only 9.84% of
  wall-clock, because the per-batch Python loop over 15,105 soft
  positions raised per-step cost +3.58% (0.18349 -> 0.19005
  s/step, VERDICT L30064). Two untested implementations recover
  the gap: (a) VECTORIZE — precompute per-row (position, token,
  prob) index tensors once, gather logp with a single
  scatter/index op per batch instead of a Python double loop; (b)
  DROP the soft targets entirely and keep only the WEIGHTED
  REPRESENTATIVE row (one-hot, count-weighted) — SOFT-NEXT-1
  already refuted soft targets as a quality lever at this recipe,
  so (b) tests whether the collapse's value was ever in the
  distribution or purely in the deduplication. Measured anchors:
  SOFT-SPEED-1 quality non-inferiority (64 v 62), SOFT-NEXT-1
  refutation (L29733). Honest breaks: (b) changes the arm's
  semantics, so it is a NEW registered arm, not a re-run; both
  need the mps-nondeterminism fence (in-run paired control, no
  cross-run identity). Testable residue: SOFT-SPEED-1b, bars =
  quality within resolution of an in-run control AND wall-clock
  saved >= 10%; arm (b) additionally answers "distribution or
  dedup?" for free. Attribution: Artin (the original lever),
  house (the overhead measurement and the two fixes).

- **BANKED (2026-08-15): RULE-ABLATE hardening and the rest of the
  suspect list** (house, off RULE-ABLATE-1's knife-edge fire). The
  carrier bar fired at exactly its threshold on one seed, so the
  claim "i_heurisch content carries the L4 cell" is
  DIRECTION-GRADE. Hardening rung: seeds 4/5 paired arms, same two
  derived shards, same horizon, mean paired delta >= 3 as the bar
  (the ladder's replication shape). Cost ~2h per seed pair on the
  Mac, no farm. The dose ladder's suspect list had THREE entries
  and only the first is now tested: rule mix (TESTED, fires),
  answer canonicalization (UNTESTED — do the two emitters
  serialize the same mathematical answer differently, and does
  normalizing kill the divergence?), survivor censoring (UNTESTED
  — the axiom farm's 8s L4 wall censored a bimodal-heurisch band;
  the sympy shard has no equivalent wall). Honest breaks: the
  ablation confounds level mix at every level (L6 starved 2.9x),
  so a seeds-4/5 replication inherits that confound; a clean
  rule-only test needs level-stratified shards, which shrinks the
  usable row pool. Testable residue: RULE-ABLATE-2 (seeds 4/5),
  ANSWER-FORM-0 (desk: diff the two emitters' serializations of
  the same integrand), CENSOR-0 (desk: what the 8s wall removed).
  Attribution: house.

- **BANKED (2026-08-16): the cosmology cross-check — the video's
  three anomalies against the lab's measured analogs** (Artin
  brought the video; house fact-checked and mapped). A pop-science
  video (JWST early galaxies, galaxy-spin asymmetry, black-hole
  cosmology) fact-checked against the literature, then read
  against the house ledger. The mapping, one line each:
  (1) "structure too big too early" v the ignition ladder — the
  2022-23 impossible-galaxy claims largely dissolved under revised
  masses (AGN contamination, bursty star formation; residual
  tension is a star-formation-efficiency parameter, not a ΛCDM
  break), the same shape as the house's loss-v-gate dissociation
  (GRAVMOE-GATE) and METALLICITY-1: apparent early capability is
  an instrument/assumption artifact until the oracle gate says
  otherwise, and ignition tracks diet QUALITY (verification
  grade), not exposure mass. (2) spin asymmetry (Shamir, MNRAS
  538:76, ~3.4 sigma, disputed; his own leading explanation is a
  Doppler selection bias) v SYMMETRY LADDER S1 + TIER-A A2+A3:
  the house found NO spontaneous rotational structure in the
  weight basis, then fenced the null as instrument-blind to a
  confirmed activation clock — both fields hold a real alignment
  signal hostage to lens selection effects; SGD ACCEPTS imposed
  symmetry (ROTATIONAL SNAP R3) but was never observed to choose
  it. (3) Poplawski torsion bounce (real Einstein-Cartan
  literature, fringe as cosmology) — already banked 2026-07-29 as
  the white-hole riff; no update to the physics claim. Measured
  anchors: GRAVMOE-GATE L14470, METALLICITY-1 L27523, S1 L8665,
  A2+A3 L12814, R3 L8610, B0 atlas L11044. Honest breaks: the
  mapping is epistemic (how anomalies resolve under better
  instruments), never mechanistic — no physics claim, standing
  concepts-as-methods fence. Testable residue: none NEW — the
  cross-check re-surfaces the 07-29 AREA-LAW PROBE (removable
  information v interface-v-bulk measure, desk-able against the
  existing deletion/snap ledger) as the one instrument this frame
  has been pointing at twice now. Attribution: Artin (the video +
  the compare ask), house (fact-check + mapping).

  CORRECTION (2026-08-16, GPT seat round 3; house verified both
  points against the sources before adopting). TWO defects in the
  bank above.
  (a) Item (1) is TOO DISMISSIVE AS WRITTEN. Sweeping the whole
  "too big too early" theme into the 2022-23 claims-that-dissolved
  bucket mis-files XLSSC 122, which is newer, LENSING-based (not
  photometric-mass-based, so the revision mechanism that softened
  the 2022-23 claims does not apply to it), and carries a
  genuinely unusual concentration c = 6.3 +- 0.5. The honest
  statement is "an interesting high-concentration tail that may
  imply faster assembly", not "the tension dissolved". The
  ANALOGY is unaffected — the epistemic mapping was never about
  which claims survived — but the astronomy summary inside it was
  wrong and is retracted as stated. HOUSE COUNTER-NOTE on the
  correction itself (verified, and it is the same lesson one
  level down): the 2026 weak-lensing follow-up (ApJL
  10.3847/2041-8213/ae447a) does NOT independently re-measure
  that concentration — its c200c = 6.3 +- 0.4 is IMPLIED by
  fitting the weak-lensing mass through a published c-M relation,
  from the same group, so it is a derived quantity agreeing with
  a model assumption, not a second instrument agreeing with the
  first. What the WL paper does add independently is MERGER
  evidence (common mass/X-ray/ICL/radio elongation, ~100-117 kpc
  SZ-peak offset, TNG-Cluster comparison favouring a post-merger
  state) — a dynamical-state confound that itself bears on how a
  concentration should be read. The authors' own framing is
  motivation-to-test ("could provide a stringent test of
  halo-structure prescriptions within LambdaCDM"), never
  falsification. Two numbers that agree are not two measurements
  — the house version of this is the standing rule that a
  weights-sha and a logit comparison can both agree while the
  sampler has already changed the reading
  (SOFT-PROMPT-1-SAMPLER).
  (b) The AREA-LAW residue was banked as if UNATTEMPTED. It is
  not: PRE-REG + VERDICT boundary-or-bulk (RESULTS L4513/L4527,
  2026-07-25) already ran the boundary-v-volume ordering test on
  the measured grid, and its clause 2 is a STANDING FENCE this
  bank should have cited — volume fails because it ignores the
  bit-dependent W* (a FEEDING mechanism), and "No boundary
  measure kept order where volume failed (width-only measures
  silent-not-correct; honest null)", with the explicit
  instruction "do NOT cite this as 'boundary won.'" So bulk
  failing is NOT evidence for boundary, and the video cannot
  upgrade the residue's evidentiary status. Any revived area-law
  probe therefore owes a PRE-DECLARED candidate boundary measure
  (e.g. residual-stream interface tensors v FFN bulk) that must
  predict removal damage WHERE PARAMETER VOLUME DOES NOT — the
  same bar the 07-25 rung set and no measure has yet cleared.
  Attribution: GPT seat (both catches), house (verification, the
  07-25 fence text).

- **BANKED (2026-08-16): the XTERM mechanism ladder, the
  intermediate-as-policy read, and transition-interface attachment**
  (GPT seat, reviewing 5332bf5..dc73020; house verified every
  ledger claim in the review before banking). Three items off the
  live XTERM-DIET-1 rung:
  (1) MECHANISM LADDER: a firing XTERM-DIET-1 confirms the
  intervention, not the mechanism — xexp (decomposition stated) and
  xstep (arithmetic in-format) are confounded by design. The ladder
  is control / xexp-only / xstep-only / both; xstep-only is the
  discriminator: if intermediate->final training alone moves the
  ORIGINAL product->final probe, competence attached to the
  representational neighborhood (format-as-routing); if xexp is
  required, the story is decomposition/state scaffolding.
  (2) INTERMEDIATE EMISSIONS AS ACCIDENTAL POLICY: the registered
  most-informative outcome is BAR 1 fires + BAR 2 misses +
  intermediate_form rises — training fixed the defect while the
  one-step interface misaligned with what the model wants to emit.
  Follow-on writes itself: feed the emitted intermediate back as
  the next Current: state and score the two-step chain (the first
  concrete bridge from births to verified multi-step execution;
  ties to the RULE-POLICY bank).
  (3) TRANSITION-INTERFACE ATTACHMENT, the generalization:
  capabilities may attach to state-transition interfaces rather
  than becoming globally callable subroutines. Anchors: atoms fix
  L4 first-ply recognition specifically (L29250, L29465),
  arithmetic learned at 61.67% stays format-local (L30467), numsum
  half-moves by proximity, i_heurisch carries L4 worth invisible
  to gradient metrics (L29916, L30127).
  Honest breaks: house correction adopted INTO the bank — the GPT
  claim that intermediate emissions cannot flatter BAR 1 is wrong
  one-way: a correct-products intermediate parses with the mid
  coefficient right and so deflates the cross-term-wrong counter
  without any evaluation; a BAR 1 fire with a large
  intermediate_form count books as composite (the probe's counter
  plus the control's measured 0 make it separable). The
  canonical-form strictness caveat is registered but empirically
  quiet at control (canonical == any-form on expand, 25 v 25).
  Testable residue: XTERM-DIET-2 (the xstep-only arm, one birth);
  the chain-scored two-step probe. Attribution: GPT seat (all
  three frames), house (BAR-1 deflation correction, verification).

  AMENDMENT (2026-08-16, same day): GPT seat's follow-up refines
  item (3) — "interface" must not collapse to string format. Three
  coordinates: (state representation, requested transition,
  supervision grain). BASICS moved state representation; XTERM
  moves representation AND supervision grain; atoms moved
  transition decomposition to one-ply; skip pairs move transition
  distance. The general form: COMPETENCE IS LOCAL IN TRANSITION
  SPACE — format-as-routing is one projection of it, and the
  transfer-v-representation-distance curve (standalone-X /
  numsum-partial / xstep-? edges) is the measurable object. The
  W+I decomposition is now REGISTERED pre-look as RIDER
  XTERM-DIET-1-DECOMP (RESULTS, this date) — promoted from bank
  note to on-the-record rider while the treatment probe had not
  run. Attribution: GPT seat (frame + rider ask), house
  (registration timing + the control anchors W=80, I=0).

  AMENDMENT 2 (2026-08-16, hours later): XTERM-DIET-1 booked BOTH
  PRIMARY BARS NO-FIRE (L30682) — the measured datum lands on the
  bank's open edge and against the house prior: in-format
  decomposed statement does not install the cross term at 1.6%
  dose, the decomposed route is never expressed (I=0), and the
  xstep surface (emit products) was learned without its content
  (evaluate them). Item (2)'s two-step probe is DEAD AS MOTIVATED
  (no intermediates to complete); item (1)'s xstep-only ladder is
  coherent but unlaunched off a null; item (3) survives with its
  coordinates re-weighted — state representation is now measured
  on both sides and insufficient ALONE; supervision grain and
  dose are the live coordinates. The transfer-v-distance curve
  gains a third measured edge: in-format statement, ~zero
  transfer at this dose.

  AMENDMENT to the RULE-ABLATE bank (2026-08-16): both untested
  suspect-list entries are now MEASURED and both fire —
  answer-form divergence 75.5% formdiff on co-solved integrands
  (OBSERVATION L30869; the mechanism is a serialization dialect,
  math disagreement 0) and survivor censoring real at 15.2% of L4
  successes in the 8-60s band (bimodality claim refuted as
  stated). The suspect list is now 3-for-3 measured: rule mix
  (RULE-ABLATE-1), answer form, censoring. Residue: ATOM-NORM
  (normalize axiom nxt through sstr(sympify()) at farm time,
  re-run the dose comparison at matched dose — the emitter-tie
  bar retests with the dialect removed).

- **BANKED (2026-08-16): the selection-function frame — five items
  off the cosmology round-3 review** (GPT seat; house verified
  every ledger claim and both design objections before banking).
  The review's organizing insight, adopted: the video's real
  transferable content is not black-hole cosmology, it is SURVEY
  SCIENCE — what an observer sees is the underlying population
  AFTER brightness limits, instrument response, classification,
  geometry and censoring. The house measured that exact structure
  in miniature overnight, which is why the frame earns a bank
  rather than a metaphor.

  (1) OBSERVATION/SELECTION CONTRACT — the proposed extension to
  the data-contract vocabulary (adopted 2026-08-16 from ECC
  mle-workflow: row grain / label timing / split policy). The
  added clause: an observed diet is never the generator, it is
  generator o representation-transform o survival/censor o
  scorer. Every farm should emit, automatically: attempted ->
  emitted survival by rule and level, wall-time distribution,
  raw-v-canonical answer-form distribution, and output length.
  Measured anchors, both from the same night: ANSWER-FORM-0
  (representation transform — 75.5% formdiff at math-disagreement
  0) and CENSOR-0 (survival function — a real 15.2% solvable band
  removed by an 8s wall, discovered only after the shard trained
  a booked rung). CENSOR-0 found a hidden selection function
  AFTER the fact; the contract's purpose is to make that
  impossible going forward. Testable residue: a farm-preamble
  helper in llmopt/lab/ that emits the four distributions, plus
  the retro-application of it to the shipped shards.

  (2) ATOM-NORM, REDESIGNED — the review's strongest design catch,
  adopted over the house handoff's own wording. The handoff put
  normalization AND a raised L4 wall in one next-step; those are
  now TWO INDEPENDENTLY MEASURED suspects and changing both
  together destroys attribution. Corrected design: derive a
  NORMALIZED TWIN from the FROZEN axiom shard (nxt =
  sstr(sympify(nxt)), same cur, same rows, same rules, same
  censored population, same dose — do not regenerate the
  mathematical sample), and run ONE serial Mac family of three
  arms: sympy / axiom-raw / axiom-normalized. Discriminator: if
  normalized closes the L4-and-gate gap while raw reproduces it,
  answer dialect is CAUSAL; if normalized changes emitted style
  while capability stays put, dialect was only a TRACER. The
  wall gets its own separate CENSOR rung, wall unchanged here.
  Rider named at design time: emitted-dialect measurement on the
  probe, so the tracer reading is available either way.

  (3) DIALECT-TRACER-0 — a new instrument, off the MoM-z14
  abundance-ratio analogy (astronomers read present composition
  as a historical tracer of prior processing). The house now owns
  two semantically neutral textual "isotopes" of the same
  mathematics: sympy dialect and axiom dialect. Design: identical
  mathematical content in two serialization conventions, train
  controlled mixtures (0/25/50/75/100%), ask whether HELD-OUT
  EMISSION DIALECT tracks the training mixture while capability
  stays fixed. If it does, the lab gains a NON-INVASIVE
  PROVENANCE TRACER for data influence — read functionally, by
  running the weights, which is the house-legal direction (the
  standing never-score-by-weight-distance law, and the closed
  weight-reader arc TENET-W1/W1-R where direction was invisible
  to every weight lens and loud in function). Honest break: a
  tracer measures INFLUENCE-ON-STYLE, never influence-on-
  capability; conflating them would repeat the marker-v-mechanism
  error this same review names.

  (4) STATE-SUFFICIENCY-0 (a.k.a. MARKOV-CLOSURE-0) — a desk rung
  the review places BEFORE any policy birth, and the house
  agrees: RULE-POLICY-0-CENSUS's most consequential finding may
  not be the 47.58% label number but the HIDDEN-HISTORY class it
  named (Integral(0, x) -> "+ 4": the target carries information
  inherited from chain context and is not computable from cur
  alone). Ask, by counting: P(next state determined | Current
  only) v Current + previous state v Current + chain metadata.
  Why it blocks: a policy pi(a|s) cannot learn a clean action map
  if s is not a sufficient state, so this prices the policy rung's
  premise before a birth is spent. Ties to the determinability
  law (underdetermined rows train confident guessing, L3401) —
  and would be its first sighting IN THE STATE REPRESENTATION
  rather than in the label.

  (5) TRAINING AGE != EFFECTIVE CAPABILITY AGE — offered as
  THEORY SYNTHESIS, not new evidence, and banked that way. The
  astronomy lesson from XLSSC 122 / MoM-z14 is that observed
  maturity constrains the ASSEMBLY RATE, not the clock. The house
  already owns four measured rows saying step count is a poor
  proxy for effective capability age: warm birth changes
  time-to-capability but not the endpoint (L2469); the
  decomposition discount is ~10x per row with primitives
  saturating where chains asymptote below (L3682); one-ply atoms
  buy a replicated +6 at 3.5% dose (L29465); and capability LEADS
  weight-space settling, 90% of final gate at step 10,800 of
  15,420 (CAP-V-TRAJ-1, L27909). Candidate law: effective age =
  accumulated USEFUL exposure, not raw steps. NOT written into
  THEORY yet — a THEORY row owes a published lineage citation
  alongside its measured basis, and the lineage for this one is
  unidentified. That is the residue.

  Honest breaks (house): (1) is a discipline, not a result — it
  cannot be "fired", only adopted; (3) and (4) are unpriced in
  wall-clock; (5) risks being a restatement of the ladder law in
  new coordinates, and must not be booked as a discovery.
  Verified astronomy figures behind the frame (house web pass,
  for anyone re-deriving the analogy): MoM-z14's bright z~14-15
  number density is 182 (+329/-105) x pre-JWST consensus models
  (arXiv:2505.11263, abstract) — the ">100x" is real and carries
  a large asymmetric error; and arXiv:2509.07695 does accommodate
  MoM-z14's mass in the Renaissance suite while GS-z14 "remains
  an outlier even after accounting for cosmic variance", so the
  early-maturity tension is object-specific rather than
  population-wide — which is precisely the shape of the house's
  own single-cell-v-family readings.
  Attribution: GPT seat (all five, and the ATOM-NORM attribution
  catch), Artin (the relay), house (verification of the four
  measured anchors in (5), the 07-25 boundary fence, the
  weak-lensing counter-note, adoption).

- **BANKED (2026-08-16): the scientific-hardening program — make every
  auditor BLOCKER a repository invariant** (GPT seat, off the
  STREAM-WDISTILL audit episode; house verified the state-of-play
  claims). The overarching rule, adopted as the organizing
  principle: **every BLOCKER an auditor ever finds graduates TWICE
  — once into a machine-enforced invariant, and once into a
  regression fixture reproducing the original failure.** The
  diagnosis is correct and checkable: the lab's protections
  currently live mostly as AGENT INSTRUCTIONS (the prereg-auditor
  and receipt-auditor checklists) rather than as things the repo
  refuses to do; scripts/book.py mechanizes only a subset (killed
  runs, dict sums, sha-less gates, unfenced sub-sigma n=1), and
  its tests already show the right philosophy — past failures as
  permanent regression cases.
  TEN PROPOSALS, banked in the seat's order:
  (1) MACHINE-READABLE PRE-REGS — experiment.yaml alongside the
  prose, authoritative for arms/configs/metrics/populations/
  aggregation/bars/refuted-if/scope/seeds/budget; a contrast_check
  diffs two arms so a bar claiming "only X differs" is REFUSED
  before a weight byte is read. Would have caught the S2-v-W32
  false width isolation mechanically.
  (2) ORTHOGONAL CLAIM STATES — split today's single maturity tag
  into validity (valid/retracted/superseded/unresolved) x outcome
  (positive/null/mixed/not-adjudicated) x maturity (single-seed/
  replicated/mechanism-confirmed/exact). The measured motivation:
  EXEC1 had to be tagged RETRACTED because the vocabulary
  (RETRACTED, NULL, MECHANISM-CONFIRMED, REPLICATED, SINGLE-SEED)
  has no way to say "valid question, invalid execution".
  (3) TYPED METRICS — a metric carries semantic type, unit,
  POPULATION, aggregation, dtype and provenance, so an
  expert:0 number cannot satisfy a bar registered for
  layer:all_experts, and an entropy_gap cannot be read as
  bits_per_weight. Would have caught BOTH the expert-0 BAR 2
  defect and the capacity-meter category error.
  (4) BOOK PRIMITIVES, DERIVE SUMMARIES — never store a ratio,
  percentage, mean or rate as the only evidence: percentage ->
  (count, n); ratio -> (num, den); mean -> (sum, n, sum_sq);
  bpw -> (artifact_bytes, weight_count). The adjudicator computes
  the published number, so a mean-of-ratios error becomes
  impossible. This is the gate-dict checksum law generalized.
  (5) ARTIFACT CLOSURE BY INTERFACE — for representation work the
  scorer accepts ONLY the decoded artifact, never the encoder's
  working floats, and the budget comes from real serialized bytes.
  Structurally eliminates fp32-decoded-while-fp16-billed and
  never-serialized arms instead of hoping an auditor notices.
  (6) BLIND DETERMINISTIC ADJUDICATOR between running and prose:
  locked pre-reg + raw receipts in, FIRE/NO-FIRE/UNRESOLVED per
  bar out, with arithmetic — and it never sees the draft verdict.
  The prereg-auditor's job then becomes proving the narrative
  matches an independent adjudication, which removes anchoring.
  (7) SCIENTIFIC-LANGUAGE LINTER with claim semantics: only,
  matched, optimal, exact, best, wins, refuted, isotropic, local
  structure, causes/mechanism, law/generalizes — each carrying a
  machine-checkable obligation, defined in a
  docs/SCIENTIFIC_STYLE.md. It would have flagged "Lloyd-optimal"
  live.
  (8) tests/science_incidents/ — every historical incident as an
  adversarial fixture with minimal prereg/config/receipt/verdict
  inputs and the expected refusal: inherited emitter label,
  moving-revision literal, smoke rows in a real receipt,
  over-budget comparator, omitted serializer metadata,
  fp32-billed-fp16, wrong metric population, ternary-in-a-2-bit-
  field, false "only" contrast, wrong theorem scope, sub-sigma
  direction, stale downstream claim.
  (9) EXPERIMENT LIFECYCLE SEAL — DRAFT -> PREREG_LOCKED ->
  RUNNING -> RUN_COMPLETE -> AUDIT_PENDING -> ADJUDICATED ->
  BOOKED, with hashes of pre-reg + driver stored before
  observation and auditor sign-off bound to receipt/draft hashes,
  so a post-look amendment MUST create a new lineage node. Makes
  the EXEC1 -> AUDIT-REPAIR split unavoidable rather than a matter
  of discipline.
  (10) EVIDENCE-GRAPH STALENESS — extend results-index links into
  a dependency graph; when a claim is retracted or materially
  amended, CI walks downstream references in FINDINGS, THEORY,
  README, RIFF and live pre-regs and marks them
  STALE_UNTIL_REVIEWED. Turns the lab's periodic manual
  claims-audit passes into continuous checking.
  DIVISION OF LABOUR, adopted: keep the specialist agents, narrow
  their remit — machines catch what can be stated as an invariant;
  auditors attack the scientific INTERPRETATION that machines
  cannot understand (does this intervention identify the claimed
  mechanism? what confound is unmeasured? does the negative
  control separate the preferred reading from the strongest
  alternative?).
  Honest breaks: (1),(2),(3),(9),(10) are real engineering, not
  afternoons, and each adds ceremony that must earn its cost on a
  lab this size; the seat's framing that "failures of scientific
  process are software bugs" is the load-bearing claim and it is a
  METHODOLOGICAL bet, unmeasured. House ordering by
  value-per-hour: (4) is a CLAUDE.md law that costs nothing and
  closes the mean-of-ratios class today; (8) is incremental and
  starts paying on the first reuse; (5) is the one that would have
  killed the biggest blocker of this episode; (6) removes the
  anchoring the house has now demonstrated twice. Testable
  residue: none of this is an experiment — it is instrumentation,
  and its success measure is BLOCKER RECURRENCE going to zero for
  any class that has graduated. Attribution: GPT seat (the whole
  program and the graduation rule), Artin (the relay), house
  (state-of-play verification, ordering).

  AMENDMENT (2026-08-16, same day, GPT seat — a correction the
  house earned): the bank above VIOLATES ITS OWN GRADUATION RULE
  in the sentence that orders the work. It calls item (4)
  "a CLAUDE.md law that costs nothing", but a CLAUDE.md law is a
  model instruction — exactly the human/checklist-level defence
  the program exists to move beyond — so it cannot count as
  graduation. Two states are now distinguished, and the
  distinction is itself part of the machinery:
    PROMOTED   the class is a documented house law
    GRADUATED  the class has an EXECUTABLE INVARIANT that refuses
               it, AND a regression fixture reproducing the
               original failure
An incident table carries the status per class, e.g.
    incident_class            law  invariant  regression  status
    mean_of_ratios            yes  no         pending     PROMOTED
    fp32_billed_fp16          yes  pending    pending     PROMOTED
    moving_revision_literal    yes  pending    pending     PROMOTED
    wrong_metric_population   yes  pending    pending     PROMOTED
    ternary_in_2bit_field     yes  pending    pending     PROMOTED
    unserialized_arm          yes  pending    pending     PROMOTED
    over_budget_comparator    yes  pending    pending     PROMOTED
    false_only_contrast       yes  pending    pending     PROMOTED
    smoke_row_in_real_receipt yes  yes        yes         GRADUATED
    inherited_emitter_label   yes  partial    pending     PARTIAL
Only the smoke class is genuinely graduated today (the
2026-08-15 CLAUDE.md rule plus .claude/hooks/smoke_guard.py plus
its verified both-directions test). Everything this episode
produced is PROMOTED at best. That is the honest starting state,
and it makes the program's own success metric — blocker
recurrence going to zero for a GRADUATED class — measurable
rather than aspirational, because a class that recurs while
merely PROMOTED is evidence for the program, not against it.
House ordering updated accordingly: build tests/science_incidents/
FIRST (it is the regression half of every future graduation and
the corpus already exists), then the executable invariants in
value order (artifact closure, typed metrics/populations, the
blind adjudicator). Attribution: GPT seat (the PROMOTED-v-
GRADUATED distinction and the catch), house (adoption, the
honest status column).

  AMENDMENT 2 (2026-08-16, GPT seat): the episode's most reusable
  lesson gets a schema. MEASUREMENT VALIDITY and COMPARISON
  ADMISSIBILITY are different predicates, and EXEC1 failed both
  in different ways for different reasons. Every bar should
  eventually carry:
      measurement_valid : true/false
      arms_admissible   : true/false
      bar_adjudicable   : measurement_valid AND arms_admissible
  Read against this episode: EXEC1 books false/false/false; the
  repair's BAR 2 books true/false/false (the measurement is fixed,
  arm A is still structurally over budget); BARs 1 and 3 after a
  clean repair book true/true/true. The whole debugging arc
  collapses into that one three-field record, which is the
  cheapest concrete piece of proposal (2)/(3) and the natural
  first field of a machine-readable pre-reg. Attribution: GPT
  seat.

  AMENDMENT 3 (2026-08-16, GPT seat): the bar schema in AMENDMENT
  2 is upgraded before it graduates into code — booleans and
  run-level scope are both wrong. EXEC1 is the proof: the run was
  not uniformly invalid (its pooled Frobenius produced usable
  diagnostic observations), and its three bars broke for
  DIFFERENT reasons with DIFFERENT arms at fault. The nucleus is
  therefore BAR-SCOPED, ARM-EXPLICIT, REASON-CODED, and
  MULTI-STATE (pending | true | false | not_applicable — a bar
  awaiting its receipt and auditors is `pending`, not `false`;
  a diagnostic with no comparator is `not_applicable`):
      bars:
        BAR1:
          measurement_valid: false
          arms: {C: admissible, D: admissible}
          bar_adjudicable: false
          reasons: [billed_fp16_decoded_fp32,
                    comparator_solver_failed_validation]
        BAR2:
          measurement_valid: false
          arms: {A: inadmissible, B: admissible}
          bar_adjudicable: false
          reasons: [baseline_over_budget, baseline_not_serialized,
                    candidate_not_serialized,
                    candidate_decoder_precision_mismatch,
                    metric_population_mismatch]
        BAR3:
          measurement_valid: false
          arms: {C: admissible, E: admissible}
          bar_adjudicable: false
          reasons: [billed_fp16_decoded_fp32]
  with bar_adjudicable = measurement_valid AND (all required arms
  admissible), computed rather than asserted. The reason codes are
  the same strings the incident table keys on, so a bar refusal
  and its regression fixture share one vocabulary. Why it matters
  beyond tidiness: six months on this answers WHICH ARM broke
  admissibility and WHY without reconstructing prose.
  Also banked, from the same reading: the wrong-metric-population
  class has now been observed TWICE, and the second time in
  ANALYSIS PROSE rather than tensor code — a repository can
  execute every float operation correctly and still difference
  two numbers from different populations. That makes it the
  natural FIRST fixture for typed metrics: a typed subtraction
  where population "expert:0" and population "experts:0:256"
  must refuse unless the caller explicitly requests a
  cross-population operation and labels it descriptive.
  Attribution: GPT seat.

  AMENDMENT 4 (2026-08-16, GPT seat — the deepest correction of
  the episode, and it invalidates the schema's own sufficiency).
  (a) CONTRAST ADMISSIBILITY MUST BE FIRST-CLASS. The nucleus so
  far was measurement_valid x arm_admissible, and that pair
  CANNOT express the wrong-metric-population class — the very
  class nominated as the first graduation. Population
  compatibility is not a property of either observation or either
  arm; it is a RELATION BETWEEN THE TWO THINGS BEING CONTRASTED.
  The proof is in this repo: EXEC1's operator metric over
  population expert:0 is valid, the repair's over experts:0:256
  is valid, both producing arms are admissible, and their
  DIFFERENCE is still inadmissible as a repair delta. So the
  nucleus gains a fourth object and the derived rule becomes
      bar_adjudicable = all required observations valid
                        AND all required arms admissible
                        AND all required CONTRASTS admissible
  computed, never asserted.
  (b) STATE VOCABULARIES ARE SEMANTICALLY TYPED, not one shared
  enum: measurement_status {pending|valid|invalid|not_applicable},
  arm_status {pending|admissible|inadmissible|not_applicable},
  contrast_status {pending|admissible|inadmissible|
  not_applicable}, adjudication_status {pending|adjudicable|
  refused|not_applicable}, outcome {pending|fire|no_fire|
  unresolved|not_applicable}. EXEC1 is why they must never share
  a boolean: it simultaneously held valid measurements, invalid
  contrasts, unresolved bars and useful diagnostics, and
  `measurement_valid=false` must never be confusable with
  `outcome=no_fire`.
  (c) REASONS NEST UNDER THEIR SUBJECT. A flat reasons list makes
  the machine parse English prefixes to learn that
  `baseline_over_budget` means arm A. Reasons attach to the arm or
  the contrast they belong to, so "which arm broke admissibility
  and why" is answerable FROM THE DATA STRUCTURE rather than by
  convention.
  (d) ARM ADMISSIBILITY != MEASUREMENT VALIDITY FOR THAT ARM. Arm
  B was a conceptually admissible candidate whose EXEC1
  measurement was invalid (decoded at unavailable precision).
  Collapsing those into one "B good/bad" loses exactly the
  distinction this episode was about.
  (e) THE TYPED-METRIC ESCAPE HATCH RETURNS A DISTINCT TYPE, not a
  labelled metric: cross_population_difference(a, b,
  purpose="descriptive") -> DescriptiveContrast(adjudicable=False),
  so the bar adjudicator REFUSES it structurally. The guarantee
  is compositional — you may deliberately look at apples versus
  oranges; you may not accidentally turn that look into evidence
  for a registered bar. The regression fixture tests all THREE
  directions: mismatched subtraction refuses; explicit descriptive
  contrast succeeds; descriptive contrast handed to the
  adjudicator refuses.
  (f) THE SUCCESS METRIC IS A RATE, NOT A COUNT. Raw recurrence
  misleads because classes get different numbers of opportunities.
  Measure escaped_incidents / opportunities_to_violate, WITHIN
  class, before v after graduation — and track REFUSALS as
  positive evidence, since "attempted 17, refused 17, escaped 0"
  is far more informative than an absence of recurrence.
  (g) HOUSE NOTE ON THIS BANK'S OWN STATUS, and it is recursive:
  everything above is DOCUMENTATION. The schema DESIGN is banked
  and upgraded; the schema is NOT implemented and nothing in the
  repo enforces it. Saying "the schema is now upgraded" would
  commit the PROMOTED-v-GRADUATED error inside the paragraph
  defining PROMOTED-v-GRADUATED. Status: PROMOTED.
  (h) LOCALITY LADDER DESIGN CONDITION, for the BAR-3 follow-up:
  every rung (random / native contiguous / row-column / neuron /
  gate-up pair) gets the same rate and byte accounting AND ITS OWN
  TRAINED CODEBOOK after its grouping transform — otherwise the
  measurement is whether one codebook happened to suit another
  representation, not grouping quality. If gate/up pairing beats
  raw adjacency, the compressible covariance lives in a SEMANTIC
  ARCHITECTURAL coordinate, which is a much stronger statement
  than "W32 beats shuffled". Attribution: GPT seat.

  AMENDMENT 5 (2026-08-16): FIRST REAL GRADUATION, and an audit
  correction to the bank's own accounting. Built per Artin's
  ordering (science-incident harness + first invariant together,
  chosen as wrong_metric_population because it exercises the new
  contrast object):
    llmopt/lab/metrics.py — typed Metric carrying semantic type,
    population, aggregation, unit and provenance; arithmetic
    REFUSES across mismatched population/aggregation/identity with
    a reason code from the incident vocabulary; the escape hatch
    cross_population_difference() returns a DISTINCT TYPE,
    DescriptiveContrast(adjudicable=False), which adjudicate()
    refuses structurally. You may look at apples versus oranges;
    you may not launder the look into bar evidence.
    tests/science_incidents/ — the harness, with
    test_wrong_metric_population.py reproducing the REAL failure
    (EXEC1 operator at expert:0 = 0.7657 v repair at
    experts:0:256 = 0.769457) across all three registered
    directions, plus the aggregation class (pooled v
    mean-of-ratios), plus the bar-population defect that was
    BAR 2's actual fault, plus a NEGATIVE CONTROL asserting that a
    same-population comparison still WORKS — without which an
    invariant that refused everything would pass every other test.
    Six tests, suite green at 866 passed.
  GRADUATED today: wrong_metric_population,
  metric_aggregation_mismatch, contrast_not_adjudicable (all
  llmopt/lab/metrics.py), joining smoke_row_in_real_receipt. The
  status table lives in tests/science_incidents/README.md so it is
  derived from the suite rather than asserted here.
  AUDIT CORRECTION to this bank's own claims (GPT seat, verified):
  the recommendation report said "six hooks"; six hook FILES exist
  but the checked-in settings.json wires five commands. The
  half-right finding is the useful one — findings_headroom.py is
  NOT a registered hook entry, yet it IS invoked by
  ledger_regen.py:67, so the invariant runs. NEITHER INVENTORY IS
  AUTHORITATIVE: counting files overcounts, counting settings
  entries undercounts, and "is this invariant live?" is not
  observable from either surface. Recorded because it is precisely
  the artifact-exists-v-invariant-active distinction the
  PROMOTED/GRADUATED vocabulary was built to make, found in the
  bank's own reporting.
  Remaining order (Artin): (2) receipt immutability — hook as
  mitigation PLUS content-addressed CI invariant and its
  historical fixture; (3) internal citation migration, anchor_guard
  as a TRANSITION toward stable entry IDs rather than perpetual
  line-number repair (measured: 34 of 123 internal L#### refs, 28%,
  no longer land on a heading); (4) machine-readable prereg/bar
  schema feeding a deterministic /adjudicate; (5) claim-linter LAST,
  once the schema gives "matched"/"only"/"independent" actual
  machine-checkable objects to interrogate rather than prose.

  AMENDMENT 6 (2026-08-16): ITEM 2 SHIPPED — receipt immutability,
  mitigation AND invariant AND historical fixture, per Artin's
  ordering. Three parts:
    scripts/gen_receipt_lock.py -> docs/receipts.lock.json —
    content-addresses every receipt path cited by RESULTS.md.
    First build: 78 cited paths, 71 present and sha256'd, 7
    CITED-BUT-ABSENT (older merge_space/microstar driver logs
    never force-added — a pre-existing backlog surfaced by the
    lock, now ratcheted so no new booking can cite evidence it
    did not commit). Changing an already-locked sha REFUSES
    without --accept "reason", so a legitimate change lands in a
    reviewable diff instead of happening silently.
    tests/science_incidents/test_frozen_receipt_mutation.py — the
    live invariant (every locked sha still matches), the absent-
    citation ratchet, the HISTORICAL fixture reproducing the exact
    2026-08-16 manoeuvre (rename the cited receipt aside, rerun
    into the freed canonical path, citation silently resolves to
    the wrong run), and a negative control so an always-failing
    check cannot pass.
    .claude/hooks/receipt_freeze.py — PreToolUse Bash mitigation,
    DENY not ask. Verified six directions: mv on a cited receipt
    DENY, append to one DENY (the doctrine names APPEND
    explicitly), redirect INTO one DENY, reading one ALLOW,
    redirecting output ELSEWHERE while reading one ALLOW, mv on an
    uncited path ALLOW. Wired into settings.json.
  GRADUATED: frozen_receipt_mutation, cited_receipt_never_committed.
  KNOWN GAP, recorded rather than papered over: citations are
  scraped from PROSE, so a receipt cited as a bare filename is
  invisible to both lock and hook — RESULTS L31402 cites
  "logs/streamwd/pass12_B1.jsonl, run_B1.log" and the second is
  unprotected. A bare-filename matcher would false-positive on
  ordinary words, so the fix is STRUCTURED RECEIPT REFERENCES,
  which is the concrete argument for item (4) rather than a defect
  to hide. Suite green 870 passed.

  AMENDMENT 7 (2026-08-16): ITEM 3 SHIPPED — internal citation
  migration to STABLE ENTRY IDS, the transition Artin specified
  rather than a perpetual line-number repair loop. Design: DUAL
  CITE. scripts/anchor_guard.py rewrites every link-form anchor
  Title -> RESULTS.md#L74 to the same link carrying
  "id:<results-index entry id>" as its markdown link title — the
  clickable line anchor survives, the id rides in the markdown
  link title, and the id (results-index.jsonl, append-stable) is
  now the source of truth while the line number is a repairable
  cache. Modes: report / --migrate (adds ids; refuses to guess on
  an anchor that matches no index heading) / --repair (recomputes
  L from id; exits nonzero on an id the index lacks). Migration
  ran inside its window: all 388 link-form anchors landed exactly
  on index heading lines (the existing heading test enforced
  this), so the line-to-id mapping was unambiguous — 388 migrated,
  0 unresolved, idempotent on re-run, --repair a no-op after.
  tests/test_docs_integrity.py::test_anchor_id_coverage_and_sync
  ratchets id-less link anchors at 0, requires every cited id to
  exist in the index, and requires the cached line to match the
  index (fix = --repair, never by hand). SCOPE: docs/RESULTS.md
  itself untouched — append-only ledger, its internal L#### refs
  (the measured 34/123 mid-entry pointers) are historical text and
  many point mid-entry DELIBERATELY; they stay as written. Bare
  non-link anchors (3 across the repo) remain outside the id
  channel, covered by the heading test only.

  AMENDMENT 8 (2026-08-16): ITEM 4 SHIPPED — machine-readable
  prereg/bar schema + deterministic adjudicator. The PROMOTED law
  (bar_adjudicable = measurement_valid AND arms admissible AND
  contrast admissible) is now executable, which is the graduation
  bar the schema design itself was held to.
    llmopt/lab/prereg.py — schema validator (unknown/typoed keys
    REFUSED loudly; refuted_if and registered_prior REQUIRED;
    every bar names declared arms and a numeric value already on
    the page) and adjudicate_prereg(): per bar exactly one of
    FIRE / NO-FIRE / UNRESOLVED with the reason chain. Two failure
    shapes kept apart BY TYPE: UNRESOLVED is a bookable scientific
    outcome (inadmissible arm, not-run, invalid measurement);
    MetricContractError is a pipeline bug (measurement is not the
    registered quantity — wrong metric/population/aggregation) and
    never books. Numeric comparison delegates to
    llmopt.lab.metrics.adjudicate, so the wrong_metric_population
    incident guard is the same code path.
    scripts/adjudicate.py — CLI; exit 0 only if every bar reached
    FIRE/NO-FIRE, exit 2 on any UNRESOLVED.
    docs/preregs/stream-wdistill-0.json — FIRST FIXTURE, marked
    RETROSPECTIVE in-file (written after receipts; not a pre-reg).
    tests/test_prereg_schema.py re-derives the booked audit-repair
    verdict from the booked numbers: BAR 1 NO-FIRE (0.3674% v
    10%), BAR 2 UNRESOLVED (arm A inadmissible, 19 bytes over B1),
    BAR 3 FIRE (+4.54%) — adjudicator and ledger must agree or the
    suite goes red. Plus: invalid measurement unresolves every
    bar; missing measurement books "not-run"; the ORIGINAL
    incident (layer bar handed an expert:0 number) raises instead
    of adjudicating.
    STRUCTURED RECEIPT REFERENCES land here too: the prereg's
    receipts list feeds scripts/gen_receipt_lock.py (lock grew
    78 to 80 paths; run_B1.log — the bare-filename gap from
    AMENDMENT 6 — is now content-addressed). Item 2's known gap
    CLOSED for any thread that ships a prereg JSON.
    .claude/skills/adjudicate/SKILL.md — the /adjudicate ritual:
    JSON at pre-reg time (same commit as the prose entry, which
    remains the registration of record), observations at verdict
    time, book the tool's per-bar lines verbatim; prose may add
    reading but may not overrule an outcome.
  Suite green 879 passed. Item (5), claim-linter, now has its
  machine-checkable objects: bars, arms, admissibility reasons,
  populations — the precondition Artin set for building it.

  AMENDMENT 9 (2026-08-16): ITEM 5 SHIPPED — claim linter, built
  LAST per Artin's ordering, conditional on item 4's objects.
  Three layers (llmopt/lab/claimlint.py + scripts/claim_lint.py):
    (1) DENY REGISTRY docs/claims.deny.json — superseded readings,
    append-only provenance, each pattern carrying its refuting
    entry id: near-isotropic; misalignment-is-the-reason;
    statistically-indistinguishable; independent gauge evidence;
    matched-bytes scalar; Lloyd-optimal; Nx-more-structure;
    RUN_TAG-as-graduation. ERROR on match.
    (2) OVERCLAIM WORDS — context-free phrases carrying proof
    obligations the linter cannot see (differ-only-in,
    statistically-significant, independently-verifies, proves,
    exactly-captured). WARN naming the obligation and incident.
    (3) ADJUDICATION CHECKS (--prereg/--obs) — the strong layer:
    ERROR when a fire/no-fire sentence contradicts the
    deterministic adjudicator, when any verdict sentence targets
    an UNRESOLVED bar, when contest wording (matched/winner/
    loses/beats) appears while an UNRESOLVED bar is in scope, and
    when prose names a bar the pre-reg lacks.
  Regression corpus = the REAL seven shipped overclaim sentences
  from STREAM-WDISTILL-0 (tests/test_claim_lint.py, incident class
  prose_overclaims_object): every one caught; the booked corrected
  reading lints CLEAN as the negative control; the correct
  BAR 1 NO-FIRE / BAR 3 FIRES / BAR 2-descriptive verdict text
  passes layer 3 with zero errors. /book now runs the linter as
  step 0 on every draft; reading corrections add their dead phrase
  to the registry in the same commit. KNOWN LIMIT, stated: the
  contest-word rule is scope-coarse (any contest word while any
  bar is UNRESOLVED) — precision over recall was rejected because
  the incident class shipped seven times in one day; false
  positives are resolved in the booking text, not by weakening
  the rule. All five automation items now SHIPPED.

- **BANKED (2026-08-16): Qwen3.8-27B as the dense no-router control
  and small-frontier subject for streaming/weight-space rungs**
  (Artin: the ask and the model pick — "very good model for its
  size"; house: the slotting).
  The mapping: Alibaba Tongyi Lab dense multimodal 27.8B, released
  2026-08-14, Apache 2.0, 262k context, weights stated for HF/
  ModelScope. Takes the slot previously penciled for
  Muse-Glimmer-30B (never banked in this ledger, memory-only):
  no-router DENSE control for MoE-anatomy comparisons, plus a
  candidate subject for weight-only streaming rungs (the
  STREAM-WDISTILL method minus the expert-sharing bars — a dense
  FFN has no expert axis, so arms C/D/E do not transfer; scalar/VQ
  arms A/B and the streaming harness do).
  Measured anchors: NONE in this lab yet. External benchmark claims
  (Terminal-Bench 2.1 73.0, OSWorld-Verified 84.3, "surpasses
  larger models") are VENDOR/BLOG numbers, unverified here — the
  house has measured nothing on this model.
  Honest breaks: one source claims the 27B open weights were
  DELAYED relative to the Max release — verify the actual HF repo
  exists and shas match the model card BEFORE any rung pins it.
  27.8B bf16 is ~56GB, over the Mac's 36GB for full residency —
  layer-streaming or quantized residency only, and the
  one-resident-30B-class rule applies beside any live job.
  Testable residue: (1) dense-control cell in any MoE-anatomy
  rung: does the routed layer's structure claim survive against a
  dense FFN of matched scale? (2) STREAM-WDISTILL-class A/B arms
  on its FFN blocks at matched bytes. (3) drafter/distill target
  for spec-decoding on Mac (36GB fits streamed or 4-bit).
  Attribution: Artin (model pick, banking ask), house (fit
  analysis, fences).

  AMENDMENT 10 (2026-08-16): EXTERNAL-REVIEW ADOPTION PASS on the
  five shipped items, with TWO DOWNGRADES the house verifies and
  accepts (GPT seat; every finding checked against the source
  before adoption, per standing rule).
  DOWNGRADE 1: AMENDMENT 8's "the law is now executable" was an
  overclaim — adjudicate_prereg() had measurement validity, arm
  admissibility, and the METRIC CONTRACT, but no first-class
  CONTRAST object: two individually admissible arms could still
  form a defective relation (dose mismatch, unpaired seeds, byte-
  convention mismatch) invisibly. Corrected claim: measurement +
  arm + metric-contract adjudication was executable; contrast
  status was PARTIAL. Closed in the same pass: observations now
  carry contrasts:{bar: {admissible, reason}} gating adjudication,
  and bars carry optional CONJUNCTS (compound predicates), which
  made 0S BAR 3's "beats the mean AND all three twins" fully
  executable — the conjunct the 0S JSON had disclosed as
  manual-check-only. Regression tests for both.
  DOWNGRADE 2: cited_receipt_never_committed was GRADUATED on a
  COUNT ratchet (<= 7 absent) — recovering one legacy file would
  free a slot a new booking could spend on new uncommitted
  evidence. Now an IDENTITY ratchet (the exact seven legacy paths
  pinned; shrinking welcome, substitution refused) plus a COVERAGE
  test (lock keys must equal the recomputed citation set — nothing
  regenerated the lock on new citations before) and ledger_regen
  now regenerates the lock on RESULTS/prereg changes. Status
  restored to GRADUATED with those three pieces landed.
  ALSO ADOPTED, same pass: (a) metric_unit_mismatch — Metric
  carried unit but never compared it, bytes could difference
  against seconds; (b) receipt-to-observations ADAPTER for 0S
  (scripts/obs_from_receipt_0s.py, committed BEFORE the 0S receipt
  existed): locked receipt -> typed observations -> adjudicator
  with no hand transcription, refusing smoke rows, partial
  populations, and WALLED arms (a partially trained VQ stack
  produces an ordinary-looking number; the adapter makes the
  logical arm inadmissible structurally); (c) claims.deny.json
  provenance fixes — two superseded_by ids pointed at the
  amendment that INTRODUCED the phrase (-READING-2) rather than
  the one that withdrew it (-READING-3), now tested against the
  index; (d) deny rules gained SCOPE regexes — "statistically
  indistinguishable" is not globally invalid science, it is
  invalid for this thread's corrected reading; scopeless rules
  stay global; (e) claim lint moved INSIDE scripts/book.py as a
  booking refusal, not just the /book ritual; (f) v2 qualification
  tolerance pinned PRE-LOOK (f1e0da8, 20:55 EDT, before the full
  v2 receipt landed at ~21:03): per-arm 5e-3 relative on
  operator_layer, 1e-6 for deterministic scalar arms, with v2's
  CUDA nondeterminism sources (index_add_, weighted bincount)
  named for the promotion gate.
  NOT adopted yet, named openly: v2 in-code correctness gate /
  shard sha / code_commit / unique output paths (prototype stays
  fenced as descriptive); evidence-graph staleness layer (banked);
  strict observation-schema validation beyond the contract checks.

- **BANKED (2026-08-16): the whole-model STREAM-WDISTILL program —
  LAYER-CENSUS, then WHOLE-0 (the streaming compression compiler),
  then MODEL-1 (the runnable artifact)** (GPT seat: the decomposition
  and the "stop calling it distillation" correction; Artin: the
  original stream-and-compress-without-inference ask and the GO;
  house: fences and the census pre-reg).
  The mapping: the layer-22 rung was instrument validation, not the
  destination. Three rungs upward: CENSUS-0 (does the codec ranking
  generalize across depth — PRE-REG booked this date, chain running),
  WHOLE-0 (stream all 43 MoE layers' routed experts shard-by-shard,
  bounded residency, emit a compressed artifact; compiler-correctness
  rung with a CONSERVATION LAW — source tensor keys = compressed
  union explicitly-passthrough, zero silently dropped), MODEL-1 (a
  runnable checkpoint; the first rung in this family that needs
  model forwards — logits/KL, perplexity, greedy-token agreement,
  routing shifts; connects weight-space error to function-space
  error to capability error). Heterogeneous codec by family: routed
  experts (this machinery), shared expert (own family), attention
  (own rung — the architecture carries low-rank/compressed attention
  structure), embeddings/head (separately budgeted), norms/router/
  small tensors (passthrough at native precision). DSpark module:
  preserve or declare excluded, never silently drop.
  Measured anchors: v2 full layer 26.5 min on the 3080 (7.7x Mac
  v1); 167 GB / 48 shards official artifact; one shard per MoE layer
  (index-verified for L2/5/12/22/33/40/42).
  Honest breaks: weight-space compression is NOT behavioral
  distillation — no path from here to a different-architecture
  student without teacher forwards; a compressed artifact is not yet
  a runnable one (decode-on-demand runtime, offline dequant, or
  compressed-domain kernels — the axiom C++ leg's natural target:
  y = C(W)x without reconstructing W).
  Testable residue: CENSUS-0's registered kill/promote; the
  conservation law as a WHOLE-0 bar; MODEL-1's exact greedy-token
  agreement against the vendor artifact (the house's own equivalence
  oracle applied to its own compression).
  Attribution: Artin (ask, GO, machines), GPT seat (program
  decomposition, family taxonomy, conservation law), house
  (census design, fences, drivers).

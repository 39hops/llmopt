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
pre-reg before fire. [SLATE UPDATE 2026-08-20: item (a) is SPENT —
PRE-REG EX4-UNIF (RESULTS 23592, 2026-08-09) -> VERDICT
(RESULTS 28597, 2026-08-13): neither primary bar; unif0 +19 /
unif1 -1 (split-draws knife-edge, |20| one solve under the +21
bar); top-80 -26 with a 9.56% recall confound; EX4-COMPOSITION-1
desk rider eliminated window overlap and read the demand-dose
curve as non-monotone. Live residue after (a): the
per-layer-count-matched UNIFORM arm (keep invp's exact per-layer
deletion counts, draw expert ids uniformly within layer, breaking
rank matching while holding placement — the one control cell not
in the ledger; ex3_del_rand0 already matches layer AND rank by
construction) and the demand-dose ladder at fixed rank class.
(b) and (c) remain unregistered and unrun.] [RESIDUE MEASURED
SAME DAY — VERDICT EX5-LAYERMATCH-0 (RESULTS L37996, 2026-08-21):
layer-only family 0/3 (median -3), fresh rank-window family 1/3
(sums -12/+25/+10, range 37 — rand0's +28 was substantially a
draw property), named crest replicates +43 3/3 on a virgin triple
and beats 2/3 of its own rank class. The benefit is scoped to
carrier IDENTITY. Remaining live: the fixed-rank-class dose
ladder, slate items (b) TRAJ anatomy + (c) physics-gate scalpel —
(b) is now the sharpest question: what distinguishes the named 80
from same-window draws that do nothing or harm.] Attribution: Grok (sharpening), three Opus
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
  structure" from "any deletion helps." [MEASURED 2026-08-13,
  residue re-pointed 2026-08-20: EX4-UNIF answered NEITHER cleanly
  — unif0 +19 v unif1 -1 is a split-draws knife-edge one solve
  under the +21 bar, so "any deletion helps" is NOT supported and
  "horizon structure" is not yet separated either; the frame's
  next honest discriminator is the per-layer-count-matched uniform
  arm / fixed-rank-class dose ladder named in the EX-ANAT-4 slate
  update. MEASURED 2026-08-21 (EX5-LAYERMATCH-0): layer-matched
  identity draws do nothing (0/3) and the rank-window class is
  draw-lottery (1/3, range 37) while the named carriers replicate
  +43 — for the horizon frame this means the interface story
  survives only in IDENTITY form: WHICH experts sit at the
  interface matters, their rank/placement class does not.] Scale corollary (Artin,
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
  Measured anchors (updated 2026-08-17): OBSERVATION
  QWEN-STREAM-PROBE-0 — first house measurement. W4 beats the
  DP-optimal scalar by 14.3% on layer 32's dense FFN; width
  inversion and locality null transport from V4; scalar ladder
  inverts (ternary beats uniform-4, cause unmeasured). External
  benchmark claims remain VENDOR/BLOG numbers, unverified.
  Honest breaks: the weights-availability fence is RESOLVED —
  Qwen/Qwen3.8-27B public, safetensors, Apache-2.0, revision
  1d4bf0f2 pinned in the probe driver; the "27B delayed" source
  was wrong. Tensor naming carries a multimodal wrapper
  (model.language_model.layers.*), and the tower mixes linear_attn
  with standard attention — the attention family is NOT plain.
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
  Testable residue: CENSUS-0's registered kill/promote — MEASURED
  SAME NIGHT (OBSERVATION 2026-08-17: PROMOTE fired, ranking stable
  at all five depths, W4-class coding leads; v2-descriptive class);
  the conservation law as a WHOLE-0 bar; MODEL-1's exact
  greedy-token agreement against the vendor artifact (the house's
  own equivalence oracle applied to its own compression). NOTE the
  0S verdict booked the same night NARROWS the codec choice: W32
  residual stacking is refuted at this rate (indistinguishable from
  the optimal scalar), so WHOLE-0's expert codec is W4-class, not
  the EXEC1-era W32.
  Attribution: Artin (ask, GO, machines), GPT seat (program
  decomposition, family taxonomy, conservation law), house
  (census design, fences, drivers).

- **BANKED (2026-08-17): model entropy is a MEASURABLE — the
  rate-distortion ladder as a weight-entropy instrument, and the
  Qwen-first runnable-artifact program** (Artin: the "dense,
  entropic, blackhole-like" intuition and the Qwen-first ask; GPT
  seat: the bridge-rung program and the compressed-domain GEMV
  design; house: the literalization and fences).
  The mapping: "how entropic is a model" has a non-metaphorical
  form — the error-vs-bits curve of a fixed codec family IS a
  distortion-rate point set, i.e. an operational entropy estimate
  of the weight distribution. Tonight's ladders are exactly that
  instrument; cross-model comparisons of the CURVES (never raw
  errors) are the legal form. Related banked instrument: the
  area-law desk probe. HONEST BREAK, stated hard: no link to
  gravitational physics is claimed — black-hole language stays
  metaphor; Poplawski-class bounce cosmology (the "gateway"
  hypothesis) is published, coherent, and untestable from inside,
  and none of the lab's star/gravity/blackhole-NAMED rungs touch
  it. Charter note: horizon/area-law MATHEMATICS as analysis
  frames = methods, welcome; physics claims = not ours to make.
  The program (adopted from GPT seat, Artin GO direction):
  v2 promotion gate -> Qwen bridge rung (light FFN depth census;
  desk byte-census by tensor family solving a <=7.5-8 GiB weight
  budget for the 10GB 3080; one representative probe each for
  linear-attn / full-attn / embeddings / LM head; scalar-inversion
  diagnosis via S2 levels + cell occupancy) -> QWEN-WHOLE-0T
  (text tower only, vision excluded, bounded delete-after-compress
  transactions, conservation law) -> SLOW reference decode +
  frozen functional battery (greedy agreement, KL, perplexity,
  small task set, readable chat) BEFORE any fast kernel ->
  QWEN-RUNTIME-1: compressed-domain W4 GEMV (256-codeword LUT dot
  products, uint8 indices, E8M0 scales — never materialize fp16
  weights; the axiom C++ leg's y = C(W)x target). V4 WHOLE-0
  returns after, as the compiler-correctness rung.
  Testable residue: (1) FFN ordering stability across Qwen depth —
  MEASURED SAME NIGHT (QWEN-FFN-CENSUS-0: stable 4/4, stop-rule
  fired); (2) whether the FFN codec destroys attention/embeddings
  — open, and UPGRADED per the 02:39 refinement: the family probes
  are RATE probes (S2@2b, W4@2b, ~4b reference each) answering
  "cheapest acceptable rate", not codec-only; linear_attn (20%)
  first and classified by TENSOR ROLE (projections v state/conv/
  gating vectors — small numerically-special tensors passthrough,
  the compiler operates on a role table, never family==codec); (3)
  greedy agreement of a ~2-bpw 27B on a 10GB card — eval FROZEN
  pre-artifact 2026-08-17 (evals/qwen_model1/, teacher-forced
  core, no-retuning rule); (4) the entropy-curve comparison
  V4-experts v Qwen-dense. ADOPTED STOP CONDITION (GPT seat,
  verbatim intent): after the four non-FFN family probes, no
  further weight-anatomy experiment may block QWEN-WHOLE-0T
  unless a probe shows catastrophic reconstruction at every
  budget-compatible rate — the program is now optimized for
  reaching "coherent generation YES/NO on the 3080", not for
  compression science without end. DEPENDENCY NOTE: the promotion
  gate must bless a SHARED codec module that the Qwen compressor
  imports — promoting streamwd_v2.py while qwen_whole copies the
  old kmeans would recreate the dual-copy seam; codebook creation
  records source tensor hash, seed, codec params, codebook hash,
  realized bytes, code commit. MTP (1.5%) scope decision is
  explicit in the WHOLE-0T pre-reg, never a silent drop.
  Attribution: Artin (ask, framing, GO), GPT seat (program), house
  (fences, instruments).

- **BANKED (2026-08-17): the Qwen runtime program — CPU reference
  oracle, Metal direct-W4 as the Mac leg, CUDA direct-W4 as the
  3080 leg, and precision-as-escalation reusing the closed Ozaki
  lineage** (Artin: the MPS ask and CPU interest; GPT seat: the
  three-path ladder, the teacher-baseline freeze catch, and the
  no-new-Ozaki correction after the house pointed at the closed
  work; house: verification against the ledger).
  The mapping: three execution paths against one ~7.x GiB
  compressed artifact — QWEN-RUNTIME-0R (slow portable CPU decode,
  the does-it-talk oracle), Metal direct-W4 GEMV (36GB unified
  memory, llmopt/kernels/metal.py lineage), CUDA direct-W4 (hard
  10GB residency). Oracle ladder: each faster backend scores
  against the frozen CPU reference, all against ONE immutable
  teacher-logit baseline produced by a single streamed vendor CPU
  pass (procedure frozen in evals/qwen_model1/SPEC.md before any
  artifact exists — without it MODEL-1 would freeze a KL test it
  cannot compute on the target hardware). Headline metric beyond
  tok/s: effective compressed-weight bandwidth (bytes touched/token
  x tok/s).
  Measured anchors: Ozaki/exactness lineage CLOSED on CUDA
  (scratch/ozaki_* family; int8-sliced exact beats native fp64 —
  CLAUDE.md precision doctrine); Metal side: exact integer carrier
  proven, exact_gemm correctness built with NO Mac wall number,
  fp32-limb Metal GEMM NOT BUILT, M-series exposes no integer
  simdgroup MMA (RIFF fp32-limb bank, 2026-08-10).
  Honest breaks: W4 inference does NOT want exact arithmetic as
  baseline — the compression error (~0.34 relative weight-space)
  dominates fp32 rounding by orders of magnitude, so exactness is
  an ESCALATION/ORACLE tool (backend-KL divergence triggers the
  ladder: higher accumulator -> exact carrier -> tiled
  exact_gemm/fp32-limb), per the standing doctrine that exact
  arithmetic is a speed/determinism lever, not capability. The
  Ozaki-W4 "synthesis" (codebooks in integer-carrier-friendly
  form) is a named idea with zero measurement, parked until the
  model talks. Hybrid CPU+GPU offload for a 10GB overflow: never
  auto-offload per-token weight traffic over PCIe; profile
  embeddings and LM head independently (head is a full matmul per
  token, "same shape as embeddings" is not a placement argument).
  Testable residue: (1) RUNTIME-0R coherent-generation YES/NO;
  (2) per-backend KL v the CPU reference (the escalation trigger);
  (3) effective-bandwidth comparison Mac-Metal v 3080-CUDA on one
  format; (4) the banked Metal exact leg finally getting its
  consumer.
  Attribution: Artin (MPS/CPU direction), GPT seat (program +
  self-correction), house (ledger verification, fences).

- **BANKED (2026-08-17): follow-ups to a multi-arm compile are byte
  RECOMPOSITIONS, not recompressions — and the decision tree fires
  on pre-committed numeric triggers, not on reading the scores**
  (GPT seat, relayed by Artin; house verified feasibility).
  The mapping: because WHOLE-0T's three rate tables compiled in ONE
  pass with shared encoders, the alternative encodings of every
  family already exist on disk; an io-attribution experiment (D =
  A + B's embed shard, E = A + B's lm_head shard) is a one-file
  swap with a merged manifest — zero new quantization randomness,
  treatment isolated at the byte level.
  Measured anchors: WHOLE-0T receipt sha pattern (A==B on 16
  shards; shard 00003 = embed alone; shard 00018 payload = lm_head
  alone, 15 MTP keys excluded at zero payload).
  Honest breaks: attention attribution does NOT get the simple
  version — attn tensors span many shards, so the B/C leg needs
  key-level container rebuilds with its own conservation law;
  and rope_calls>0 in the teacher gate proves module EXECUTION,
  not mathematical dependence of outputs on cos/sin — accepted as
  sufficient for this incident class, no dependency tracer unless
  a new problem appears.
  Testable residue: PRE-REG QWEN-MODEL1-TREE (T1/T2/T3 triggers,
  20% relative CE+KL conjunction, registered before any score).
  Attribution: GPT seat (recomposition + pre-committed triggers),
  Artin (relay + teacher-traversal push), house (receipt
  verification, trigger arithmetic, registration).

- **BANKED (2026-08-17): weight-space "friendliness" is a property
  you can FORCE — precondition the model into a codec-friendly
  gauge before quantizing** (Artin's ask "can we take a model and
  force it to have consistent weightspace?"; house formalization).
  The mapping: the same function lives at many weight arrangements
  (the never-score-by-weight-distance law — permutations,
  rescalings, rotations are gauge freedom). A rotation R applied as
  W -> WR with R^-1 absorbed into the adjacent op leaves the
  function IDENTICAL while reshaping the stored distribution the
  codec sees — this is the published incoherence-processing family
  (QuIP#-class, SpinQuant-class). V4 is the existence proof in our
  own receipts: its stored experts are dequantized-MXFP4 (vendor
  quantization-aware pipeline), and SCALAR-MASS-CENSUS-0 measured
  exactly how differently that stored distribution treats a fixed
  alphabet (64% v 80-84% central mass) — a training/vendor pipeline
  already SHAPED that weight space; we would be doing it
  deliberately, at zero training cost, with orthogonal transforms.
  Measured anchors: SCALAR-MASS-CENSUS-0 (both-sides mass);
  QWEN-FAMILY-PROBE-0 (homogeneity = today's gauge, not fate);
  2026-07-06 weight-reader ablation (invariance-not-imposition).
  Honest breaks: rotations reshape SCALAR-friendliness cheaply, but
  our W4 already exploits 4-dim structure — the win over rotated
  scalars is unmeasured and could be small; block-E8M0 scales
  interact with rotation (a rotation that evens out blocks can
  shrink exponent variance, or wreck it); grok's "consistent enough
  that the ranking holds" conflates codec-RANKING transport
  (measured) with distribution SHAPE (model-relative, exactly as
  Artin suspects).
  Testable residue: ROTATE-THEN-CODE rung — same frozen probe
  ladder on rotation-preconditioned tensors (Hadamard or random
  orthogonal, function-identity verified by the no-op precondition)
  v the booked unrotated curves; fires ONLY after the MODEL-1
  curve lands (compression freeze holds).
  Attribution: Artin (the force-consistency question), grok seat
  (the phrasing that surfaced it), house (gauge formalization,
  anchors, fences).

  CORRECTION (2026-08-17, same day, GPT seat review — adopted
  before any rotation work exists): three fences replace the
  original wording. (1) "function bit-identical" is retracted —
  WR then R^-1 x is ALGEBRAICALLY equivalent in real arithmetic
  and generally NOT bit-identical in floating point (changed
  multiply/add order and operand representations); the no-op gate
  is a registered numerical TOLERANCE pre-quantization, not a
  bit-identity. (2) Rotations push through ELIGIBLE linear-linear
  interfaces only — generic R does not commute through elementwise
  nonlinearities, RMSNorm in arbitrary bases, gating products,
  attention structure, or linear-attention state; Qwen's tower is
  not a chain of dense matrices. (3) "Free" means ZERO TRAINING
  COST only — some rotations fold offline into weights, others
  need an online activation transform whose runtime cost depends
  on placement and fusion (a per-token 5120-wide rotation could
  eat the codec win in a direct-W4 kernel). Also NARROWED: the V4
  claim "the vendor QAT pipeline SHAPED that weight space" is
  causal attribution beyond the receipt — pre-MXFP4 V4 weights
  were never measured, so training, the MXFP4 transform itself,
  or both could produce the 64% v 80-84% shift. V4 remains an
  existence proof that the STORED REPRESENTATION presented to a
  codec can differ radically; the cause is unresolved. Additional
  measured targets when ROTATE-THEN-CODE fires: block-max/exponent
  distribution pre/post (rotation moves the E8M0 scale
  distribution independently of codebook geometry — two separate
  payoff channels), plus scalar AND vector distortion, realized
  bytes, the tolerance no-op, MODEL-1, and unfolded-transform
  runtime cost.

- **BANKED (2026-08-17): RESIDENT-DRAFT, STREAMED-VERIFY — the
  compressed artifact accelerates its own teacher** (Artin's ask
  "surely there is something... a smaller compact model in RAM that
  queries/decodes the weights"; house formalization).
  The mapping: a teacher-class rollout is autoregressive — 1 token
  per full 52GB weight sweep. Speculative decoding with the
  RESIDENT compressed artifact (A, 6.5GB, RAM-speed) as draft and
  the STREAMED vendor as verifier checks k drafted tokens per
  sweep. Greedy spec decoding is TOKEN-IDENTICAL to the target by
  verification (the house equivalence law, llmopt/eval/), so an
  UNTRUSTED draft still yields exact vendor tokens — the
  chicken-and-egg of using the unscored artifact to speed its own
  judge is broken by exactness: acceptance rate prices speed,
  never correctness. Marries two standing threads (spec-decoding =
  gate law; streamed compression).
  Measured anchors: none for the pair; components measured
  separately (teacher sweep ~57s at 24-batch; entropy-adaptive
  draft length banked in the spec-decoding thread).
  Honest breaks: teacher-FORCED scoring (corpus/prefixes) gains
  nothing — it already processes all positions in one sweep; the
  win is rollout-only. Draft quality unknown until MODEL-1 (an
  artifact that can't talk drafts at ~0% acceptance and the sweep
  count degrades to baseline, never below). Not applied to the
  live v2d run — a fifth restart to build draft infra for a
  one-time pass loses more wall than it saves.
  Testable residue: when any SECOND teacher-class pass is needed
  (prompts_v2, a new revision, the family-attribution re-run), the
  vendor pass runs resident-draft/streamed-verify and books
  acceptance rate + realized sweeps/token v the 1.0 baseline.
  Attribution: Artin (the ask, and the "we have something in this
  repo" instinct — correct), house (the spec-decoding
  identification, exactness argument, fences).
  UPDATE (2026-08-18, VERDICT QWEN-RK-CENSUS-0): the practical
  program is DEAD for the current artifacts, which fail the two
  prerequisites INDEPENDENTLY: A fails the router screen
  (R_k@1024 0.11-0.21 v bar 0.7 on all six sampled layers,
  RK-CENSUS-0) and B fails the free-generation screen (0/60,
  EFFORT-QUANT-0). Speculative acceptance rate itself was never
  measured — these are the upstream screens, not the acceptance
  number. The exactness argument stands (correctness was never at
  stake); revive only with an artifact that passes a router/
  free-generation screen first.
  CORRECTION (2026-08-17, same day, GPT seat): "degrades to
  baseline, never below" holds for TARGET SWEEPS/TOKEN only —
  end-to-end wall can lose, and the lab measured exactly that in
  the spec-decoding thread (target passes nearly halved, acceptance
  up, wall still lost to ~40% added draft work). The residue's
  booking requirements gain a third mandatory number: acceptance
  rate, target sweeps/token, AND end-to-end wall/token, separately.

- **BANKED (2026-08-17): identity beats aggregates — the lab-wide
  meta-law** (GPT seat synthesis from the full FINDINGS corpus,
  relayed by Artin; house verification of the cited anchors).
  The mapping: equal mass/rate/coverage does not imply equal
  functional information. Instances already booked: MoE carrier
  experts (aggregate recall/coverage lenses failed; swapping a
  small NAMED expert population moved dozens of solves), dense
  head cells (specific cells load-bearing, identities change
  across births), atom-diet composition (rule/emitter identity
  over dose), STAR-PROFILE (weight-homogeneous families,
  4x-different functional fragility), and the Qwen family probe's
  own conclusion (codec-homogeneous weight space cannot name the
  sensitive family).
  Honest breaks: a meta-law is a lens, not a result — it predicts
  nothing quantitative; each instance carries its own regime tags
  and none transports numbers to the others.
  Testable residue: if MODEL-1's C-branch fires, "attention needs
  4 bits" is predicted to be TOO COARSE — the follow-up
  recomposition should find a small subset of attention
  projections/layers carrying most of the extra-rate value, the
  carrier-expert pattern at 27B scale.
  Attribution: GPT seat (the synthesis), Artin (relay + standing
  ask to mine FINDINGS), house (anchor verification, fences).

- **BANKED (2026-08-17): the weight-reader as a compression
  ALLOCATOR — teach the model that reads weights to predict where
  precision matters** (Artin's ask "can't our model that reads
  model weights help compress? teach it to optimize weights?").
  The mapping: the lab owns a measured result that raw weights are
  READABLE (2026-07-06 weight-reader: 80.8% raw, 88.4% with
  permutation augmentation — teach invariance, don't impose it).
  Compression's binding open problem after the family probe is
  ALLOCATION: weight space is codec-homogeneous, so no weight
  statistic names the functionally sensitive family — but that is
  a statement about the SIMPLE statistics we tried (mass, op
  error), not about all functions of the weights. A trained reader
  could learn the map (tensor/row features) -> (measured
  functional damage) that hand statistics miss; identity-beats-
  aggregates predicts the signal lives in specific rows/heads, and
  a reader is exactly the instrument class that can see identity.
  Two grades: (1) ALLOCATOR — predict per-tensor/per-row rate
  sensitivity, spend bits where the reader says; (2) OPTIMIZER —
  propose weight EDITS (rotation/gauge moves) that improve
  codability at fixed function, verified by the no-op gate + the
  frozen eval, never by weight distance (the house law).
  Honest breaks: the training LABELS are function-space damage
  measurements, which are exactly what is scarce — MODEL-1 + the
  D/E recomposition produce the FIRST labeled examples, and n will
  be tiny for a long time; a reader trained on weight-space error
  learns the wrong target by construction (never-score-by-weight-
  distance); charter-clean (math engines) but compute-priced like
  any training rung.
  Testable residue: after MODEL-1 books, the margin-stratified
  flip data + per-family attributions become the first label set;
  a desk-grade probe (frozen features -> predict which family the
  tree promoted, leave-one-out) prices the idea before any
  training run. Fires only after the A/B/C curve; compression
  freeze holds.
  Attribution: Artin (the ask, and the reader-to-compressor
  bridge), house (grading, label-scarcity break, lineage anchors).
  EXTENSION (2026-08-17, same day, GPT+Opus seats): the LABEL
  FACTORY replaces the starved desk probe. "Predict which family
  the tree promoted, leave-one-out" is ~3 labels and prices
  nothing. Instead, after A/B/C books: single-tensor recomposition
  B_i = B + C's precision on attention tensor i (and C_-i for
  necessity), per layer/projection, measuring dX, dK,
  d(large-margin flips), d(bytes) — hundreds of labels of the form
  "what did these specific bytes buy in function space," at zero
  recompression cost because both encodings of every tensor
  already exist on disk. MODEL-1 turns from judge into label
  generator. Two fences: (1) NON-ADDITIVITY — identity-beats-
  aggregates predicts singleton repairs will not sum; hold out
  random SUBSETS from the start and price composition error before
  any allocator claim. (2) JUDGE CONTAMINATION — any optimizer
  phase searches against a DEVELOPMENT eval and books against an
  untouched final MODEL-1 (the EX1-C30 discovery-contamination
  precedent, clean-3-fresh-seed protocol). Permutation-augmentation
  transports as "teach known symmetries," not the toy MLP's neuron
  permutations — transformer row/head permutations are symmetries
  only with the coupled transformation elsewhere.

- **BANKED (2026-08-17): the Eddington-throttle riff gained an
  observational instance — JWST black-hole-star candidates**
  (Artin's ask "star and black hole... super massive
  black-hole-suns, how is this not similar?"; house verification
  of the mapping). The 2026-era JWST "little red dot" objects are
  hypothesized as supermassive black holes inside dense gas
  envelopes shining star-like — a configuration stable exactly
  because radiation pressure balances gravity at the Eddington
  limit: consumption self-gates, which is the banked row-25 frame
  ("the universe runs a dispatcher too") now with a candidate
  photograph. Honest breaks: the BH* interpretation is a live
  hypothesis, not consensus; the gravitational-lensing side (mass
  curves spacetime, light follows geodesics) has NO lab mapping
  and is not claimed to; no physics rung fires — the lab's physics
  leg remains ZX-calculus. Testable residue: none for us —
  astronomy's to test.
  FENCES (Opus seat, verified citation: Rusakov, Watson et al.,
  Nature 649, 574-579, 2026-01-14): that paper's measured MECHANISM
  is electron-scattering broadening in dense ionized cocoons at
  1e5-7 solar masses — the Eddington-balance STABILITY framing
  belongs to the envelope models, not the observation; and the
  little-red-dot population is actively contested (a 2026-07 ApJ
  paper reframes it again). The bank is "a live thread whose
  TOPOLOGY instantiates the riff", never a confirmation.
  Attribution: Artin (the ask, twice — the
  original throttle question and this instantiation), house
  (Eddington mapping both times).
  EXTENSION 2 (2026-08-17, GPT+Opus seats, same day): (a) LABELS
  are the scarcity, so the factory goes HIERARCHICAL — a cheap
  frozen prefix/margin micro-battery across EVERY singleton B_i,
  stratified by apparent impact and family/layer with random
  negatives kept, full X/K corpus evaluations spent only on a
  PREREGISTERED subset, and the cheap-diagnostic-predicts-full-
  damage calibration measured BEFORE expanding. (b) Hold out whole
  LAYERS, not just random tensors — leave-one-layer-out is the
  only test separating a learned identity table ("layer 43's
  o_proj matters") from a transferable allocator. (c) Score
  recovery PER MARGINAL BYTE (dX/db, dK/db), not raw sensitivity —
  the real problem is constrained rate-distortion, and a tensor
  that helps a lot for many bytes can lose to one that helps less
  for almost none.

- **BANKED (2026-08-17): cross-model KV-cache transfer — the KV relationship between family models is substantially LINEAR** (Artin brought arXiv:2608.03893; house mapped to ledger).
  The mapping/the math: per-head ridge regression (closed-form, no training) maps a source model's KV cache into a target model's, with RoPE stripped first so the map is position-agnostic; top-k source-layer selection per target layer. Paper numbers: 73-98% target accuracy retained across 4 matched-KV pairs, 2.7-25x faster than re-prefill; Qwen3 14B->32B single source layer explains 56% key / 32% value variance; calibrated on only 500 sequences.
  Measured anchors (house): none yet — external paper numbers only. Adjacent house results: weight-reader (raw weights readable 80.8%, RESULTS 2026-07-06 — cross-net LINEAR readability precedent), RESIDENT-DRAFT/STREAMED-VERIFY bank (a draft/verify pair sharing prefill via a mapped KV is exactly this paper's mechanism).
  Honest breaks: paper needs matched KV heads/dims (same family, dense full-attention only) — our 27B tower is 48/16 hybrid linear/full attention, out of the paper's scope for 48 of 64 layers; per-pair k was tuned on eval benchmarks (leakage the authors admit); house has no second family member resident to test against.
  Testable residue: (1) house-scale probe — does the linear map hold between a model and its OWN quantized artifact (A v vendor weights, same shapes by construction)? If KV of the compressed tower is linearly reachable from the uncompressed one, that is a cheap function-space compression instrument (KV-space MODEL-1 analogue); (2) spec-decode: map teacher KV into a small drafter instead of drafter prefill.
  Attribution: Artin (the pointer, "KV cache between diff models?"), paper Cross-Model KV Cache Transfer in LLM Families (arXiv:2608.03893), house (ledger mapping + residues).

- **BANKED (2026-08-17): momentum-space phase portraits, done right — perturb the PHASE STATE, hold the laws fixed, measure neighboring-trajectory divergence** (Artin's double-pendulum ask; GPT seat formalized; house verified repo claims).
  The mapping/the math: training state S=(W, m, v, data cursor, RNG, scheduler, t) is a deterministic discrete dynamical system S_{t+1}=F(S_t) — dissipative (AdamW damps), NOT Hamiltonian, so no conservation but attractors/basins/finite-time Lyapunov exponents all live. Per-neuron polar split w=ru: radial momentum p_r=u^T p (grow/shrink), tangential p_perp (rotation), giving pendulum-like coordinates (r,p_r) and (theta,p_theta). Better y-axis than raw exp_avg: the EFFECTIVE update p_eff = -lr*(m_hat/(sqrt(v_hat)+eps) + wd*W) — where the model is ABOUT to move. The atlas experiment: one checkpoint S*, 2-D grid W_0=W*+a*e_q, m_0=m*+b*e_p (everything else frozen incl. RNG/data order), each pixel colored by finite-time Lyapunov rate lambda_T = (1/T) log(delta_T/delta_0) from an epsilon-twin run under IDENTICAL batches. Dual divergence lanes mandatory: weight-space D_W AND function-space D_F (frozen probe set, KL) — the 2x2 table (stable/chaotic x weight/function) separates gauge chaos from real chaos. Liouville check: track perturbation-parallelepiped volume; lambda_1>0 with sum(lambda)<0 = dissipative chaotic attractor, the interesting outcome.
  Measured anchors: PHASE-PORTRAIT-1 (booked) — mean weight speed fell ~100x (1.06e-4 -> 1e-6), collective settling; 12,288 gate rows, ckpt every 900 steps with optimizer state. House-verified correction (grep scratch/phase_portrait_precompute.py:71-74): vel and mom were stored as ROW NORMS — direction/sign discarded, so PP-1 was (theta,|theta_dot|) not (theta,theta_dot); two opposite-moving neurons folded to one state. That is plausibly why PP-1 read as pure settling. init-is-the-address + merge-space results fit the damped-attractor picture.
  Honest breaks: mps float training is RUN-LEVEL NONDETERMINISTIC at fixed seed (booked 2026-08-15) — the epsilon-twin design REQUIRES deterministic replay, so this runs on CPU (19M is cheap) or cuda with determinism flags, never on Mac mps; a "closed system" here is closed only computationally, not physically; 10^7-dim projections to 2-D planes can manufacture or hide structure (basis must be registered pre-look); weight distance is not function distance (house law) — the D_F lane is not optional.
  Testable residue: PHASE-PORTRAIT-2 / STABILITY-ATLAS-1 on the 19M system — (1) re-run PP-1 capture keeping VECTOR p_eff at 25-100-step cadence; (2) the (a,b) atlas colored by functional finite-time Lyapunov rate; (3) the volume/Liouville probe. Kill condition: if lambda_T <= 0 everywhere at every reachable epsilon and T, "training chaos" is dead for this system and the atlas is a flat basin map. Prior (house, weak): globally contracting with SOME positive finite-time directions early — the dissipative-chaos quadrant.
  Attribution: Artin (double-pendulum/momentum-space ask, magnitude+direction memory), GPT seat (formalization: polar split, p_eff, dual-lane Lyapunov, Liouville test, atlas-not-lawscan correction), house (repo verification: PP-1 stored norms not vectors; mps fence; anchors).

- **BANKED (2026-08-17): 2-bit model as ROUTER — run the compressed artifact, discover which weights the computation actually leans on, load only THOSE at high precision** (Artin's ask "find where it routes to the actual model and only load those weights"; house mapping).
  The mapping/the math: this is the RESIDENT-DRAFT/STREAMED-VERIFY bank (2026-08-17) plus a weight-granular twist: instead of drafting TOKENS for a full-precision verifier, use the cheap resident 2-bit tower to identify the ACTIVE weight subset (rows with high activation mass, blocks with large contribution to the residual stream) and stream only that subset at fp16/raw precision — a per-input mixed-precision artifact assembled on demand. Adjacent published shape: contextual sparsity / DejaVu-class predictors; adjacent house shape: the hierarchical label factory in the weight-reader-as-allocator bank (B_i = B + C's precision on tensor i) is the OFFLINE version of the same question (which bytes buy the most quality), this riff is the ONLINE per-input version.
  Measured anchors: rung 3 backend KL 4.2e-8 (the 2-bit tower's hidden states track the CPU reference tightly, so its routing signal is cheap and available); no measurement yet of whether its ACTIVATION pattern predicts the full model's (that is the whole question). "fp16/bf16 double double": the house's exact-arithmetic lineage (fp32-limb / ozaki int8-sliced, RESULTS 2026-08-10 counter-book) is double-double-style limb arithmetic for EXACTNESS, a different lever than this precision-routing riff — named to keep the two from blurring.
  Honest breaks: MoE-style routing has a discrete gate to read; a dense tower has none, so "routes to" must be OPERATIONALIZED (top activation mass? largest |contribution|? attention concentration?) before anything is testable; contextual-sparsity papers show FFN activations are predictable from the PREVIOUS layer, but they predict the SAME model's sparsity, not a quantized proxy's fidelity to it; streaming fp16 rows per token re-introduces exactly the PCIe/decode traffic the resident design eliminated — the riff pays off only if the hot set is small AND stable across steps.
  Testable residue: desk-able census on receipts we can already produce — run A and the CPU reference on the same prompt, rank FFN rows by activation mass in each, measure overlap@k of the hot sets; high overlap = the 2-bit model is a faithful router and the hybrid artifact is buildable, low overlap = the riff dies for free. /desk shape, zero training.
  Attribution: Artin (the routing ask + double-double memory), house (operationalization, anchors, breaks).

  CORRECTION (2026-08-17, same day, GPT seat + house verification):
  the kill-test as first banked compared A against "the CPU
  reference" — but the CPU reference IS artifact A decoded through
  qcodec (backend KL 4.2e-8), so hot-set overlap between them is
  another backend parity test, not a router test. The comparison
  that answers the question is COMPRESSED A v PINNED VENDOR TEACHER
  on teacher-forced common prefixes (same input_ids, same
  positions). Also corrected: the routed unit is not an "FFN row"
  but the coupled intermediate channel i = (gate_proj row i,
  up_proj row i, down_proj COLUMN i), with hot signal z_i =
  act(W_gate_i h) * (W_up_i h); score both Jaccard overlap@k of
  top-|z| sets AND the teacher-mass-captured fraction R_k =
  sum_{i in TopK_A} |z_i^T| / sum_i |z_i^T| — R_k is the decisive
  number (identity can differ while captured mass stays high; the
  hybrid works iff R_k is high). Kill bar unchanged in spirit:
  low R_k kills the riff for free.

- **BANKED (2026-08-17): per-byte allocation lens — total recovery and marginal value per byte are DIFFERENT orderings, and the tree's steps must be read in both** (GPT seat, on the MODEL1-TREE receipts; house re-derived every number).
  The mapping/the math: an allocation step's worth is dQ/dBytes, not dQ. On the frozen receipts: io (A->B) +0.592 GiB buys 0.384 nat X/GiB and 0.227 nat K/GiB; attention (B->C) +1.680 GiB buys 0.348 and 0.105. Attention dominates TOTAL recovery (rel X 70.2% v 21.4%) while io is ~10% better per byte on X and ~2.2x on K. Both statements are true; only the per-byte one speaks to an allocator.
  Measured anchors: VERDICT QWEN-MODEL1-TREE receipts (logs/qwenmodel1/); byte costs from the frozen WHOLE-0T artifacts; OBSERVATION QWEN-MODEL1-POSTHOC-DIAGNOSIS books the arithmetic.
  Honest breaks: per-byte ratios divide two point readings (355/92 positions, no sampling fence) and inherit both ends' record-sensitivity floors; marginal value is measured at ONE operating point on a presumably concave curve — io's 2.2x K-efficiency at +0.59 GiB says nothing about the NEXT 0.59 GiB of io spend; family-mean efficiency can hide a high-value subset (in_proj_qkv may not equal the linear-attn average).
  Testable residue: PRE-REG QWEN-ATTN-ATTRIB-1 (this session) carries U_X/U_K per arm and the iso-rate Q-v-B contrast as registered quantities.
  Attribution: GPT seat (the lens + arithmetic), Artin (relay + "not the same thing as best allocation per byte"), house (verification, fences).

- **BANKED (2026-08-17): future-tree gate design — split INSTRUMENT-ALARM from LOW-RATE-OUT-OF-RANGE** (GPT seat; Artin adopted; MODEL-2-class lesson, explicitly NOT applied to MODEL-1).
  The mapping/the math: the MODEL-1 uniform-damage gate (X_A > 1.0 -> stop) conflates two states it cannot distinguish: shared instrument damage (all arms similarly bad) and a lowest-rate arm genuinely outside the fidelity regime while the ladder above it is healthy. Future trees should branch: X_A > 1 with all arms similarly bad -> INSTRUMENT-ALARM; X_A > 1 with X_B < 1 and X/K monotone decreasing -> LOW-RATE-OUT-OF-RANGE (drop A as baseline, allocate from the first admissible arm). The registered virtue being preserved: neither branch is chosen by session discretion after seeing scores.
  Measured anchors: MODEL1-TREE fired the conflated gate at X_A=1.061 over a strictly monotone ladder — the exact case the split would have classified as LOW-RATE-OUT-OF-RANGE.
  Honest breaks: "similarly bad" needs its own registered predicate (another threshold to freeze pre-look); a monotone ladder can coexist with an instrument defect that scales with codec rate (decode bug in the w4 path only would damage A>B>C monotonically!) — the split gate is better, not airtight; the near-miss temptation this bank exists to resist is re-scoping gates AFTER data (refused here for MODEL-1, permanently).
  Testable residue: the next whole-model tree registration (MODEL-2 class) carries the two-gate design; its instrument-side discriminator should include a rate-independent oracle (e.g. one tensor family decoded to raw in the lowest-rate arm) so "damage tracks rate" can be separated from "decode bug tracks rate".
  Attribution: GPT seat (the distinction + predicates), Artin (immutability ruling), house (the w4-only-bug counterexample, discriminator proposal).

- **BANKED (2026-08-18): conditioning-as-frame — the effort sentence as a frame-of-reference, and whether a dense forward is "aware of its whole universe" in a way a sparse one is not** (Artin, pre-caffeine 7am, relativity analogy; house counter-notes).
  The mapping/the math: Artin's frame — a model changing its deliberation length because it READS "xhigh" in its own context resembles an observer whose measured quantities depend on frame of reference; the special-relativity flavor is that there is no frame-independent "behavior of the model", only behavior-given-context. The literal version the house endorses: the conditioning context IS the frame; QWEN-EFFORT-0 measured one sentence of frame moving ~1300 tokens of trajectory. The MoE half: a dense model runs every parameter every token while a routed model activates a subset, so "the dense model is conscious of its full universe" = every-weight-participates-in-every-decision.
  Measured anchors: OBSERVATION QWEN-EFFORT-0 (dose 0/1024/1312 by system sentence, ceiling accuracy); the vendor knob is prose, not architecture.
  Honest breaks (house): conditioning is not consciousness in any load-bearing sense the lab can measure — the same mechanism explains a thermostat reading its setpoint; the MoE contrast does not hold as awareness — a routed model's ROUTER conditions on the full context too, and dense-v-sparse changes which FFN weights fire, not what information the decision conditions on; house language rule keeps ledger claims mechanistic (plain technical language, booked 2026-08-12). The relativity analogy is an ANALOGY-AS-METHOD (charter-legal) — no physics claim.
  Testable residue: the measurable fragment is instruction-following-into-behavior dose-response curves (already live) and, if ever wanted, a matched dense-v-MoE effort-knob transport test on open checkpoints — banked, unregistered.
  Attribution: Artin (frame-of-reference/consciousness riff, MoE-sparsity contrast), house (mechanistic counter-notes, thermostat break, dose anchors).
- **BANKED (2026-08-18): RESIDUAL-RECON — the quantization residual
  as a compressible correction field** (Artin's "fabricate/guess the
  missing 16 digits" ask; GPT formalization; house fences).
  The mapping: with the vendor BF16 resident during research, the
  residual R = W_teacher - W_A is computable per tensor. Three
  levels of "fake precision": (1) better dequantization (bin-center
  -> conditional-mean decode), (2) structured weight-space
  correction W_A + R_cheap where R_cheap is low-rank/sparse (rank-16
  UV^T on a 500MB tensor = tens of MB — precision as a correction
  field), (3) function-space correction (choose tiny R so
  f_{Q(W)+R} tracks f_W on calibration data — CAL-FEAS-0 territory,
  registration already parked). A teacherless variant: train a
  residual predictor g(Q(W), context) shipped WITH the artifact (a
  small neural dequantizer applied blockwise at layer-load in the
  streaming runtime, not per-matmul).
  Cheap first tests (desk, no model run): SVD spectrum of R per
  family, sparse-tail fraction, block entropy, neighbor-conditioned
  predictability, residual entropy conditioned on the W4 code.
  HOUSE FENCE (the law that bites): weight-space compressibility of
  R is a STRUCTURE CENSUS only — never score capability by weight
  distance; any promotion claim runs through MODEL-1/2 X/K on a
  held-out surface. The measured hook: the repair ladder's per-byte
  table gives exact targets to beat (a 30MB correction beating the
  461MB band promotion would be the headline).
  Testable residue: RESIDUAL-STRUCTURE-0 census (spectra + tails
  per family, priced ~one Mac evening, no compose).
  Attribution: Artin (the ask), GPT seat (three-level formalization,
  teacherless variant), house (fences, census framing, pricing).
  MEASURED 2026-08-19 (VERDICT QWEN-RESIDUAL-STRUCTURE-0): level 1
  DEAD (conditional-mean ceiling 0.086% max over 402 tensors) and
  the teacherless GLOBAL-table dequantizer DEAD (leave-one-out
  family cosine ~ 0.0015); level 2 NARROWED to per-tensor targeted
  corrections on early attention write-back projections (o_proj L3
  low-rank, out_proj L0 heavy tail — ~2 MB patch class); level 3
  (function-space, CAL-FEAS) untouched by the census.
- **BANKED (2026-08-18): ATTN-ROUTER-CENSUS — can a cheap router
  find where exact full attention wants to look?** (Artin's
  graph/vector-DB-attention ask; GPT formalization; house fences).
  The mapping: only 16/64 layers are quadratic full attention (48
  linear) — the target is not "replace attention", it is "route the
  16". Record exact attention masses a_{t,j} from the teacher on
  frozen sequences; for candidate cheap routers (recency, block
  summary cosine, hidden-state similarity, LINEAR-ATTENTION STATE
  similarity — the in-model router candidate, approximate-K, graph
  links) measure M_k = attention mass captured by the router's
  top-k candidate set. RK-CENSUS shape exactly: M_256 ~ 0.95 ->
  build sparse/hierarchical attention (block-summary retrieval,
  attention-B-tree, GPU/RAM/SSD tiers with coalesced block reads);
  M_256 ~ 0.25 -> kill before any engineering. The architectural
  hook is house-specific: the interleaved linear-attn state already
  summarizes history — the cheap substrate may BE the index for the
  expensive one.
  Fences: teacher-forced capture on frozen lists; per-layer,
  per-head reads; no SSD/graph implementation before the census;
  block-level (not token-level) retrieval is the mechanical
  candidate; oracle = recorded exact attention, later
  token-identical greedy equivalence for any implementation.
  Testable residue: ATTN-ROUTER-CENSUS-0 (capture + router ladder,
  priced ~one Mac session on the resident vendor).
  Attribution: Artin (graph/tensor-db attention, "algorithmic
  inference" instinct), GPT seat (census formalization, block
  hierarchy, M_k), house (RK-shape, linear-state-as-router
  candidate, fences).
- **BANKED (2026-08-18): FULL-ATTN-UPLIFT — turn the hybrid into
  (more of) a Transformer: convert linear-attention layers to
  softmax attention and ask what it buys the model's behavior**
  (Artin's "can we turn this thing into a Transformer?" ask; house
  formalization).
  The mapping: the vendor tower is 48 linear / 16 full. The
  conversion direction the literature travels is full->linear
  (linearizing distillation, e.g. LoLCATs-class); Artin's arrow is
  the REVERSE: uplift k linear layers to softmax attention —
  initialize q/k/v/out from the linear layer's projections where
  shapes permit, then FUNCTION-SPACE distill against the locked
  teacher (never weight matching — house law) so the swapped layer
  reproduces the original computation before any capability
  question is asked. Variants: full swap; "gradient" swap = a
  learned per-layer gate blending linear and softmax paths (the
  gate schedule over depth is the dial); depth-targeted swap using
  the LBAND result (early-linear carries the most repair value —
  is early also where softmax expressivity would bind?).
  Honest breaks (house): O(N^2) cost lands exactly where
  ATTN-ROUTER-CENSUS is trying to remove it — the two banks pull
  opposite directions on purpose (uplift asks what expressivity is
  WORTH, routing asks what it COSTS); the vendor chose the hybrid
  deliberately and uptraining beyond distillation-parity is a
  training program, not an inference rung; any behavior claim
  needs the free-generation screen class, not teacher-forced only.
  Cheap first fragment: ONE layer swapped + distilled to
  teacher-parity on the frozen corpus (feasibility + wall-clock
  pricing), before any capability arm. Oracle: X/K on a held-out
  surface + token-identical greedy equivalence for the unswapped
  path.
  Attribution: Artin (the uplift arrow, "effect on thinking"
  framing), house (init-from-linear + distill-to-parity shape,
  depth-targeting link, fences).
- **BANKED (2026-08-19): W4-GROUP-DOT — one-index/4-weight kernel
  dot for the w4 GEMV** (GPT seat, out of the BLE-FREEGEN abort's
  runtime review). The w4 codec stores one u8 index per GROUP of 4
  weights, but the current qcuda GEMV logically reloads that index
  for four scalar elements; a variant that loads the index once and
  dots the 4-vector against 4 x elements may cut instruction/gather
  overhead. Same shape for s16: one u8 packs TWO codes sharing a
  block scale — load once, unpack hi/lo, two multiplies. Fences:
  benchmark against the current parity-gated qcuda kernels on real
  shapes, strict parity gate, never promote from intuition; phase-2
  work behind the qcuda-tower routing fix (spec
  2026-08-19-qcuda-tower-runtime).
- **BANKED (2026-08-19): UNSLOTH-MATERIALIZATION — temporary-state
  elimination as the unifying speed/memory lever** (Artin's unsloth
  pointer; GPT seat analysis w/ citations; house mapping). Unsloth's
  "2x faster / 70% less VRAM" decomposes into: fused kernels that
  never materialize intermediates, checkpoint activations offloaded
  to pinned system RAM with double-buffered reload, Cut Cross
  Entropy (never materialize the [T x V] logits — at Qwen's ~248k
  vocab that is ~4 GB per 8k tokens), and sequence packing with
  cached attention metadata. None of it changes what the model IS;
  all of it changes what the execution graph MATERIALIZES — the
  exact defect class of the same-day BLE-FREEGEN abort (0.87 GiB
  s16 payload materialized as 6.875 GiB dense FP32; 12.7x recovered
  by decoding-as-consumed). The unifying trade, candidate house
  law once a second measured instance lands: pay bytes now v pay
  computation later, under a fidelity constraint — gradient
  checkpointing, the residual correction-field idea, sparse-KV
  routing, and compressed-weight decode are all instances.
  The candidate law, stated operationally (GPT correction 2026-08-19:
  "pay bytes v compute" was too narrow — fusion reduces both):
  MINIMIZE MATERIALIZED DATA MOVEMENT subject to fidelity + latency
  constraints; never materialize a representation unless the next
  computation needs it. Tower abort/recovery = first measured
  instance; an independent second mechanism required before any
  THEORY row.
  Named residues, cheapest first:
  (1) DESK: padding-fraction census on house training batches
  (train_mathnative bucketing) — an OPPORTUNITY census, zero GPU
  cost; realized wall recovery depends on linear-v-attention mix,
  length distribution, and packed-kernel overhead, so packed wall
  is measured AFTER the census, never inferred from it;
  (2) DOUBLE-BUFFERING, two distinct shapes: the CPU streaming
  scorer can literally prefetch payload i+1 during layer i; the
  CUDA tower's payloads are already VRAM-resident, so its version
  is decode-AHEAD (decode row chunk i+1 into workspace B while the
  GEMM consumes chunk A, swap) — profile before assuming overlap
  wins, decode and GEMM may contend for SM/bandwidth;
  (3) plan_residency -> general MEMORY PLANNER (tensor x codec x
  placement x representation, activations included) — already
  spec'd for weights in 2026-08-19-qcuda-tower-runtime, this bank
  widens the ambition; (4) the decomposed benchmark (HF/TRL+FA2 v
  unsloth, feature-by-feature, dVRAM/dwall per mechanism) — PARKED,
  3080-days class, needs its own GO.
  SPEED ROOFLINE (the 2k tok/s joke, priced): BLe's compressed GPU
  payload ~7.38 GB against the 3080's ~760 GB/s gives a fantasy
  one-pass-per-token decode ceiling ~103 tok/s before any
  compute/attention/overhead; 2k tok/s would need ~14.8 TB/s of
  weight streaming. Kernel work has real headroom from ~10 tok/s,
  but breaking the one-weight-pass-per-token wall is ALGORITHMIC
  AMORTIZATION — speculative/multi-token acceptance (the house's
  banked spec-decoding = gate-law thread), never endlessly faster
  scalar GEMV.
  FENCES: their 2x/70% is workload-conditional (HF's own table:
  12-74%); VRAM moved to host RAM is not memory saved; their 2x
  inference claim is v transformers-native, NOT v our qcuda tower —
  never quote their numbers as portable without a paired in-house
  run; CCE is inapplicable to our full-logit teacher ORACLE records
  but applies to any large-vocab hard-target LM training generally
  (full-distribution KL is a different, though chunkable, object).
- **BANKED (2026-08-19): CHEAP-READOUT-CENSUS — does a low-precision
  head preserve the teacher's token choice inside a small candidate
  set?** (GPT seat, out of the 2k-tok/s roofline discussion; house
  pricing). Qwen's lm_head is 1.27B weights, functionally important
  (io attribution: head carries prefix-K) AND touched every token —
  a bandwidth floor no kernel removes. The census: R_k = P(teacher
  top1 inside the CHEAP head's top-k) and teacher-mass captured, for
  k in {16, 64, 256, 1024}, cheap head = arm A's w4 lm_head rows.
  CORRECTED IN PLACE same day (GPT fatal design check, verified at
  source): the census is NOT zero-run — cheap-head logits =
  W4_head @ h and no run ever froze final hidden states (the
  teacher pass saves logits arrays only; scorer receipts save
  aggregates). The cheapest valid shape is the ARM-STATE census:
  rerun A on the frozen MODEL-1/MODEL-2 positions saving PRE-head
  hidden states (minutes-class on the Mac CPU — A's MODEL-1
  scoring wall was ~483 s), hold each h_A fixed, compute candidate
  sets under the w4 head, read the teacher's desired token/mass
  from the frozen teacher logits, and optionally apply the vendor
  head to the SAME h_A as a control — separating BODY-STATE damage
  from READOUT ranking damage. The head-only oracle on TEACHER h
  would need a teacher rerun; not paid first. Two claim levels,
  never conflated: (1) w4 FULL-VOCAB proposal + exact top-k rerank
  — R_k validates this (cheap w4 sweep replaces an expensive
  high-precision sweep); (2) SUBLINEAR router/index + exact rerank
  — the thing that avoids scanning 248k rows at all; level-1 R_k
  does NOT prove level 2, whose static-vector router gets its own
  census if level 1 reads well. Still ahead of ATTN-ROUTER in the
  queue: static vocab vectors + frozen teacher distributions =
  the cleaner routing problem. Design hardenings (GPT, pre-launch):
  PRE-HEAD state is defined OPERATIONALLY — the literal tensor
  entering lm_head, captured by hook, never a guessed API field;
  identity fixture gates everything (captured h_A through the w4
  head must reproduce native A logits/top1 under frozen tolerance
  BEFORE any R_k is computed); the reading is three-object on
  identical frozen positions — T (frozen teacher logits), A_W
  (w4_head @ h_A), A_T (vendor_head @ h_A) — where the R_k gap
  A_W-v-A_T isolates readout-representation loss conditional on
  the same state, and what remains under A_T reads as upstream
  body-state damage seen through the vendor head, NOT an additive
  causal decomposition; M_k mass capture + teacher-margin-
  stratified R_k ride along free; if promoted to a rung, LEVEL-2-GO
  thresholds freeze BEFORE h_A is produced (never post-hoc "R_256
  looks good enough").
  FENCES: candidate-set recall is NOT generation equivalence
  (sampling/margins live in the tail); the RK-census lesson carries
  — measure capture before building any router; per-surface, both
  X and K classes of positions reported separately.
- **BANKED (2026-08-19): TRAFFIC-BUDGETED-ARCH — prototype the
  bytes-moved-per-accepted-token objective at HOUSE scale** (Artin's
  "what could we even get" + GPT's architecture sketch; house
  scoping). The 2k-tok/s regime needs the assumption killed that
  one emitted token costs one full weight pass; published parts
  exist (speculative decoding 2-3x, Medusa 2.2-3.6x, MoD, DeepSeek
  V3's 671B-total/37B-active + multi-token objective) but nobody
  trains 27B here. The HOUSE version: train small conditional-
  capacity models (mathnative substrate, sympy oracle) with the
  traffic objective stated as a CONSTRAINED problem (minimize bytes
  moved subject to L_teacher <= epsilon — loss below a ceiling, or
  equivalently fidelity F >= F_min; or sweep explicit traffic
  budgets) reported as the Pareto curve of teacher fidelity v
  measured bytes/token — a lambda-weighted sum is a toy-optimizer
  convenience, never the scientific readout (lambda arbitrariness).
  The metric the whole program points at: TEACHER FIDELITY PER BYTE
  MOVED PER ACCEPTED TOKEN. The allocation results (io/early-band
  value, metric-split, band marginals) become design data, not just
  compression facts. FENCES: multiplicative speedup stacks (route x
  speculate x kernel) multiply BEST cases — the factors degrade
  each other's ceiling (routed targets change drafter fidelity;
  acceptance falls on hard tokens); any capability claim runs
  through the oracle, never through weight/byte counts; 27B-scale
  training is out of scope on house hardware, permanently.
- **BANKED (2026-08-19): OPERAND-PROVIDER-LAW — every registered
  metric enumerates its mathematical operands and the exact frozen
  artifact/provider for each; a missing provider refuses the
  prereg** (GPT seat; house adoption case). Caught twice in one day
  at review time instead of design time: MODEL-2 r1's arm algebra
  (BLe already contained io — the operand existed but its content
  was double-counted) and the CHEAP-READOUT "zero-run" claim
  (logits_cheap = W4_head @ h, and NO artifact provides h — the
  operand had no provider at all). Executable residue: extend the
  machine prereg schema (llmopt/lab/prereg.py) so each registered
  measurement may carry operands: [{name, provider: <frozen path
  or committed producer>}], and claim_lint/prereg validation
  refuses a registered measurement whose operand names a provider
  that does not exist. Design-time refusal, review-time relief.
  FENCES: the field proves existence, not sufficiency (a provider
  can exist and still be the wrong surface/precision — fences and
  auditors keep that job); retrofit is forward-only, never applied
  to booked preregs. RESIDUE LANDED (2026-08-20, mac-only session):
  optional `operands` field shipped in llmopt/lab/prereg.py —
  validate() checks shape (name/provider, repo-relative, unique),
  verify_operands()/load() refuse a provider absent from disk with
  the repo root inferred from the docs/preregs location; three
  focused fixtures in tests/test_prereg_schema.py including the
  CHEAP-READOUT missing-h shape. v1 is existence-only per the
  fence; claim_lint wiring not yet done (open follow-up).
- **BANKED (2026-08-19): TRAJECTORY-SIDECAR — free-generation
  instruments preserve the exact generated token IDs as a canonical
  receipt sidecar (per-row offset/count/hash into one binary file;
  decoded text derived on demand), forward-only** (GPT seat, from
  the FREEGEN-2 null). The screen driver parsed answers and
  discarded the raw generations, so the 30 xhigh rows that rode the
  3072-token cap CANNOT be autopsied post hoc — the most
  scientifically interesting bytes of the run were never persisted.
  Cost ceiling is trivial (60 rows x 3072 int32 < 1 MB). FREEGEN-2's
  own receipts stay frozen as booked; the fix applies to every
  future freegen driver. Mitigation available meanwhile: greedy
  decoding is deterministic on a fixed runtime + artifact, so any
  single row can be regenerated verbatim for autopsy as an
  unregistered probe.
- **BANKED (2026-08-19): TRAJECTORY-PRESERVATION-AXIS — treat
  free-run trajectory/attractor behavior as an INDEPENDENT
  deployment axis that teacher-forced X/K fidelity is never assumed
  to proxy** (GPT seat reading of the FREEGEN-2 null; house
  concurs). Measured basis, one arm: BLe buys substantial scored-
  stream X/K recovery over B yet shows zero registered free-run
  recovery (0/30 terminations, 0/60 correct). One arm is evidence,
  not a law — but every future deployment-promotion case must carry
  its own free-run bar rather than inheriting one from scored-stream
  wins. Corollary carried from the same review: FREEGEN-2 LOWERS the
  prior on another early-attention repair as a direct behavior
  rescue; the RESIDUAL targeted-patch follow-up stays interesting
  specifically as "can structured correction directions succeed
  where broad s16 precision did not", and any patching of BLe
  (rather than A) requires a fresh residual census — the A/w4
  residual structure does not transport to BLe's payload untested.
- **BANKED (2026-08-19): 3080 artifact-garden cleanup ON HOLD by
  Artin ruling — no broad cleanup until the dependent rungs drain**
  (Artin + GPT seat, same review). A is needed by CHEAP-READOUT,
  BLe by phase-2 runtime work, vendor/comparison artifacts by
  follow-ups; the 3080 has more free storage than the Mac so the
  pressure is low. Inventory-then-delete happens AFTER those rungs,
  under the standing BOARD housekeeping gate, on explicit GO.
- **BANKED (2026-08-19): NO-REGRET-RETRY-CONTROLLER — a tiny online
  controller over retry strategies, aimed squarely at the measured
  detect-retry limit cycles** (GPT seat, from the BLE2-XHIGH-AUTOPSY
  reading; Artin's market-maker frame). The autopsy showed BLe's
  effective retry policy is w(same strategy)=1 forever even after
  its own critic fires — the worst possible online learner. The
  rung shape: greedy by default; when self-error is detected AND a
  cycle detector says the trajectory is re-entering the same basin
  (n-gram recurrence — trivially implementable, we measured exact
  periods 88/242), a controller with a handful of strategy arms
  (brief temperature burst, top-2 branch, alternate precision/
  route, representation restart, symbolic check) picks an escape
  move; the oracle scores the outcome; multiplicative-weights
  update. Metric: correct attractor escapes per extra
  tokens/bytes/wall. No 27B retraining, no new artifact. FENCES:
  regret guarantees apply to the controller's decision problem,
  never to correctness — settlement is always the oracle;
  whether plain temperature alone escapes the attractor is the
  cheaper prior question and needs its own registration first;
  controller experiments start on the house substrate or as a
  registered BLe retry rung, not as a deployment claim.
- **BANKED (2026-08-19): SEQUENTIAL-RESOURCE-CONTROL imports — one
  umbrella for the quant/control mathematics GPT ranked, unified by
  "don't spend the expensive operation unless evidence says its
  expected value exceeds its cost"** (GPT seat; Artin's CTA/market
  riff — his read: the market is both the data stream and the
  referee, which is online learning, not batch training). Members,
  each needing its own registered rung before any is treated as
  more than an analogy: (a) attractor-escape control — minimum
  perturbation only when a retry orbit is detected, Lyapunov/
  adaptive-control framing; (b) rate-distortion with ORACLE
  distortion — R(D_X), R(D_K), R(D_trajectory) as the theoretical
  ceiling question over the compression program, distortion in
  function space never weight L2 (house law already); (c)
  sequential hypothesis testing / optimal stopping — accumulate
  evidence per token, stop thinking when extra compute stops
  moving the answer; (d) HJB / inventory-control framing — compute
  as capital, uncertainty as inventory, switching costs priced;
  (e) filtering — compact recursive state z_t = f(z_{t-1}, x_t)
  over cheap per-token observables (entropy, top1-top2 margin,
  cycle score, retry count) instead of re-reading history. The
  trajectory-preservation axis gives (b) its third distortion
  coordinate. NOT copied from finance: proxy-P&L as reward —
  settlement stays oracle correctness, the quant machinery is for
  allocation/stopping/exploration only. Long-arc companion:
  verified-progress-per-compute for theorem search (local oracles:
  type-checks, finite cases, exact counterexamples) is the FA Law
  restated over proof tactics — banked as a frame, no engine
  commitment (charter: math + physics only, which this satisfies).
- **BANK AMENDMENT (2026-08-19): CHEAP-READOUT-CENSUS gains the
  MIPS framing** (GPT seat). The 248k-token output head is a
  maximum-inner-product search; hashing/quantization/bandit MIPS
  methods with approximation guarantees exist, and the quant shape
  is cheap-scan-then-exact-price (248k -> ~1k cheap filter -> exact
  head only there). CHEAP-READOUT's arm-state census is the
  empirical gate BEFORE importing any of that theory: if the cheap
  readout's candidate sets don't contain the teacher's argmax at
  high recall, MIPS machinery has nothing to stand on here.
- **BANKED (2026-08-19): JSPACE-PRESERVATION-CENSUS — does
  quantization preferentially damage workspace-like
  deliberate-reasoning representations even when teacher-forced X/K
  improve?** (Artin's global-workspace question + GPT seat
  tightening; Anthropic global-workspace/Jacobian-lens research as
  the published anchor). DEFINITIONS TIGHTENED AT BANK TIME (house's
  first chat framing was loose, corrected here): the residual
  stream is NOT a J-space — it is the shared activation fabric; the
  published J-space is a privileged, limited subspace of
  verbalizable internal patterns found with the Jacobian lens
  (reportable, deliberately controllable, causally used, broadly
  connected) and most residual states do not qualify. The think/CoT
  channel is NOT J-space either — the published work explicitly
  distinguishes workspace from scratchpad; Qwen think is an
  external scratchpad that may expose workspace-relevant content.
  The autopsy reading stays HYPOTHESIS: explicit "I'm wrong" text
  shows error INFORMATION survives compression; it does not prove
  successful workspace broadcast or failed specialist recruitment —
  circuit language waits on an internal/causal measurement. THE
  CENSUS: if a Qwen J-space analogue is findable, compare
  vendor/A/BLe on frozen math states — projection preservation
  first, causal preservation later — and ask whether SMALL
  workspace-targeted corrections beat whole-tensor s16 (the
  RESIDUAL-RECON question restated in a privileged basis).
  LADDER, cheap-first: reproduce a published open implementation on
  a small open model -> small Qwen-family model -> only then price
  27B; qcuda needs NO autograd if workspace directions are
  discovered offline and monitored/intervened at inference.
  LONG-ARC PAYOFF: small high-fidelity control/workspace channel +
  aggressively compressed conditional bulk specialists — a
  Global-Workspace-inspired ARCHITECTURE, and a stronger one than
  calling the residual stream a workspace. FENCES: benign
  brain-analogy under the charter (analysis frame for our models'
  representations, zero organism capability); target model is
  dense/non-MoE with a HYBRID linear/full attention stack (house's
  own ATTN-ATTRIB families), never described as vanilla all-full
  attention.
- **BANKED (2026-08-19): LOOP-STATE-READOUT — capture the literal
  pre-head h_t ALONG BLe's loop trajectories and measure the
  actionable distribution around self-error/retry points** (GPT
  seat bridge, post CHEAP-READOUT-0). CHEAP-READOUT measured
  teacher-forced MODEL-1 states on arm A; FREEGEN measured BLe's
  self-generated xhigh states; together they support "single-step
  candidate information surprisingly robust, long-horizon
  self-conditioned behavior catastrophically fragile" — they do
  NOT establish that all damage lives in recurrent dynamics, and
  they do not exclude readout failure on OFF-MANIFOLD loop states.
  The bridge census: along regenerated loop trajectories, capture
  h_t at and around detect/retry points ("Wait", "Let me
  reconsider") and read top-k composition, teacher-style margin,
  entropy, and candidate-set overlap before v after — does the
  self-error text CHANGE the actionable distribution, or have the
  alternative candidates disappeared from the head's view? Directly
  connects the trajectory-control program to the
  [[jspace-preservation-census]] hypothesis. Cheap: the tower
  runtime regenerates loops deterministically; a head-input hook is
  the same instrument CHEAP-READOUT already built.
- **BANK AMENDMENT (2026-08-19): CHEAP-READOUT level-2 census
  reports the PARETO of R_256 v rows/bytes TOUCHED, never router
  recall alone** (GPT seat). The point of level 2 is sublinear
  traffic; a router with high recall that still touches most of the
  248k rows buys nothing. Also carried: level-1's knife-edge bar
  crossing (0.98034 by one position) is the least informative part
  of the verdict — the descriptive shape (7 corpus misses at k=256,
  1 at k=1024, prefix at ceiling, ~96% teacher mass) is what design
  decisions should read.
- **BANK NOTE (2026-08-19): CYCLE-IMPULSE reading law, fixed before
  receipts land** (GPT seat review of the prereg; run live,
  untouched): bar-1 FIRE means at least one run escaped the FROZEN
  recurrence criterion — never "attractor fixed"; bar-2 FIRE means
  one correct perturbed trajectory, nothing more. At booking, the
  token sidecars get a descriptive post-impulse basin
  classification: same cycle / new cycle / semantic restart /
  nonrepeating wrong / eos wrong / correct — unregistered color,
  gates nothing.
- **BANK AMENDMENT (2026-08-19): LOOP-STATE-READOUT pre-registration
  hardenings, fixed before any registration** (GPT seat; operand-
  provider law applied at design time — the law's first scheduled
  win). (1) OPERAND/PROVIDER: frozen teacher logits DO NOT EXIST for
  BLe's self-generated loop prefixes — "teacher-style margin" in the
  bank means only the margin FORMULA applied to BLe's own logits and
  gets renamed LOCAL top1-top2 margin; intact-teacher preference at
  loop states requires a new vendor forward on the exact prefix and
  is priced separately. The cheap intermediate: W_BLe @ h_BLe v
  W_vendor @ h_BLe isolates head representation conditional on the
  same damaged loop state and is NOT teacher behavior — never
  labeled as such. (2) RUNTIME IDENTITY: CHEAP-READOUT's capture
  fixture qualified the CPU/reference path on arm A; LOOP-STATE runs
  on the qcuda-tower with a fused s16 lm_head — same concept, NOT
  the same qualified instrument. A tower-specific capture identity
  fixture is required: captured h_t through the fused lm_head must
  reproduce the native tower last-position logits/top1 under a
  tolerance frozen in the prereg before any loop-state metric is
  read. (3) STORAGE: sparse h/logit windows around detector/
  self-error/retry events with exact token-position alignment, never
  blanket per-token dumps. (4) METRIC SHAPE: compare distributions
  at pre-error / error / retry / repeated-step positions — local
  margin, entropy, top-k Jaccard/churn, KL/JS movement, rank
  trajectory of the relevant tokens — built to discriminate "the
  error flag moved almost no actionable policy" from "alternatives
  entered the distribution but greedy still chose the same top1."
  (5) MECHANISTIC LADDER, cheap-first: BLe head on BLe h -> vendor
  head on the same h -> full vendor body on the exact loop prefix
  only if the first two legs leave the question open.
  CYCLE-IMPULSE WORDING FENCE carried with it: impulse FIRE/NO-FIRE
  reads "locally escapable v locally restoring under the registered
  token-space perturbation" — never noise-floor v sharp-geometry;
  that deeper diagnosis belongs to loop-state/internal measurements,
  not token-space impulses.
- **BANK AMENDMENT (2026-08-19): LOOP-STATE-READOUT metric law —
  the clean attractor measurement is CYCLE-ALIGNED, never
  adjacent-position** (GPT seat, second tightening). Raw KL/JS
  between pre-error/error/retry positions is temporally confounded:
  each is a different next-token task after a different prefix —
  those stay as descriptive event-window color only. The clean
  comparison is homologous-state recurrence: on exact loops,
  logits/distributions at t v t+88 and t v t+242 at corresponding
  cycle positions; on the semantic-restart item, align repeated
  computation landmarks (the failed step) across restarts. Report
  JS/KL, top-k overlap, top1 equality, margin/entropy deltas ACROSS
  HOMOLOGOUS ATTEMPTS. Correction response is quantified by
  comparing successive retry attempts at the same semantic step:
  repeated "I'm wrong" with P_retry(i) ~ P_retry(i+1) is the strong
  evidence of a stuck actionable policy. The vendor-head-on-BLe-h
  leg runs chunked/offline over the sparse captured states — tower
  residency stays untouched. Standing note: OPERAND-PROVIDER is
  prospectively useful but human-enforced until the prereg-schema
  operands field lands (banked forward fix). [LANDED 2026-08-20:
  the operands field shipped in llmopt/lab/prereg.py — enforcement
  is now machine-side at load time for any pre-reg that declares
  operands; declaring them stays a house habit, not a schema
  requirement.]
- **BANK AMENDMENT (2026-08-20): LOOP-STATE-READOUT specimen order +
  hidden-state recurrence** (GPT seat, post CYCLE-IMPULSE booking).
  Items 0 and 4 are the PRIMARY rigid-orbit specimens (0/192 bursts
  escaped — the measured hard cases); item 3 is the loose
  semantic-orbit CONTRAST, not the lead. Beyond homologous
  logit/policy recurrence, capture homologous PRE-HEAD h at t, t+L,
  t+2L... and report hidden-state recurrence too (cosine/relative
  L2; layerwise only if cheap) — distinguishes an internal-state
  orbit from mere output-policy recurrence. Controller note carried:
  after CYCLE-IMPULSE, the temperature burst is the WEAK BASELINE
  for exact orbits, not an equal-status NO-REGRET arm — structured
  interventions (representation restart, alternate route,
  precision escalation) carry the next controller branch.
- **BANKED (2026-08-20): PRIMITIVE-EVIDENCE DOCTRINE — receipts
  persist primitive evidence sufficient for independent
  recomputation, and intervention caps must never disable
  observation** (GPT seat; earned twice in one night). Measured
  basis: the CYCLE-IMPULSE producer's run_escaped said 18/18; the
  token-ID sidecars let the honest 5/18 be recovered by offline
  replay WITHOUT a rerun — while the same driver's burst cap
  silently disarmed the detector, making the final burst's escape
  true by construction (the second face of the same defect: the
  intervention limit turned off the measurement). Corollaries
  already banked as forward fixes: escape/derived scoring lives in
  adjudicators that recompute from primitives, never in the run
  loop; per-run sidecars get refuse-if-exists guards; artifact shas
  go in start_provenance. Pairs with [[trajectory-sidecar]] (the
  bank whose first use made the recovery possible).
- **BANK OUTCOME (2026-08-20): LOOP-STATE-READOUT registered and
  adjudicated as QWEN-LOOP-STATE-0** (RESULTS L36345): internal-orbit
  bar NO-FIRE KNIFE-EDGE (0.9963/0.9701 v 0.99, not in the 0.90
  refuted band), stuck-retry bar FIRES (98.4% top1, JS 1.7e-5 nats).
  All five prereg hardenings + the cycle-aligned metric law + the
  specimen order executed as banked; the vendor-head ladder leg 2
  stays priced, unrun. Controller note updated: item-4 color (state
  variation at HIGH-margin positions, cosine decreasing with lag)
  makes drift-amplifying intervention a priced arm beside
  representation-restart in the [[no-regret-retry-controller]]
  branch. Hidden-state-recurrence amendment (2026-08-20) partially
  served: pooled + k-split cosine landed; layerwise stays unrun.
- **BANKED (2026-08-20): CONVEX-CELL FACT + MARGIN-IS-NOT-DEPTH law**
  (GPT seat, LOOP-STATE-0 review; verified in-house). Each
  linear-head argmax region {h : (Wh)_y >= (Wh)_j for all j} is a
  convex polyhedral cell, so the same winner at two h endpoints
  implies the straight segment between them stays in the cell —
  while the model's actual trajectory between the states need not
  follow that segment. And raw logit margin z_y - z_j is NOT
  distance to the y/j boundary — that is (z_y - z_j)/||w_y - w_j||
  and needs head row norms. Any future "deep in the cell" claim
  measures normalized boundary distance or does not say depth
  (deny-listed; AMENDMENT QWEN-LOOP-STATE-0-COLOR2).
- **BANKED (2026-08-20): SYMBOLIC-DYNAMICS wording law for loop
  readings** (GPT seat). What LOOP-STATE-0 measured is a PERIODIC
  SYMBOLIC (token) ORBIT over a drifting hidden state: q(h) =
  argmax Wh is periodic at L while h only approximately recurs.
  "Quotient dynamics" is the stronger unmeasured claim that the
  transition descends to equivalence classes (q o F = G o q) —
  never use it for this result class. Companion wording fence:
  BLe's tower is hybrid attention, so hidden drift is attributed to
  "context/state continuing to evolve each lap" — never
  specifically to KV-cache accumulation (full-attn KV, linear-attn
  recurrent state, and positional context all evolve).
- **BANK PROMOTION (2026-08-20): artifact digest in
  start_provenance is a MUST-FIX before the next registered run**
  (Grok + GPT seats converging on receipt-audit S2): no receipt
  field currently distinguishes which weights ran beyond ART_DIR
  resolution. Implemented same day in llmopt/lab/provenance.py —
  see the commit this bank rides in; the next driver consumes it.
- **BANK NOTE (2026-08-20): LOOP-STATE-0 calibration + convergence
  color** (external-review asks, receipted in loopstate_color2.json):
  homologous cosine 0.998/0.984 v nonhomologous baseline 0.32-0.54
  (the recurrence contrast is huge; the 0.99 bar was
  uncalibrated-strict, informs the next registration's thresholds);
  item-3 successive-attempt JS medians fall monotonically 1.9e-4 ->
  2.8e-6 (the retry loop DEEPENS across attempts — new color for
  the [[no-regret-retry-controller]]: later interventions face a
  more converged policy, so early intervention is BETTER-MOTIVATED;
  matched-cost early-v-late efficacy is unmeasured — corrected in
  place 2026-08-20 per AMENDMENT QWEN-LOOP-STATE-1-SCOPE).
- **BANK OUTCOME (2026-08-20): mechanistic ladder leg 2 booked as
  QWEN-LOOP-STATE-1-HEADSWAP** (RESULTS L36594): both bars fire,
  prior correct — the intact vendor head reproduces the stuck
  retry policy on identical states (0.984375, numerically the
  BLe-head fraction) and agrees on 99.6% of captured positions;
  every failing attempt-pair under every top1 basis lies in retry
  attempt 0->1. The [[no-regret-retry-controller]] branch now has
  readout repair MEASURED OUT at the k=1 level on these specimens;
  state intervention (drift amplification / representation
  restart) is the live lever. Ladder leg 3 (vendor body on the
  exact loop prefix) stays priced, unregistered. Forward fix
  banked: headswap driver derives HOMOLOGOUS geometry from the
  params file on any re-run (literal-v-derived class, verified
  equal this run).
- **BANKED (2026-08-20): LEVEL-2 EXACT GEOMETRIC MIPS CENSUS — the
  preferred first level-2 registration is an EXACT
  branch-and-bound top-256, never approximate ANN** (GPT seat;
  Artin Mac GO 2026-08-20). Design: cluster the decoded W4 head
  rows; each cluster stores center c_j and full-space radius r_j;
  per query h the upper bound U_j = h . c_j + ||h|| r_j
  (Cauchy-Schwarz) orders clusters; exact-score clusters in
  descending U and STOP once the running kth score tau >= max
  unscanned U — certifying the exact global top-256 with zero
  additional fidelity loss. Qualification bar: 100% equality with
  brute-force W4 top-256 on a small-vocab surface BEFORE any
  full-surface read. Scientific output is the TRAFFIC Pareto
  (bank amendment 2026-08-19): q50/q90/q95/max rows scanned and
  actual payload+index bytes touched, per corpus/prefix
  population. If high-dimensional bounds force a near-full scan,
  BOOK THE NULL; if they prune hard, that is exact sublinear
  readout. Index uses head rows only, no evaluation labels.
  Clustering/index recipe + traffic thresholds FREEZE in the
  prereg before any full-surface result is read. Links
  [[cheap-readout-census]]; supersedes-in-preference any
  approximate-recall router variant for the first registration.
- **BANK OUTCOME (2026-08-20): HEADSWAP-IMPULSE-0 booked** (RESULTS
  L36833): 5/5 recurrence-return, 0/5 original-orbit rejoin, 0/5
  repair (cap-limited). The [[no-regret-retry-controller]] law
  extends: measured micro-perturbations (random bursts, one
  vendor-informed token) redirect trajectory/recurrent-pattern
  IDENTITY while the recurrence tendency survives — no durable
  escape or repair demonstrated (recurrence != exact periodicity;
  one impulse tail was non-periodic under its caps) — measured for
  the vendor-informed one-token arm at the five registered loci (n=1
  each; generic/top-2/oracle/adversarial one-token arms UNMEASURED,
  the alternate-token control is its own registration); multi-token
  and state-level arms (drift amplification, representation
  restart) are better-motivated, one-token arms as a class are not
  ruled out — corrected in place 2026-08-20 per AMENDMENT
  QWEN-HEADSWAP-IMPULSE-0-WORDING. The
  matched alternate-token control (generic v vendor-informed, same
  positions) stays a separate registration. Ladder leg 3 (vendor
  body) remains priced, unregistered — its motivation now runs
  through trajectory-formation causality, not local readout.
- **BANKED (2026-08-20): HOMEOSTATIC-INFERENCE umbrella — can
  inference be given cheap internal instability signals plus
  inference-time negative feedback (computation, precision,
  routing) that pushes the dynamics back into a productive
  regime?** (Artin + GPT seat; framed from cybernetics and
  physical self-regulating systems — stars, equilibrium,
  attractors, the double-pendulum initial-conditions observation
  that early structure constrains what a system can become while
  local feedback regulates how it behaves after. CHARTER FENCE
  RESTATED: biological systems entered this conversation as
  CONCEPTS ONLY — the lab builds no bio or chem capability, ever;
  cybernetic/physics framings are the admissible methods.)
  Guiding formulation (GPT): "successful complex systems combine
  strongly constrained structure formed early with local negative
  feedback that continuously keeps their dynamics inside a viable
  region." The measured basis already in the ledger, stated as a
  FAILED-FEEDBACK PATTERN (corrected in place 2026-08-20, GPT
  seat: internal sensor/controller/actuator decomposition is
  UNMEASURED — do not claim a causal internal sensor, and the
  actuation capacity is not literally gone since perturbations do
  redirect trajectories): error-indicating information remains
  OBSERVABLE in the stream (explicit self-error text), while the
  effective ENDOGENOUS corrective response is insufficient/stuck
  (retry policy at JS ~1e-5, convergent across attempts;
  vendor-informed single-token exogenous actuation insufficient at
  the tested doses). The umbrella therefore asks
  for: (1) cheap online instability signals (recurrence detectors,
  local margin/entropy trajectories, hidden-state recurrence —
  all instruments this program already built); (2) actuation
  levers beyond token space (precision escalation per the banked
  precision-as-escalation runtime program, routing, representation
  restart, drift amplification); (3) a viability region defined by
  measurable dynamics, not vibes. Subsumes/links
  [[no-regret-retry-controller]], [[sequential-resource-control]],
  [[jspace-preservation-census]]; the LOOP-STATE chain is its
  measurement arm. OPERATIONAL FIRST LADDER (GPT seat, adopted — operational
  before metaphorical): (1) external recurrence detector (built);
  (2) ACTUATOR QUALIFICATION — verify a precision/state escalation
  applied at a frozen recurrence event has useful authority at
  all; (3) closed-loop controller — always-low v always-high v
  detector-controlled escalation, with an explicit hysteresis/exit
  rule, scored on oracle correctness + recurrence + bytes/wall.
  The CANDIDATE win condition (this is a BANK, not a
  pre-registration — thresholds get numeric form only at
  registration): detector-controlled fidelity comparable to
  always-high at materially lower resource cost, both terms to be
  frozen numerically in the prereg. The detect-retry
  pathology is COMPRESSION-ASSOCIATED (observed under compressed
  BLe; an intact-body matched counterfactual would be needed for a
  causal claim), which keeps precision escalation a strongly
  motivated actuator HYPOTHESIS, not an established mechanism.
  Analogy fence: the initial-conditions/dynamical-systems analogy
  and the cybernetic negative-feedback analogy stay DISTINCT until
  feedback is actually installed and measured.
- **BANKED (2026-08-20): PRECISION-SWITCH STATE CONSISTENCY — a
  live precision/weight switch that retains the cache is a
  dynamically MIXED-PRECISION STATE MACHINE, never merely "high
  precision when unstable"** (GPT seat; design law for any
  [[homeostatic-inference]] actuator registration). At detector
  time t the model carries model-dependent cached/recurrent state
  S_t^BLe (full-attn KV, hybrid linear-attn recurrent state,
  positional context); switching weights while keeping the cache
  computes F(W_high, S_t^BLe), NOT the counterfactual
  F(W_high, S_t^high) — and switching back later creates
  mixed-history state. Valid instrument, but name it what it is.
  Actuator qualification therefore SPLITS: LOW (continue BLe);
  HOT-HIGH (switch the high-precision route from the detector
  onward, PRESERVING the current BLe state); REFRESH-HIGH (at the
  detector, replay/reconstruct the same prefix under the high arm
  for an internally consistent state, then continue); optional
  later BOUNDED-REFRESH (recompute only a recent window/state
  slice). The mechanistic payoff of the split (narrowed in place
  2026-08-20, GPT seat): HOT fails while REFRESH succeeds = the
  high arm's benefit REQUIRES a high-arm-consistent
  reconstructed history/state — the low arm's accumulated
  state/history is implicated, but state-v-weight causality is
  NOT uniquely identified by this contrast alone; HOT succeeds =
  a cheap actuator exists; both fail = precision escalation
  demotes. REQUIRED SANITY FIXTURE before REFRESH-HIGH is
  interpretable: REFRESH-LOW — replay the same prefix under BLe
  itself, reconstruct its state, and the continuation must
  reproduce the deterministic LOW baseline exactly. Qualification
  matrix: LOW / REFRESH-LOW (sanity) / HOT-HIGH / REFRESH-HIGH.
  First actuator qualification stays ONE-WAY (intervention
  through eos/cap); high->low hysteresis enters only after
  actuator authority is established. Closed-loop comparisons must name the exact
  high arm and distinguish always-high-FROM-START from controlled
  hot/refresh switching; more precision is NOT assumed monotonic
  from the existing X/K results.
- **BANK OUTCOME (2026-08-20): LEVEL-2 EXACT GEOMETRIC MIPS booked
  as its own registered null** (RESULTS L37116): certification
  machinery correct (1362/1362 exact), pruning absent (q50 0.989+
  at every frozen K), modeled bytes ratio > 1 everywhere — under
  isotropic k-means-ball Cauchy-Schwarz geometry at d=5120 the
  bound is too loose to locate the candidate set cheaply. The
  [[cheap-readout-census]] LEVEL-2 question stays open only
  through DIFFERENT-geometry routes (anisotropic/score-aware
  bounds, structured K) or approximate-recall routes — each its
  own registration, none launched before pricing index cost
  (already dominant at K=4096). House prior called the null;
  its prefix-prunes-better clause was wrong (median visitation
  bit-identical; tails differ by a few rows).
- **BANKED (2026-08-20): BLEM-DECODE-PERF probe — BLem decodes ~7x
  slower than BLe on the qcuda tower (HOT item0: 3072 tokens at
  ~1.5 tok/s v BLe's ~11 tok/s same run, same driver loop) and the
  cheap explanations fail arithmetic** (GPT seat + house, live
  HOMEO-ACTUATOR-0 observation): total fused routes identical (401
  both; 48 routes changed class W4->s16, so even a several-x
  per-route s16 penalty bounds well under 7x), and BLe previously
  sustained ~9.9 tok/s down to ~0.65 GiB free so headroom alone is
  out. CORRECTED IN PLACE (2026-08-20, GPT seat, house-verified
  same day): (a) route count is not a cost bound at all — the 48
  promoted tensors are ~1.8B weights, ~0.443 GiB under the ~2 bpw
  W4 vector codec v ~0.873 GiB s16 (delta = the receipted
  0.4296 GiB), so the byte growth is ~2x on those routes — still
  short of 7x by itself, but a severe s16 kernel/occupancy
  pathology on those shapes remains open; (b) the HOMEO receipts
  do NOT cleanly separate HOT v REFRESH decode rate — HOT's wall_s
  starts after state restore (continuation only) while
  REFRESH-HIGH's includes BLem teacher-forced prefill, so
  restored-state v BLem-global attribution is CONFOUNDED in those
  rows. The surviving observation: HOT BLem continuation ~1.5
  tok/s v BLe >= ~10 tok/s, cause UNDIAGNOSED. Causal diagnosis
  belongs to the dedicated perf probe with separate prefill/decode
  timers and per-shape W4 v s16 GEMV profiling — the banked qcuda
  packed-pair s16 kernel + geometry sweep slot is directly
  relevant. SECOND CORRECTION same day (GPT seat): RH0 275s v HOT0
  2020s does NOT yet separate restored-state from cold-tower —
  HOT0 was also the freshly built BLem tower's first long
  workload; RH0 ran on a warmed tower. The in-run natural control:
  HOT3/HOT4 restore BLe states onto the ALREADY-WARMED BLem tower
  — HOT3-slow/RH3-fast (replicated on item4) = restored-state
  attribution strong; HOT3-fast = HOT0 was warmup. Probe design
  rider: compare native v restored cache PHYSICAL properties
  (dtype/shape/stride/contiguity/storage/device, module-local
  runtime state) plus per-layer timing, not numerical contents.
  No kernel changes while any registered run is live. WORDING
  (post-booking review): the HOMEO sanity gate certifies the
  serializer roundtrip UNDER BLe only — say "no evidence of
  corruption" for the cross-tower restore, never "proven
  performance artifact"; the 7x gap stays undiagnosed until this
  probe runs (AMENDMENT -RUNTIME-WORDING). PROBE RAN SAME DAY,
  cause LOCALIZED (OBSERVATION QWEN-BLEM-DECODE-PERF-0):
  restored-cache x second-tower x cache-position>~1790 conjunction;
  state values, layout/contiguity, s16-kernel cost, and serializer
  FUNCTIONAL corruption all exonerated — but CPU-restored cache
  provenance is itself condition (a) of the trigger conjunction, so
  the roundtrip path stays causally implicated even though its
  outputs are correct (WORDING CORRECTED 2026-08-20 post-review:
  the earlier flat "serializer exonerated" overstated; second-build
  causality stays observation-grade, allocator attribution
  unmeasured); benign branch of the
  -DIAGNOSIS-SCOPE fork confirmed; phase-5 allocator-counter slot
  is the only remaining open item.
- **BANK OUTCOME (2026-08-20): HOMEOSTATIC-INFERENCE stage 2
  (actuator qualification) measured — the precision actuator
  DEMOTES at the BLem dose** (VERDICT QWEN-HOMEO-ACTUATOR-0, the
  registered refutation: 0/6 high-arm escapes, HOT and same-prefix
  REFRESH alike, every branch's first escalated token identical to
  the frozen loop-continuing token). Consequence for the ladder:
  stage 3 (closed-loop controller) does NOT launch on one-band
  precision escalation; the [[precision-switch-state-consistency]]
  matrix performed exactly as designed (sanity gate 3/3 through
  the treatment mechanics; both-fail branch taken). The umbrella
  survives — its actuator FAMILY question is now sharpened:
  larger precision doses need hardware that fits them, and the
  non-precision levers (routing, representation restart, drift
  amplification) are the live candidates. Dose fence: this null is
  BLem-scoped; it neither closes precision escalation at larger
  doses nor touches the other lever families.
- **BANK OUTCOME (2026-08-20): ALTERNATE-TOKEN CONTROL blocked by
  its own admissibility gate — CONTROL-MATCH-FAILED, redesign is a
  new registration** (VERDICT QWEN-ALTTOKEN-CONTROL-0): no
  gap-matched non-vendor-head control exists at the five loci
  because BLe's near-top set is contained in vendor's top-256
  there. Redesign options, each a separate registration with its
  confound named: (a) drop the vendor-top-256 exclusion (control
  may then be vendor-plausible — weakens the vendor-specificity
  contrast), (b) rank-matched instead of gap-matched control
  (matches selection position, not perturbation size), (c) widen
  the gap tolerance (unmatched perturbation magnitude). The
  head-set containment fact is banked as its own lead: candidate
  sets survive compression at near-tie states — a cheap-readout
  and precision-doctrine data point.
- **BANKED (2026-08-20): TOPSET-OVERLAP CENSUS — observation-only
  BLe v vendor top-set geometry at the five frozen loop-state h,
  BEFORE any alternate-token redesign is chosen** (GPT seat +
  house; priced trivial: reuses the qwen_alttok_derive.py logit
  machinery, CPU minutes, no tower, no branches). Per locus,
  persisted primitives: top-K recall/Jaccard between the two heads
  for K = 1..256; both rank lists; rank correlation over the
  union; the FIRST non-special BLe token outside vendor's top-256
  and its logit gap to BLe top1 (the gap boundary the failed
  control ran into); both recomputed argmaxes; per-token exclusion
  attribution (special v vendor-top-256). Hypothesis it prices:
  compression may preserve the CANDIDATE MANIFOLD while perturbing
  within-set ordering — if so, the candidate set, not the argmax,
  is the compression-stable object, and that geometry chooses the
  next control design. Redesign triage updated per the same
  review: "widen tolerance" DEMOTED outright (12-19 logits is not
  a match); rank-matched control likely hits the same
  incompatibility (the near-top region is inside the exclusion
  union); the plausible next causal control drops the
  vendor-top-256 exclusion and is framed NARROWLY as "exact
  vendor-token identity v another equally BLe-plausible
  candidate", never "vendor-information v non-vendor-information".
  AMENDED (2026-08-20, mac-only session): NOT mac-runnable as
  banked — verified: the loop-state npz primitives
  (logs/qwenloopstate/loopstate_arrays_id*.npz) and the BLe head
  rows (~/qwen_whole0t/BLe) exist only on the 3080; the Mac holds
  only the vendor lm_head slice + tokenizer. Census stays queued
  for a 3080 window (any npz transfer must preserve the booked
  arrays_sha256 in loopstate_rows.jsonl). FORWARD LESSON banked
  with it (GPT seat + house verification): derivation drivers
  should PERSIST COMPACT SUFFICIENT STATISTICS at emit time —
  qwen_alttok_derive.py computed the full Z/Zv logit matrices and
  kept only the control-table scalars; a small
  top-1024-per-head-per-locus sidecar (ids + logits) would have
  made this census machine-portable for CPU pennies. Same class as
  the stream-your-rows corollary: what a driver discards at emit
  time is what the next question needs.
- **BANKED (2026-08-20): PRECISION-CREST-TRANSPORT — does the MoE
  deletion crest survive quantization precision?** (Artin's ask
  "find a way to isolate weight-quant-precisions with MoEs"; house
  design + pricing; GPT sequencing). The mapping: every booked
  EX-ANAT number lives on the 4-bit MLX artifact; a quant-precision
  x deletion interaction has never been run (ledger searched
  2026-08-20, zero hits). Design: a SELF-CONTAINED 6-bit family
  (mlx-community/Qwen3-30B-A3B-6bit, ~24GB — runs on the 36GB Mac;
  8-bit ~32GB refused under the one-resident-30B rule, bf16 ~60GB
  impossible) re-running full + named80 + one rank mask + one layer
  mask on the same seeds, comparing deletion EFFECTS across
  precisions, never raw gates (instrument fences: gate comparisons
  never cross instruments; sigma never transports across formats).
  Measured anchors: 4-bit crest named80-full = +55 pooled
  (RESULTS 22454); mask compositions frozen in
  logs/ex5/mask_manifest.json. Honest breaks: a precision
  difference in effect size confounds rounding of the router
  logits with rounding of the expert weights — this design cannot
  separate those two; and ~24GB more disk needs a cleanup pass
  first (33GB free measured). Testable residue: if the crest
  vanishes at 6-bit, the named-carrier interference story is
  partly a quantization artifact; if it holds, the mechanism is
  weight-precision-robust. SEQUENCED (GPT seat, adopted): do not
  run until EX5-LAYERMATCH establishes whether there is a 4-bit
  mechanism worth transporting. Attribution: Artin (ask), house
  (design/pricing), GPT (sequencing fence).
- **BANKED (2026-08-21): DENSE-ROUTER READOUT — if the MoE router
  "knows" a reusable carrier set, can the same structure be read
  out of a DENSE model, and is a router FOR a dense model
  possible?** (Artin's ask, 2:26 AM post-EX5: "can there be a
  router for non-moe models? are its dense layers similar to
  experts?"; house formalization). The mapping: a dense FFN is one
  always-on expert; the published kin (MoEfication — post-hoc
  clustering of FFN neurons into co-activating groups + a learned
  selector; contextual sparsity/DejaVu — per-input activation
  sparsity predictable from the previous layer) says dense layers
  DO carry implicit expert structure. Three standing house banks
  already touch this and this bank links them: the dense no-router
  control (Qwen3.8-27B bank, 2026-08-16 — deletion-crest absence
  prediction), the functional co-routed dense-block bank, and the
  resident-draft contextual-sparsity riff whose honest break is
  the load-bearing one here: A DENSE TOWER HAS NO DISCRETE GATE TO
  READ — "routes to" must be OPERATIONALIZED (top activation mass?
  largest |contribution| to the residual?) before any carrier
  claim is testable. Measured anchors: EX5-TRAJ-ANATOMY-0 gives
  the feature template a dense readout must reproduce (phase
  signature, failure enrichment) for the analogy to hold. Honest
  breaks: the MoE carrier result is about a TRAINED router's
  selections; an imposed post-hoc clustering on a dense model
  tests a different object (house-built structure, not
  vendor-trained structure), so a null there would NOT refute the
  MoE finding; activation-group deletion in a dense model has no
  clean keepset semantics (neurons overlap groups). Testable
  residue: (1) the already-banked dense no-router deletion
  control; (2) MoEfication-lite on the house dense tower — cluster
  FFN activations on the gate corpus, delete the top co-activating
  "implicit carrier" group, paired gate v random-group deletion —
  a genuinely new rung, desk-priceable first (cluster stability
  census before any gate). Attribution: Artin (ask + the "router
  knowledgeable" frame), GPT (context-dependence phrasing), house
  (kin links, breaks, residue).
- **BANKED (2026-08-21): CROSS-MODEL EXPERT TRANSPLANT — extract
  implicit experts from the dense 27B and graft into the
  Qwen3-30B-A3B MoE** (Artin's ask, same exchange; house pricing).
  The honest breaks come first because they are load-bearing:
  (1) SHAPE INCOMPATIBILITY — A3B experts are narrow MLPs on a
  2048-d stream; the dense tower's FFN lives on a much wider
  stream; no slice of the dense FFN drops into an expert slot
  without a learned projection, at which point the object is a
  DISTILLED expert, not an extracted one; (2) the two models have
  different tokenizers/training corpora, so "the same function"
  must be verified by the oracle on outputs, never by weight
  geometry (the never-score-weights-by-weight-distance law);
  (3) the house cannot train 27B-scale, so any graft is
  frozen-weight surgery + small learned adapters at most. Viable
  narrowed form worth pricing: FUNCTION-level transplant — distill
  a dense-model capability slice into ONE new expert of the MoE
  (train only that expert's weights + router bias on an
  oracle-verified corpus, everything else frozen) and gate it
  paired v an untrained-expert control. This is a capability-graft
  rung, several steps behind the anatomy program; parked behind
  EX5 slate (b)/(c) and PRECISION-CREST-TRANSPORT. No measured
  anchors yet; residue is the pricing desk (adapter parameter
  count, corpus size, Mac wall). Attribution: Artin (ask), house
  (breaks + narrowed form).
- **BANKED (2026-08-21): EXACT-BF16 OUT-OF-CORE 27B — AirLLM as
  existence proof, MTP self-speculation as the revival key for the
  dead RESIDENT-DRAFT/STREAMED-VERIFY program** (Artin's ask via
  the AirLLM sighting; GPT seat design; two Opus survey/measurement
  agents; house verification). QUANTITATIVE FEASIBILITY NOTE, all
  numbers measured on this Mac or config-derived with sources:
  - PREMISE CORRECTION: the full BF16 dense checkpoint is ALREADY
    LOCAL (~/qwen_vendor, 55.56GB, 18 shards, 64 layers, 26.9B
    text params, bfloat16) including the trained MTP tensors
    (~0.85GB; shipped separately as model_mtp.safetensors
    upstream — absent from the main safetensors index, so an
    index grep wrongly reads "no MTP head").
  - TEXT-ONLY COLD PATH IS 49.7GB, not 54-55.6: the 5.9GB vision
    tower never pages in on text decode (free 11% GPT left on
    the table). DeltaNet blocks are BIGGER than full-attention
    blocks (0.703 v 0.682GB); the 48 linear layers are 33.7GB of
    the stack and ARE the streaming problem — "skip the cheap
    linear layers" intuition is backwards.
  - MEASURED: M3 Pro (150GB/s class, spec-inferred), 36GiB;
    SSD 4.2GB/s single-stream, 8.3GB/s at queue depth 4 (parallel
    reads are a 1.98x lever); BF16 lossless ceiling 1.51x
    (10.6 bits/weight order-0, two independent tensors agree to
    0.3%; mantissa byte exactly incompressible at 8.000 bits, all
    headroom in the 37-value exponent plane, 2.39x alone;
    byte-split before zstd is worth 1.10 b/w free); DISK IS THE
    BINDING CONSTRAINT: 17GiB free cannot hold a 35.6GB lossless
    re-encode beside the 55.6GB source (delete-after-compress or
    external volume is a precondition).
  - HONEST LADDER (tok/s): naive stream 0.12-0.14; +MTP
    (third-party acceptance 65.5% at k=3, E[tokens/sweep]=2.37 —
    NOT k+1) ~0.3; +4-stream reads ~0.6-0.7; +lossless 1.51x and
    28GB resident ~2.6 — only the FULL stack approaches the
    full-resident BF16 ceiling of ~3.0 tok/s (which itself does
    not fit in RAM); the house lossy 6.98GB tower ceilings at
    ~21.5 tok/s resident. The exact path's entire value is the
    token-equivalence guarantee arm A explicitly does not book
    (WHOLE-0T fences, RESULTS L32980).
  - DELTANET ROLLBACK (good news + the likeliest silent bug):
    speculative verify needs ONE pre-verify state snapshot,
    48 layers x 1.57MB = 75.5MB bf16 (151MB fp32) = 0.25-0.5% of
    a sweep — immaterial. But DeltaNet state is ORDER-DEPENDENT
    and NOT truncatable like KV: any implementation that
    truncates it like a cache produces wrong-but-plausible tokens
    with no crash. Full-attention KV (16 layers, 65.5KB/token)
    truncates normally.
  - KIN SCORED AGAINST EXACTNESS: Apple LLM-in-a-flash bundling
    transfers, its ReLU-sparsity payload does not (SwiGLU here);
    PowerInfer-2 pipelining transfers, TurboSparse does not;
    DejaVu does not (approximate by construction); MTP/EAGLE
    self-draft transfers COMPLETELY (acceptance prices speed,
    never correctness); llama.cpp mmap + DeepSpeed prefetch are
    the plumbing. Field tripwire: ollama#17776 measured MTP
    variants SLOWER than non-MTP on Apple Metal (draft cost >
    savings — the exact three-number booking law from the
    2026-08-17 correction at the resident-draft bank: acceptance,
    sweeps/token, AND end-to-end wall).
  - REVIVAL: the RESIDENT-DRAFT/STREAMED-VERIFY bank (2026-08-17,
    marked DEAD 2026-08-18 v QWEN-RK-CENSUS-0) revives — both
    failed screens indicted artifact A as router/generator; a
    VENDOR-TRAINED MTP head bypasses both. Links: dense no-router
    control bank (2026-08-16, same model, revision pin), 2-bit
    router riff (exactness kills its DejaVu route independently),
    cross-model KV bank (same 48/16 hybrid fence), standing
    per-token-offload caution (this bank answers it with measured
    stream numbers, not assumption).
  Honest breaks: unified bandwidth spec-inferred (Metal
  measurement forbidden during the live EX6 run); acceptance is a
  third-party INT8 figure, unmeasured for BF16; MLX batched
  multi-position DeltaNet verify EXISTENCE unverified; AirLLM
  publishes zero tok/s numbers (nothing to counter-book).
  Testable residue, in order: (1) preflights — mtp.* tensor
  shapes, MLX DeltaNet batched-verify support, disk-space
  decision; (2) a one-layer streamed-GEMV timing probe (cold
  mmap v resident, both stream counts) prices the pager without
  building it; (3) if numbers hold, the Axiom/MLX pager design
  doc. AirLLM counter-run explicitly NOT queued (no published
  number to check). Attribution: Artin (ask), GPT (design + MTP
  lead), Opus seats (survey + measurement), house (verification,
  ladder arithmetic, revival call).
  BANK CORRECTIONS (2026-08-21, GPT seat review, house-verified
  arithmetic; applied before any number becomes a design
  constant): (1) DELTANET ROLLBACK — one pre-verify snapshot S0
  is insufficient for PARTIAL rejection in one sweep: accepting a
  k'-prefix needs the recurrent state AFTER that prefix, so
  either per-depth snapshots S1..S3 (~3 x 75.5MB = 226MB, still
  immaterial) or recompute-from-S0 during the same layer
  residency visit (zero extra I/O, extra arithmetic only) — the
  silent-bug note stands, the single-snapshot sufficiency claim
  is corrected. (2) MTP E[tokens/sweep] = 2.37 assumed a CONSTANT
  conditional acceptance 0.655 at every depth; an aggregate rate
  cannot identify E — public Qwen3.8 Q4/Q8 traces show strongly
  depth-dependent acceptance. 2.37 is an ILLUSTRATIVE assumption;
  pricing requires our own BF16 P(A1), P(A2|A1), P(A3|A1,A2)
  acceptance-length distribution. (3) RESIDENT SEMANTICS — the
  ladder's "~2.6 tok/s at 28GB resident + lossless" conflated
  two designs: 28GB of DECODED bf16 resident leaves ~21.7GB raw
  = ~14.4GB compressed cold traffic (~1.4 tok/s at QD4 + E=2.37,
  does not close); the promising shape is 28GB of the COMPRESSED
  representation resident (49.7/1.51 = ~32.9GB packed, ~4.9GB
  cold), which only works if the codec admits packed-to-GEMV
  without a materialize-then-multiply pass that re-doubles
  memory traffic. The decisive residue cell is therefore ONE
  LAYER: lossless-packed weights -> direct/fused exact-BF16 GEMV,
  scored as EFFECTIVE EXACT-BF16 GB/s, never compression ratio
  alone (1.51x is an entropy ceiling, not an achieved
  random-access runtime codec). (4) The 8.3GB/s QD4 figure
  requalifies as cache-cold physical I/O before use. EXACTNESS
  ORACLE defined now: the slow sequential one-token out-of-core
  BF16 path, no MTP, is the reference; batched verify, packed
  codec, MTP, and fused kernels must each reproduce its greedy
  tokens; if batched DeltaNet arithmetic differs from sequential,
  speculative positions are processed SEQUENTIALLY within each
  layer residency visit rather than sacrificing exactness.
  (5) DISK does not block bootstrap: shardwise transactional
  encode -> decode/sha verify -> atomic promote -> delete source
  shard fits in 17GiB free. Residue order updated: the
  packed-GEMV one-layer cell is now residue (2) alongside the
  streamed-GEMV timing probe; depth-conditional MTP acceptance
  measurement joins the preflights.
  KIN UPDATE (2026-08-21, Artin sighting + GPT seat): NInfer added
  as PRIMARY systems kin above AirLLM. AirLLM stays the
  "larger-than-memory executes at all" existence proof; NInfer is
  the specialization blueprint — from-scratch C++/CUDA,
  checkpoint-closed to Qwen3.8, custom .ninfer artifact laid out
  for execution, model-specific Gated-DeltaNet/small-T decode
  kernels, CUDA-graph replay, native MTP3, resident packed
  weights (~20GiB NVFP4). Observed on a 5090: ~71 tok/s MTP0,
  ~151-220 tok/s MTP3, acceptance highly workload-dependent —
  empirical support that the objective is COMMITTED TOKENS PER
  TARGET WEIGHT TRAVERSAL, not raw draft acceptance (the same
  three-number law as the 2026-08-17 correction). Distinctions
  that fence the analogy: NInfer has NO offload — it does not
  solve the memory deficit, and its artifact is lossy NVFP4 where
  ours must stay bit-exact BF16. Transferable invariants to
  price for Mac/Axiom: one fixed text target, mmap-able .axiom
  artifact in execution order, separate T=1 / T=2..4-verify /
  prefill kernels, explicit GDN running/chunk/snapshot contracts
  with per-prefix state outputs, logical-v-physical byte
  accounting, kernel-consumed lossless representation (never
  safetensors-shaped materialization), eventually C++/Metal in
  the hot path over high-level MLX. Mac-specific edge: unified
  memory removes the obligatory host-to-device copy — price
  direct Metal reads from resident/mmap packed pages + explicit
  cold-page prefetch, but MEASURE macOS VM/page-fault behavior
  rather than trusting mmap residency. Residue insert: a
  NInfer-specific survey (artifact layout, GDN snapshot/chunk
  contracts, MTP state handling, traffic accounting) runs AFTER
  the EX6 receipts freeze; bank transferable invariants only,
  never CUDA details.
- **BANK UPDATE (2026-08-21): the horizon frame's interface story
  gets a PHASE coordinate** (VERDICT EX6-PHASE-0, RESULTS 38436).
  The named carriers' harm lives in CONTEXT ASSEMBLY: masking them
  during the prompt phase alone recaptures the full deletion crest
  (+47 v +48), masking them during generation does nothing (-2,
  sub-floor), effects additive. For the horizon frame: what
  crosses the interface while the state is being BUILT decides
  usable v interfering; by generation time the damage is already
  in the state. EX5's decode-time failure enrichment demoted to
  symptom. Next honest discriminator: a mechanism rung on WHAT the
  carrier-routed prefill state gets wrong (state-space, not gate
  counts) — unregistered.
- **BANKED (2026-08-21): EXPERTDB / ROUTE-DB — a control-plane
  expert database for MoE residency/prefetch, with the exactness
  boundary drawn so a bad prediction is a CACHE MISS, never a
  wrong token** (Artin's riff, GPT sharpening, house corrections).
  The mapping: for the 30B MoE, a per-(layer, expert) record —
  exact weight location/bytes/sha, demand, PHASE AFFINITY (EX6
  made phase a first-class coordinate: prompt-mask +47 v decode
  -2), task fingerprint (5-domain traj breadth), carrier
  membership, cache frequency, load cost. The ROUTER stays
  authoritative for selection; the DB predicts residency and
  prefetch only. Explicitly NOT generic RAG and NOT ANN-on-the-
  exactness-boundary — the ledger carries two naive-vector
  negatives that kill that route: QWEN-MIPS-CENSUS-0 (exact
  5120-D head indexing visits q50 0.989+ of vocab, modeled bytes
  > brute force) and QWEN-RK-CENSUS-0 (oracle dense top-k leaves
  32-63% activation reconstruction error). Measured anchors:
  EX5-LAYERMATCH (named identity transports while aggregate
  rank/layer lenses fail — so the DB keys on IDENTITY, not
  class), EX6-PHASE (phase asymmetry is real physics for a
  phase-aware predictor), EX5-TRAJ-ANATOMY (demand, phase
  fractions, first-touch already computed per slot). HOUSE
  CORRECTIONS to the GPT draft: (a) the "causal ablation effect"
  column is populated at MASK grain only (one mask, one dose) —
  per-expert causal effect is UNMEASURED and EX5 specifically
  showed aggregate proxies mislead; the schema must mark that
  column unpopulated rather than bake in false precision; (b)
  deployment honesty — the 4-bit MoE (17GB) is FULLY RESIDENT on
  the 36GB Mac, so an expert cache prices for the 10GB 3080, for
  bigger MoEs, or for BF16-grade artifacts, not for the current
  Mac inference path. Testable residue, rung 1 ZERO-MODEL-COST
  (desk, CPU): replay the frozen routing traces
  (logs/opus/moe_gt1_traj_v2.jsonl + the five domain trajs)
  through LRU / LFU / per-layer-LFU / phase-aware / prompt-route
  predictors at several resident budgets; score bytes/token, miss
  stalls at measured load costs (per-expert byte sizes exist in
  the compose manifests), and 1/2/4-layer-ahead prefetch
  precision/recall — prices whether the cache is worth building
  before any implementation. Kin to compare when rung 1 books:
  MoE-Infinity-class request-level expert caching, LFU +
  speculative expert-prefetch work, PowerInfer, DejaVu,
  LLM-in-a-Flash (exact-v-approximate boundary stated per kin).
  Attribution: Artin (riff), GPT (control-plane framing + rung
  design), house (negatives linkage, grain + deployment
  corrections).
- **BANKED (2026-08-21): IMPLICIT-EXPERTDB for the dense 27B —
  activation/causal-fingerprint channel clusters as a
  drafter/prefetch predictor UNDER the exact packed-BF16
  verifier, never as the answer path** (Artin + GPT, same
  exchange; extends [[DENSE-ROUTER READOUT]] and the EXACT-BF16
  OUT-OF-CORE bank). The optimization target is COMMITTED TOKENS
  PER EXPENSIVE BF16 SWEEP (the NInfer objective), with retrieved
  channel groups predicting what to prefetch/draft while the
  streamed exact target verifies — potentially composing with the
  native MTP route rather than replacing it. Honest breaks
  carried from the parent banks: clustering must be by
  activation/causal fingerprint, never raw weight cosine (the
  never-score-weights-by-weight-distance law); a dense tower has
  no gate to read, so "routes to" needs operationalizing before
  anything is testable; and the DejaVu-style skip route stays
  dead under exactness — groups may only inform ORDER and
  RESIDENCY, never omission. Residue: blocked behind the
  out-of-core preflights; first cell is a fingerprint-stability
  census (do channel clusters transport across the gate corpus v
  the domain trajs?) at desk cost. Attribution: Artin (ask), GPT
  (drafter framing), house (boundary + sequencing).
  RUNG 1 MEASURED (2026-08-21, OBSERVATION ROUTE-DB-REPLAY-0 +
  AMENDMENT -SIM): the registered K<=32 read SPLIT (phase-static
  clears 110 MB on 3/6 traces); the robust descriptive read is
  plain LRU@K48 (~6.4GB): 63.7-103.7 MB/decode-token on all six
  traces with zero fitted table. Phase-split tables look strong
  but their numbers are LOWER BOUNDS (phase-boundary table churn
  uncharged) and the Belady variant they tied is insert-on-miss,
  not a true ceiling; prefetch read = previous-token same-layer
  reuse 0.40-0.54 (depth claim retracted — one predictor measured
  thrice). Fences: self-fit tables; MB/token portable, ms columns
  Mac-SSD only, 3080 stalls need their own NVMe->H2D measurement.
  Rung 2: within-domain prompt holdouts with charged transitions
  + BELADY-BYPASS, then the 6x6 transport matrix; sharp target =
  transported phase knowledge making K32 behave like plain K48.
- **BANKED (2026-08-21): DOMAIN-CAPABILITY-COUPLING — does domain
  change the model's CAPABILITY, or only its routing?** (Artin's
  ask, morning after the phase night: "physics/math domains
  benefit each other first hand"). What the ledger actually
  holds: domain-dependent ROUTING is measured six ways (the
  gt2/gt3/gt4 trajectory suites, Jaccard lenses, the carriers'
  4.3/5 domain breadth, ROUTE-DB's per-domain locality spread
  with prose the outlier) — but CAPABILITY is measured on exactly
  ONE domain (the 120-item math gate, sympy oracle). Every
  deletion/phase verdict this week reads "capability" through
  that single lens; whether the math-demand-derived named-80 are
  math-specific or domain-general carriers is UNMEASURED. The
  mapping (Artin's observation, made falsifiable): if domains
  share substrate, a math-derived intervention should transport
  to a non-math gate; if capability is domain-partitioned, the
  same deletion should be neutral-or-harmful off-domain. Honest
  breaks: "physics/math benefit each other" has no booked
  same-instrument anchor yet (training-diet coupling results are
  a different substrate — the small math-natives, not the 30B
  MoE); a second domain gate needs its own ORACLE, and the house
  has one ready-made candidate: code scored by the toolchain
  (llmopt/codegen/llvm.py — assemble/run, no string match; native
  clang exists on the Mac), while a physics gate would need a
  generator + oracle built first (charter: physics is a standing
  engine domain). Testable residue, in order: (1) desk census —
  the carriers' routing share on the code/phys trajs is already
  on disk (EX5-TRAJ breadth says they ARE routed off-domain;
  count their demand share per domain); (2) the transport rung —
  a code gate (toolchain oracle, fresh seeds) under full v
  named-80-deletion v a control mask: does the math crest
  transport to code capability? Same-instrument, same-machine,
  paired. (3) only then the training-side coupling question
  (diet mixing on the math natives — different substrate, its
  own registration). Attribution: Artin (question + the coupling
  intuition), house (falsifiable split routing-v-capability,
  oracle inventory, residue order).
  RUNG 2 MEASURED (2026-08-21, OBSERVATION ROUTE-DB-REPLAY-1):
  the phase-table route is DEAD for residency — held out and
  churn-charged, phase-static@K32 runs 1.6-4.7x WORSE than plain
  LRU@K48 (sharp target fails on all six traces) and loses to
  same-K LRU on 4-5/6. LRU DYNAMICS transport (rung-2 warm start
  was train-half fitted; rung-1 empty-start numbers nearly
  identical): warm LRU@K48 = 65-95 MB/decode-token on unseen
  prompts. Phase info not nulled generally — PHASE@K48 edges LRU
  on proofs; the DEAD call scopes to K32-replaces-K48. True
  BELADY-BYPASS ceiling shows ~2x implementable headroom over
  LRU at both budgets — in reuse structure, not phase tables.
  Implementation call now rests on: LRU@K48 (~6.4GB) + a
  per-machine I/O measurement (3080 NVMe->H2D unmeasured); the
  6x6 domain matrix moot for residency, still live for
  [[DOMAIN-CAPABILITY-COUPLING]].
  CENSUS MEASURED (2026-08-21, OBSERVATION TOPSET-OVERLAP-0, run
  on the 3080 short-test window): the hypothesis CONFIRMS — the
  candidate manifold survives compression (top-256 recall
  0.957-0.973, Jaccard 0.918-0.947, Spearman 0.979-0.983; top-8
  shared 0.875-1.0) while rec@1 = 0 at all five near-tie loci.
  BLe's top-256 is 245-249/256 inside vendor's top-256, zero
  specials. DROP-TOP256 REDESIGN PRICED DEAD: the first
  admissible outside token sits at BLe rank 204-223, 12.3-18.6
  logits down — the exact CONTROL-MATCH-FAILED class.
  [CORRECTED 2026-08-21, AMENDMENT MORNING-BLOCK-WORDING: the
  "rank-2 in-manifold control" first proposed here is the
  TREATMENT — rank2 == vendor token at 5/5 loci (binary top-2
  near-ties); rank-3 is 0/5 gap-matched under the +-0.05 law
  (gaps 0.35-3.64 v targets 0.007-0.24). An identity-matched
  third-token control DOES NOT EXIST at these specimens; the
  thread parks or a tri-tie desk search finds fresh loci.] Portable top-1024
  sidecars for both heads at every locus live in the receipt —
  the census never needs the 3080 again (the sufficient-
  statistics lesson, applied).
  REFRAME (2026-08-21, GPT seat post-rung-2, house-verified
  anchor): per-DOMAIN tables are NOT the target architecture —
  the six traces are evaluation slices, not a partition of the
  model's function, and the ledger already holds the
  cross-domain-coupling anchor (VERDICT MOE-GT-5, RESULTS L20179:
  the union-of-bases mask resurrects the dead math core 0 -> 55
  with mostly VERBAL experts; "math capability, masked, lives in
  the verbal branch" — necessity across domain labels, exactly
  what Artin's DOMAIN-CAPABILITY-COUPLING intuition names).
  EXPERTDB reframed: COMPLETE metadata over all 48x128
  identities; the route predictor is statistical —
  score(layer, expert | x) = global_core + phase_residual +
  latent_route_mixture(x) + temporal_reuse — with latent modes
  LEARNED from routing fingerprints, never named by domain;
  prompts may mix modes and components may cross labels
  (labels are interpretation AFTER fitting, only). RUNG 3 (next,
  zero model cost): ROUTE-BASIS census over prompt-level routing
  signatures — frozen representation, effective rank / NMF-PCA
  spectrum, component memberships, cross-domain nearest-neighbor
  rate, and held-out demand prediction under four priors (global
  / global+phase / named-domain / latent-mixture). The decisive
  systems question unchanged — can transported structure make
  K32 behave like plain LRU@K48 — now asked of SHARED LATENT
  factors, with the rung-2 repairs retained (charged transitions,
  Belady-bypass warm starts, honest async prefetch when it
  lands). If latent structure transports, the "vector DB" role
  is retrieval over compact route fingerprints for residency
  priors — ANN errors cost misses, never tokens.
  RUNG 3 MEASURED (2026-08-21, OBSERVATION ROUTE-BASIS-0):
  routing signatures are ~6-dimensional (PCA eff-rank 5.98);
  under the prefill-only boundary NAMED-DOMAIN priors predict
  held-out decode residency best (K48 coverage 0.939, latent-mix
  r=16 close at 0.922, global/raw-prefill 0.75-0.76, oracle
  0.980); cross-domain NN rate 0.0 at prompt grain — but the NMF
  basis is genuinely CROSS-LABEL exactly in the math/phys/proofs
  family (three blended components), the routing-level signature
  of [[DOMAIN-CAPABILITY-COUPLING]]'s intuition. At r=16 latent
  does not yet subsume named; open cells: r sweep + hybrid
  residuals, and the static-prior + LRU composition replay for
  honest MB/token.
- **BANKED (2026-08-21): GENERAL-ROUTING-FACTORIAL — is routing
  separability SEMANTIC or TEMPLATE-DRIVEN? An observation-only
  factorial over (topic x linguistic form x requested operation)
  on the frozen public MoE** (Artin's concern via GPT seat; house
  verification + charter handling). The critique it operationalizes
  is sharp and aimed at our own result: ROUTE-BASIS-0's
  cross-domain-NN rate of 0.0 was measured on six GATE-CORPUS
  slices that share templates within slice — math<->physics
  coupling alone cannot answer Artin's question because those
  corpora share nomenclature, symbolic syntax, and answer form,
  and the ledger already holds form-sensitivity evidence (CE-400
  format-BOUND; the EFFORT-tag trajectory result). The design:
  a FROZEN benign prompt corpus factored explicitly across
  TOPICS (history, geography, astronomy, literature/language,
  everyday science, general factual QA) x FORMS (direct QA /
  explain / definition / MCQ / completion) x OPERATIONS (recall /
  causal explanation / comparison / classification), with MATCHED
  PAIRS constructed so topic and wording dissociate. Primary rung
  is ROUTING-ONLY: variance decomposition + similarity + held-out
  prediction of fingerprints against topic v form v operation v
  interactions. CHARTER HANDLING (explicit, not delegated): the
  GPT draft floats "small benign conceptual biology/chemistry
  slices if charter interpretation permits" — the house DEFAULT
  here EXCLUDES them; the factorial dissociation works fully on
  the benign-topic set above, and whether passive routing
  observation on conceptual bio/chem prompts is inside the
  "concepts as methods, zero harmful applicability" clause is an
  ARTIN RULING, banked as an open decision, never a mid-task
  interpretation. [RULED 2026-08-21, same day: Artin includes
  benign conceptual biology and chemistry as FIRST-CLASS BALANCED
  topic levels; the charter gains the evaluation/engine
  distinction (CLAUDE.md, "measured, never developed") with every
  hard prohibition retained verbatim and the anti-Goodhart fence:
  bio/chem performance is never a search/selection objective, and
  causal-perturbation evaluation requires the intervention frozen
  independently of (or preregistered before reading) the bio/chem
  scores. Content stays textbook/conceptual; propositions and
  factorial cells freeze before any model call.]
  Non-negotiables carried verbatim: no training,
  no fine-tuning, no tools, no domain solver, no generation
  pipeline, no capability improvement, no
  pathogen/wet-lab/synthesis/molecular-design content in any
  slice; this is passive interpretability of a frozen public
  model. If a capability leg is EVER justified it uses frozen
  benign MCQ/exact-answer items with ANSWER IDENTITY as the
  evaluator (no new domain engine), registered separately from
  the observational rung. Testable residue: (1) corpus authoring
  + freeze (committed prompt set with the factor table, before
  any model call); (2) the traj-capture run (Mac, TRAJ=1
  machinery, one pass); (3) the decomposition — does topic
  survive form-matching, does form survive topic-matching, and
  does the ROUTE-BASIS domain separability collapse under
  template control? Attribution: Artin (the concern), GPT
  (factorial design), house (charter fence + the
  template-critique linkage to ROUTE-BASIS-0).
  PRIMARY RUNG MEASURED same day (OBSERVATION GRF-0): topic
  separability is SEMANTIC — held-out topic accuracy 0.775 v
  chance 0.125 under full template control — but the gate
  corpora's templates were inflating ROUTE-BASIS-0's purity (NN
  topic 0.69 here v effectively 1.0 there). The phase split
  mirrors EX6: prefill routing reads the FORM (predicted 1.00
  from prefill alone), decode routing tracks the TOPIC (contrast
  0.1505 v form 0.0058). Matched pairs: content-coherence 0.944
  v form-coherence 0.702. Bio/chem slices behaved as ordinary
  topic levels; measurement-only, zero capability surface. Open
  cells: answer-identity leg (separate registration), per-layer
  depth profile of the form->topic handoff, latent refit on the
  template-controlled corpus.
- **BANKED (2026-08-21): TIME-AS-STRUCTURE — explicit learned
  representations of causal order, horizon, budget, and
  time-to-future-use, as opposed to positional order or
  timestamps** (Artin's frame, GPT co-development; house
  assessment + inventory). "Time" here means PARTIAL ORDER /
  HORIZON / BUDGET, never seconds; no race-condition, security,
  or vulnerability work under this bank, ever. IN-HOUSE INVENTORY
  (where the concept already lives implicitly, verified in the
  ledger): search-engine node budgets and solve-at-fixed-budget
  economics (L4 failures are 10/11 WALL timeouts —
  expression-size economics; size-cap pruning arms); the magic
  estimator's cost/variance slots ("prediction pays only where
  variance lives"); EX6's phase asymmetry (prompt v decode is a
  causal-order coordinate, now measured twice); ROUTE-DB's
  temporal locality (LRU beat every fitted table; the
  Belady-bypass machinery already computes EXACT next-use
  distances as its eviction key); the prev-token prefetch ceiling
  0.40-0.54; training-side temporal structure (capability leads
  settling, rho 0.979; the LR absorption floor; schedule
  direction a non-factor). The pattern across all of them: the
  house MEASURES temporal structure constantly but has never made
  TIME-TO-EVENT the prediction target. NOVELTY ASSESSMENT
  (house): genuinely new as a target; nothing booked predicts a
  time-to-X distribution. Two candidate formulations, ranked:
  (1) ROUTE-TIME — from the frozen routing traces, the exact
  next-use distance of every (layer, expert) at every step is
  already computable (the replay2 next_use machinery IS the label
  factory); the falsifiable claim is clean: routing
  history/prefill features predict time-to-next-use better than
  recency (LRU rank) and popularity (demand rank) baselines on
  held-out prompts, scored by rank correlation + top-K
  eviction-decision agreement with Belady. Zero model cost,
  frozen labels, desk-runnable — THE RECOMMENDED FIRST RUNG, and
  it feeds the EXPERTDB latent predictor directly (a
  time-to-next-use head is the principled replacement for the
  dead phase tables). (2) MAGIC-TIME — replace/augment the
  solved-at-budget binary with a SOLVE-BY-BUDGET SURVIVAL CURVE
  over node horizons, learning a distribution of logical
  time-to-solution from root features; oracle-verifiable via the
  existing sympy pipeline and priced by the existing
  wall-timeout economics, but needs the search corpus + estimator
  retrain — SECOND, behind a desk census of whether solve-time
  variance is even predictable from root features (the
  magic-estimator bank's own law: prediction pays only where
  variance lives). Honest breaks: ROUTE-TIME's labels are
  workload-conditional (gate-corpus traces; the GRF captures add
  a template-controlled workload); a learned time-to-use head
  that only rediscovers recency is the null outcome and must be
  reported as such; nothing here claims biological/temporal
  cognition — plain-language rule applies. Testable residue:
  (a) ROUTE-TIME label-factory desk rung (predict-v-recency-v-
  popularity, held-out); (b) the MAGIC-TIME variance census;
  (c) if (a) beats recency, wire the head into the EXPERTDB
  rung-2 replay as a fifth implementable policy and re-ask the
  K32-v-K48 question. Attribution: Artin (frame + "time as
  structure, not wall-clock"), GPT (survival-curve and
  next-use-distribution formulations), house (inventory, novelty
  check, ranking).
  RESIDUE (a) AND (c) MEASURED (2026-08-21, VERDICT
  ROUTE-TIME-0): (a) fired — finite-horizon next-use prediction
  beats the age-only hazard by median +0.041 AUC at H=8 (~half a
  count-saturating popularity term — "decayed frequency" updates
  only on activation, so it is a function of count, not
  token-time decay; prev-token co-activation the WEAKEST
  baseline — house mechanism call wrong); (c) failed 0/6 —
  learned eviction at K32 loses to LRU@K48 everywhere and to
  same-budget LRU@K32 on 5/6. CORRECTED same day (AMENDMENT
  -ISOLATION, outside audit): the original holdout was
  stream-entangled and the closed loop could evict current-token
  experts; the isolated + protected rerun reproduces both
  readings (median +0.0406 v +0.0407; still 0/6, LRU@K32 loses
  5/6), so the defects were not load-bearing. The
  beyond-popularity component is LARGEST at short horizons
  (H<=4, +0.042 median) not long ones; "prefetch territory" is
  demoted to hypothesis (H64 AUC lives on ~4% never-reused
  negatives). Sharpened claim: temporal structure in expert
  reuse is a PREDICTION resource, not (at this policy class) an
  EVICTION resource. New residue: victim-decision-conditional
  training objective; prefetch as hypothesis. Residue (b)
  MAGIC-TIME still open.
  NOTHINK TRANSPORT MEASURED same day (OBSERVATION GRF-NOTHINK-0):
  the reweighting SURVIVES answering and the registered refutation
  does not trip — decode topic contrast STRENGTHENS to 0.1856
  (2.7x form) with a valid capture (0 think markers, 20/200
  ceiling, 40/40 MCQ letters). House E2 call wrong in the
  interesting direction: answering routing is MORE topic-organized
  than thinking routing (deliberation dilutes topic structure).
  The two captures now bracket both regimes; depth rung (where
  along 48 layers does form hand off to topic) is unblocked, and
  the answer-identity leg has its data on disk awaiting its own
  registration.
  RECEIPTS AMENDMENT + HORIZON RIDER (2026-08-21, AMENDMENT
  GRF-NOTHINK-0-RECEIPTS): the thinking-v-answering difference is
  maximal at the START of decode — thinking routing begins
  topic-free (contrast 0.0025 at N=8 v answering's 0.1441) and
  converges with generation time. First measured
  [[TIME-AS-STRUCTURE]] bridge: topic-informativeness of routing
  v position-in-generation is a regime property. Fences: the
  empty-think scaffold treats PREFILL too; GRF-0 scopes to
  greedy/96; MCQ answer-identity readiness is 32/40 under the
  frozen anchored extractor.
  CORRECTION (2026-08-21, AMENDMENT GRF-NOTHINK-0-WINDOWS): the
  rider2 horizon numbers were CUMULATIVE first-N signatures, so
  "converges with generation time" is RETRACTED — equal-width
  sample-matched windows on a fixed >=32-step cohort (n=99) show
  the thinking topic contrast peaking in window 9:16 (0.0867)
  then declining (0.0438, 0.0388), not strengthening. What
  survives sample-matched: the FIRST 8 decode tokens of thinking
  routing carry near-zero topic contrast (0.0047; weak topic
  information still decodable, LOPO 0.253 v 0.125 chance — topic
  is not the organizing cosine axis) while answering's are
  already topic-loaded (0.1331, LOPO 0.626). Magnitudes scoped
  to the form-selected >=32-step cohort (n=99); the regime
  comparison is paired within it. The bridge demotes to a
  measured early-horizon regime difference. Extractor audit: 32/40 stands,
  all matches are "X) option" restatements, zero article-A prose
  false positives.
  [TIME-AS-STRUCTURE bank cross-link, same day: the GRF horizon
  rider delivered the bank's first measured instance — see the
  GRF bank's RECEIPTS entry; routing topic-informativeness as a
  function of generation position differs by regime, which is a
  time-indexed observable no wall clock touches. DEMOTED
  2026-08-21 (AMENDMENT GRF-NOTHINK-0-WINDOWS): the cumulative-N
  reading did not survive equal-width windowed adjudication —
  the bank's measured instance is now the EARLY-HORIZON regime
  difference (first-window topic contrast 0.0047 thinking v
  0.1331 answering, sample-matched), not convergence.]

- **BANKED (2026-08-21): "Haven't we tried this with the mathnative
  models? Like Collatz — 3x+1, odd/even; some prompts re-input
  return LESS text than input. What is odd/even in our case? Did
  precision matter — fp16 training loss as why the model is
  imperfect?"** (Artin, the EX6-MED postmortem afternoon).
  The mapping: a mathnative derivation IS a rewrite dynamical
  system — each step maps expression -> expression, and solving =
  reaching a normal form (the fixed point). The Collatz analog is
  real: CONTRACTION steps (simplify, cancel, integrate-termwise)
  are the x/2 moves; EXPANSION steps (series expand, split, the
  banked euler-move / Liouville-jailbreak family) are the 3x+1
  moves that GROW the term before a later contraction pays it
  back. "Odd/even" in our case = whether the current expression
  admits a direct contraction or needs an expansion move first —
  a property of the term, as parity is of the integer.
  Termination of mixed expand/contract chains is exactly the
  Collatz-shaped open question; our farms dodge it with step
  budgets and honest-UNDECIDED oracles.
  Measured anchors: euler-move/ceiling-mover family banked (series
  continent, 2026-07-21); step-model = associative memory of
  state->rewrite pairs (2026-07-17); no Collatz-like LENGTH
  census has ever been run (output/input token ratio per level is
  unmeasured — honest gap).
  Precision half: the house precision doctrine (CLOSED
  2026-07-24) says birth precision above TF32 is a NON-FACTOR for
  capability — "fp16 training loss makes the model imperfect" is
  not supported in our regime; imperfection is structural, not
  numeric. BUT today's EX6-MED postmortem adds the complement:
  fp16 near-ties flip token DECISIONS across evaluation
  schedules (11/240 diagonals, flips at masked-softmax tight
  margins) — precision is a DECISION-STABILITY lever even where
  it is not a capability lever. That lands exactly on the
  doctrine's one named retest slot (exact-mode gate v rounded
  gate, same weights, when exact inference lands).
  Honest breaks: no measured Collatz dynamics in any booked
  result; the parity analogy is a frame, not a law; completion
  length is confounded by problem level and format.
  Testable residue: (a) desk census of output/input length ratio
  over the frozen gate batteries, by level and by solve/fail;
  (b) classify solved traces by expansion-move usage and test
  whether failures cluster where expansion is required (the
  "odd" class); (c) the standing exact-inference retest slot
  covers the precision half.
  REFINEMENT (same day, review-adopted + Artin's follow-up):
  expansion/contraction is the step OUTCOME; the true parity
  analogue to search for is a cheap LOCAL STRUCTURAL PREDICATE
  of the term that selects which rewrite family is required
  (parity is computable without running the orbit — the analogue
  must be too). The length census should carry token ratio AND
  sympy AST measures (count_ops, tree depth) so text formatting
  cannot masquerade as dynamics. Artin's sharpening: 3x+1 is a
  PERFECT function on EXACT inputs (pure integers), so its
  orbits are deterministic; a model is an imperfect function on
  a lossy substrate, so its orbits are only approximately
  deterministic — which connects the frame to the house's
  deterministic integer battery (exact instruments have
  Collatz-grade orbit determinism; float inference does not) and
  to the exact-inference retest slot. Precision wording
  corrected: today's MED lesson is FINITE-PRECISION /
  EVALUATION-ORDER decision sensitivity on the 4-bit MLX stack —
  dtype attribution (fp16 v 4-bit dequant v accumulation order)
  was not measured.
  Attribution: Artin (Collatz frame + precision question +
  perfect-function sharpening), house (rewrite-dynamics mapping,
  doctrine cross-links), GPT (parity-as-predicate refinement).
  REFINEMENT 2 (same day, Artin + GPT): three layers now named —
  the task's rewrite RELATION R (math admits many next states,
  unlike Collatz's unique successor), the learned POLICY pi
  choosing among them, and the EVALUATOR E_I realizing the
  choice. Determinism corrected: finite precision does NOT mean
  nondeterministic orbits — a fixed implementation defines a
  perfectly deterministic F; what MED measured is
  CROSS-SCHEDULE NON-INVARIANCE (two implementations of the
  same F disagree at sensitive near-ties), and the deterministic
  integer battery is stronger precisely because its fixed
  arithmetic/rounding/tie-break contract aims at bit-equality
  ACROSS implementations, removing evaluator variation and
  exposing the policy's dynamics alone. Artin's perfectibility
  claim, banked: when the training corpus never NEEDS the
  truncated digits, the model CAN be ideal on that closed system
  — anchored by the precision doctrine (birth precision above
  TF32 a non-factor: the data didn't need the digits) and the
  calculated-model thesis (closed systems admit computable
  weights); imperfection enters where rounding/truncation
  matters or the case was never in the diet. His correlation
  question (absorption x lr x training data) is HONEST-OPEN:
  diet exposure share, lr schedules, and absorption were each
  measured as separate levers in the mathnative threads, but
  their joint correlation structure was never booked as one
  experiment. Parity analogue re-scoped for a relation: predict
  whether progress REQUIRES an expansion-family move v an
  available contraction (family-level, not unique-rewrite).
  Later exact-battery observables banked: excursion height,
  transient length, basin structure, orbit merges, non-solving
  cycles.

- **BANKED (2026-08-21): THE ABSORBED-INFORMATION COLLAPSE — do
  exposure frequency, LR, and update precision jointly reduce to
  one cumulative eta*|g|/ulp(w) statistic per rule?** (Artin's
  "weren't absorption, lr, training data all correlated?" +
  GPT's factorial design; ledger check first, per Artin's "chance
  we tested this already" — partially YES).
  Already measured (single-axis legs, all booked): THE ABSORPTION
  LAW (absorbed fraction ~= c/LR, four decades, c ~= 2.8e-9) with
  its consolidation ALREADY stating the joint knob — "LR and
  precision are ONE knob: what matters is LR*|g|/ulp(w)";
  CAP-V-TRAJ-2 (LR absorption floor below which steps/data buy
  nothing); BASICS-DIET (exposure share redistributes resident
  capability). Artin's correlation instinct is thus a measured
  mechanism, not an open guess — what was NEVER run is the JOINT
  test.
  The unrun delta, banked as a rung design: controlled
  closed-system factorial varying rule exposure frequency p, LR
  eta, and accumulator/update precision at fixed architecture
  and total compute; per rule family record exposure count,
  gradient/update magnitudes, absorbed-update fraction,
  cumulative surviving update mass, and held-out rule mastery;
  the registered question is whether capability curves COLLAPSE
  against a cumulative dimensionless statistic built from
  eta*|g|/ulp(w) (the law predicts they should; a non-collapse
  names the missing variable). Distinguish unseen INSTANCES from
  absent/underidentified RULES — perfect generalization needs
  every rule identified, not every instance seen.
  Perfectibility wording tightened (GPT correction, adopted):
  TF32~=FP32 shows precision ABOVE the regime's numeric
  requirement is unnecessary; it does not show the current
  architecture/width/optimizer reaches a perfect gate. The
  calculated-model thesis remains an existential/compilation
  program, not a closed empirical result. Artin's claim survives
  as: perfectibility is not precision-blocked in our regime; it
  may still be architecture- or diet-blocked.
  Honest breaks: |g| distribution stability across exposure
  frequencies is assumed by the law's derivation and unmeasured
  at low p; rule-family gradient attribution needs per-rule
  batches or gradient tagging (instrumentation cost unpriced).
  Testable residue: the factorial above, desk-priceable via
  /desk before any GPU spend.
  Attribution: Artin (correlation instinct + tested-already
  check), GPT (factorial design + collapse statistic), house
  (ledger verification: the statistic already exists as the
  absorption law's consolidation).
  CORRECTION (same day, review-adopted): two overclaims above are
  retracted in place. (1) "correlation instinct = measured
  mechanism" overstates — LR x precision via eta*|g|/ulp(w) is a
  measured TWO-AXIS mechanism and diet is an INDEPENDENTLY
  measured third axis; diet joining the same sufficient statistic
  is precisely the unrun joint hypothesis the factorial tests.
  (2) "if blocked, architecture- or diet-blocked" dropped a
  blocker: optimization reachability is separate, and the
  absorption law itself shows numeric resolution CAN block
  (low-LR updates at training time; near-tie decisions at
  inference). Scoped precision claim: above-TF32 did not matter
  for the TESTED birth regime; nothing establishes precision can
  never block elsewhere. FOUR-GATE PERFECTIBILITY FRAME banked:
  representability -> diet identification -> optimizer
  reachability -> numerical realization (extra digits irrelevant
  only once training-absorption AND inference-decision margins
  are both adequate). Factorial statistic hierarchy registered:
  cumulative surviving update mass is a CANDIDATE statistic, not
  a definition of information (equal sum|delta| can differ in
  cancellation/direction); ladder = intended-update survival
  fraction -> realized/intended update norm -> directional net
  rule-associated update. Named clean outcome: absorption curves
  collapse but capability does NOT — which would isolate
  optimization geometry/interference as the missing variable.

- **BANKED (2026-08-21): machine-readable ARTIFACT VISIBILITY
  CLASSES in the prereg schema — always-readable /
  sealed-until-qualification / sealed-forever-on-failure**
  (house + GPT, from the EX6-MED watcher-unblinding incident).
  The mapping: a prereg already declares bars, fences, and
  operands; it should also declare, per output artifact, WHO MAY
  READ IT WHEN. A watcher (or any consumer) then enforces
  blinding mechanically from the prereg instead of from a
  checklist — the /watch skill's classes, promoted from prose to
  schema (llmopt/lab/prereg.py, next to `operands`).
  Measured anchors: AMENDMENT EX6-MED-0-QUALFAIL-2 (the
  unconditional watcher dump that permanently unblinded a
  qualification-failed run); the receipt_freeze hook's
  structured-reference lesson (prose-scraped citations leave
  gaps that only machine-readable references close).
  Honest breaks: enforcement is only as strong as the consumers
  that honor the field; a Bash tail outside the ritual still
  leaks — the schema makes the ritual checkable, not impossible
  to bypass.
  Testable residue: schema field + validator in prereg.py;
  /watch reads it and generates the gated dump command;
  receipt-auditor checks that sealed artifacts never appear in
  session-visible logs before their gate.
  Attribution: GPT (generalize-the-classes ask), house (incident
  + schema placement), Artin (adopted the review).

- **BANKED (2026-08-21): TEMPORAL-DISPLACEMENT-MATCH control —
  equal FORCED native-topk replacements at z1/z2/z3, independent
  of named80 recall** (Artin, post AMENDMENT EX6-TEMPORAL-0-SCOPE).
  The mapping: EX6-TEMPORAL-0's arms matched temporal-CALL dose
  but not realized displacement — the named-80 mask replaced
  ~2.09x more native expert slots at z1 (excluded top-8 mass
  1.82% v 0.86-0.88% at z2/z3, booked recalls). The control:
  a deterministic perturbation that forces an EQUAL NUMBER of
  native-top8 replacements at each temporal position (e.g. demote
  the top-1 pick to the vendor's rank-9 expert at N fixed layers,
  N identical across z1/z2/z3), decoupling position from
  displacement magnitude. If z1's advantage survives
  displacement-matching, temporal sensitivity is intrinsic; if it
  vanishes, TEMPORAL-0's step was carrier-family-demand-shaped.
  Measured anchors: VERDICT EX6-TEMPORAL-0 (step +21/-3/0);
  AMENDMENT -SCOPE (the 2.09x displacement gap).
  Honest breaks: forced replacements are a DIFFERENT perturbation
  family from keepset masking — a null would not directly rescope
  the named-80 result, only the intrinsic-sensitivity reading;
  replacement choice (rank-9 v random-kept) is itself a design
  degree of freedom that needs registering.
  Testable residue: 4-arm rung (NONE + forced-k at z1/z2/z3, same
  seeds/gate), bar on Delta_z1_forced v Delta_z2_forced at
  matched replacement counts.
  Attribution: Artin (the control ask + the scope correction that
  motivates it), house (recall arithmetic + design sketch).

- **BANKED (2026-08-21): outcome-blind per-layer z1/z2/z3
  NATIVE-DEMAND census — where along the 48 layers does z1's
  excess carrier demand live, and does it align with the causal
  depth band?** (Artin, same review). The mapping: the ~2.1x
  excess outside-keepset demand at z1 is a POOLED-over-layers
  number; an unmasked (NONE-arm) capture of per-layer,
  per-temporal-position native top-8 hits against the named-80
  keepset is outcome-blind, cheap (one NONE pass with counters),
  and produces a 48 x 3 demand map. If the layers where z1
  over-demands the carrier family COINCIDE with the depth band
  that carries the +21 (ranked residue 2, z1 depth masks), the
  demand map becomes a cheap predictor of causal locus —
  family-demand explanation (b) gains mechanism; misalignment
  supports intrinsic-sensitivity (a).
  Measured anchors: AMENDMENT EX6-TEMPORAL-0-SCOPE (pooled 2.09x
  gap); EX6-LOC-0-LEVELS (difficulty-dependent structure that a
  per-layer read could stratify).
  Honest breaks: demand is correlational — alignment would be
  suggestive, never causal on its own; the census reads the
  UNMASKED router, so it cannot see mask-induced rerouting
  cascades.
  Testable residue: 48x3 census + registered alignment read
  against the depth-band rung's verdict (rank correlation or
  band-overlap count, named before either fires).
  Attribution: Artin (census ask + alignment hypothesis), house
  (instrument shape).

- **BANKED (2026-08-21): CYBERNETIC-MATH / MATH-CYBER-0 — a true
  closed causal loop for the math models, where the model's ACTION
  determines its next mathematical state and diet** (Artin, GPT
  co-shaping the frame; inspired by Cortical CL1/DishBrain
  closed-loop embodiment; BANK ONLY, no desk/prereg/code until
  Artin returns post-OS-update).
  The mapping: ordinary supervised loss v a causal loop are
  DIFFERENT LEARNING REGIMES. Small exact rewrite world: state =
  canonical expression; action = rewrite family + locus; the exact
  oracle returns legality / equivalence / solved / complexity /
  cycle feedback WITHOUT revealing the correct action; online
  updates allowed within trajectories. The loop closes because the
  model's action selects its next state and thereby its own
  training stream.
  Future factorial (banked design, unpriced): OFFLINE / ENV-LOOP /
  DIET-LOOP / FULL-CYBER at matched transitions, tokens, optimizer
  steps, device — plus the critical REPLAY arm trained OFFLINE on
  the exact experience stream FULL-CYBER generated. ONLINE > REPLAY
  isolates the value of causal feedback/order; REPLAY ~= ONLINE
  says the loop mainly discovered a better curriculum. Later
  extensions: exact next-state/world-model prediction head;
  homeostatic difficulty controller; LR/diet controller reading
  update absorption eta*|g|/ulp(w) (ABSORBED-INFORMATION bank).
  Measured anchors: none yet — the frame is unmeasured; nearest
  kin are the expert-iteration LOOP-LOG rounds (model-generated
  data, but not action-conditioned state), the RULE-POLICY-0
  label-timing census (engine-recoverable feedback), and the
  Collatz/rewrite-dynamics bank (relation/policy/evaluator split).
  Registered informal prior, on the record because it was called
  loudly (Artin, pre-work): FULL-CYBER "scores perfectly" — the
  loop wins outright. House counter-prior: the REPLAY arm is the
  dangerous control; curriculum-discovery may carry most of the
  gain (the family's direction-call record is 1-for-5).
  ADJUDICATED (2026-08-23, VERDICT
  MATH-CYBER-1-ACTIVE-EPISODIC-0): the PERFECT prior is REFUTED
  at the first measured dose (36/40 HOLDOUT) and the total
  closed-loop feedback effect measured NULL (exact tie v frozen
  theta_0, 36 v 36; 38 success-gated updates changed 6 choices
  total, all outcome-neutral). Scope: one dose point (lr 1e-4,
  episode-boundary, success-only), one seed chain; the
  ONLINE-v-REPLAY discriminating contrast of this bank remains
  UNTESTED — the adjudicated contrast was ACTIVE v NO-UPDATE.
  Honest breaks: DishBrain-class embodiment claims are contested
  in the literature — the analogy is a design generator, not
  evidence; "cycle feedback without the correct action" must be
  audited against the verified-AND-distinct law (identity rewrites
  and trivial cycles are the known reward-hack class, bit three
  times); online-updates-within-trajectory collides with the mps
  run-level nondeterminism fence (paired arms must share
  substrate noise in-run).
  Testable residue: the four-arm-plus-REPLAY factorial with
  matched budgets; the discriminating contrast is ONLINE v REPLAY
  on the same experience bytes.
  Attribution: Artin (loop vision + perfect-score prediction),
  GPT (factorial + REPLAY control shaping), house (fences,
  kin-mapping, counter-prior).

  AMENDED IN PLACE (2026-08-22, Artin + GPT review, pre-work):
  (i) MATHWORLD CONTRACT LAYER banked above the experiment: the
  environment is a learner-independent closed-loop contract —
  declarative actions, deterministic causal ordering, atomic
  transition admission/rollback, observable receipts,
  simulator/execution equivalence, logical time. MINIMAL VIABLE
  CONTRACT (adopted): transition function + admission rule + one
  transition receipt row, with canonical logical (episode_id,
  step_id) and state hashes from rung 0 so causal ordering is
  intrinsic to receipts. No CL-API-complete surface before
  /desk. The verified-AND-distinct law becomes a CONTRACT
  property (identity rewrites/trivial cycles rejected at
  admission), checked once, not per experiment. MATH-CYBER-0 is
  then ONE experiment implemented against the contract.
  (ii) ONLINE-v-REPLAY CORRECTION (GPT, house-verified): the
  original "ONLINE > REPLAY isolates causal feedback on the same
  bytes" is NOT generally true — a deterministic learner
  replaying the exact tuples, order, and immediate
  per-transition updates from the same initial weights
  reproduces ONLINE's update trajectory; that arm is a
  RECONSTRUCTION/QUALIFICATION TWIN, not a treatment (fenced by
  the mps run-level nondeterminism law: reproduction is
  trajectory-class on Mac, bit-class only on deterministic
  substrates). Superseding CONTROL LADDER, banked: ACTIVE-ONLINE
  (act -> transition -> immediate update); ORDERED-REPLAY (exact
  tuples/order/immediate updates — reconstruction control);
  DELAYED-REPLAY (same tuples/order, updates delayed/chunked —
  isolates within-trajectory adaptation timing); SHUFFLED-REPLAY
  (same experience multiset, chronology removed — order/
  curriculum sensitivity); and separately ADAPTIVE-DIET v
  FIXED-DIET generated streams scored by training fresh
  identical offline learners (value of loop-discovered data).
  Frame: each arm CUTS AN EDGE in X_t = (s_t, theta_t, C_t) —
  replay cuts theta_t -> future world; delayed replay also cuts
  feedback_t -> theta_{t+1}; fixed diet cuts learner/controller
  -> next experience distribution. The primary science question
  becomes: WHICH FEEDBACK EDGE BUYS LEARNING? Artin's
  perfect-score prediction stands as registered on ACTIVE-ONLINE;
  the house counter-prior transfers to the ladder (curriculum
  edge carries most of it).
  (iii) SECOND NEUROAI ANCHOR: Patel et al., "A Computational
  Perspective on NeuroAI and Synthetic Biological Intelligence"
  (arXiv:2509.23896) — SOFTWARE NeuroAI only; organizes
  neuro-symbolic reasoning, open-v-closed-loop learning,
  RL/active inference, feedback-driven adaptation, homeostatic
  regulation, stability/plasticity + experience replay. STANDING
  THREAD FENCE, adopted verbatim: steal computational invariants
  ONLY when they map to a measured learning problem — no spikes,
  dendrites, FEP, organoids; no generic biomimicry. Sharpened
  architecture: MathWorld (exact symbolic transition contract) /
  MathNative (learned policy + optional next-state head) /
  Controller (sensors over mastery/surprise/update-absorption
  eta*|g|/ulp(w); actuators over diet/LR/difficulty/replay/
  horizon). Cite via /cite when a THEORY row lands.
  STATUS UPDATE (2026-08-22, in place): rung 0 IS MEASURED —
  OBSERVATION MATH-CYBER-0-RUNG0 (python, Artin GO): the
  legal-action contract lives on successors(), replay
  qualification 101/101 causal rows, 35/40 L4-7 episodes solved
  by scripted greedy-hce under the fixed 12-decision budget;
  action identity = name#child_hash and wall_cap-as-safety-event
  are now CONTRACT SEMANTICS (both earned by replay-bar
  failures). Control ladder + proposal-mode remain banked,
  unimplemented. The frozen receipt interface is the axiom
  C++-replica handoff artifact; axiom stays one contract rung
  behind by design.
  ESTIMAND REFINEMENT (2026-08-22 evening, in place, per
  AMENDMENT MATH-CYBER-1-DESK-0-COVERAGE): the bank's single
  REPLAY arm splits into three registered estimands — (i)
  ORDERED-REPLAY with identical experience/order/immediate
  updates = RECONSTRUCTION QUALIFICATION, not a causal contrast;
  (ii) fixed-experience DELAYED/SHUFFLED replay = the
  optimization timing/order estimand (the bank's REPLAY arm
  proper); (iii) interactive ACTIVE v no/delayed online update =
  the total closed-loop feedback effect (the bank's ONLINE >
  REPLAY contrast), where experience may diverge endogenously.
  Learning signal frozen pre-prereg: terminal-success-gated
  trajectory updates (legality is not feedback; hce stays
  baseline/diagnostic, never the silent reward), refined same
  evening to EPISODE GRANULARITY (AMENDMENT
  MATH-CYBER-1-SUBSTRATE-DESK-0-SCOPE): terminal success is
  known only post-episode, so the active arm is ACTIVE-EPISODIC
  (retroactive positive updates on chosen transitions, landing
  before the next episode) and terminal-success learning is
  never called within-trajectory adaptation. The informal
  perfect-score registration pinned no seed band; seeds
  9100-9109 are now CALIBRATION (instrument design used them),
  and confirmatory PERFECT adjudication runs on a fresh
  post-freeze seed band.

- **BANKED (2026-08-22): RRUN — sha-pinned sandboxed remote
  execution for the 3080/WSL leg** (GPT proposal via Artin;
  concept banked, adopt INCREMENTALLY, nothing built).
  The shape: `rrun cuda <sha>` / `rrun cpu <sha>` / `rrun msvc
  <sha>` — Mac hands the WSL runner a GIT SHA (never a working
  tree), the runner executes inside a disposable Docker worker
  (CPU or RTX3080 via nvidia container toolkit) with no network
  and no host mounts; the Windows/MSVC test runner stays a
  SEPARATE non-admin seat (VM/snapshot class) because
  torch.compile's MSVC leg cannot live in a Linux container.
  WHY THE HOUSE LIKES IT (mapped to booked incidents): the
  sha-pinned interface kills the sync-drift class outright
  (verify-pulls-by-HEAD-hash, the dirty-step_chains silent-abort
  incident); a baked image pins the knob doctrine in one place
  (max_split_size_mb:128, TORCH_DISABLE_NATIVE_JIT=1, MSYS
  toolchain paths) instead of per-driver env incantations;
  disposable workers make the remote leg stateless, which is
  the WSL-as-thin-execution-target doctrine enforced by
  construction rather than by discipline.
  HONEST COSTS, on the record: image maintenance is a new
  standing chore (CUDA/toolkit/torch version pins drift);
  Docker images and layer cache eat C: through the vhdx that
  never shrinks (the WSL disk reality — budget before adopting);
  the Windows VM/snapshot runner is the heaviest third and
  PARKED until the other two earn their keep; container
  overhead on the 3080 must be measured once (expected ~nil for
  long jobs, but the house books, never assumes).
  ADOPTION ORDER banked: (1) the rrun INTERFACE alone —
  sha-checkout-then-run wrapper over the existing wsl.sh, no
  Docker, immediate sync-drift payoff; (2) the CPU Docker
  worker (cheap to validate the image discipline); (3) the CUDA
  worker; (4) the MSVC seat, only if MSVC-leg incidents recur.
  Fence: all of this is Artin's own two machines on his home
  network — the sandbox is for reproducibility and blast-radius
  hygiene, not because anything here is untrusted.
  Attribution: GPT (architecture), Artin (the ask + relay),
  house (incident mapping, adoption order, cost fences).

- **BANKED (2026-08-22): MATH-BASIS / prerequisite-basis
  hypothesis — the z1 x B43 routing locus selects/preserves a
  reusable computational basis of primitive mathematical
  transformations from which later reasoning composes**
  (Artin, the frame; GPT sharpening; MATH-BASIS-0).
  The mapping: do NOT equate school subject with layer depth,
  and do NOT say B43 "stores arithmetic." The sharper reading of
  VERDICT EX6-DEPTH-1 (+20 at z1 x block 43, matched-volume B46
  inert, intervention BEFORE downstream composition rescues) is
  that the locus gates a PRIMITIVE-TRANSFORMATION BASIS whose
  availability later reasoning depends on — which would explain
  early-position leverage without any topic story.
  Measured anchors: EX6-DEPTH-1 (+20/+1 at matched-mean
  displacement, pair outcome-identical to the late band);
  EX6-TEMPORAL-0 (launch-step specificity); BASICS-DIET-1
  format-distance transfer (competence radiates by format, a
  prior datum consistent with basis-not-topic structure).
  Registered predictions, on the record: (1) PRIMITIVE-BASIS
  account — the B43 effect associates with specific prerequisite
  rules and transports UPWARD to harder tasks requiring them;
  (2) ORCHESTRATION account — primitives themselves unaffected,
  effect emerges only with compositional depth; (3)
  INSTRUMENT-SPECIFIC account — no transfer outside the current
  gate. A future transport corpus factorializes PRIMITIVE
  DEPENDENCIES (prerequisite/dominator rules), never mere
  arithmetic/algebra/calculus labels.
  Honest breaks: "prerequisite structure" is defined relative to
  the house engine's rule vocabulary (successors()), not the
  model's internal features — a null under this vocabulary does
  not rule out a basis in a different decomposition; the
  conceptual bridge to MATH-CYBER (MathWorld exposes exact legal
  rewrite primitives; the neural policy may select an internal
  action basis over analogous transformations) is HYPOTHESIS,
  not finding. Naming fence: no "division/algebra expert" talk
  without the expert-identity census x rescue-structure cross
  AND a same-B43 matched-identity intervention.
  Testable residue: (a) rescue-anatomy census — FIRED 2026-08-22
  same day, OBSERVATION MATH-BASIS-0-CENSUS: FLAT at the named
  thresholds (no engine-vocabulary enrichment; depth-1-dominated
  corpus, resolution fence dominates — cheapest support absent,
  nothing adjudicated; the transport corpus is the real
  instrument); (b) the
  outcome-blind B43 expert-identity census; (c) if (a) enriches
  AND (b) concentrates, the matched-identity intervention; (d)
  the primitive-dependency transport corpus.
  Attribution: Artin (hypothesis + the insane-arc energy), GPT
  (prediction triple + fences), house (census design).

- **BANKED (2026-08-22): B43-JSPACE-BRIDGE — z1 x B43 routing as
  a possible WRITER/GATE into a J-space-like high-downstream-gain
  subspace** (Artin + GPT relay of Anthropic's 2026 J-space work;
  NOT RUN; citation to be /cite-verified before any THEORY row —
  the house has not independently checked the paper).
  The mapping (terminology fenced): B43 is NOT claimed to be
  "the J-space" or one mathematical-primitive expert. The
  relayed frame: J-space = a privileged global-workspace-like
  representation set found by the Jacobian lens
  J_l = E[d h_final / d h_l], readout ~ unembed(J_l @ h_l),
  carrying silent intermediate math steps and causally mediating
  higher-order reasoning. The plausible connection: the z1 x B43
  routing locus may WRITE into such a high-downstream-gain
  subspace — which would explain +20 v +1 GEOMETRICALLY (matched
  local displacement, unmatched downstream gain) without any
  semantic claim.
  Measured anchors: EX6-DEPTH-1 (+20/+1 at matched-mean
  displacement, locus-specific); none yet on the J side (no
  Jacobian quantity has been computed on this Qwen).
  CORRECTED LADDER (2026-08-22 in-place, superseding the first
  draft's global-projector design — paper section J-Space defines
  J-space as points expressible as SPARSE NONNEGATIVE
  combinations of J-lens vectors, typically k <= 25: a union of
  k-dimensional cones / sparse subframe, NOT one linear subspace
  with a global orthogonal P_J; the P_J-v-(I-P_J) mediation rung
  is retracted before ever firing): (1) GAIN — capture post-MoE
  native-v-B43/B46 deltas, compare G_l = |J_l delta_l| /
  |delta_l|, called DOWNSTREAM LINEAR GAIN, never J-space
  membership; (2) LINEARIZATION QUAL — compare J_l delta_l
  against the actually observed downstream/final residual delta;
  top-k MoE makes the finite intervention piecewise/nonlinear,
  so the Jacobian approximation itself must qualify before any
  gain number is read; (3) J-CONTENT — sparse nonnegative
  decomposition of each delta over that layer's J-lens vectors,
  comparing reconstruction/coefficient structure B43 v B46 and
  rescue v non-rescue; (4) MEDIATION — intervene on the
  implicated sparse J-lens coordinates via paper-faithful
  coordinate manipulation, never a fictitious global projection;
  (5) cross with MATH-BASIS prerequisite structure (its census
  read FLAT at the current gate's grain) — only after geometry +
  causal coordinates + task structure does "mathematical basis
  representation" become canonical vocabulary.
  Honest breaks (counter-fence, on the record): the relayed
  J-space sits in a model-specific INTERMEDIATE layer band while
  B43 is LATE in this 48-block MoE stack — structural analogy
  motivated, direct identity UNMEASURED; the Jacobian lens on a
  4-bit MoE with routing discontinuities is itself an
  instrument-design question (J through top-k selection is
  piecewise); everything here is relayed description of external
  work until /cite verifies it.
  CORRECTED IN PLACE (2026-08-22, AMENDMENT
  EX6-B43-IDENTITY-0-SCOPE, house-verified v the idcensus
  receipt): delta_43 is the NET POST-MoE REROUTING DELTA of
  excluding expert 71 — deleted term + variable-entrant term +
  kept-expert renormalization term — NEVER "expert 71's output
  displacement direction"; the ladder gains a step (0)
  outcome-blind DOSE CENSUS (native normalized removed mass,
  router margins, actual masked entrant, common-expert
  reweighting, post-MoE delta norm; B43/B46 matched slot COUNT
  only, ranks 0 v 2-7, removed mass unmeasured — "same-sized
  write" deny-listed) with the argsort-v-argpartition dual-
  selector precondition; and a TEMPORAL FENCE: same-token
  J-gain null does not refute later KV/autoregressive
  mediation.
  Testable residue: step (0) dose census, then the five-step
  ladder above; step 1 is a cheap capture probe, step 2 is the
  first real instrument build.
  Attribution: Artin (the bridge ask), GPT (gain discriminator +
  projection-patching design + counter-fence + delta
  decomposition + temporal fence), house (fences,
  MoE-discontinuity caveat, cite gate, receipt recount).

- **BANKED (2026-08-23): MAGIC-CYBER-HARDNESS — the difficulty
  estimator meets the MATHWORLD frontier** (Artin's call on the
  theta_0 verdict night — "isn't this exactly what the magic
  predictor said for the difficulty of the questions?" — GPT
  design shaping, house fences; BANK ONLY, no run).
  The observation that seeds it: theta_0's misses are
  INSTANCE-picked, not level-picked (L7 10/10 while L4/L6 carry
  the five failures — the magic-estimator lesson that variance
  lives at identifiable hard instances, booked across the judge
  slots). Two-stage design, frozen at bank time:
  - RETRO (descriptive, zero-risk): join the FROZEN historical
    MAGIC checkpoint's hardness scores against the CALIBRATION
    solve/fail pattern, WITH a support audit first — the
    estimator was trained on engine-search features whose
    support may not cover MATHWORLD L6/L7 states; OOD/support
    status must be reported per level, and the prior
    MAGIC-CURRICULUM transport-failure scar is carried
    explicitly (the estimator has FAILED a transport test
    before; that scar is the prior, not a footnote).
  - PROSPECT (the real test): predictions for the future HOLDOUT
    band computed and SEALED before any HOLDOUT outcome exists
    and never visible to treatment; PRIMARY target = the FROZEN
    arm's solve/fail rank (ACTIVE secondary — its policy moves
    during ADAPT, so its frontier is the harder prediction).
    Do NOT retrain MAGIC for this — the historical-prediction
    test is only meaningful with the frozen historical artifact.
    PIN PROTOCOL (hardened 2026-08-23, pre-PROSPECT): before any
    PROSPECT scoring, ONE historically canonical MAGIC artifact
    is pinned by exact path + sha256, estimator/version lineage,
    feature-extractor code sha, normalization, hardness scalar
    definition, and a frozen support/OOD rule — selected on
    HISTORICAL PEDIGREE only, never by CALIBRATION/HOLDOUT
    correlation; no version switching after the pin. PROSPECT
    scores are computed and sha-sealed AFTER the target roots
    materialize but BEFORE any policy outcome on them exists,
    and stay inaccessible to treatment. NOTE: the first
    ACTIVE-EPISODIC HOLDOUT band (9400-9409) launches WITHOUT a
    sealed PROSPECT (no pin existed pre-launch), so PROSPECT
    targets a FUTURE fresh band; RETRO stays descriptive.
  Conceptual bridge, METHODOLOGY-ONLY (charter fence, verbatim):
  cheap structural hardness surrogate -> reduced/local candidate
  space -> exact oracle adjudication -> adaptive/variational
  update. The shape is general search methodology; no
  chemistry-domain development, ever, under the standing
  charter.

- **BANKED (2026-08-23): SUCCESS-ONLY-FIXED-POINT — the
  self-confirming attractor of positive-only gating** (GPT
  naming off the ACTIVE-EPISODIC-0 verdict; house-measured
  anchors; a fixed-point-LIKE attractor, explicitly NOT an exact
  parameter/policy fixed point). Measured anchors (VERDICT
  MATH-CYBER-1-ACTIVE-EPISODIC-0): ACTIVE entered ADAPT at
  38/40-class competence; all 38 successes produced
  self-imitation updates and both failures produced zero
  gradient; weights and scores moved, but observed same-state
  top-1 movement was 2/75 ADAPT + 4 HOLDOUT, all
  solve/fail-neutral; final paired solve indicators identical
  36/40 (clean subset 29/33 tie). Mechanism: deterministic
  argmax + positive-only success gating reinforces exactly the
  already-selected successful actions, failed states receive no
  direct corrective signal, and cross-state parameter
  generalization is the only escape route. Registered
  prediction: marginal frontier movement FALLS as initial
  competence approaches ceiling. Fence: one dose, one substrate,
  one seed chain — a measured attractor observation, not a
  universal law.

- **BANKED (2026-08-23): COUNTERFACTUAL-CREDIT — the next
  discriminating feedback operator is information TOPOLOGY, not
  a larger LR** (GPT proposal off the null; bank only). At a
  state, choose alternatives OUTCOME-BLIND (e.g. frozen-policy
  top-1 v top-2), force each once, continue BOTH with the same
  frozen continuation policy over the remaining registered
  horizon, and let the exact MathWorld solve/fail adjudicate the
  fork; train preference ONLY from outcome-DIFFERING forks. hce
  never the reward. Any future comparison against success-only
  must match update/gradient dose. The question the bank holds:
  can difference-bearing feedback move the frontier that
  confirmation-only feedback measurably did not?

- **BANKED (2026-08-23): FRONTIER-REPAIRABILITY — desk the
  failed-case mechanism before any counterfactual treatment**
  (GPT, bank only; design exact receipts before running). For
  each failed trajectory: does ANY visited state carry a
  one-step legal deviation that SOLVES when followed by frozen
  theta_0 for the remaining registered horizon? Separates
  locally policy-repairable failures from deeper
  horizon/context/world limitations, and prices
  COUNTERFACTUAL-CREDIT's best case before it runs.
  Companion note: ONLINE-v-REPLAY stays alive but DEPRIORITIZED
  — replay timing is weakly discriminating until the feedback
  operator itself produces a behavioral effect.

- **BANKED (2026-08-23): TRANSITION-BASIS / VISUAL-NGRAM —
  temporal information can be EDGE-LOCAL rather than
  state-local, and video representation should carry transition
  tokens, not only frame tokens** (Artin intuition, GPT
  formalization, house bank). The claim: some semantic facts are
  invariant to individual-frame content and distinguishable ONLY
  from temporal order — direction, reversal, acceleration,
  collision causality, handoff, appearance/disappearance — so a
  bag/multiset of frames destroys them BY CONSTRUCTION (the
  Tenet test: forward v reversed playback share the exact frame
  set). Representations: ordered local tuples
  G_k(t)=(F_t..F_{t+k-1}) (frame n-grams) or derived transition
  operators Delta_t=Phi(F_t,F_{t+1}). Methodological connection:
  MathWorld's state-v-edge distinction (action = name#child_hash
  EDGE identity), TIME-AS-STRUCTURE (order/horizon first-class),
  and the same question underneath all three — is the
  computational object the state or the transition? Known prior
  art the bank sits on (not novel machinery, novel framing for
  this lab): optical flow / two-stream nets, motion history
  images, 3D-conv tubelets, video transformers. Measured
  anchors: NONE yet — this bank has no booked result; the
  prior-art claims are literature, not house measurements.
  Honest breaks: (1) modern video transformers already consume
  ordered frame tokens, so "ordered beats bag" alone is not
  news — the live question is the CHEAPNESS claim (transition
  tokens as a smaller sufficient statistic per unit
  accuracy/wall than dense full-frame processing); (2) per-frame
  captioning pipelines may erase exactly the transition
  information under study, so no "watching" claim attaches to
  caption-then-summarize; (3) video is OUTSIDE the lab's two
  standing engine domains (math/physics) — any rung here is a
  methods/representation experiment on synthetic
  oracle-verifiable data, never a movie-understanding
  capability program. Testable residue (first rung, synthetic
  and oracle-verifiable): generate short videos whose frame
  MULTISET is identical across labels while temporal order
  differs (LEFT-to-RIGHT v RIGHT-to-LEFT, expand v contract,
  A-before-B v B-before-A, collision v its time reversal);
  compare BAG (unordered pooled frames — information-
  theoretically blind on matched-set pairs by construction, the
  registered control), ORDERED-FRAMES (chronological frame
  tokens), DELTA (pairwise transition tokens), K3 (local
  three-frame tubelet); measure accuracy per input token and
  per wall; the bank fires if DELTA/K3 recover temporal
  direction at materially lower token/wall cost than
  ORDERED-FRAMES. Future extension (banked, not designed):
  hierarchical movie encoding = keyframe/state tokens +
  transition tokens + shot/event summaries + long-range
  retrieval. Attribution: Artin (the ask + the overlap/n-gram
  intuition), GPT (formalization + arm taxonomy), house
  (prior-art grounding, charter fence, control-arm framing).
  IN-PLACE NOTE (2026-08-23, GPT + Artin follow-up, house
  cross-link): treat the proposal explicitly as a TEMPORAL
  CHANGE OF COORDINATES. For literal finite differences,
  (F_0, Delta_0..Delta_{T-1}) is information-EQUIVALENT to
  ordered frames (a bijection), while Delta-only carries a
  constant-sequence/nullspace ambiguity — so the high-value arm
  is ANCHOR+DELTA, never transition-only. The efficiency
  hypothesis therefore requires COMPRESSIBILITY/SPARSITY of the
  residual, not mere differencing: a dense residual has the same
  dimensionality as a dense frame. Bar currency:
  tokens/bytes/FLOPs/wall at MATCHED semantic accuracy.
  First adversarial control: global camera motion (a global
  transform makes every pixel-delta dense while the semantic
  delta is one token, "pan left"); eventual decomposition =
  global transform + local residual events. Cross-linked as the
  general repo question STATE BASIS v TRANSITION BASIS.
  HOUSE CASH-OUT (MathWorld, per-question): MathNative already
  trains on edges (diet rows are cur/nxt pairs; the anchor is
  the parent in the prompt) — but it SCORES dense states: every
  candidate is a full child expression, and that is precisely
  why the model_ctx_overflow failure class exists (L6-s9403,
  L6-s9100: one long candidate encoding kills the whole
  decision). The world's own delta representation already
  exists — the action label "rule@target" — and is tiny and
  closed-vocabulary. Testable residue added: an ACTION-SCORING
  arm (score/emit rule@target deltas against the parent anchor,
  world expands the state) v the standing STATE-SCORING arm,
  same episodes, same oracle; deletes the overflow class by
  construction and prices whether the transition basis is the
  cheaper sufficient statistic IN OUR OWN ENGINE. Sparsity
  premise measurably holds here: median chosen-edge target 76
  tokens v multi-hundred-token child states
  (LONGCTX-POPULATIONS). Attribution: Artin ("tell it once what
  reality is, then how reality changes" + the per-question math
  ask), GPT (change-of-coordinates note, nullspace caveat,
  matched-accuracy currency, camera-motion control), house (the
  overflow cash-out + action-scoring arm).
  CORRECTION (2026-08-23, GPT review, house-verified line-by-line
  against source; the note above stands as record, four of its
  claims are corrected here):
  (1) "MathNative already trains on edges" RETRACTED as stated —
  the corpus is EDGE-INDEXED but the trained representation is
  DENSE-STATE: the LM sequence is "Current: {FULL parent}\n
  Hints: none\nStep: {FULL child}\n" and inference scores full
  child serializations. Training is not action/delta-based.
  (2) "rule@target is the delta" OVERCLAIMED — rule@target is
  NOT a complete action: multi-branch rules (i_parts (u,dv)
  splits, i_usub candidates) emit several children under one
  label (llmopt/search/derivation.py replay docstring: "Labels
  are NOT unique... Replay therefore BACKTRACKS over all
  same-label children"), which is exactly why rung-0 action
  identity is name#child_hash. E(parent, rule@target) is not a
  function; name#child_hash is unique but an opaque POINTER, not
  a reconstructive program. A true ACTION-BASIS needs a compact
  deterministic action PROGRAM a with exact qualification
  E(s,a)==s'.
  (3) "sparsity premise measurably holds (median 76-token
  deltas)" RETRACTED — the LONGCTX P2 medians measure the FULL
  scoring sequence (parent prefix + full child + newline), not
  action-label or residual length. Delta sparsity in this engine
  is UNMEASURED.
  (4) "deletes the overflow class by construction" NARROWED —
  compact unique actions remove CHILD-SERIALIZATION-induced
  overflow only; the parent prefix still occupies ctx and
  parent-context overflow survives. Any desk must first
  decompose existing overflow specimens into parent-prefix
  length v per-child scoring length v label length v candidate
  program length.
  CORRECTED RESIDUE — MATHWORLD ACTION-BASIS (the STATE-v-
  TRANSITION-BASIS house instance), desk first, no model, no
  run without GO: on the existing frozen semantic corpus
  (states.jsonl/actions.jsonl), design the minimal COMPLETE
  program schema (rule + branch parameters sufficient to
  deterministically reconstruct exactly one child), require
  unique/reconstructive coverage of EVERY candidate, report
  token-length distributions and compression ratios v STATE
  (overflow specimens especially), and quantify whether
  program scoring removes the child-length nuisance the MINLEN
  diagnostic exposed. Scientific motivation (GPT, adopted):
  the current theta_0 score is the likelihood of an entire
  serialized successor, CONFOUNDING transition quality with
  successor textual length/predictability; ACTION-BASIS asks
  for the likelihood of the transformation itself — possibly
  worth more than the context savings. Arms if it ever runs:
  STATE (score full successor) v PROGRAM (score compact
  deterministic action program, exact E(s,a)==s'
  qualification).
  MEASURED (2026-08-23, VERDICT MATH-CYBER-1-FRONTIER-DESK-0,
  updates the FRONTIER-REPAIRABILITY bank above): the desk ran
  on the six FROZEN theta_0 failures — 2 ONE-DEV-REPAIRABLE
  (L7-s9303: 17/84 rescues, min rank 2, rank-2 rescues at 4/12
  sites; L6-s9300: 1/228 at rank 6), 1 ONE-DEV-NOT-REPAIRABLE
  (L4-s9401, itself a 2-state cycle — loop structure alone does
  not imply a one-deviation escape), 3 UNDECIDED (censoring/
  world mismatch). COUNTERFACTUAL-CREDIT's single-fork
  top1-v-top2 best case is therefore NOT empty on this band: an
  outcome-differing rank-2 fork is counted at L7-s9303 steps
  0-3. Repairability priced; the treatment stays unproposed
  without GO.

- **BANKED (2026-08-23): FAILURE-TRIGGERED RETROSPECTIVE CREDIT
  — the preferred next feedback operator: fork only on failure,
  only at the pre-existing top-2, only where outcomes differ**
  (GPT proposal off the FRONTIER-DESK-0 counts, house bank; no
  prereg, no treatment yet). Design: run an episode with the
  frozen/pre-update policy; on FAILURE only, revisit the
  recorded states and test the OUTCOME-BLIND pre-existing
  top-2 alternative under the same remaining horizon and the
  pre-update continuation policy; only an outcome-DIFFERING
  fork (chosen fails, alternative solves) creates a pairwise
  preference label; censored or world-noncomparable forks
  create NO label. hce never the reward. Measured anchors
  (FRONTIER-DESK-0, RESULTS L43136): the design's target class
  is non-empty on the spent band — rank-2 outcome-differing
  forks exist at L7-s9303 steps 0-3 (min margin 6.42), and the
  rescuing information there is GENUINELY counterfactual (the
  rank-2 winner needs continuation depth 8, invisible to the
  world at the decision). Honest breaks: (1) L6-s9300's rescue
  is NOT this class — an already-solved legal child missed by
  the argmax controller is a TERMINAL-DOMINANCE controller
  defect (see PRE-REG MATH-CYBER-1-TERMINAL-DOMINANCE-0), and
  must never be used to justify deep-k exploration; (2) any
  future comparison against success-only updating must be
  dose-matched (the COUNTERFACTUAL-CREDIT bank's standing
  fence); (3) label yield on this band is small (4 sites, one
  root) — a fresh-band yield census prices the operator before
  any training. Testable residue: mechanism-complete smoke +
  prereg of the fork-labeler on outcome-spent data; the
  discriminating question stays the COUNTERFACTUAL-CREDIT
  bank's — can difference-bearing feedback move the frontier
  that confirmation-only feedback measurably did not?
  Attribution: GPT (operator design), Artin (relay + GO
  structure), house (anchors, breaks, terminal-defect
  separation).

- **BANKED (2026-08-23): TERMINAL-FIRST — the corrected
  controller baseline: exact terminal override before the
  learned scorer** (GPT proposal off the FRONTIER L6-s9300
  anatomy; house measured + banked; the TERMINAL-DOMINANCE-0
  registered conditional, condition met). Law: if any legal
  child is terminal-solved (the world's own is_solved
  predicate, no model, no hce), a solve-maximizing controller
  selects a terminal child deterministically; the learned
  scorer is consulted ONLY when no immediate solved child
  exists. Measured anchors (VERDICT
  MATH-CYBER-1-TERMINAL-DOMINANCE-0, RESULTS L43378): 21/130
  terminal-child states missed by the shipped argmax (all at
  steps 0-1, 15 at rank 2), 2 episode-costing; derived exact:
  theta_0 becomes 36/40 CALIBRATION / 39/40 FROZEN ADAPT /
  36/40 HOLDOUT under the override. Honest breaks: baseline
  for FUTURE rungs only — never a retroactive rescore of
  booked verdicts (all booked numbers are measurements of the
  argmax controller); on-path census, off-path terminal
  children unmeasured; the override changes the controller's
  exploration distribution, so any future learning rung
  re-prices its own frontier. Testable residue: every future
  MATH-CYBER controller arm carries TERMINAL-FIRST as its
  default decision rule (or registers why not), and the
  ACTION-BASIS desk should report terminal-recognition
  separately from transition scoring. Attribution: GPT
  (invariant + baseline design), house (census instrument,
  counts, derivation), Artin (relay/GO).
  ADOPTED (2026-08-23, Artin-relayed outside GO on edff502f):
  TERMINAL-FIRST is the DEFAULT controller for all future
  MATH-CYBER arms, prospective only — every existing
  argmax-controller verdict stands untouched; any future arm
  omitting the override must explicitly register why.
  MEASURED + PARKED (2026-08-23, VERDICT
  MATH-CYBER-1-LABEL-YIELD-0, updates the FAILURE-TRIGGERED
  RETROSPECTIVE CREDIT bank above): the fresh-band yield census
  came back NO-GO (house prior fired) — TERMINAL-FIRST +
  theta_0 solves 76/80 on the sacrificial band, leaving 1
  label-bearing failed episode and 4 distinct corrective facts
  v the >=3 / >=5 GO bars. The operator is QUALIFIED (6/6) but
  STARVED: label scarcity is baseline competence, not labeler
  defect. Parked with registered revivals: a ~3x wider or
  harder sacrificial band, or top-k>2 forking under a fresh
  qualification on FUTURE bands only. Bands 9500-9519
  permanently outcome-spent; treatment bands >= 9600.
  MEASURED (2026-08-24, OBSERVATION
  MATH-CYBER-1-ACTION-BASIS-DESK-0, updates the corrected
  ACTION-BASIS residue above): the minimal (label,
  sibling-index) program schema is COMPLETE (725/725, all 242
  same-label collisions index-resolved) at 5.75x median
  compression — but NOT promoted: targets-as-serialized-
  sub-expressions keep the tail dense (program CV 1.69 > child
  1.20; MINLEN confound survives) and the parent prefix (p90
  555 tokens) is outside any action basis. Live refinement:
  schema v2 with INDEXED targets (bounded length by
  construction) + the parent-side state-naming lever, each
  under its own blind-threshold desk.
- **BANKED (2026-08-24): MATH-CYBER COMPUTE-OPTION-CHAIN / DISTRIBUTIONAL-ACTION-VALUE — score each exact legal ActionProgram by its solve-probability CDF over a compute-budget strike ladder, and allocate search by MARGINAL solve mass per unit world cost, not by a static action rank** (Artin's options-chain analogy; house formalization).
  The mapping / the math: for each exact legal ActionProgram a from state s, define T_a = continuation compute-to-verified-solution under an explicitly pinned continuation controller + world snapshot; the object of interest is the budget CDF F_a(B) = P(T_a <= B) on a FIXED strike ladder (e.g. expansions B in {12, 24, 48, 96} — any actual ladder needs its own desk/prereg). The useful derivative is marginal compute value Delta_B(a) ~ [F_a(B+D) - F_a(B)]/D, optionally the conditional hazard of solving in the next tranche given survival so far. Search allocation then spends the next expansion on the branch with the highest expected marginal solve gain per unit world cost. Analogy dictionary (instrument-talk only): digital call = solve by budget B; call spread = solve mass unlocked between B1 and B2; put = unresolved/dead past B; delta = marginal solve probability per extra compute.
  Measured anchors: the controller ladder booked that residual failures are budget-censored, not rank-2-rescuable (REGRET-LDS-DESK-0 + WALLLIFT: 5/6 roots exhaust 96 expansions unsolved, world materialization 73-89% of wall — so marginal-compute allocation is exactly the lever the measurements point at); v4/v5 ActionPrograms are fully qualified (ACTION-SITE 725/725, ACTION-SEMANTICS 725/725) with med 5 / max 8 tokens under ActionGCTok (ACTION-OPCODE-QUAL-0), removing the full-child length nuisance that poisoned every earlier per-action scoring design (MINLEN class); PROGRAM-DIET-COVERAGE-0 gives 73,324 relabeled rows for any program-conditioned head.
  Honest breaks: the analogy carries NO financial option-pricing machinery (no no-arbitrage, no measure change — F_a is just a CDF) and NO quantum speedup claim; T_a is defined only relative to a pinned continuation controller, so F_a is controller-relative, not a property of the state; labels cost exact bounded continuation runs (expensive; the strike ladder IS the budget bill); hazard estimates at large B are censoring-dominated (the WALLLIFT lesson).
  Testable residue: (1) a PROGRAM-DISTVALUE head — state + ActionProgram -> monotone solve-CDF over budget strikes, labels minted by exact bounded continuation, monotonicity enforced or measured; (2) a desk pricing label minting cost per strike before anything trains; (3) allocation A/B: marginal-value tranche allocation v static-rank beam at matched world budget. No run or prereg now — bank only, revival after STATE-v-PROGRAM.
  Attribution: Artin (options-chain frame, "the chain over budgets is the object"; also the July reverse-LLMUE/pincer "complete legal superposition + learned amplitudes" framing this converges with — see the reverse-LLMUE distribution spec and scratch/pincer_dist_probe.py); house (CDF/hazard formalization, anchor audit). Cross-link: the July pincer bank queued in docs (queued-workspace-experiments: temporal pincer / distributional engine) — v5 ActionPrograms are the compact action basis that makes that architecture sane, which is the convergence Artin named on 2026-08-24.

- **BANKED (2026-08-24): CAUSAL-STATE-GEOMETRY — derivation search as a learned high-dimensional causal geometry: states embed as points, ActionPrograms as local transition vectors, one distinguished compute/depth coordinate, forward/reverse policies as opposite causal flows, and the distributional action value F_a(B) as a scalar field over the action cone** (Artin's Interstellar/Tenet thought; house formalization).
  The mapping / the math: state s -> z(s) in a learned latent; each legal ActionProgram a acts as a local transition/vector v_a at z(s); the "time" axis is a distinguished COMPUTE/DEPTH coordinate, never literal physical time; the forward policy and a reverse/pincer policy are opposite flows along the causal ordering; the COMPUTE-OPTION-CHAIN object F_a(B) = P(solve by budget B) becomes a scalar field on the reachable cone. Candidate geometry: hyperbolic/Lorentz-model embeddings, motivated ONLY by the combinatorial fact that derivation trees branch exponentially (hyperbolic volume growth matches tree growth); a plain Euclidean latent is the MANDATORY control arm in any eventual test.
  Measured anchors: none yet for the geometry itself — the frame stands on the booked representation chain that makes it POSABLE (v4 ActionPrograms qualified 725/725 with <= 8-token serializations; PROGRAM-DIET-COVERAGE 73,324 matched rows; the forward/reverse meetability half maps onto the banked temporal-pincer riff and the reverse-LLMUE distribution spec/scratch/pincer_dist_probe.py).
  Honest breaks: explicitly NO literal spacetime, black-hole, or quantum claim — the physics vocabulary is a geometry-selection heuristic, nothing propagates from GR/QM; "causal" here means the DAG order of derivations, not physical causality; hyperbolic embeddings routinely lose to well-tuned Euclidean baselines, hence the mandatory control; nothing here is trainable until STATE-v-PROGRAM lands.
  Testable residue (future only, after STATE-v-PROGRAM; no experiment authorized): does a geometry-constrained latent predict (1) derivational distance between states, (2) forward/reverse meetability (pincer closure), or (3) the solve-CDF F_a(B), more sample-efficiently than an unconstrained Euclidean latent of matched capacity? Kill condition at that desk: no metric beats Euclidean at matched parameters.
  Attribution: Artin (the Interstellar/Tenet causal-geometry frame and its link to the options-chain field); house (formalization, control-arm law, break audit).

- **BANKED (2026-08-25): AXIOM-PRINTER-EXTENSION — extend Axiom's sympy-order sstr printer beyond its zoo envelope to lift interchange admission** (Axiom's proposal, relayed in their Tranche-A reply; banked by house, NOT LIVE).
  The mapping / the math: Axiom's Tranche-A run (their b90cad71) showed the ActionProgram ABI sound natively while admission collapsed on printer spelling — sstr of sympify(axiom_string) reproduces the house string byte-exactly on the dominant divergence class (ordering-only), so a generic printer extension (never fixture-conditioned, gated on their sstr_fixture.tsv staying byte-green) converts most rejections into admissions.
  Measured anchors: counter-booked in OBSERVATION MATH-CYBER-1-AX-ABI-COUNTERBOOK-0 — 6/64 core and ~393/2,888 stress unique parents admitted today; their Tier-B bound projects ~47/64 and ~2,620/2,888 after extension, with the genuine constructor/canonicalization class (~27% of remaining divergence) as the residual floor unless a srepr-level transport lane is added.
  Honest breaks: the payoff bound is Axiom's Tier-B diagnosis, not a house re-derivation; the structural class is real transport loss no printer fixes; lifting admission does not test portability beyond SymPy-stored-order engines; nothing here gates the paired STATE-v-PROGRAM experiment.
  Testable residue: after extension, does unique-parent admission reach the Tier-B bound, and do term_index/site_ordinal mismatch rates on the enlarged admitted set stay at the stored-arg-order-divergence level measured today (99/2,894 and 7/2,894)?
  Attribution: Axiom seat (proposal + Tier-B bound); house (bank, counter-book, fences). Needs its own GO on the Axiom side; no llmopt work item.

- **BANKED (2026-08-25): I-USUB-SEMANTICS-DESK — a semantic u-substitution parameter for i_usub, gated BEHIND the STATE-v-PROGRAM experiment** (house, from the SVP-EVALBAND-0 finding; Artin/GPT gate).
  The mapping / the math: on 4 fresh 9600-9619 parents i_usub emits TWO accepted children at a single site, so the canonical (rule, site) carries no parameter for it — the same shape i_parts had before u_choice; the natural candidate parameter is the substitution choice (which u the rule tried), read from the rule's own generative loop like trace_unprod.
  Measured anchors: VERDICT MATH-CYBER-1-SVP-EVALBAND-0 (4/82 solved-episode decisions excluded with count; i_usub single-site branching 0 in the 725 corpus, leg previously vacuous).
  Honest breaks: NOT runnable before the paired experiment — the residue was discovered through the primary evaluation band, so repairing the schema against it would turn that band into development data; any repaired schema needs a NEW untouched primary band; nothing here touches the frozen training artifact or canonical schema.
  Testable residue: does a (rule, site, sub_choice) parameter decode all i_usub multi-child sites with zero collisions on a fresh band, corpus-regression clean, like ACTION-FINAL did for term_index?
  Attribution: house (finding + bank); gated on Artin's GO after STATE-v-PROGRAM adjudication.

- **BANKED (2026-08-26): the smallest causal control separating semantics from target length/entropy in the STATE-v-PROGRAM win** (GPT via Artin, the ask + candidate designs; house, desk pricing).
  The mapping / the math: three confound channels — (A) semantic/compositional ActionProgram structure, (B) short/low-entropy target supervision, (C) scoring-length normalization. The LENGTH-CONTROL desk priced (C): summed-lp (alpha=0) halves the median gap in both bands yet PROGRAM stays ahead in all 30 band x birth x alpha cells, so (C) carries ~half the magnitude and none of the direction. The live confound is (A) v (B), which only a new paired birth can split.
  Candidate treatments (design only, none selected): PROGRAM-PAD — canonical program + deterministic semantically-null filler toward STATE length (probes raw token burden; disclosed limit: easy filler does not match STATE entropy); OPAQUE-SHORT-ACTION — stable opaque action code at PROGRAM length (probes compact action identity v compositional factorization; disclosed limits: codebook memorization, no compositional generalization probe).
  Measured anchors: VERDICT SVP-LENGTH-CONTROL-DESK-0 (SCORING-LENGTH-PLAUSIBLE, gap medians 21->9 old / 20->8 new at alpha=0, program_ahead_everywhere true); VERDICT SVP-GENERALIZATION-SCORE-0 (true target T p50 29 v 6); the 3.63x training continuation-token asymmetry (SVP-ADJUDICATION-0).
  Honest breaks: any new treatment needs its OWN untouched evaluation band (the 9600 and 9700 bands are spent for design purposes); PROGRAM-PAD changes the training-loss geometry (per-row normalization over longer masks), so it is not a pure length knob; neither treatment separates (B)'s two sub-channels (length v entropy).
  Testable residue: does PROGRAM-PAD (matched length, same semantics) retain the advantage (=> semantics/entropy, not token burden)? does OPAQUE-SHORT-ACTION (matched length, no composition) lose it (=> factorization load-bearing, not mere brevity)?
  Attribution: GPT (control designs), Artin (relay + GO), house (desk that localized the live confound, bank). Gated on its own GO; nothing trained.
  AMENDMENT (2026-08-26, same day — SVP-ACTION-COVERAGE-DESK-0): OPAQUE-SHORT-ACTION as written above is NOT IDENTIFIABLE — training contains only 33 distinct whole-action tuples, ~32% of eval-band candidates are whole-action OOV (incl. entire rule families cancel/euler/factor/i_apart/trigsimp never seen as targets), so training-observed categorical codes would confound compact identity with OOV exposure (REDESIGN-REQUIRED, booked). Redesign direction: codes assigned by a deterministic outcome-blind law over the FULL enumerable action space, or atom-level codes; the three-way ladder (canonical / renamed-atoms-compositional / opaque-whole-action) is now the preferred design and the renamed-atoms arm inherits coverage exactly (GPT proposed the three-way; house priced identifiability). Coverage floor for any future arm: 0 decisions on either band are decided by coverage alone (every covered label faces >=1 covered rival).
  AMENDMENT 2 (2026-08-26, same day — SVP-TOKEN-CHANNEL-DESK-0): the renamed-atoms arm is RETIRED as redundant — ActionGCTok encodes each rule as a single dedicated opcode ID and the model trains from scratch, so renaming rule atoms is a vocabulary permutation distribution-identical to a seed change, not a semantic-surface ablation (GPT review caught it; house verified against the tokenizer and censused the channels: rule opcodes 15/15 action-only, sharing confined to digit/space/newline/"u" byte tokens). Superseding design: common 34-token dedicated code alphabet, fixed width 8, FACTOR-OPAQUE (per-coordinate deterministic codes, preserves cross-state reuse, removes digit sharing) v HASH-OPAQUE (whole-tuple sha-derived code, destroys factorization, same alphabet/width/exposure law, defined for any legal tuple, law frozen before any future band). Clean contrast FACTOR v HASH = reusable factorization v compact arbitrary identity; canonical-366 arm rides as reference at equalized vocab. Booked at the TOKEN-CHANNEL-DESK observation.

- **BANKED (2026-08-29): EXPERIMENTAL THROUGHPUT as a research capability metric — hypotheses adjudicated per compute-hour / dollar / wall-clock, not benchmark intelligence alone** (Artin, prompted by a Jane Street interview on fast domain-specific models; ties to the FA Law "speed is intelligence" line he wants revived).
  The mapping / the math: FA Law v2 already states intelligence = RATE at which verified variance becomes compressed structure; this riff operationalizes the rate for the LAB ITSELF — the unit of scientific progress is the adjudicated hypothesis (a pre-registered bar mechanically resolved FIRE/NO-FIRE), and the capability metric is adjudications per unit cost. A small fast domain-specific instrument that resolves 50 bars/day can out-research a frontier model that resolves 2, exactly the Jane Street shape (narrow, fast, in the loop).
  Measured anchors (house practice, not yet a booked metric): the SVP chain resolved prereg->birth->calibration->heldout in single sessions with scoring walls of ~2 s per 96-state gate and ~1 hr births; the /desk and /probe ladders exist precisely to raise adjudications-per-dollar by moving questions down the cost ladder. No booked entry yet MEASURES throughput as a number.
  Honest breaks: throughput is gameable by cheap trivial bars (Goodhart — 50 vacuous adjudications beat 1 decisive one unless bar QUALITY is priced in); wall-clock counts idle queue time that is scheduling, not intelligence; the metric prices adjudication, not hypothesis GENERATION quality, and the lab's binding constraint has often been design/audit wall (human+model deliberation), not compute; cross-lab comparison inherits all the device/sigma fences.
  Testable residue: instrument the ledger — adjudications (VERDICT entries with mechanical bars) per session-day and per GPU-hour over the repo history, trend it, and test whether skill/desk adoption dates coincide with throughput inflections; a future rung could pre-register a throughput target for a fixed question set under a fixed budget.
  Attribution: Artin (metric + FA-law revival ask), house (formalization, breaks, residue).

- **BANKED (2026-08-29): SPX/SPY STRIKE-REACHABILITY FROM CURRENT OPTION-MARKET STATE — can a compact domain-specific model estimate how hard a target strike K is to reach (touch) or finish ITM by horizon T from the CONTEMPORANEOUS option-market state, rather than a long raw price-history sequence?** (Artin: idea + current-state-positioning hypothesis; GPT: formalization/falsification structure). BANK ONLY — no experiment, no data purchase, no trading, no code under this bank.
  The mapping / the math: two SEPARATE targets, never mixed — (1) terminal probability P(S_T > K | X_t) for calls / P(S_T < K | X_t) for puts; (2) first-passage/touch probability P(tau_K <= T | X_t). Representation candidate: recenter the option surface around the queried strike K; per nearby strike i — normalized displacement from target, normalized displacement from spot (preferably in implied-vol/sqrt(time) units), IV, skew/local smile geometry, bid/ask or liquidity, volume, open interest, timestamp-valid Greeks, and positioning/GEX features explicitly labeled ESTIMATES (OI alone does not reveal dealer sign). Global state: spot/futures, time to horizon/expiry, ATM IV, term structure, broad volatility state, time of day, other contemporaneously observable state only if justified.
  Explicitly banked as TESTABLE HYPOTHESES, not truths: "the past does not matter"; "GEX controls SPX"; "market makers are positioned per public OI-derived GEX"; "better probability forecasts imply profitable trades". None is assumed.
  Key baseline law: the surface already implies a risk-neutral terminal distribution — any model must beat or add information beyond a contemporaneous option-implied baseline, not rediscover option pricing. Staged questions: A option-implied baseline calibration for realized terminal/touch outcomes; B does a tiny nonlinear current-state model improve heldout log loss/Brier/calibration beyond A; C does estimated positioning add beyond surface-only; D does raw history add beyond current state ("past unnecessary" survives only if history adds no material out-of-sample information); E under fixed compute/data, do ticker-/horizon-specific tiny models beat one pooled model. Metrics: log loss, Brier, reliability, calibration by strike distance / horizon / volatility regime — NEVER classification accuracy as the main metric. Model prior: tiny MLP / 1D strike-local convolution before any transformer; the research value rides the [[experimental-throughput bank, 2026-08-29 above]] (cheap model, many controlled ablations, fast falsification).
  Measured anchors: none — nothing in the lab record touches this domain; the DATA-CONTRACT vocabulary (grain / label timing / split policy) and the leakage incident corpus are the transferable house instruments.
  Honest breaks: OUTSIDE THE STANDING CHARTER (math + physics engines only) — any rung here needs an explicit Artin charter ruling first, this bank creates no eligibility; data is the likely bottleneck and gets its own feasibility desk BEFORE any modeling (timestamp-correct historical chains, bid/ask/volume/OI, reproducible historical IV/Greeks, exact expiry/calendar handling, corporate actions, no end-of-bar/end-of-day leakage, overlapping-horizon dependence handled, survivorship/delisted contracts, GEX/dealer positioning as INFERRED variables); calibrated prediction != economic edge — spreads, slippage, commissions, execution constraints are a LATER, separate question; overlapping horizons break iid assumptions in every naive eval.
  Testable residue: (smallest first desk, gate for everything else) can a leakage-clean SPX dataset be built for a finite date window where each row is (timestamp, target strike, horizon, contemporaneous state) -> {touch label, terminal-ITM label} with every feature provably available at timestamp t? Only if that data desk fires does a model rung become eligible. Possible future scientific question, no claim yet: does estimated positioning contain incremental information about realized strike reachability beyond the contemporaneous implied-volatility surface?
  Attribution: Artin (idea, current-state positioning hypothesis, staged-question ask), GPT (formalization, falsification structure), house (bank, charter fence, throughput link).

- **BANKED (2026-08-29): MODEL CONFLUENCE / REPRESENTATIONAL UNIVERSALITY — when independently optimized learners face the same structured environment, objective, and efficiency pressures, do they converge toward equivalent latent abstractions even when microscopic implementations differ?** (Artin: confluence/closed-system/universe intuition; GPT: dynamical-system/universality formulation + falsification fences). BANK ONLY — no rung, no experiment, no claim of established law.
  The mapping / the math: view a trained model as a finite dynamical system — architecture + learned parameters + representation define a state space, reachable trajectories, invariances, attractor-like behaviors, decision boundaries. Three DISTINCT outcomes, never conflated: (1) STRONG CONFLUENCE — different training/representation paths recover approximately equivalent internal latent variables; (2) BEHAVIORAL CONFLUENCE — internal mechanisms differ but input-output behavior converges on-support AND off-support; (3) NON-CONFLUENCE — different inductive biases produce materially different extrapolation despite equivalent underlying information. Deeper analogy: physics UNIVERSALITY — different microscopic systems share macroscopic behavior when only a small set of structural invariants matters; AI systems may converge on abstractions imposed by ENVIRONMENTAL structure rather than implementation identity.
  Measured anchors: the SVP chain is relevant ONLY as a counterexample to naive strong confluence — FACTOR and HASH encode the same whole-action information through an invertible bijection at equal width/data/model/compute, nearly agree on covered calibration (94/96 v 96/96), yet seed-16001 diverges sharply on withheld coordinate combinations (F 84/96 v H 41/96, RESULTS L54242) and REVERSES on the novel degree-class population (F 15/48 v H 33/48, L54553). Information equivalence alone does not force learned-function equivalence; representation geometry changes reachable generalization — and which representation wins is POPULATION-DEPENDENT. NOT promoted to a universal model-confluence law. Adjacent booked instrument: the 2026-07-06 weight-reader ablation (same function, many weight arrangements; never score weights by weight distance) is the house's standing microscopic-nonidentity anchor.
  Honest breaks / fences: a model is not literally a universe or a physically closed system; training data/objectives/tools are external boundary conditions; agentic/tool-using models are OPEN systems during deployment; "same information" does not imply same gradient geometry or learned behavior; similar outputs do not establish identical mechanisms; cross-lab architectural trends evidence common optimization pressures, not one inevitable architecture. Markets analogy kept SEPARATE from the [[SPX strike-reachability bank, 2026-08-29 above]]: competitive markets as distributed inference through one price surface, arbitrage pressure as behavioral convergence — but markets are REFLEXIVE (predictions/trades alter the environment), which breaks the fixed-environment premise of confluence.
  Testable residue (future questions, not experiments): do different seeds converge to functionally equivalent latent subspaces? do different architectures on the exact same task recover equivalent low-dimensional variables? does representational similarity rise with capability across architectures? which features are universal v representation-specific? does convergence hold on-support while breaking specifically under OOD/compositional tests (the SVP result predicts YES — that is the sharpest house-testable residue)? can two non-isomorphic internal representations be related by a simple invertible map? does efficiency pressure favor a smaller/more canonical latent state? Philosophical residue, unclaimed: a significant component of intelligence may be not storing more raw reality but discovering coordinate systems in which environmental regularities become simple (coheres with FA Law v2's compression-rate framing; no measurement).
  Attribution: Artin (confluence/closed-system/universe intuition), GPT (dynamical-system/universality formulation, falsification fences), house (SVP counterexample anchoring, weight-reader link, bank).

- **BANKED (2026-08-29): RIFF BUNDLE — six related organizing ideas around coordinates, time, and predictive state** (Artin: coordinate/time/minimal-state/randomness intuitions; GPT: law-compression, multiple-time distinction, predictive-state formalization, coordinate-reuse-v-extrapolation synthesis; house: bank + SVP anchoring). Speculative organizing ideas, not established laws; no rung, no experiment, no interruption of the frozen seed-17001 replication chain.

  **(1) NATURAL COORDINATES / LAW-COMPRESSION.** Intelligence may partly consist of finding a coordinate system in which time, causality, dynamics, and prediction become SIMPLE — information-equivalent representations need not make the underlying law equally simple. For z = phi(x), the candidate quality principle is not minimum representation description length alone but JOINT compression of state AND dynamics: R* ~ argmin L(state under R) + lambda * L(transition/prediction law under R). A tiny representation that makes the transition law complex may be inferior to a slightly larger one in which the law is simple. Residue: does better generalization correlate with lower transition-law complexity in learned coordinates? can two information-equivalent encodings be ranked by the complexity of the extrapolation map between states? does training preferentially discover coordinates that linearize/factorize task dynamics? SVP relevance: FACTOR/HASH are information-equivalent with sharply different extrapolation — COMPATIBLE WITH but not establishing a law-complexity account. No claim.

  **(2) MINIMAL PREDICTIVE STATE.** Seek z_t = phi(H_t) with P(Y_future | H_t) = P(Y_future | z_t); among all sufficient z_t, the smallest/cheapest. Distinguish ambient hidden dimension v intrinsic predictive-state dimension v raw physical spacetime dimension — a 10,000-dim activation vector may carry a task manifold of far lower dimension. Interpretation candidate: "understanding" ~ maintaining a compact sufficient predictive state rather than memorizing raw history. No assumption the minimum is literally four scalar coordinates. (Predictive-state / causal-state lineage — computational mechanics' epsilon-machines are the published relative; no THEORY row until measured.)

  **(3) INFORMATIONAL EFFICIENCY -> RANDOM RESIDUAL.** Maximum informational efficiency can LOOK like maximum randomness: once every predictable dependency under information set I_t is extracted, observation = compressed structure + residual with predictable information in the residual -> 0 relative to I_t. Domains: efficient markets, scientific residuals, compression, model errors. HARD FENCE: random-looking output does NOT prove efficiency — hidden structure may simply remain undiscovered (the inference runs one way only). Residue: under controlled models, does residual compressibility / conditional predictability decrease monotonically as calibration quality improves? (Coheres with FA Law v2: verified variance converted to structure leaves incompressible residue behind.)

  **(4) MULTIPLE TIMES IN A MODEL.** Distinguish t_world (when an event occurs in the modeled environment), t_causal (dependency ordering/partial order), t_compute (where the model is in its own inference trajectory). These need not coincide: a model revisits an earlier world-time event at a later computation step; near-simultaneous world events can have directed causal order. Residue: does explicitly disentangling these temporal coordinates improve learning under irregular physical time, causal reasoning, planning, or markets? Do not assume ordinary positional encoding is equivalent to explicit continuous world time. (Adjacent bank: the TRANSITION-BASIS / VISUAL-NGRAM riff — temporal info edge-local, not state-local.)

  **(5) COORDINATE-REUSE / COORDINATE-EXTRAPOLATION TRADEOFF.** Motivated by the booked seed-16001 contrast; HYPOTHESIS ONLY. Observed (both booked): strict in-support missing-combination test FACTOR 84/96 v HASH 41/96 (RESULTS L54242); separate fresh degree-11 P-OUT covered-I0/t2 test FACTOR 15/48 v HASH 33/48 with 0 F-only / 18 H-only discordants, significant reverse split, FACTOR's successes entirely at the two boundary forms and 0/32 across interior novel forms (L54553). Mechanism hypothesis, NOT claimed: explicitly factored coordinates give a strong inductive bias for recombining KNOWN coordinate values into unseen tuples while producing brittleness when coordinate VALUES leave learned support; arbitrary whole-action identity lacks combinatorial reuse but may degrade differently/more smoothly under some value-OOD populations. Honest breaks: the strict and P-OUT tests differ in population AND question; the degree-10 residue differed in seed and population; seed x P-form-class x term interactions are NOT identified. The future rung this names (do not design yet — seed-17001 replication has priority): an explicit 2D crossing of tuple-combination support {seen, unseen} x coordinate-value support {seen, unseen}, holding seed/population/term and all else fixed.

  **(6) REPRESENTATION AS INDUCTIVE PHYSICS.** A learned representation is not storage: it defines which transformations are local, which states are near, which combinations are cheap, which extrapolations need long paths — the coordinate system imposes an "inductive physics" on gradient learning (neighborhoods, reachable states, reusable transformations, support boundaries). Information equivalence therefore does not imply learning-dynamics equivalence. Philosophical framing only until converted into measurable quantities. (Sibling of the [[MODEL CONFLUENCE bank, same day above]]; the SVP pair is the house's one measured instance of the non-equivalence.)

- **BANKED (2026-08-29): INVARIANT EFFECT / NON-INVARIANT TRACE — macroscopic representation effects may be stable across realizations while local error mechanisms are realization-dependent** (Artin, riding the seed-16001/17001 replication pair; house: bank + anchors). Descriptive riff, NO mechanism claim.
  The observation: across seeds 16001 and 17001 the high-level FACTOR advantage on the strict missing-combination endpoint replicates (both STRONG-FACTOR), while magnitude (44/1 v 20/8 discordants) and cell-level t2/t3 anatomy do NOT (16001: t3 HASH-collapse to 2/48 with FACTOR ahead at t2; 17001: FACTOR's edge at t3 46v33 with pooled t2 even and t2-IN 13v20).
  Measured anchors: RESULTS L54242 (16001 STRONG-FACTOR), L55052 (17001 REPLICATES-STRONG-FACTOR, realization table + anatomy divergence, booked descriptive).
  Honest breaks: n=2 realizations of one protocol, one HASH permutation, one dataset — "macroscopic" here means one preregistered contrast, not a family of effects; the anatomy cells are n=24 with no subgroup tests, so "non-invariant trace" partly reflects small-cell noise, not only mechanism variance; the P-OUT reverse split shows the macroscopic effect itself is population-dependent, so stability is claimed across INITS only, never across populations.
  Testable residue: at a third realization (or the P2 permutation arm), preregister a cell-anatomy stability statistic (e.g., rank correlation of per-cell F-H gaps across realizations) alongside the primary — if the primary keeps replicating while the anatomy statistic stays at chance, the riff graduates to a measured claim; coheres with the [[MODEL CONFLUENCE bank]] (behavioral convergence without mechanism identity) as a within-protocol instance.
  Attribution: Artin (riff), house (anchors, breaks, residue).

- **AMENDMENT (2026-08-30) to three banks — the KNOWN-SET-P2-ONLY result lands (RESULTS L56104)**: (1) COORDINATE-REUSE/COORDINATE-EXTRAPOLATION TRADEOFF: the "arbitrary whole-action identity lacks combinatorial reuse" premise is now PARTIALLY REFUTED as a universal — HASH-P1 at seed 18001 completed the withheld combinations at 83/96 (F 85/96) while sibling bijection HASH-P2 collapsed to 43/96; reuse-through-factorization cannot be the only route to completion, and whatever route the P1 realization found is bijection- and seed-dependent (P1 across seeds: 41/63/83). The banked 2D support-crossing rung gains a third axis candidate (bijection realization) if ever designed. (2) INVARIANT EFFECT / NON-INVARIANT TRACE: strengthened AND sharpened — the "invariant" part now attaches to the FACTOR ARM'S LEVEL (84/75/85), not to the F-v-H contrast, whose magnitude swings from 2 to 43 solves across realizations because the CONTROL moves; the banked stability-statistic residue should target arm levels, not gaps. (3) MODEL CONFLUENCE: the SVP counterexample to naive strong confluence is now itself realization-dependent — information-equivalent representations sometimes diverge enormously (F v H2, F16001 v H) and sometimes behaviorally converge (F v H1 at 18001: 2/0 discordants); non-confluence is not a stable property of the representation PAIR but of the (representation, seed, bijection-realization) triple. All three banks keep their original text; this amendment names what the 18001 permutation run measured.

- **BANKED (2026-08-30): LEARNED GEOMETRY / STATIC GEOMETRY SPLIT** (Artin; house: bank + anchors). The codebook defines a static geometry before learning, but the trained model induces its OWN conditional geometry over that codebook through token probabilities and gradients — two identical static codebooks under different initializations can therefore acquire different learned geometries. Measured anchors: GEOMETRY-DESK-0's inversion (static support anti-aligns with outcomes, RESULTS L56392) and P1's 41/63/83 across seeds at fixed geometry (RESULTS L54242 / L55052 / L56104). Honest breaks: "learned geometry" is not yet operationalized — the token-onset probe (design frozen this day) is its first candidate instrument; n=2 permutations, n=3 seeds. Testable residue: do learned-geometry summaries (onset distributions, delta trajectories) separate realizations that static geometry cannot?
- **BANKED (2026-08-30): REPRESENTATION CONDITIONING AS ONSET STABILITY** (Artin; house: bank). Hypothesis only: a well-conditioned representation may produce LESS realization-sensitive token-level decision trajectories even when final accuracy is similar — FACTOR's stable arm levels (84/75/85, RESULTS L54242/L55052/L56104, banked not claimed) would then coincide with stable onset anatomy across seeds, while the HASH arms' swings would show as onset instability. Measured anchors: none yet (the token-onset probe's FACTOR rider is the first look). Honest breaks: n=3 seeds cannot support a variance claim; onset stability could equally reflect margin magnitude, not conditioning. Testable residue: the probe's registered strengthen/weaken conditions (RESULTS, TOKEN-ONSET-PROBE-DESIGN-0). AMENDMENT (2026-08-30, same day — TOKEN-ONSET-PROBE-0 booked): first descriptive anchor landed — the three FACTOR accuracy(k) curves are shape-identical (flat 29/96 then a k=8 step to 84/75/85) while the HASH arms' curves swing across seeds and bijections; the registered STRENGTHEN conditions for initialization-conditioned optimization were met as written. This is a description of curves, not a FACTOR-stability claim; still hypothesis, n=3 seeds / n=2 permutations.

- **BANKED (2026-08-30): ORDERED COORDINATE CHARTS — for an autoregressive learner, a representation is not only a set of information-equivalent coordinates but an ORDERED FACTORIZATION of their joint distribution; coordinate permutation preserves information while changing conditional learning geometry** (Artin; house: bank + anchors). Measured anchors: the FACTOR token-8 step function at all three seeds with position 8 the structural separating coordinate on 67/96 states (RESULTS L56656); the serialization-imposed fence stands — the step is partly structural. Honest breaks: no field-order intervention has run, so "order changes learning geometry" is untested on the FACTOR side (the HASH evidence — same information, wildly different dynamics — concerns arbitrary codes, not coordinate order); the claim is chart-level, not a theorem. Testable residue: the FACTOR-FIELD-ORDER intervention designed this day (RESULTS, FACTOR-FIELD-ORDER-DESIGN-0) — SEMANTIC-SOCKET v ORDER-LOAD-BEARING v MIXED via the two frozen binaries; coheres with [[LEARNED GEOMETRY / STATIC GEOMETRY SPLIT]] (order is a static property whose consequences are learned).

- **BANKED (2026-08-30): FUNCTIONAL COMPRESSION REPAIR — compress, measure functional damage, restore ONLY the damaged functionally important subspace** (Artin: concept + repair families + fair-arm design; house: bank, falsifiable form, confounds, kill condition). BANK/ASSESSMENT ONLY; nothing downloaded, compressed, trained, repaired, or benchmarked.
  The mapping / the math: for weights W and a frozen compressed approximation W_c, the raw error E = W - W_c; full repair recovers W and zero benefit. Hypothesis: FUNCTIONALLY IMPORTANT compression damage may occupy a much lower-dimensional subspace than the full tensor error. Measured anchors: none — nothing in the lab record touches compression repair. Candidate repair families (mechanism candidates, none frozen as the method): (1) WEIGHT-RESIDUAL LOW-RANK — W_r = W_c + E_r with E_r the truncated-SVD rank-r approximation of E; (2) ACTIVATION-AWARE LOW-RANK — argmin over small-rank Delta W of ||(W_c + Delta W)X - WX||^2 on frozen teacher activations X under a fixed byte budget; (3) DIRECTIONAL/FUNCTIONAL-SUBSPACE — measure teacher-v-compressed residual-state discrepancies on a frozen harmless calibration population, extract a low-dimensional damaged basis U_r, and preferentially restore mappings into it (conceptually Delta W = U_r U_r^T (W - W_c) — well-typed ONLY for matrices writing into the residual stream; non-residual-output matrices (QKV, up/gate) need an intervening map, scoped at the desk — or an activation-weighted regression restricted to U_r). Core question: how low-dimensional is COMPRESSION-INDUCED FUNCTIONAL ERROR — deliberately distinct from "how low-rank are the weights" and from "how low-rank is E in Frobenius norm"; a model may tolerate enormous weight-space error while being sensitive to a small functionally privileged subspace. Geometric primitive lineage: the directional-ablation update W' = (I - lambda vv^T)W (the public refusal-direction / directional-ablation weight-surgery lineage, Arditi et al. 2024-class and its open tooling, of which Heretic is one implementation) motivates the INTERVENTION PRIMITIVE only — the interest here is the inverse-style REPAIR operation; nothing about censorship or safety-direction removal is in scope, and no one-dimensional "intelligence direction" is assumed.
  Candidate future primary object: repair_fraction = (S_repaired - S_compressed) / (S_original - S_compressed) on a frozen benchmark score S (0 = no recovery, 1 = full; UNDEFINED and unread when compression does not damage past the sigma bar; values > 1 or < 0 reported raw, never clipped), ALWAYS paired with teacher-v-model KL on a harmless calibration set, repair bytes / original bytes, compressed bytes / original bytes, and wall/inference cost when materially changed. The useful frontier: recovered capability v TOTAL storage cost v repair rank/bytes. Fair future arms from ONE frozen compressed model: A compressed baseline; B RANDOM low-rank delta control at the same budget; C weight-residual SVD repair; D activation-aware repair; E functional-direction repair — all repair arms at the SAME stored-parameter budget, teacher as ceiling, compression frozen BEFORE any repair method sees evaluation outcomes, no repair tuned on the final heldout benchmark.
  STRONGEST FALSIFIABLE FORM (house formalization): at a fixed compression severity whose benchmark damage gap S_original - S_compressed exceeds ~3 sigma of the frozen benchmark's noise floor (seed count named in the adopting prereg), there exists a rank r whose TOTAL adapter cost — sum over ALL repaired matrices of dtype_bytes * r * (d_in + d_out), adapter dtype stated (2 bytes at fp16) — is <= 5% of the compressed model's bytes at its stated precision, such that a prospectively chosen rank-r repair (family C, D, or E) achieves repair_fraction >= 0.5 while equal-budget controls achieve < 0.1; controls = the RANDOM low-rank delta AND the harder zero-delta and E-restricted-random baselines (a naive random control is nearly free to beat). Refuted if no family beats the harder controls materially at any small rank across severities.
  KILL CONDITION (early, cheap): if on the smallest desk model BOTH the activation-aware family (D) AND the functional-direction family (E) fail to beat the harder controls at equal bytes on ANY compression severity, the low-dimensional functional-damage hypothesis dies for that model class and the bank gets its refutation named in place (D optimizes an activation-reconstruction PROXY, not the benchmark — a D-only null kills nothing by itself, per this bank's own KL-is-not-task-success confound).
  MAJOR CONFOUNDS (named): low weight-reconstruction error != preserved capability; low teacher KL != task success; repair adapters can recover BENCHMARK behavior without reconstructing the original internal computation (behavioral v mechanistic repair must be kept distinct in any booking); calibration-set choice leaks task distribution into "task-general" corrections; total-storage accounting (compressed + repair) is the only honest denominator; for MoE, total parameter-storage compression v active inference FLOPs are separate claims; benchmark noise floors bound any repair_fraction reading (house sigma discipline applies).
  SMALLEST FUTURE DESK (assessment, not run): a small dense local model on the Mac where original weights fit in memory, several compression severities freeze cheaply (e.g. int-quantization ladders), teacher and compressed activations capture in one pass, equal-byte repair arms are practical, and the existing harmless math/general-reasoning evaluation applies — the 0.5B-class dense residents in the lab's distill stack are the natural candidates; prefer this before any Kimi-class MoE. Later MoE assessment (separate, unpriced): shared-base + expert-residual compression, expert-specific repair rank, router/activation-conditioned repair, storage-v-active-compute split. CHARTER NOTE: evaluation scope is the existing benchmark world; any bio/chem content stays measure-only per the standing evaluation/engine ruling — no domain capability is developed.
  Testable residue: the desk above; the falsifiable form; and whether the damaged-subspace dimension GROWS or stays flat as compression severity increases (a flat dimension across severities would be the striking outcome).
  Attribution: Artin (concept, repair families, fair-arm design, MoE split), house (falsifiable form, kill condition, confound census, desk assessment).

- **BANKED (2026-08-30): FUNCTIONAL ERROR DIMENSION — the meaningful compressibility of a trained network may depend less on the dimensionality of its raw weight tensor than on the dimensionality of the FUNCTIONALLY CONSEQUENTIAL error introduced by approximation** (Artin; house: bank). A useful compressor may therefore be: coarse model approximation + small task-general functional correction. Measured anchors: none in the lab record (the weight-domain relative is the booked never-score-weights-by-weight-distance law, 2026-07-06 — same function, many weight arrangements — which already separates weight geometry from function). Honest breaks: "functionally consequential" is calibration-set-relative; the dimension may not be well-defined if the damage is nonlinear or distributed. Testable residue: the [[FUNCTIONAL COMPRESSION REPAIR]] desk directly measures it.

- **BANKED (2026-08-30): DIRECTIONAL ABLATION / DIRECTIONAL REPAIR DUALITY — ablation suppresses a selected output subspace (W' ~ (I - UU^T)W); repair asks whether a compressed model recovers teacher behavior efficiently by restoring only selected damaged subspaces (W_r ~ W_c + UU^T(W - W_c))** (Artin; house: bank). Conceptual geometric DUALS, not exact algorithmic inverses — the ablation U is chosen to remove a behavior, the repair U to restore one, and neither implies the other's subspace is meaningful. Measured anchors: none; the actual repair law is to be chosen prospectively at the assessment desk. Honest breaks: the duality is a framing device — a repair that works may do so through mechanisms unrelated to any ablation-style geometry. Testable residue: whether family E (functional-subspace repair) outperforms family C (weight-residual SVD) at equal bytes — a direct test of whether the FUNCTIONAL basis beats the WEIGHT basis.

- **AMENDMENT (2026-08-30) to ORDERED COORDINATE CHARTS — first measured anchor lands (RESULTS L57428)**: coordinate permutation preserved information and covered competence (CANONICAL 94/96, PARAM-FIRST 93/96) while significantly changing withheld-combination completion (85 v 78, exact McNemar p=.0390625, one-state boundary-adjacent) and moving the learned decision step exactly to the relocated coordinate (k_step 8 -> 3, both arms matching their pre-training structural predictions; the PF alignment margin is one state). Order IS part of the representation for this learner — in both mechanism and outcome — at this single known-set realization; the frozen map's registered wording, 'identity and ordering interact', is the anchor's claim ceiling. The original bank text stands; this amendment names what the 19001 intervention measured.

- **BANKED (2026-08-30): EXTERNAL-READER QUESTIONS 0 — eight outside-reader pressure-test questions from two independent external model reads of FINDINGS.md, banked with house corrections against the actual ledger** (external readers: the questions; Artin: the relay + bank ask; house: ledger corrections, fences, smallest-test pricing). BANK/ASSESSMENT ONLY — nothing here reorders the active queue; the live field-order replication chain stays authoritative until adjudicated.

  (1) ACTIONPROGRAM RANKING -> GENERATION -> CLOSED LOOP. The established result class is sibling-action RANKING under frozen candidate sets (pessimistic top-1 over supplied legal candidates — the entire SVP scoring lineage); nothing measures autonomous generation. Banked transport ladder: A sibling ranking (measured) -> B one-step free ActionProgram generation -> C closed-loop generation + verified execution/search. Questions: does the semantic-program representation advantage survive when the model must GENERATE a valid action rather than rank a supplied set; and separately, does any one-step generation edge survive compounding state-distribution shift in closed-loop solving? Honest breaks: ranking != generation != solve-rate; syntax validity, semantic replay, search branching, and error compounding are NEW failure modes with no measured anchor; deterministic engine verification stays mandatory at every rung. Measured anchors: v4/v5 ActionPrograms fully qualified (ACTION-SITE/SEMANTICS 725/725, ACTION-OPCODE-QUAL-0); ranking results per the SVP record. Testable residue: a minimal FREE-ACTION-1 bridge (one-step generation, engine-checked validity + semantic replay, paired F/H arms) BEFORE any closed-loop rung; not the next rung until the field-order chain adjudicates.

  (2) EXACT COMPUTE — WHERE DOES THE REMAINING ERROR LIVE? Reader proposed the nonlinear/softmax carry path rather than matmul precision limits exact compute. HOUSE CORRECTION: the mechanism half is already MEASURED, not open — EXACT1-SMALL booked frozen-carry error as the softmax carry quantization, grain-independent by mechanism (anchor-to-Q32 distance == anchor-to-Q64 to the last decimal at every certified step; ring grain fully absorbed at p=32) — the reader's "perhaps" is the ledger's verdict (RESULTS L23852 + amendments L23910/L23948). What IS open, banked as hypothesis: CAUSAL ISOLATION of the approximation itself — holding exact matmul fixed, does replacing only the nonlinear/normalization/softmax approximation law (candidates: current carry law, lookup table, piecewise-linear, rational/integer; none adopted) materially move the measured error floor? And the fractional attribution question: what share of end-to-end exact-compute error is matmul v normalization/softmax v carry quantization v other nonlinear stages? Honest breaks: EXACT1-SMALL is the deterministic integer battery, two cells sharing one implementation and one device — mechanism reading, not a replication route; the frozen-carry floor curve (linear in steps, super-width growth ~2.4/2.7/8.3) is descriptive, one run per width. Testable residue: a carry-ladder arm swapping ONLY the softmax approximation at fixed ring precision against the booked floor curve (the curve exists to be beaten — FINDINGS already names it as the target for the next engine pre-reg). Scope fence: precision-capability questions stay inside the CLOSED precision doctrine's named retest slot; this is an ERROR-ATTRIBUTION instrument question, not a capability revival.

  (3) TERNARY IGNITION SURFACE. Ledger state: ternary capability is trained-in, never projected-in (post-hoc quantization craters the body while QAT ties fp32); WIDTH BUYS TOLERANCE (every class's ternary retention rises d64->d512, eye 0% -> 93%); parity reached in some regimes (RESULTS L2397, L25772, L25979, L26031). Banked open question: is there a width-dependent IGNITION BOUNDARY for ternary training, or only a smooth change in P(parity)? Candidate object: P(parity | width, seed, diet, training law) — deliberately NOT assuming a deterministic threshold w* exists; a smooth probabilistic surface and a sharp boundary are distinguishable outcomes. Honest breaks: existing width ladder is retention-after-projection plus a small QAT set, single-seed at most widths, house-crystal regime (capacity fence L10676 applies — no transport to web-trained models); "parity" needs a preregistered bar per diet or the surface is unreadable. Testable residue: a width x seed ignition grid at ONE frozen diet with preregistered parity bars — priced by /desk before any GO; no run authorized here.

  (4) MoE CARRIER SEMANTICS / CAUSAL PATH TRACING. Ledger state: carrier-expert interventions are causally SUFFICIENT at the ablation grain (named 80-expert deletion beats full +55 pooled 3/3 and beats matched-rank random +27; portability/saturation/redundancy decomposition booked — RESULTS L21943/L22007/L22034 + the replicated crest set). HOUSE CORRECTION of the reader's implicit over-read: nothing yet establishes WHAT computation a carrier expert performs, nor whether KV/attention mediates the effect — sufficiency-at-a-grain is not mechanism identity. Banked question: what SMALLEST downstream variable transports the carrier intervention? Tracing ladder: expert output -> residual stream -> K/V projections -> attention-head outputs -> later residual/logits — via intervention/patching, never activation correlation alone. Transport question banked beside it: does an analogous carrier phenomenon occur in other OPEN-WEIGHT MoEs — with actually comparable MoE architectures as controls (a dense model is not an MoE control; Muse-Glimmer-30B is the banked NO-ROUTER dense control for the router-v-features split, a different role). Honest breaks: all carrier evidence is FORMAT-BOUND / FREE-RUN-GATED / one deployment artifact family; patching grain on a 30B-class MoE is priced by memory (one-resident rule) and needs its own desk. Testable residue: single-layer path-patch desk — replace the carrier expert's output into ONLY the residual v ONLY K/V at the measured block, frozen gate battery as readout.

  (5) HOUSE-CRYSTAL -> SCALE TRANSPORT VIA DIMENSIONLESS LAWS. Reader default ("run it at 7B") corrected to the stronger banked question: which measured micro-model laws admit DIMENSIONLESS or normalized transport — width/support-complexity ratios, margin/perturbation-scale ratios, representation-support density, init sensitivity normalized by scale, error tolerance per effective capacity? Goal: find scale-collapsing relationships or demonstrate the small-model law BREAKS (either is a result). Measured anchors: the capacity fence itself (PACKED CRYSTAL C6, RESULTS L10676 — house-crystal laws do NOT transport by default; that fence is the null this question would test against); width-buys-tolerance is the most collapse-shaped existing curve. Honest breaks: causal isolation degrades as models grow (paired-arm births get expensive; mps/cuda fences bite); a collapse over 3 points is numerology — the residue must name minimum ladder length. Testable residue: pick ONE law (ternary retention v width is the natural first), write its dimensionless form, extend the ladder by two scale points, preregister the collapse bar.

  (6) SUPPORT x COORDINATE FACTORIZATION x AUTOREGRESSIVE ORDER. Reader ask ("more diverse action-support diet / hybrids") AMENDED to the current frontier — the ledger has already moved past FACTOR-v-HASH accuracy: one fixed bijection is not representative of arbitrary-code controls (H-P1 41/63/83 across inits, H-P2 43 under the same init — RESULTS L56104); static code geometry does not explain the split and even anti-aligns (L56392); token-onset anatomy differs by representation and realization (L56656); field-order intervention causally moves the learned decision position while preserving information and per-action multisets (L57428). Banked question: how do SUPPORT (which combinations trained), COORDINATE FACTORIZATION (which chart), and ORDER (which conditional factorization) INTERACT — a three-axis design space in which every existing run is a face or edge. Honest breaks: full factorial is unpriceable; the 2D support-crossing rung and the bijection-realization axis are already banked separately (coordinate-reuse bank + its amendment); order axis has ONE measured cell pair (canonical v param-first, one seed + one live replication). Testable residue: after the field-order replication adjudicates, a /desk pricing the smallest informative CORNER cell (e.g. permuted-order x P1-support) rather than the cube.

  (7) ADAPTIVE REGRET / CONDITIONAL ABORT CONTROLLER. Ledger state: regret probes discriminate trajectory fate early (trace-fate signal RESULTS L1060) and a GLOBAL engine-level abort law already pays (2.1x, -2.4% completeness, in solve() by default — L1203); the LDS controller ladder booked residual failures as budget-censored (REGRET-LDS-DESK-0 + WALLLIFT: 5/6 roots exhaust expansions unsolved). Banked question: does a CALIBRATED CONDITIONAL abort controller dominate the global law on the same frozen workload — capturing most throughput gain while pricing false aborts explicitly and preserving more completeness? Policy metrics banked (AUC explicitly NOT the objective): cost per solved instance, completeness loss, false-abort rate on eventually-solvable trajectories, wall/node savings, calibrated expected regret. Honest breaks: the global law's 2.4% completeness cost is small — headroom for "dominates" may be a few instances on the measured workload; any controller trained on trajectory features must respect the label-timing data contract (fate labels join future information). Testable residue: frozen-workload paired replay — global v conditional on identical trajectory sets, completeness-matched wall comparison; connects to the [[experimental-throughput]] metric bank.

  (8) EXTERNAL LEGIBILITY AS AN INSTRUMENT (documentation bank, no scientific claim). Two independent readers found the evidence vocabulary precise but hard to enter. Banked artifact question: can a short external-reader map (RESEARCH-MAP.md / START-HERE layer) explain the major programs, what each established, what remains open, and how maturity tags / gates / fences / desks / rungs work — WITHOUT simplifying or rewriting FINDINGS.md itself? NOT BUILT under this GO: the repo has no designated slot and the task is not trivial (front-facing voice rules + honesty-ledger regen hooks make it a real artifact with its own review). Honest break: a map that drifts from the ledger is worse than no map; it would need the same CI-anchor discipline as FINDINGS bullets. Testable residue: none scientific — an acceptance test would be "an external reader locates the evidence for a named claim in under 10 minutes."

  (META) PROTOCOL COST. Banked outside-reader challenge: when does procedural rigor increase information per compute, and when does it merely reduce iteration throughput? Working distinction preserved: cheap exploratory desks stay cheap (the /desk and /probe ladder IS the existing answer at the cheap end); expensive claim-bearing / treatment-changing runs earn prereg + dual-audit overhead. Possible future process metric: adjudicated information gain (or killed hypotheses) per compute-hour and per wall-hour, protocol overhead separately accounted. Honest breaks: killed-hypotheses-per-hour is gameable by proposing weak hypotheses; the metric itself would need a fence against that; no anchor exists. Testable residue: retrospective desk over the existing ledger — count verdicts, nulls, retractions, and audit-caught errors per program against wall receipts already booked; zero new compute.

  Attribution: two external model readers (questions), Artin (relay, bank framing, priority fence), house (ledger corrections, anchors, breaks, residue pricing).

- **AMENDMENT (2026-08-30) to ORDERED COORDINATE CHARTS — the seed-20001 init-only replication FAILS both endpoints (RESULTS L57839)**: under an independent initialization of the identical frozen world, PARAM-FIRST's decision step landed at k=5 (not the structural k=3; gain(3) = -12) and completion SIGNIFICANTLY REVERSED (PARAM-FIRST 87/96 v CANONICAL 80/96, 0/7 discordants, p=.015625) against seed 19001's significant ORDER-DAMAGE (85 v 78, 8/1, p=.0390625). The bank's core claim — order is part of the representation for this learner — SURVIVES and is arguably strengthened (ordering changed outcomes in BOTH realizations), but the first anchor's direction does not transport: WHICH ordering wins and WHERE the relocated socket fires are initialization-dependent; the stable parts across all realizations are CANONICAL's late k=8 one-step mechanism and the qualitative PARAM-FIRST early-block transient disruption (29 -> 11-12 at 19001, 29 -> 8 at 20001). 'Identity and ordering interact' stays licensed only at per-realization scope; no ordering is 'better' at n=2 opposite-signed significances. Original text and first amendment stand; this names what the replication measured.

- **BANKED (2026-08-30): AUTOREGRESSIVE CREDIT ROUTING — an ordered semantic representation fixes which coordinates are AVAILABLE at each prefix, but finite training need not assign predictive discrimination to the structurally earliest distinguishing coordinate; initialization and optimization can route sequence-level credit through later conditional fields under the same representation, data, and training law** (Artin: the desk ask + the routing framing; house: families A-E, anchors, fences). Connection: [[ORDERED COORDINATE CHARTS]] — information-equivalent ordered charts define possible conditional factorizations; the learned model selects a realization-dependent predictive ROUTE through them (a learner-induced gauge/chart-symmetry breaking frame, named as analogy only).
  Measured anchors: FIELD-BLOCK-CREDIT-DESK-0 (RESULTS L57971) — CANONICAL PARAM-block routing stable across seeds (+56/+46); PARAM-FIRST routed through PARAM at 19001 (+59) and through RULE at 20001 (PARAM +17, RULE +70, 70/96 RULE-RESCUED), while the PARAM block still contributed positive fixed-rival credit at 20001 (median +1.11, 66/96 positive) — the collapse is dynamic-rival rank loss, not missing parameter information; completion sign flip coincides with the routing shift (association, n=2).
  Honest breaks: two seeds, one world/population/intervention — no universality, no population-variance estimate; "credit" is a logprob decomposition, not a causal mechanism; k_step and block deltas are cumulative-ranking descriptions (the desk's own narrowing: structural first-separation does not identify the used field, and "socket" language overstates); the gauge-breaking connection is a naming frame with zero independent anchors.
  Testable residue: (1) FREE-ACTION-1 (assessed same commit, RESULTS L57971) — if greedy generation cannot exploit later-field rescue, PF free-generation validity should drop toward its end-of-PARAM top-1 (17/96 at 20001) rather than its ranking total (87/96), a large predicted gap; (2) a multi-seed order battery would measure the routing distribution rather than two draws — unpriced, gated on Artin; (3) routing stability under the P1/P2 hash arms (existing raws) is a zero-cost extension desk if ever wanted.
  Attribution: Artin (ask, routing + gauge frames), house (desk implementation, anchors, breaks, residue).

- **BANKED (2026-09-03): RENDER ATLAS AS A CAYLEY-GRAPH LANDSCAPE — the exhaustive 720-policy global role-permutation atlas is a complete scalar field on the Cayley graph of S_6 under adjacent transpositions (720 vertices, degree 5, 1,800 edges), so "basin", "component", "ridge", "barrier", "cliff" and "ruggedness" for renderer preference can be given exact finite-graph meanings with zero new logits** (Artin, the frame and the eleven-bank design; house, feasibility and freezing).
  The mapping / the math: vertices = role permutations, edges = one adjacent-role swap (Kendall distance); fields T_c and B_c from the booked atlas per checkpoint. Banks: A component geometry at the frozen NEAR / MAJORITY / STRONG thresholds; B shortest paths and widest-path (mountain-pass) barriers with a set-to-set convention; C steepest-ascent attraction basins per field and under the frozen lexicographic maximin tuple; D edge-conflict field (Delta per checkpoint on every edge, local Pareto optimality of R488, common-monotone RAW-to-R488 path); E robustness radii; F distance-to-optimum profiles; G role-position order-grammar census; H total variation and Laplacian spectral energy at predeclared cuts; I Pareto-front induced graph; J state-level decision-boundary edges (needs the untracked chunk streams); K 2 x 2 seed-representation landscape similarity.
  Measured anchors: RESULTS L65657 (atlas verdict: each checkpoint reaches 96/48 individually, near sets 6/46/91/13 with empty intersection, STRONG 0/720, R488 sole MAJORITY at T 78/88/92/96 B 30/40/44/48, six positive Spearman pairs); assessment RESULTS entry MATH-CYBER-1-RENDER-ATLAS-GEOMETRY-BANK-ASSESSMENT-0 (graph gates verified, feasibility spot-check of components unbooked).
  Honest breaks: a finite Cayley graph, not a manifold; the field lives on renderer strings, not on the model's weights or representation; four seed-correlated discovery checkpoints, no population claim; graph adjacency is a choice (adjacent transpositions), and a different generator set gives a different landscape; nothing here says which surface feature a model uses; semantic_beyond_all_surface_identifiable = false.
  Testable residue: (1) whether R488 and each checkpoint's optimum set share a MAJORITY / NEAR component or are separated by a barrier of frozen depth; (2) whether a RAW-to-R488 path exists with no checkpoint's B decreasing (COMMON-MONOTONE PATH ABSENT otherwise); (3) whether steepest ascent from RAW under each checkpoint's field reaches a global optimum and how large the optimum basins are; (4) whether W-first failure is explained by W position alone (the 120 W-at-1-to-2 edges); (5) whether spectral energy sits in low or high Laplacian modes at the predeclared cuts; (6) whether same-seed or same-representation pairs are more similar on nodes and on edges. Nominated first analysis: MATH-CYBER-1-RENDER-ATLAS-CAYLEY-LANDSCAPE-0 (banks A + B + C + D + K), zero logits, not launched.
  MEASURED (2026-09-04, VERDICT MATH-CYBER-1-RENDER-ATLAS-CAYLEY-LANDSCAPE-0, RESULTS L65907, folded here one booking late): residues (1), (2), (3), (6) measured on the four discovery and eight fresh checkpoints, never pooled. (1) R488 shares a MAJORITY component with an optimum on 3 of 4 discovery checkpoints and is DISCONNECTED from every 19001 CANONICAL optimum with a widest-path barrier of 30 pairs (a B = 0 valley); (2) COMMON-MONOTONE PATH ABSENT, RAW's monotone-reachable set is RAW alone; (3) strict ascent from RAW reaches a global optimum on 2 of 4 discovery and 6 of 8 fresh B fields, largest strict-ascent optimum basin 4.3 % of starts; (6) no same-seed or same-representation structure at n = 2 / 4 pairs. Residues (4) (bank G) and (5) (bank H spectral cuts) remain open.
  CORRECTION (2026-09-04, AMENDMENT ...-CAYLEY-LANDSCAPE-0-PLATEAU, RESULTS L65979): the L65907 reading "rugged: hundreds of local maxima" is withdrawn as a B-ruggedness claim. Bank C counts strict-ascent FIXED POINTS on integer-valued fields; 137 to 494 of the 204 to 526 per B field are interior vertices of the B = 0 floor (480 to 626 of 720 renderers per field score B = 0). Ruggedness is now stated only through the plateau-quotient interior-maximum count, a new step P0 frozen in OBSERVATION MATH-CYBER-1-RENDER-ATLAS-MORPHOLOGY-BANK-ASSESSMENT-0 (RESULTS L65937), which also prices banks E + F + H + I and nominates MATH-CYBER-1-RENDER-ATLAS-MORPHOLOGY-0 (P0 + E + F + H + I, zero logits, not launched). Attribution: Artin (the plateau-quotient requirement before any local-maximum reading), house (desk census, P0 definitions).
  MEASURED (2026-09-04, VERDICT MATH-CYBER-1-RENDER-ATLAS-MORPHOLOGY-0, RESULTS L66016): residue (5) closed — every field's energy sits in the low Laplacian modes (10 %-cut fraction 0.27 to 0.78 against relabeling maxima at most 0.16, invariant at the eigenspace boundary) and every field is smoother than all 200 relabelings of itself. Plateau quotient: 10 to 61 quotient maxima per B field (8 to 51 interior) against 204 to 526 fixed points, and the T fields collapse the same way (10 to 59 against 163 to 492); discovery B classes RUGGED / FLAT / INTERMEDIATE / FLAT; the largest ceiling basin under quotient ascent holds 77 % to 95 % of starts on the discovery B fields, so the strict-ascent 4.3 % of L65907 was a floor artefact; every in-set anchor and every median optimum is one swap from leaving its set; both discovery fronts split into 11 components. Residue (4) (bank G) remains open; banks G and J stay banked.
  MEASURED (2026-09-04, VERDICT MATH-CYBER-1-RENDER-ATLAS-DECISION-ATLAS-0, RESULTS L66073): bank J run on all twelve streams (re-hashed, untouched). No render-invariant state anywhere; per-state boundary fraction median 0.096 to 0.190 on the discovery cohort; the wrong action is the pair partner's gold in 92.7 % or more of wrong cells; COS_LOW states are the hard ones (median renderers correct per state 65.5 to 169.5 of 720 against 545 to 649 for SIN_LOW), so the house prior that B0 is the fallback (drawn from the W-first family) was backwards; pair partners co-flip LESS than unrelated states on 12 of 12; zero-flip edges are 97 % to 99 % of zero-delta-T edges on the discovery cohort. Banks A to F, H, I, J now measured; bank G (role-position census) remains the only unmeasured bank.
  MEASURED (2026-09-04, VERDICT MATH-CYBER-1-RENDER-ATLAS-ROLE-CENSUS-0, RESULTS L66121): bank G run, residue (4) closed — W-first failure is not explained by W position alone: the W-position-only fit explains 0.06 to 0.35 of the both-correct variance on discovery and the full 36-indicator fit 0.33 to 0.84, while the effect shared by all four discovery checkpoints is the relative order of HI_D (the SIN_LOW negative-bearing role) and W (HI_D before W +5.6 to +16.9 pairs; HI_D last worst; W last best). The W-first family extends one swap inward only on the PARAM_FIRST checkpoints (W-second family 94 / 120 / 92 / 120 at T 48 B 0) and is universal on 4 of 8 fresh checkpoints. All eleven banks A to K are now measured; the Cayley-landscape bank is closed as a program (further residue would need a new bank, not a re-run).
  BANK L (2026-09-05, OBSERVATION MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-ASSESSMENT-0, RESULTS L66150): matched transposition response — every bank-G precedence effect (a marginal difference of 360-renderer means) decomposes exactly into matched in-place swaps of the two roles with the other four fixed: 120 direct adjacent transpositions (Cayley edges) plus 96 / 72 / 48 / 24 swaps at gaps 2 to 5, weights 1/3 and 2/3 (identity verified on the manifest, all 15 pairs). Testable residue (7): whether "HI_D before W" (+5.6 to +16.9 pairs, L66121) is carried by the direct HI_D<->W edge (direct mean at least the non-adjacent mean, sign-consistent over the 120 contexts) or by ordering context (slot, neighbours, gap), and which theta's states carry the gain (SIN_LOW, whose negative-bearing role is HI_D, is the house prior). Zero logits, policy tables plus the L66073 decision matrices. Nominated rung MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0, not preregistered, not launched. Attribution: Artin (the matched adjacent-swap framing and the direct-v-context question), house (the exact decomposition identity, the gap weights, the readout list).
  MEASURED (2026-09-05, VERDICT MATH-CYBER-1-RENDER-ATLAS-TRANSPOSITION-RESPONSE-0, RESULTS L66198): residue (7) closed against the house priors — "HI_D before W" is NOT carried by the direct HI_D<->W transposition: the 120 adjacent swaps are the weakest gap class on all four discovery checkpoints (direct +0.8 to +10.4 v non-adjacent +8.0 to +20.2; gap-5 swaps up to +46.5), mostly exact zeros (67 to 104 of 120), never B-raising on all four, best in slot 0 / 1 / 3 and not the last; the B changes are carried by COS_LOW states flipping (the swap lowers SIN_LOW correctness and raises COS_LOW correctness on 4 / 4 discovery and 8 / 8 fresh, by more on 11 of 12). All four bars fail 0 / 4; fresh reproduces (7 / 8, 8 / 8). The marginal precedence effect is a separation effect read through the binding COS_LOW states, not a generator-level effect. Bank L closed.
  BANK M + MEASURED (2026-09-05, PRE-REG L66224 / VERDICT MATH-CYBER-1-RENDER-ATLAS-NESTED-SWAP-DECOMPOSITION-0, RESULTS L66247): Artin's nested framing (gap -> endpoint slot -> intervening identity and order; 15 cells of 24; sums of squares as exact partition identities, no regression) run on the matched HI_D<->W swaps. Answer: the large gap-4 / gap-5 effects are mostly NOT position effects — the within-cell part holds 57 % to 67 % of the sum of squares on 3 of 4 discovery checkpoints (20001 CANONICAL is the exception at 21 %, its HI_D-first slot carrying 61 %); the gap-5 cell is all-zero, near-constant or spread 0 to 48 depending on the checkpoint; the role placed just after W (gap-4 cell (0, 4)) or just before HI_D (cell (1, 5)) separates sub-cell means by up to 31 pairs. Bars (a), (c), (d) fail on at least 3 of 4, (b) on 2 of 4; fresh reproduces the position-share failure (7 / 8), not the W-last-cell ordering (2 / 8). Attribution: Artin (the nested combinatorial decomposition and the endpoint-v-interior question), house (the SS identities, the role-at-position split, the readouts). Bank M closed; the matched-swap program (L, M) has now resolved the bank-G precedence effect down to cells of 24 and finds it context-carried at every level it can resolve.
  Attribution: Artin (the landscape frame, all eleven banks, the priority ladder and the language fences), house (graph verification, artifact pinning, cost and provenance assessment, frozen conventions).

- **BANKED (2026-09-05): DATA -> WEIGHTS -> FUNCTION ON A REAL OPEN CHECKPOINT LIFECYCLE, WITH CROSS-SCALE TRANSPORT AS THE FALSIFIABLE RIDER** (Artin: the program frame, the K2 Horizon candidate, the renormalization / timeframe analogy stated as analogy only, the cross-scale observable list, the contamination law, the marginal-v-context lesson carried from the render-atlas arc; house: artifact verification, ranking, cost ladder, the zero-training first rung).
  The mapping: a released checkpoint ladder with stage boundaries (K2 Horizon pretrain -> mid_1 -> mid_2 -> post-train, 33 to 68 full-weight tags per size, verified 2026-09-05) is a free DATA-STAGE variable; the weight delta between adjacent stage tags is the WEIGHT-DELTA; oracle-verified house gates at each tag are the CAPABILITY-DELTA. Cross-scale transport = whether normalized structural observables of the stage deltas at 0.9B (depth-relative location, per-module share, stable rank, Hill alpha, singular-vector IPR, delta effective rank) predict the same observables at 3.7B and 7B.
  Measured anchors: none yet. Ledger has zero Pythia / OLMo / checkpoint-lifecycle / IPR entries. Related house results: crest on one vehicle (FINDINGS L1331); MERGE-SPACE independent-birth zero at d64 (FINDINGS L1263-1304); never-score-by-weight-distance law (CLAUDE.md); weight-reader null TENET-W1 (FINDINGS L1596); effective rank falls with capability (RESULTS L2322).
  Honest breaks: stage boundaries confound data with context length (8K -> 32K -> 128K) and unknown per-stage token budgets (IFM publishes none); the 0.9B final stage is teacher distillation, not RL, so its post-train delta is cross-kind; the vocab changes 64K -> 250K between 0.9B and 3.7B; the K2 mixture recipe is not published, so the interventional arm (diet surgery) cannot reproduce IFM's mixture and must be a house-defined continuation at 0.9B full-parameter on the Mac with LoRA-only promotion at 3.7B/7B (rank-constrained deltas are a named confound); the renormalization framing is an analogy with no measured anchor; sigma never transports across widths (house fence), so every cross-scale bar is a paired-within-device rank statistic on normalized profiles, never a raw-number comparison.
  Testable residue: (1) K2-HORIZON-STAGE-DELTA-CENSUS-0 at 0.9B (spec §6 bars a/b/c); (2) the 3.7B transport bar registered before any 3.7B tensor is read; (3) the matched-factorial diet-surgery design (family mass x sequencing x co-occurrence with SS partition identities as in L66247) armed only if (1) and (2) fire; (4) the latent-bridge rescue of the d64 independent-birth zero (affine residual-stream bridge, random-rotation control); (5) a second-vehicle port of the crest protocol with anti-demand and identity-over-aggregates controls; (6) a data-cutoff desk before any 2026 finite-certificate rediscovery task. Assessment: OBSERVATION NEXT-PROGRAM-ASSESSMENT-2026-09-05-0; full text docs/superpowers/specs/2026-09-05-next-program-assessment.md.
  MEASURED (2026-09-05, VERDICT K2-HORIZON-STAGE-DELTA-CENSUS-0, RESULTS L66337): residue (1) closed at 0.9B — instrument qualified (no-op precondition 255 / 255 + identical gate ids, verifier 0 discrepancies, 6,453 s wall), all three bars NO-FIRE, prior (b) backwards (pretraining resembles the first context extension at Spearman 0.854; the two context extensions do not, 0.269). New lineage facts: `_final` tags are bit-identical relabels of the last numbered tag; the specialist merge is a 1.3e-3 relative edit and the distillation 7e-5, both mid-stack MLP, yet the gate rises 33 / 16 / 32 -> 43 / 23 / 40 at the merge tag (which also switches YaRN on; one-seed rider puts +4 of +10 on the config) and is flat under distillation; the distillation delta is the only localized one (IPR 94x uniform). The renormalization analogy remains unmeasured; residue (2) (3.7B -> 7B transport) stays open and needs a per-tag download-census-delete stream (32 GB free) and a harder gate tier (tier 2 floored at every 0.9B tag).
  MEASURED (2026-09-05, VERDICT K2-HORIZON-RESIDUES-0, RESULTS L66404): the merge-tag gate rise decomposes as YaRN config +6 / +6 / +5.5, weights +4 / +1 / +2.5, interaction −4 / 0 / −5 (sub-additive; the L66337 additive split refuted); lag-1 anti-alignment of consecutive deltas is a chain-wide property at the published spacing (21 / 21 within-stage pairs, pretraining −0.30 on every 10k pair, lag 2 null, 50k spacing +0.13), not a boundary artefact. Tags on the hub are mutable (six re-pointed with identical shards during the run): fetch by commit + shard sha. The six-tier gate ladder (OBSERVATION ...-GATE-LADDER-0-CALIBRATION, L66427) is family-bimodal at 0.9B, so a family-level floor / ceiling readout law is frozen before any 3.7B / 7B logit. Residue (2) (3.7B -> 7B transport) remains the open one.
  MEASURED (2026-09-06, VERDICT K2-HORIZON-TRANSPORT-0-DISCOVERY-3.7B, RESULTS L66460): the 3.7B discovery half of residue (2). D1 and D2 fire (lag-1 anti-alignment holds at 100k spacing and grows along training; pretraining and first-extension profiles agree at 0.976); D3 and D4 fail (the localized edit is the first context extension, IPR 12x, not SFT at 3.6x; SFT lowers 3 of 4 open families). The 0.9B distillation-is-localized hypothesis did not transport to the 3.7B SFT boundary. T4 has four candidate cells (B1 prod_diff +6, B2 xexp_int +4, B4 expand2 +3, B5 expand2 −4.5) and books NOT-RESOLVABLE at 7B by the registered six-cell law. Next: wall-clock audit, then the 7B prereg with T1-T5 values frozen from L66460.

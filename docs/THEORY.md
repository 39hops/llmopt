# The grounding map — house laws and their published lineage

Purpose (Artin, 2026-07-19): "the most important part is proving
we aren't just vibing this." Every load-bearing house result,
mapped to the established mathematics/literature it stands on or
extends. House-original contributions marked [ORIGINAL] — those
are the publishable deltas.

| House law / result | Published foundation | Status |
|---|---|---|
| Closed system = (terms, rules, decidable check) | Term rewriting systems (Church-Rosser 1936; Baader & Nipkow 1998) | direct instance |
| Training relaxes to a Boltzmann distribution over the oracle-sculpted loss | Stochastic quantization (Parisi-Wu 1981); SG Langevin dynamics (Welling & Teh 2011); imaginary-time Schrodinger <-> diffusion Monte Carlo (standard QMC) | direct application |
| Gauge law (never score weights by distance; lottery redrawn per birth) | Permutation symmetry of NNs (Hecht-Nielsen 1990); linear mode connectivity / git-re-basin (Ainsworth et al. 2022); ground-state degeneracy (standard QM) | known + [ORIGINAL: measured 5 independent ways incl. mid-lineage redraw; 5th = the Lloyd-Max race 2026-07-25 — the explicitly distance-optimal (MSE k-means) codebook never beats naive uniform at any bit-width, the first DIRECT test with a distance-optimal quantizer] |
| Democracy/oligarchy phases; holographic code | Superposition (Elhage et al., Anthropic 2022); population coding (neuroscience, Georgopoulos 1986) | known frame + [ORIGINAL: NNUE as measured counterexample; phase set by feature/neuron ratio in ONE closed system] |
| Ternary parity at 1.58 bits (wiring thesis) | BitNet b1.58 (Ma et al. 2024); binary-weight universality (ancient perceptron results) | replication at micro-scale + [ORIGINAL: from-birth vs post-hoc bracket 69 vs 24; alphabet-as-lens result] |
| Natural width W* (tokens-per-width ceiling) | Compute-optimal scaling / Chinchilla (Hoffmann et al. 2022) | closed-system analogue + [ORIGINAL: capability COLLAPSE above W* at fixed corpus (400M=30); W* as calculability/learnability balance] |
| W -> inf limit: calculable but unlearnable | Neural tangent kernel (Jacot et al. 2018); lazy training (Chizat et al. 2019) | direct |
| Exposure curve solves_L = 24(1-e^(-eff/k)) | Exponential learning curves (Amari 1993; Hestness et al. 2017 scaling-law lineage) | same family + [ORIGINAL: clade-transfer terms from measured phylogeny; blind L8 prediction passed] |
| Template/warm-birth (+8 solves at ep1, time machine not basin) | Net2Net function-preserving transforms (Chen et al. 2015); mean-field init theory | mechanism known + [ORIGINAL: statistics computed from corpus signature, not copied from a trained net] |
| Function-preserving growth (gen-6 arm B, identity-gate proof) | Net2Net (Chen et al. 2015); progressive growing (Karras et al. 2017) | direct + [ORIGINAL: calculator-sized, template-sprayed] |
| Committee/mass spectrum (frequency -> localization -> robustness) | Feature frequency vs superposition allocation (Elhage 2022 toy models); loss of plasticity at m->inf (Dohare & Sutton, Nature 2024) | consistent + [ORIGINAL: diet-invariant selectivity table across substrates; Schrodinger-mass reading] |
| Tail-dies-first under compression | Long-tail forgetting under quantization/pruning (Hooker et al., "What do compressed nets forget", 2020) | replication + [ORIGINAL: same law at bits/rank/anchor boundaries in one system] |
| Sub-Gaussian weights explain int4-losslessness | Quantization outlier literature (SmoothQuant etc.) inverted | [ORIGINAL reading: kurtosis invariant as the doctrine's cause] |
| Circuit compilation targets the wrong invariant | Tracr (Lindner et al. 2023); grokking circuits (Nanda et al. 2023) | known tools + [ORIGINAL: microstate non-preservation argument from biography data] |
| Verified self-play data engine | Expert iteration (Anthony et al. 2017); AlphaZero (Silver et al. 2017); STaR (Zelikman 2022); MAI-Thinking-1 closed-world envs (2026) | same family; ours oracle-exact |
| Symbolic-integration ML | Lample & Charton 2019 (notation A/B lineage) | anchor for banked work |
| Learned quantization levels (L* contestant) | LSQ (Esser et al., ICLR 2020) | direct plan |
| Prediction pays only where variance lives (starved judges; prior-wash both sides of node cost) | Value-of-information (decision theory); no published twin known for the two-sided engine measurement | [ORIGINAL — candidate paper on its own] |
| Bits-dimension exchange (eff. bits ~ b + 1/2 log2 d) | TWO independent derivations meeting one measured table: interference/concentration-of-measure (quasi-orthogonal directions) AND classification counting (Cover 1965 function-counting / VC dim d+1: b*d >= log2 of demanded distinctions); AM capacity ~2 bits/param (Gardner 1988) independently inside the measured 1.58-4 bracket | [ORIGINAL: the alphabet tournament table as the law's data — 19M ternary 60 < fp32 64 < P2 66; 45M ternary ties then wins cold 73 v 71; B@768 rerun (2026-07-24) CLOSED the file — AMENDMENT: the exchange holds only for COMPLETE alphabets (zero included); binary-at-width degrades with dose (45->28 at 6ep) — width repays resolution, never structure; SECOND CAVEAT (boundary-or-bulk regression 2026-07-25): the exchange form is LOW-BIT-ONLY — raw b+1/2 log2 d with b=32 mis-ranks the crossover grid identically to volume (fp32@768 predicted first, measured last); the law lives at <=~4 bits and needs a saturating b_eff before any raw-bit extrapolation] |
| Ternary = the minimal COMPLETE weight vocabulary ({-1,0,+1} = oppose/irrelevant/support; zero is load-bearing) | BitNet b1.58 (Ma et al. 2024); sparsity-as-feature-selection lineage; hypercube-corner geometry of binary nets | [ORIGINAL: the abstention argument MEASURED — binary crater 54/120 @ 36.73 validity (capability -6, validity -18: silence is structure) + the one-dot portrait (PR/d = 1.000000 every neuron, norms identical to 1.8e-7)] |
| The feedback ladder (verified-bit -> how-wrong -> what-instead -> why) | Potential shaping (Ng et al. 1999); process supervision / step-level reward (Lightman et al. 2023); DPO preference pairs (Rafailov et al. 2023) | known rungs + [ORIGINAL: exact-oracle grading at every rung (size-Phi shaping measured NULL — penalized the legitimate uphill step; distance-to-solved Phi = rung 2b, pre-registered UNRUN; wave-contrast pairs are SOUND preferences, free per wave; hints twice-nulled = the why must arrive as gradient); regret probe AUC 0.914 as the model-can-read-its-wrongness leg] |
| Late-layer plasticity (freeze early layers during online learning: cheaper AND safer) | Surgical fine-tuning (Lee et al. 2022); critical-period/early-layer stabilization (Achille et al. 2019); layer-wise probing lineage | consistent + [ORIGINAL: the control-rod A/B — identical LR-ladder abuse, late-only 71/120 vs full-backprop 57/120, predicted by the measured depth-monotone delta-mass profile (0.194 -> 0.260, layers 0->11)] |
| Split law + its cause (expert decorrelation = INIT DEFAULT preserved; routing MI is where training writes; sparse assignment NOT the cause — soft routing changes nothing) | MoE specialization lineage (Switch, Fedus 2021; DeepSeekMoE, Dai 2024); random-vector near-orthogonality in high dim (concentration of measure) | [ORIGINAL: the three-arm + soft falsification chain (UMOE-1/2, n=2 + n=1) with the init-corr desk check 0.0016; production N3/B4 observation reframed — decorrelation is absence of force, not achievement] |
| Turbulent-medium law + trainable spacetime (residual stream amplifies perturbations by default; a one-line contractivity tax flips it flat at ~zero gate toll) | Lipschitz-constrained nets (spectral norm reg., Miyato et al. 2018; Parseval nets, Cisse et al. 2017); dynamical isometry lineage (Saxe et al. 2014) | known tools + [ORIGINAL: the GRAV chain — influence unscreened/amplifying measured (n=2), unlawful in router coords too, then FLIPPED by training (falloff 0.67->2.24 v 0.26->0.44) with gate -4 and calibration untouched] |
| Calibration is diet-borne (determined verified rows train honesty; noise dilution -> monotone ECE loss toward DOUBT; consistent falsehood predicted for the DK flip) | Calibration of LMs (Kadavath et al. 2022); Kalai & Vempala 2024 (hallucination from underdetermination — the same law's other face) | [ORIGINAL: ECE 0.0049->0.0386 dose-response at f=0/.1/.3 with the DK gap going MORE negative; no-DK baseline ECE 0.0068/AUROC 0.989 on a 0.9M model, replicated across 3 births] |
| The clock-placement law (Fourier/rotational structure marks WHERE the computation runs, not whether the task is solved) | Nanda et al. 2301.05217 (grokking builds Fourier clocks when modular arithmetic is FORCED single-pass); mechanistic-interpretability circuit literature | [ORIGINAL two-sided completion, FOURIER-2b/3 2026-07-31: single-pass shortcut competence -> 276-351/512 periodic neurons; TAUGHT-ALGORITHM chain competence (k=9 acc 0.83) -> 0/512, clock never forms; no competence -> no clock (k=7). Prosthetics-replace-anatomy at the representation level; clock COUNT is exposure-sensitive (351->142 at same acc), presence is the robust readout; GRADED 07-31 (B6, 3 arms): partial competence -> partial clock (51-115/512 at k=8 acc 0.54-0.77), the presence/partial/absent ladder tracks per-modulus competence WITHIN an architecture (cross-arch the count inverts: 0.77-acc arm carries a smaller clock than a 0.63-acc arm — presence/absence is the only cross-arch readout); exact zeta-8 phases NOT adopted even on the periodic diet — teach-don't-impose, strongest form] |
| The delta doctrine (paired same-device same-checkpoint deltas; absolutes are fragile) | Matched-pairs experimental design (standard statistics); our own fp16-near-tie doctrine extended to frontier probes | [ORIGINAL: two headlines killed by two controls in six hours — transfer control (champion 9/24 = every 'gain' was inherited) + device control (same checkpoint 18/24 cuda v 9/24 MPS: near-tie coin flips resolved by hardware); B=16 frontier probes adopted] |
| LLMUE: continuous full-weight learning on oracle-signed self-generated experience; flip-quantized plasticity | Test-time training w/ verifier selection (VDS-TTT, arXiv 2505.19475); TTT/continual-learning lineage | category exists + [ORIGINAL: exact-oracle (sound, not learned) data purity; full-weight + immune-system stability (vs frozen+LoRA); THE FLIP CENSUS — learning counted in discrete quanta (100,884 flips -> +2 proxy), no precedent found] |
| Exact-over-approximate crossover (stay-in-RNS lazy pipeline; estimate-exit/exact-exit split) | Ozaki scheme (Ozaki et al. 2012); Ootomo int8-TC DGEMM line; RNS/CRT (Garner 1959; Szabo-Tanaka); Shewchuk 1997 (expansions, adaptive precision) | known pieces + [ORIGINAL: the composition MEASURED on gaming silicon — zero-rounding GPU matmul verified vs big-int; int8 crossover (exact beats fp64 accuracy at 1.35x wall, tri beats wall 2x); break-even ~6 layers = exact chains CHEAPER than approximate; dd-exit floor 2^-107 identified to the bit; fractional-CRT 17x estimate exit = the propose-verify law appearing inside arithmetic itself] |

| Determinability law (underdetermined rows train hallucination; audit rows for determinability, not just correctness) | Well-posedness (Hadamard 1902 — the target must be determined by the given data); calibrated LMs must hallucinate on underdetermined targets (Kalai & Vempala 2024) | [ORIGINAL: measured WITHIN one model — kin appends re-spelled determinable 96% vs shm appends left underdetermined 47%, same birth, format the only variable; series rung-1's memorized-factorial fill-ins as the mechanism specimen] |
| Decomposition discount (~10x per row; one-primitive kinds saturate at S_max=1 where chains asymptote below it) | Scratchpads / intermediate-computation training (Nye et al. 2021); curriculum learning (Bengio et al. 2009); process supervision (Lightman et al. 2023) | [ORIGINAL: exposure-constant fit — k_efold ~1.9-2.4k rows for primitives vs ~20.8k chain-era; the 15->67->88->98 series arc as the dose-response curve; perfection re-priced ~10-100x in wall-clock] |

## Anchor-paper notes: VDS-TTT (arXiv:2505.19475, Moradi et al., May 2025)

The closest published system to LLMUE; differences banked
factually for paper #4's related work.

Their method: per test query, sample N in {2,4,8,16} candidates
(temperature sampling) -> a LEARNED verifier (math-trained reward
model) scores them -> keep the top candidate only above threshold
tau (0.99 easy benchmarks / 0.9 AIME) as a pseudo-label -> SFT on
it, updating LoRA ADAPTERS ONLY (rank 128; rank 8 low-resource;
q/k/v/o + MLP), base frozen; iterate across test batches.
Results: up to +32% relative (GSM8K/MATH-500/AIME/AMC; Llama-3.2
1B/3B, R1-Distill-Qwen-1.5B, Llama-3.1-8B); AIME 0.54% -> 4.22%
on the 1.5B. Own stated limits: verifier is math-trained and
fails off-domain (code/QA); no verifier calibration or soundness
analysis given.

| Axis | VDS-TTT | LLMUE (here) |
|---|---|---|
| Verifier | learned scorer, threshold tau | exact symbolic oracle (sound by construction) |
| Label status | pseudo-labels (their term) | signed rows (wrong label impossible) |
| What updates | LoRA adapters, base frozen | full weights |
| Stability mechanism | touch-almost-nothing | two-tier gates + snapshots + rollback |
| Substrate | continuous fp | fp AND discrete (flip-quantized; census: 100,884 flips -> +2 proxy) |
| Data regime | general pretrained model, mixed diet | closed-system-native, single grammar, vocab 40 |
| Domain reach | general (verifier fragile off-domain) | domain-locked, domain-perfect |


## THE ABSORPTION LAW: absorbed fraction ~= c / LR (c ~= 2.8e-9)

Measured leg (house, 2026-07-21/22, four decades, same model/food):
fraction of fp32 updates where w + delta == w exactly — LR 1e-4:
0.0007% / 1e-5: 0.030% / 1e-6: 0.278% / 1e-7: 2.790%. x10 per
decade to within read noise; the constant c ~= 2.8e-9 makes the
law predictive (e.g. 1e-8 -> ~28% of updates silently discarded).

Derivation (why 1/LR exactly): an update is absorbed when
|delta| < ulp(w)/2. With delta = LR * g and the gradient/weight
magnitude distributions fixed by the model+food (they don't move
with LR over a burst), the absorbed mass is P(|g| < ulp(w)/(2*LR))
— the CDF of |g| evaluated at a threshold ∝ 1/LR. The x10/decade
observation says |g| has a ~flat density over these four decades
(log-uniform tail), so the CDF is linear in the threshold: fraction
∝ 1/LR. Citation leg: this is loss-of-significance / swamping in
floating-point summation (Higham, *Accuracy and Stability of
Numerical Algorithms*, ch. 4; Kahan 1965 compensated summation is
the classical fix; mixed-precision master weights — Micikevicius
et al. 2018 — is the modern one, our fp64-masters arm B measured
it recovering ~5x flips at 2.5e-6).

Consolidation — what this law now explains/connects:
- **LR and precision are ONE knob** (schedule law corollary): what
  matters is the update-to-ULP ratio LR*|g|/ulp(w). Halving LR and
  adding one mantissa bit are the same move. "Higher precision"
  and "lower LR" are not two design axes; bits(needed) ~
  log2(w/(LR*g)).
- **Why low-LR regimes (metabolism/GRPO at 1e-6..2.5e-6) are the
  fp64-master regime** and birth (3e-4) is not: absorption at 3e-4
  is ~1e-5 of updates — noise; at 1e-6 it's 0.278% concentrated in
  exactly the small-gradient (= converged, = subtle) directions.
  Absorption is a low-pass filter on learning: it deletes the
  FAINT signals first.
- **Rarity link**: rare-shape rows produce small, infrequent
  gradients — the first mass to fall under the threshold. The
  fp32-vs-ternary rarity split (episodic memory needs resolution)
  is the same mechanism one level up: quantization is absorption
  with a coarser ULP. Ternary's absmean threshold IS an absorption
  threshold; the rarity-routed-precision riff is "route the faint
  signals around the filter."
- **Slow-leak kinship**: absorbed updates are invisible per-step
  and cumulative in effect (the missing mass never arrives) — same
  threshold-blindness class as tripwire slow leaks and the
  ration-drift failure; absolute anchors / master accumulators are
  the shared fix.

Rule check: measured leg = four-decade house scan + arm B recovery;
citation leg = Higham/Kahan/Micikevicius. Row stands.

## Paper candidates (need: replication runs, related-work rigor)

1. **"The Closed-System Equation"** — signature + width determine
   the model up to gauge; calculator, template, warm-birth,
   scaling invariants, perfection price. (The flagship.)
2. **"Alphabet as Lens"** — ternary-from-birth parity, no latent
   polarization, tail economics, the tournament (once run).
3. **"Two Engines, One Oracle"** — the axiom/sympy fair fight
4. **"LLMUE"** — sound-verifier test-time learning + the flip
   census (anchor: VDS-TTT; the discrete-plasticity result is the
   novel core).

Rule for this file: no row without a measured house result AND a
real citation. Rows are removed if either leg fails scrutiny.

| The interaction law, DIMENSION-SPLIT form (2026-07-26 amendment): diversity-per-step binds single-pass training, and the recoverability of composition damage depends on WHICH dimension carries it — epochs buy back BATCH composition (gradient correlation, washed out by revisit averaging) but NOT CONTEXT composition (packed contexts change the learned task; revisits cannot un-teach a task) | Small-batch/gradient-diversity generalization (Keskar et al. 2017; gradient diversity, Yin et al. 2018); example packing (Krell et al. 2021); In-Context Pretraining (Shi et al. 2023 — reports GAINS from related-context packing at LLM scale: a live tension with our micro-scale sign, noted not resolved) | [ORIGINAL: the batch 2x2 (53/57/45/45: -12 at 1 pass, ~0 at 3ep) PLUS the format ladder's 3E column (context packing still costs at 3ep: randpack -5, sim-max delta -16, traces -21; zero rank flips across schedules); Muon failed to buy diversity in-optimizer (10/34 v 45); similarity-MAX packing worse than random in-context both schedules (-17/-11) — banded-delta form untested] |
| Opposition is the computational primitive (per-weight sign is load-bearing; zero and resolution are secondary) | Dale's law + E/I balance (neuroscience); BitNet's ternary {-1,0,1}; excitatory-only nets' known weakness (Parisien et al. 2008 sign-constrained learning) | [ORIGINAL: the full sign-granularity ladder in ONE system — per-weight sign 54 (S4, no zero) / channel sign 0 at any LR (Z1S + hot) / no sign 0 (Z1); Dale's-law channel signs buy training STABILITY (loss 27.7 -> 1.10) and zero capability; + the phase contrast: sign symmetry holds exactly in democracy crystals, breaks toward the readout in the NNUE oligarchy] |
| Retention at low dose is free (metabolic feeding purely additive; no measurable forgetting at LR 1e-5 / 17 steps / 120 min) | Catastrophic-forgetting lineage (McCloskey-Cohen 1989; Kirkpatrick EWC 2017) as the contrast class; sparse-replay stability results | [ORIGINAL: dense per-level retention columns every ~5 min for 120 min, ZERO decay in any column INCLUDING isolated clades, while resolution rose 6->10/100 — the forgetting literature's tradeoff does not bite at this dose; tau unfittable (no decay exists to fit)] |
| Speed doctrine, two-tier (lossless levers always-on; priced-loss substrates for exploration; full-fat only for records) | Mixed-precision training (Micikevicius 2018); proxy-task/small-scale search (mu-transfer lineage, Yang et al.) | [ORIGINAL: the priced substrate — d256 at -5 solves with measured per-level sigma, adopted via an explicit gate (-27 v -3 bar); the sqrt(2)-sigma decision rule prices every cross-cell claim; lossless tier measured: KV 3.4x, TF32 parity, grad-ckpt] |
| Symmetry is a third compression axis (weight-sharing), orthogonal to bits and dimensions: trained dense crystals RETROFIT into group-commutant structure (complex/quaternion/Z2/circulant) at sublinear capability toll, though SGD never chooses symmetry spontaneously | Group-equivariant CNNs (Cohen & Welling 2016 — symmetry imposed at birth); circulant projections in deep nets (Cheng et al. ICCV 2015 — dense->circulant at training time); complex/quaternion networks (Trabelsi et al. 2018; Parcollet et al. 2019) | [ORIGINAL: the RETROFIT direction — project a TRAINED crystal onto the commutant + 1 warm epoch: toll 2x:-1, 4x:-4, 8x:-6 of 65 (S1-S4, 2026-07-28); anti-mass null at every group (no spontaneous symmetry); holography edge structure-dependent (Z2 49 v complex 57 at equal mass); commutant locally stable under lambda=0, weakening with group size; exact embedding at 2x width gates exactly 65] |
| Capability is compression-axis-asymmetric: at a fixed diet the function lives in ATTENTION WIDTH (sharp cliff), while MLP capacity is nearly pure slack (no cliff down to inverted SwiGLU), and ALL compression axes (bits, sharing, bands, tiers, ffn) spend from one shared slack pool that vanishes at the width floor | FFN-as-key-value-memories (Geva et al. 2021 — MLP stores retrievable patterns, consistent with MLP-as-slack under a small closed diet); attention-only transformers (Sukhbaatar et al. 2019 all-attention layer); lottery tickets (Frankle & Carbin 2019 — wide-parent reachability, cf. projected tier beating native narrow) | [ORIGINAL: the paired-axis anatomy in ONE system — d-cliff 56->48 = -13 v ffn 224->48 = -4 (no knee, n=1/point cuda); width floor d56=d64 replicated n=3 both endpoints; free-compression point present at d256 (-2) and GONE at the floor (-9); matryoshka tier free at d256, -6 at floor; projected eighth-tier 52 BEATS native d48 44 (2026-07-29)] |
| The precision knee is a sigma-ratio constant: quantization grids are priced in weight-sigma units (free while grid <~ 0.5 sigma, biting by ~1.0 sigma), so denominators must scale ~sqrt(d) and every Q fence is width-bound | Absmax/blockwise scaling in LLM.int8 (Dettmers et al. 2022) and quantile/NF4 codes (Dettmers et al. 2023 QLoRA) — both normalize the grid to the weight distribution before coding; classical SQNR theory (grid-to-sigma ratio governs distortion) | [ORIGINAL: one constant measured across three crystals and two widths — d56 Q=16 free (0.48 sigma) / Q=8 bites (0.96); d256 Q=64 EXACTLY free 65/65/65 (0.25) / Q=16 bites (1.0); 19M knee Q(16,24] = 0.65-1.0 sigma (2026-07-29); unifies the rational-snap knee with the width axis; EXTENDED geometry-blind by the polar-split cell: Cartesian and polar grids at matched measured bits/weight cost equal capability (polar 4.75-bit 61 v uniform 4.62-bit 60 on cplx_none; 8 angle bins free) — one scalar governs the axis (2026-07-29 day)] |
| Calibration-free quantization is an AT-CAPACITY property: born-crystal weights are max-entropy at their scale (code entropy = Gaussian capacity to <1%), so closed-form sigma-allocation matches Hessian-based methods at matched bits, MSE-optimal codebooks buy nothing, and a zero-inference capacity meter (span-bits minus code entropy; kurtosis) predicts from disk whether a model is in the regime — web-dense LLMs are not (outlier tails = exploitable structure; use max-anchored/calibrated), MoE routed experts are (DeepSeek-V3 kurt 3.07) | Rate-distortion/max-entropy classics (Shannon 1948; uniform quantizer near-optimality at high rate, Gish & Pierce 1968); GPTQ (Frantar et al. 2022), AWQ (Lin et al. 2023), HQQ (Badri & Shaji 2023) as the calibrated contrast class; MoE expert specialization (Fedus et al. 2022) | [ORIGINAL: the C-series (2026-07-29) — packed crystals gate at parity at ~5 bits/wt with entropy within 1% of capacity; GPTQ/AWQ/HQQ with real Hessians tie the closed form on crystals and beat it 33x on Qwen-0.5B (mechanism: fixed-width sigma grids pay bits to the worst outlier); Lloyd-Max k-means tied uniform on crystals 07-25; capacity meter separates the classes from disk (crystals 0.96-1.61 bits / experts 2.33 / web-dense 3.6-3.9); cross-device integer determinism measured (2 seeds)] |
| Attention heads are intrinsically incompressible relation channels (deletion catastrophic at every width; count and rank tolls do not train away), while MLP capacity and attention rank are provisioning-dependent slack | Head-pruning literature finds MANY prunable heads at BERT/WMT scale (Michel et al. 2019; Voita et al. 2019) — a live tension: their models are over-provisioned for their tasks, ours sit near capability floors on a closed diet; induction-head/circuit work (Olsson et al. 2022) for heads-as-discrete-mechanisms | [ORIGINAL: the width-controlled census — single-head deletion craters BOTH d56 (4-17 of 63) and d256 at 4x room (11-30 of 65); heads-2 at fixed width -7; born-r32 -8 = post-hoc -9 (no train-through heal); v ffn flat 224->48 and rank 75% free at d256 (2026-07-29); HARDENED by the head-tension cell: born-8-heads d64 gates 58 (heads x2 free) yet ALL EIGHT essential (best deletion -11, worst -37) — the provisioning frame falsified; load spreads to fill provision (2026-07-29 day); REFINED by the 64-cell autopsy: essentiality is CELL-SPARSE — ~13/64 (layer,head) cells load-bearing (one deletion -34), ~51/64 slack; every index-column contains a critical cell, which is why column deletions always cratered (2026-07-29 eve)] |
| Fixed-point training fails by UPDATE STARVATION, not early flooring: large early gradients punch through any quantizer, then convergence strangles as updates shrink below the weight grid — the fix is a wide weight accumulator (store wide, compute narrow), lr DECAY actively deepens the failure, and the required width grows with model scale (shift 8 suffices for an FFN mini, 12 for a full block) | Limited-precision training (Gupta et al. 2015 — 16-bit fixed point needs stochastic rounding for exactly this floor); high-precision weight copies in low-precision training (Courbariaux et al. 2015 BinaryConnect; Micikevicius et al. 2018 fp32 master weights — our Q_w is the integer-exact form of the master-copy idiom) | [ORIGINAL: the mechanism measured at 3 scales in ONE exact system (2026-07-31/08-01): R2 toy (lr 1e-3 floors to zero; runs only at 1/20), R3a mini (SHIFT=0 nz-updates decay 0.999->0.014, loss stalls 3 orders above SHIFT=8; 8 ties 12), R2b block (plateau at SHIFT=8 revived by SHIFT=12 to 10233 still-falling; BOTH decay arms hurt — starvation, not lr-bounce); deterministic to the digit, C++-replicated cross-lab, so the curve is exactly reproducible] |
| MoE routers are DOMAIN-ORGANIZED at deployment: per-domain expert coalitions differ beyond within-domain resampling noise, coalition distance tracks corpus distance, and demanded-coalition COVERAGE (not expert count) is the load-bearing capability quantity — with at least one intermediate regime between dead (44.7% coverage -> 0/120) and healthy; the beats-full "crest" of masked routing is domain-specific (math yes at 7/7 seeds, mechanics null), so coalition structure is general but the interference the mask removes is not | Expert specialization in MoE (Jacobs et al. 1991 adaptive mixtures; Shazeer et al. 2017; ST-MoE, Zoph et al. 2022 — encoder experts specialize by token type); domain/task expert clustering observed in modern MoEs (OLMoE, Muennighoff et al. 2024; DeepSeekMoE, Dai et al. 2024 — fine-grained expert specialization as a design goal); pruning-by-frequency as a known-lossy importance proxy (magnitude/frequency vs importance, LeCun et al. 1990 OBD lineage) | [ORIGINAL: MOE-GT-2 (2026-08-04) — decode-only coalition Jaccard math-phys 0.804 vs math-code 0.543 against split-half nulls 0.930/0.871/0.653, ordering REGISTERED from corpus token overlap (0.329 vs 0.097) before the runs; cross-masks at ~78-80% coverage score 19-21/120 vs random-45%'s 0/120 (R6) and own-crest 76-87 (R4/R5); crest transport to mechanics NULL (+3 pooled vs +7 bar, underpowered fence booked); crest location survives an unspent seed (argmax 45.3%, +22). Fences travel: one seed per domain, Jaccards are upper bounds (tie-fill), functional-vs-topical claim UNRESOLVED (diagnose-checker artifact, GT2-REVIEW)] |

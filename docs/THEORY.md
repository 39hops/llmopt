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
| Gauge law (never score weights by distance; lottery redrawn per birth) | Permutation symmetry of NNs (Hecht-Nielsen 1990); linear mode connectivity / git-re-basin (Ainsworth et al. 2022); ground-state degeneracy (standard QM) | known + [ORIGINAL: measured 4 independent ways incl. mid-lineage redraw] |
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
| Bits-dimension exchange (eff. bits ~ b + 1/2 log2 d) | TWO independent derivations meeting one measured table: interference/concentration-of-measure (quasi-orthogonal directions) AND classification counting (Cover 1965 function-counting / VC dim d+1: b*d >= log2 of demanded distinctions); AM capacity ~2 bits/param (Gardner 1988) independently inside the measured 1.58-4 bracket | [ORIGINAL: the alphabet tournament table as the law's data — 19M ternary 60 < fp32 64 < P2 66; 45M ternary ties then wins cold 73 v 71; B@768 test wounded by data confound, rerun queued] |
| Ternary = the minimal COMPLETE weight vocabulary ({-1,0,+1} = oppose/irrelevant/support; zero is load-bearing) | BitNet b1.58 (Ma et al. 2024); sparsity-as-feature-selection lineage; hypercube-corner geometry of binary nets | [ORIGINAL: the abstention argument MEASURED — binary crater 54/120 @ 36.73 validity (capability -6, validity -18: silence is structure) + the one-dot portrait (PR/d = 1.000000 every neuron, norms identical to 1.8e-7)] |
| The feedback ladder (verified-bit -> how-wrong -> what-instead -> why) | Potential shaping (Ng et al. 1999); process supervision / step-level reward (Lightman et al. 2023); DPO preference pairs (Rafailov et al. 2023) | known rungs + [ORIGINAL: exact-oracle grading at every rung (shaped Phi live; wave-contrast pairs are SOUND preferences, free per wave; hints twice-nulled = the why must arrive as gradient); regret probe AUC 0.914 as the model-can-read-its-wrongness leg] |
| Late-layer plasticity (freeze early layers during online learning: cheaper AND safer) | Surgical fine-tuning (Lee et al. 2022); critical-period/early-layer stabilization (Achille et al. 2019); layer-wise probing lineage | consistent + [ORIGINAL: the control-rod A/B — identical LR-ladder abuse, late-only 71/120 vs full-backprop 57/120, predicted by the measured depth-monotone delta-mass profile (0.194 -> 0.260, layers 0->11)] |
| The delta doctrine (paired same-device same-checkpoint deltas; absolutes are fragile) | Matched-pairs experimental design (standard statistics); our own fp16-near-tie doctrine extended to frontier probes | [ORIGINAL: two headlines killed by two controls in six hours — transfer control (champion 9/24 = every 'gain' was inherited) + device control (same checkpoint 18/24 cuda v 9/24 MPS: near-tie coin flips resolved by hardware); B=16 frontier probes adopted] |
| LLMUE: continuous full-weight learning on oracle-signed self-generated experience; flip-quantized plasticity | Test-time training w/ verifier selection (VDS-TTT, arXiv 2505.19475); TTT/continual-learning lineage | category exists + [ORIGINAL: exact-oracle (sound, not learned) data purity; full-weight + immune-system stability (vs frozen+LoRA); THE FLIP CENSUS — learning counted in discrete quanta (100,884 flips -> +2 proxy), no precedent found] |

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

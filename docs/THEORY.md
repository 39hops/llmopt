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

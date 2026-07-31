# FINDINGS — the curated research results

One entry per major measured finding, newest program last. Every
line traces to a dated booking in `RESULTS.md` (the append-only
verdict ledger, which also holds the nulls, retractions, and
amendments this file omits). Dates are 2026. Rule of the house:
pre-registered predictions, paired arms, oracle-verified scoring,
honest negatives booked with the same care as wins.

## The derivation engine (07-06 → 07-12)

- **73.6% → 360/360 in four days** on held-out, string-seeded,
  sympy-oracle-verified calculus — every gain a named component
  (search wisdom, autopsy-derived operator rules, a theorem from
  1835). The benchmark is closed. (07-09)
- **The component taxonomy**: ranking moves is *grammar* and fits
  in a bigram dictionary (beats the fine-tuned 0.5B in-search at
  zero cost); *confidence* — knowing when you're sure — is the
  GPU's entire job (+15 solves, the largest premium measured);
  width partially substitutes for confidence. (07-07/08)
- **The FA Law: verified speed is intelligence.** At fixed wall,
  cheap nodes convert to solves — killing one heurisch call in the
  verifier (~2000x cheaper edges) bought beam width, which bought
  capability. Routed engine: 141/150 @ 167s vs 337/429s pure arms.
  (07-11)
- **Self-teaching is a step function** to the reachable-set
  ceiling: the entire climb happens in ONE round of expert
  iteration, from either a mature or tabula-rasa start; only new
  operators move the ceiling, and the limit of self-teaching is
  the limit of self-checking. (07-08)
- **The magic detector**: Liouville 1835 as integration's
  Gottesman-Knill — certified non-elementary integrands are
  provably dead within the operator closure; the only prune in
  the repo with zero false positives. (07-08)
- **Rule subsumption**: the 13-rule library rests on four
  generators (power, substitution, the linear-algebra move,
  constants); by-parts — the textbook's crown jewel — is fully
  subsumed by one matrix solve. (07-08)
- **The engine beats sympy on every level** (L5 100%, L6 59v56,
  L7 56v42), using sympy as a gated subcontractor and sympy's own
  differentiator as judge. (07-11)
- Imitation can't beat the teacher at the teacher's own job, but
  it can make the teacher ~4x cheaper (the policy gate); one
  DAgger round lands the imitator exactly at its teacher. (07-09)

## The micro-model program (07-15 → 07-20)

- **Priors vs drag**: a from-scratch 19M model with a 45-token
  hand-built vocabulary reaches ~20x the step validity of a
  pretrained 0.5B on closed-system math at 1/26th the parameters —
  pretraining priors were overwhelmingly drag. (07-15)
- **THE GATE, NOT THE LOSS** (7+ independent instances): training
  loss is blind to capability holes of 10+ points (packing),
  precision debits, and diet damage — cross-entropy weights by
  frequency, so rare-but-load-bearing capability contributes
  ~nothing to loss *by construction*. Oracle gates are the only
  honest score. (07-16 onward)
- **Generational training**: lineages are not patched, they are
  REBORN on accumulated verified experience — the corpus is the
  organism, weights are disposable. Then sharpened: **growth beats
  rebirth** (+5 at equal epochs) — template-sprayed warm growth is
  the standard capacity move. Rebirth without novelty just
  reshuffles the same ceiling. (07-17/19)
- **Natural width W\***: capability peaks at a corpus-dependent
  width (~45M for the gen-4 chain corpus; ~19M for token-light
  primitive federations) and *actively degrades* beyond it
  (tokens-per-width starvation; 400M gates 30/120 with L4=0).
  Full width curve measured 0.5M→400M: 38→57→65→65→**69**→65→49→30,
  floor W_min ~8.4M. (07-16→07-25)
- **The closed-system equation** (v1): capability is a computable
  saturating function of corpus signature × width —
  solves_L = 24·(1−exp(−eff_L/k(W))) with phylogeny transfer
  terms; blind-tested on L8 before farming. Perfection is
  purchasable at finite corpus (~650k effective rows, cut ~10x by
  the decomposition discount). (07-18/19)
- **The math phylogeny**: rule-bigram clustering finds technique
  lineages, not difficulty levels — and the isolated ansatz clade
  (no descent path) explains why L4 is the first casualty of
  every starvation mode (under-width, under-feeding, packing).
  Compression (gzip-NCD) independently isolates L4's dialect.
  (07-17)

## The crystal laws (07-17 → 07-27)

- **Democracy**: closed-system crystals are sub-Gaussian
  (kurtosis 2.4±0.2, a scale-invariant 19M→400M), isotropic
  (near-perfect phase ring), full-rank, expander-topology (no
  cliques — what a brain would build if wires were free), and
  **maximum-entropy**: the ternary census reads exactly 1.58
  bits (the 3-state max); the integer twin reads 6.65 bits/weight
  vs the Gaussian capacity formula log2(σ√(2πe)) = 6.755. All
  structure lives in the arrangement; the marginals are full.
  (07-17→07-27)
- **Holography**: remove the family-selective committee — nothing
  lost; read only it — almost everything present. Definitions are
  locatable as preferences, not dependencies: "the power rule is
  not a place, it is a chord." (07-17)
- **The lottery is redrawn, the statistic conserved**: committees
  re-form with zero index overlap across births AND within one
  lineage's extended training; per-family selectivity strength is
  a property of the corpus, not the network. The microstate is
  the one thing the system does not preserve. (07-18)
- **Never score weights by weight distance** (7 legs, sharpest:
  gauge-aligned distance measures ANCESTRY — three same-function
  seed births sit orthogonal (√2) even after Procrustes, while
  27-solve-different same-init models sit in a tiny ball).
  Distance is a lineage detector, nothing more. (07-06→07-26)
- **The alphabet is a lens, not an attractor**: underneath every
  trained alphabet (fp32, ternary STE) lives the same smooth
  Gaussian object; quantization is the projection at inference.
  Corollary doctrine: quantize the notches, never the axes —
  bits are redundant (int4 lossless at 8x), dimensions are sacred
  (rank-128 bleeds, rank-32 is dead). (07-17/18)
- **Templates are a time machine, not a better basin**:
  statistically-calculated init (+8 solves at 1 epoch) converges
  to the same destination ~1 epoch sooner; the corpus fixes the
  ceiling. Teach invariance, don't impose it (permutation
  augmentation 88.4% > canonical sorting 82.4%; prefix notation
  −3 with a mechanism; hints lose to no-hints). (07-18, 07-21+)

## Alphabets and precision (07-17 → 07-27)

- **Ternary parity**: a 1.58-bit from-birth model ties (then
  beats cold-for-cold) its fp32 twin — discrete lattices learn
  slower, not worse; the ~10MB crystal is the lab's second-best
  model. Discrete plasticity is real: 100k whole-weight flips
  bought +2 solves in 20 minutes, photographable. (07-18/20)
- **The alphabet tournament** (matched recipe): B 54 / T 60 /
  M4 61 / M5 62 / **P2 (3.17-bit ladder) 66 — beats fp32's 64**.
  Zero is load-bearing at 1 bit (silence is structure; binary at
  width is dose-*degrading*, no dimension budget buys back
  silence); at born-2-bit, resolution-absence is the fatal axis,
  zero-absence tolerable (S4 58). Alphabet choice is
  width-dependent (W* ~ 1/bits). (07-20→07-26)
- **Rotation does not pay** — five independent nulls: complex
  ties real on math (63v64), on the phase-carrying ZX grammar
  (32v31, the alphabet-follows-domain hypothesis died clean), on
  phase-free NNUE features, in the weight-FFT euler read, and at
  2.4x scale. Complex interiors are free, never profitable;
  linear models contain rotational ones (WJ=JW is a constraint)
  and training doesn't choose it. (07-26/27)
- **Precision doctrine** (closed, with its honest scope): birth
  precision is a non-factor above TF32 (bf16↔fp64 bracketed
  indistinguishable; the dynamical cliff sits at 8–10 mantissa
  bits and packing — not bf16 — was the −13); online/low-LR
  learning needs fp64 masters (the absorption law: fraction of
  updates lost to rounding ≈ 2.8e-9/LR across four decades);
  nothing below fp64's floor reaches the deployed function
  (exact-vs-fp64 masters: bit-identical, 132,566 = 132,566
  flips). Amended 07-27: all claims hold *above instrument
  sigma*; the sole named retest is one exact-mode gate (E3).
  (07-17→07-27)

## The exact stack (07-23 → 07-27)

- **Zero-rounding GPU matmul, verified**: block-aligned int8
  slicing on tensor cores reproduces exact big-integer products
  bit-perfectly; int8-exact is 4.5x more accurate than native
  fp64 at 1.35x its wall, and 2x *faster* than fp64 at the 1e-9
  grade. There is no finite-precision format left to race — any
  format either rounds or wastes; the exact pipeline is always
  precisely the answer's width. (07-23)
- **Stay-in-RNS**: residue-channel chains defer all carries —
  exact computation beats *approximate* fp64 past ~6 layers
  (13 vs 43 ms/layer, and fp64 was wrong). Exactness is a speed
  lever, not a capability lever. (07-23)
- **The rational twin**: every 2-D weight snapped to the best
  fraction p/q — parity at Q=64 on two crystals/devices/widths;
  minimal parity twin Q=48 (~10.5 bits/weight); the crack is a
  cliff, and **the sensitivity wall is a per-crystal property**
  (the same 3.7e-3 error that halved a Mac 19M cost a 45M one
  point). (07-27)
- **The integer twin**: W = P/512 with integer P (int16-range) at
  full parity — the forward pass is an integer GEMM over one
  shared denominator; entropy ~6.7 bits/weight, losslessly
  compressible to ~42MB from 200MB. (07-27)
- **The snap anatomy — where the wall actually lives**: a full
  coarse snap changes 2 of 2,512 next-token decisions, both at
  logit margin 1.6e-4 (median 8.9) — weight noise only decides
  pre-existing coin flips, and the gate cliff is flips/token ×
  chain length. Precision below the near-tie scale buys nothing;
  the capability lever is near-tie density (calibration) and
  verification, not digits. (07-27)
- **Born-rational**: training *on* the exact lattice from step 0
  (STE, weights = s·p/q, q≤6) is **capability-free across two
  seeds** (+5, then −2: parity; the seed-1 win demoted by its
  own replication) — every weight an exact small fraction from
  birth at zero measured cost vs fp32, deploy tax ~1, snap-back
  lossless by construction. Born-on-lattice still beats post-hoc
  snapping ~2x in bits (5.5 vs 10.5). Completes the set:
  exactness is free at birth, inference (integer twin), and the
  training step (disagreement #2) — a determinism/format lever,
  never a capability lever. (07-27/28)

## Data and diet laws (07-22 → 07-24)

- **The determinability law**: underdetermined rows train
  hallucination — 96% vs 47% within one model, format the only
  variable; audit rows for determinability, not just correctness.
  Confidently-wrong generation is reproducible three ways
  (underdetermined data, sign-only quantization, hot
  excitation-only training). (07-22+)
- **The ladder law / decomposition discount**: one-primitive
  emissions train to ~100% where chains sit at 15% — ~10x cheaper
  learning per row; primitives saturate at S_max=1. Series arc
  15→67→88→**98%** purely by re-spelling rows. Operand complexity
  is a second axis (emission size gates what volume can't buy).
  (07-22/23)
- **The bridge law, strong form**: NOTHING transfers without
  demonstrated shared steps in context — not co-residency, not
  inference-time composition, not cross-vocabulary reach (the
  desert test: 128 proposals, 0 reaches for a resident skill).
  Transfer that does exist lands level-locally where the shared
  steps apply. Territory = vocabulary (a grammar the tokenizer
  can't spell is unreachable at any capability). (07-24)
- **Exposure rations**: adding grammars dilutes resident exposure
  share (L3 dips repaired by 2x maintenance rations); diet
  exposure SHARE, not just content, is a control knob. Naked
  textbook forms fail where textured generator states succeed —
  "common" means common-in-corpus. (07-24)
- **Contamination scars** (three incidents): stable string seeds
  only; exclude=-guarded splits; widen the generator space before
  trusting a split; audit miracle-shaped numbers first. (ongoing)

## Federation of grammars (07-22 → 07-24)

- **One 19M crystal, five grammars at spec simultaneously** —
  math gate in-band, series 99.2% (beats its own specialist),
  energy conservation 100.0% (a physical law learned to
  saturation, as decidable arithmetic), poly 89%. The union
  equation's interference coefficient reads ~zero below W*;
  geometry is grammar-free (universality at 3 grammars). (07-24)
- **Width does not pay for primitive federations**: token-light
  rows keep W* at 19M-class; the same discount that makes
  federations cheap to farm makes them cheap to feed. (07-24/25)
- **ZX/graph grammar**: first graph-language crystals work at
  first contact (parsefail 0/480, boundary-anchored
  serialization); capacity was never the constraint (2.4x width
  bought zero); the 19M seed fence is sd ~4.2 — every 19M ZX
  claim must clear it. (07-26/27)

## Learning dynamics (07-16 → 07-26)

- **The packing hole**: token-budget packing costs ~10-13 solves
  at matched-or-lower loss, at two widths — and at matched step
  counts it does 2x the damage of naive step starvation
  (batch-composition mechanism). --fast never ships without
  --nopack. (07-16→07-26)
- **The streaming 2x2**: the epoch is load-bearing (−8 at the
  best single-pass schedule); cooldown is real (+4); batch
  homogeneity costs −12 at one pass and ~0 with revisits —
  diversity-per-step is the binding resource of single-pass
  training. Clade-gated self-pacing with budget recycling
  recovers ~60% of what epochs buy. (07-26)
- **Closed-system RL edits the policy, not the mind**: an entire
  2.4x verified-RL climb wrote ~6% of one SFT run's weight
  movement (CKA 0.9998 everywhere but the last layer); RL's
  durable product is the *mined corpus*, not the weights.
  Reward hygiene learned the honest way (identity-rewrite hack;
  verified AND distinct at every learning layer). (07-15/17)
- **The exchange is load-bearing and bidirectional**: a model
  cannot teach itself what it cannot sample (150 min
  self-practice = +1 wall; 23 engine demonstration rows = +4 in
  ten minutes), and the model's only two wins were the engine's
  only two losses (union 12/12). Model logs walls → engine farms
  them → model eats chains. (07-23)
- **Late-layer metabolism** is cheaper AND safer (+14 gate points
  of erosion resistance at 60% less backward); flip persistence
  distinguishes structure from churn (flip-backs ~5%, monotone
  net). (07-21/23)

## The axiom federation (07-18 → 07-27)

- **Two oracles auditing each other** find what neither finds
  alone (the first parity audit caught one real bug on each
  side); ten+ consecutive independently-adjudicated clean row
  batches; the hybrid solver mints verified rows at ~6x sympy's
  rate (443/480 qualification at 5x speed). (07-18→07-20)
- **FX-V1 exact inference**: fixed-point NN forward with declared
  rounding at five certified sites — two zero-shared-code
  implementations bit-identical; exact mode is already *faster*
  than rounded fp32 at rung 1. The E4 parity arc closed with
  every disagreement named (budget censoring, ply semantics, a
  domain fence — no soundness exposure either side), and the
  house's own bigram prior now guides their engine (3→0 decided
  misses, byte-pinned by sha). (07-27)
- Substrate fence doctrine: every cross-repo replay names file,
  row count, sha256, and full arm config — two instrument
  mismatches caught in one day. (07-27)

## Methodology (the meta-findings)

- **Pre-registration + paired arms + controls beat every
  predictor.** Two would-be headlines killed by two controls in
  six hours; three instrument bugs in 24h all caught by control
  arms, none by headlines; both the house's and the reviewer's
  on-record bets have each been booked wrong — the protocol is
  the asset. (07-21→07-25)
- **Instrument fences**: never compare gates/probes across
  devices (2x device dependence measured at the frontier); sigma
  never transports across devices/widths (2.5 cuda / ~1.0 d256);
  single-seed pairs need ≥5 solves; fp16/near-tie flips at
  margins ≤~0.02 are ties, not bugs. (07-21+)
- **Auditors must match the verifier's semantics** — four
  auditor-was-the-bug incidents, each caught by provenance
  tracing; wall-censored rows are not facts (censored ≠ False,
  and the doctrine transports across engines); killed workers
  must stream their rows out or the killed class is invisible.
  (07-20→07-27)
- **L4 is the canary**: the first compositional level is where
  the variance lives (158 RESULTS mentions vs ~70 for
  neighbors) — every intervention expresses there first; read it
  first. Companion law: prediction pays only where variance
  lives. (07-27)

## The calibration program (07-28, one night)

- **Near-tie density is a measurable model quality**: flips-per-
  token under a Q=16 rational snap (400 rows, ~1 min) rank-
  predicts full snapped-gate damage at rho 0.883 across 6
  crystals — a 100x-cheaper robustness instrument. Snap operators
  are instruments: a scaled vs direct snap is a different lattice
  (the v1/v2 lesson). (07-28)
- **Pick-trained crystals are near-deterministic over the valid
  set**: >97% of valid mass on ONE of ~7 engine-enumerated moves
  (0.06 bits vs 2.85 available), and determinism RISES with
  capability. The branching-entropy floor is unmeasurable because
  the training regime never builds the distribution; calibration
  training on the forward crystal is a diagnostic, not a lever
  (parity at matched-dose replacement — the altpairs dilution tax
  was share, not labels). Calibration belongs to the scorer.
  (07-28)
- **Generation is tie-scarce; damage is tie-amplification**:
  near-tie decode steps (<0.02 margin) fire ~once per 30 chains
  — branching has nothing to grip. Hardware decides frontier
  probes not through tie-dense generation but through rare ties x
  long horizons (the snap-anatomy damage law, confirmed from the
  decode side). The economics finding underneath: greedy decode
  captures 90% of the 8-wave's solves at 12% of its tokens —
  greedy-first, wave-on-retry is the farm/probe speed candidate.
  (07-28)

## Night-28 (07-28, overnight, both machines)

- **The exact tail's causal footprint, measured element-wise**:
  exact-dd vs fp64-masters training endpoints differ by 321 fp32
  last-bit latent casts (max 3.7e-9) and ZERO of 50.3M deployed
  sign cells, with identical calibration fingerprints —
  "deployed-function-identical" proven at the weight level. The
  precision hierarchy is now closed end to end. (07-28)
- **Axiom is the default enumerator** (scoped adoption): forward
  successors bridge sound 200/200 on the house oracle at 3.8x;
  exact set-parity between two CAS's algebra moves is
  structurally unreachable (normal forms) and unnecessary. (07-28)
- **Null revivals priced**: Muon is gate-toxic at every schedule
  (its CE-gate dissociation is a standing instrument); backward
  rows have a safe dose (10% neutral, 50% toxic); the born-2-bit
  zero premium stays sub-bar at double dose. Magnitude/rational
  STE lattices deploy at ZERO tax (P2 72=72 at 45M-class, Z[i]
  65=65) where the phase lattice paid -4 — lattice-geometry-
  dependent deploy tax, candidate law. (07-28)

## The symmetry ladder (07-28, one session, all pre-registered)

- **Symmetry is a third compression axis** (weight sharing),
  next to bits and dimensions: trained dense gate matrices
  retrofit into group-commutant structure at sublinear toll —
  complex 2x: -1, quaternion 4x: -4, circulant 8x: -6 of 65.
  The 8x rung means dense layers of a trained crystal can be
  RETROFITTED into convolution structure. (07-28)
- **SGD never chooses symmetry, but accepts it**: anti-mass
  reads at the generic null for every group tested; yet after
  projection + one warm epoch the commutant is locally stable
  even at lambda=0 — with stability WEAKENING as the group
  grows (drift 0.014 complex -> 0.136 at C8). (07-28)
- **The holography edge is structure-dependent**: Z2's 50%-mass
  cut cost double the complex 50% cut (49 vs 57); C8's 87.5%
  cut destroys (2/120). What matters is which functional
  directions are deleted, not the mass fraction. (07-28)
- **Exactness endpoint measured**: the W(+)W doubled model
  gates EXACTLY 65 (anti-mass < 1e-12 by theorem) — exact
  rotational conversion exists at 2x width; at-width
  conversion pays the measured toll instead. (07-28)

## The chaos program + matryoshka (07-28/29 overnight)

- **Training is chaotic in initial conditions, smooth in
  hyperparameters, degenerate in capability**: twin
  disagreement flat across 4 orders of eps at ~78% of seed
  saturation; LR x BS map a smooth plateau; all twins gate
  plateau-level. The attractor is a quality SHELL. Even eps=0
  reruns diverge (cuda nondeterminism = sigma). (07-28/29)
- **EMA tames it and pays**: Polyak 0.999 contracts the shell
  58% AND raises gates (+12/+6 d64, +6 d256) — adopt-candidate.
  Soups of independent births crater (no weight-space
  convexity); EMA's running average along ONE trajectory is
  the working form of averaging. (07-29)
- **The matryoshka crystal works**: joint loss CE(W)+CE(P(W))
  gives one tensor with a dense tier at 65 (zero price) and a
  params/8 circulant tier at 60. Like-family tiers nest free;
  cross-family compression (quantization on sharing) pays
  double — slack is a shared budget across UNLIKE axes.
  (07-29)
- **Capability ~ spectral mass**: under the C8 isotypic
  decomposition, cumulative band reconstruction gates
  19/49/63/64/65 — holography along the frequency axis; band
  masses exactly generic. (07-29)
- **The sharing toll is grammar-dependent at scale**: 45M math
  -5 (scale-stable) but ZX 36->17 — graph-grammar capability
  is far less symmetry-compressible than tree-grammar. (07-29)

## The slack atlas + the escalation engine (07-29, one day+evening)

- **The width floor and the attention machine**: capability
  holds to d56 at 1/3 params (cliff sharp in (48,56]), FFN flat
  224->48, depth flat at layers {4,8,12,16} (params-matched
  shallow-wide ties deep-narrow), per-level dose flat at
  quarter-cuts — the crystal is bound by NONE of depth, dose,
  FFN, or bits. The binding resources: attention geometry x
  diet hardness x decode policy. (07-29)
- **The sigma-priced snap law, then its true form**: the
  quantization knee sits at grid = 0.5-1.0 sigma across every
  crystal — and the distortion collapse shows the full
  equation is two-parameter: kept ~ f(k_c x D/sigma^2), with
  k_c a ~30x per-crystal fragility spread whose meter already
  exists (flips/token, rho .883). Geometry-blind three ways
  (Cartesian = polar at matched bits; 8 angle bins free; a
  45-degree grid rotation against star-anisotropic weights
  costs ~1): coordinates appear nowhere in the chain bits ->
  distortion -> flips -> solves. (07-29)
- **Heads: a sparse critical circuit**: single-head deletion is
  catastrophic at every width and provision (all 8 essential at
  double provision — load spreads to fill), but the 64-cell
  autopsy resolves it: ~13/64 (layer,head) cells load-bearing
  (one costs -34 alone), ~51/64 slack; every index-column
  contains a critical cell. Precious = the circuit, not the
  grid. (07-29)
- **Delegation works, step-locally**: call-span hints carrying
  the IMMEDIATE step's engine-computed value buy +15 (~3 sigma)
  at zero params; end-value spans buy noise. Prosthetics must
  answer the question being asked. (07-29)
- **The escalation ladder is a capability lever**: eighth ->
  half -> dense tier-retry on one matryoshka tensor gates 62 v
  dense 57 at 94% effective params — decode-time policy buys
  back the tier tax and then some. Hardness comes free as
  "first rung that solves". (07-29)
- **The farmer's honest null**: a full-reverse model INVERTS
  the step grammar in-distribution (107/1000 forward-verified)
  but is memorization-dominant (11/1000 novel) — self-farming
  needs multi-ply trees, solved-state seeding, or scale.
  (07-29)
- **E-series closed cross-lab**: margin-certified 50-row
  battery (every token margin >= 0.05 by construction) decoded
  byte-identical by axiom's exact mode — 50/50, sha-pinned,
  zero adjudication needed. Instrument design replaced
  arbitration. (07-29)

## The packed crystal (07-29 eve/night)

- The month's snap laws became a real artifact: sigma-law
  packed crystals gate at parity (d64h8 EXACT) at ~5 bits/wt,
  6.15-6.65x smaller than fp32, with code-stream entropy
  within 1% of Gaussian capacity — born weights are
  max-entropy at their scale.
- Nothing calibrated beats the closed form on crystals: GPTQ
  (real Hessian), AWQ, HQQ all tie at matched bits.
- The bit-packed 5-bit GEMV (2.39x v fp16) BEATS the
  byte-aligned kernel: below ~8 bits, unpack ALU is free next
  to bandwidth — the disk format is the runtime format.
- Cross-device determinism measured: integer-GEMM hash
  bit-identical MPS/cuda (exact-in-fp32 integer carrier);
  fp logits differ; greedy streams match anyway.
- External fence, mechanistic: on Qwen-0.5B per-tensor sigma
  loses 33x to HQQ; per-row rescue null; step sigma/8
  recovers 11.6x. Sigma-grids are optimal exactly where
  weights are at capacity (crystals); web-LLM outlier tails
  are exploitable structure that max-anchored grids harvest.
- Tiered pack: nested bytes real (escalation -15% bytes per
  solved row) but the joint-STE matryoshka crystal pays where
  EMA parents packed free — fragility is crystal-priced,
  again; the flips meter should gate what we pack.

## The flagship morning (07-30): determinism end-to-end + the bound as bytes

- A full transformer decode now hashes BIT-IDENTICAL across
  Apple and NVIDIA silicon — every logit of every step
  (shipped tables + exact-integer arithmetic, no libm; ~280
  lines). Price measured: 96.7% argmax agreement v fp, with
  disagreements at coin-flip margins; ~10-40x reference-speed
  cost. Cross-lab verification becomes "ship one file."
- rANS made the entropy bound physical: house crystals at
  ~9x fp32; a 30B production MoE at 16.48GB (3.67x bf16),
  packed + coded on a laptop, zero calibration, lossless.
- The parity headline survived n=3 on d64h8 and honestly
  narrowed at the width floor (weak seeds pay); the flips
  card predicts at architecture level.
- The capacity dial completed: premium monotone in M across
  6 groups; expert capacity monotone in PER-EXPERT SIZE
  (Artin's law: Qwen3 5M/2.93 -> K2 40M/2.01); K2 flat in
  depth; experts decorrelated in weights, correlated in
  routing (the split law) — compression closed post-hoc,
  systems levers open.

## The full 30th (07-30): one frontier expert, two labs, and the day the metaphors got measured

- ONE EXPERT FROM 2.8T: a Kimi-K3 routed expert pulled by
  safetensors byte-range (17.5 MB), discovered to be LATENT
  (3584-dim, ~33M/expert), and executed EXACTLY — the full
  SwiGLU chain sha-identical on cpu/mps/cuda, natively on
  Moonshot's shipped MXFP4, zero requantization. Their 4-bit
  codes carry 3.643 bits/param (~9% from capacity).
- CROSS-LAB: axiom reproduced both P3 hash digests bit-for-bit
  from the sha-pinned tables (FX-V1-H). Four backends, two
  labs, zero tolerance columns. Replication as a boolean.
- THE GRAVITY CHAIN (one afternoon, five verdicts): expert
  influence is unscreened and AMPLIFIES with depth (no
  gravity; not lawful in router coordinates either) — but the
  turbulence is a TRAINING DEFAULT: a one-line contractivity
  tax flips the medium flat at ~zero gate cost. Spacetime is
  trainable; nobody had asked.
- CALIBRATION IS DIET-BORNE: a 0.9M crystal has ECE 0.0068
  and detects its own errors at AUROC 0.989 — no
  Dunning-Kruger, slightly humble, replicated across three
  births. Corrupting 10/30% of targets degrades ECE 5x/8x
  monotonically — toward DOUBT, not arrogance (noise teaches
  hedging; consistent falsehood, banked, is the DK candidate).
- THE SPLIT LAW'S CAUSE DISSOLVED TWICE: not the balance loss
  (lb == free), not sparse assignment (soft routing changes
  nothing) — expert decorrelation is the INIT DEFAULT
  preserved (0.0016 at birth); only shared gradient paths
  couple anything. Meanwhile the systems lever is REAL:
  co-routing prediction beats popularity by +13.8 recall
  points and MI-prefetch cuts cache misses 27% on real OLMoE
  traces.
- THE SCAFFOLD PATTERN (n=1 each, seed-2 running): every
  structured-sharing arm (thin channel 48, expert tree 48,
  Hebbian usage-attraction 52 — the collapse WON, beating
  dense) improves the gate while leaving NONE of its designed
  weight-space signature: the channel is eval-inert (zeroing
  it changes nothing), the tree has no phylogeny, gravity
  homogenized the experts and routing MI survived at 245x.
  Structure helps as TRAINING DYNAMICS, not as anatomy.
  Params confound named for channel/tree; gravmoe is clean.
- Also: no single "mass" exists (usage, robustness, influence
  mutually decoupled; basin geometry an architectural
  invariant identical across 4 models to 3 decimals), and a
  first partial in-vivo sighting of Fourier digit features
  (11/512 neurons, half their periodic variance in one
  frequency).

# FINDINGS — the curated research results

This is the external-reader layer over [`RESULTS.md`](RESULTS.md). Each bullet
is one measured claim with one [`GLOSSARY.md`](../GLOSSARY.md#evidence-maturity)
maturity tag. Retractions/nulls remain visible. Scope omission never broadens a claim.

## The derivation engine

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] At n=1 held-out battery, the
  string-seeded, oracle-verified engine reached 360/360; this closes that
  benchmark, not calculus in general. ([360/360 — THE BENCHMARK IS
  SOLVED](RESULTS.md#L99 "id:undated-360-360-the-benchmark-is-solved").)
- [REPLICATED] [REGIME-SCOPED: calculus search] At fixed wall time, cheaper
  verified nodes repeatedly converted into more solves across n=15 matched
  cells per budget; the routed engine then strictly dominated both pure arms.
  ([The racing arc](RESULTS.md#L74 "id:undated-the-racing-arc-all-same-held"); [The router: strict dominance,
  adopted](RESULTS.md#L910 "id:undated-the-router-strict-dominance-adopted-verified").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] At n=1 fixed battery, the
  bigram supplied move grammar; model confidence was distinct and width partly
  substituted for it. ([The component taxonomy](RESULTS.md#L234 "id:undated-the-component-taxonomy-what-actually-carries").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Expert-iteration
  arms from mature and tabula-rasa starts both reached their operator-closure
  ceiling in one round; adding operators, rather than more self-teaching,
  moved that ceiling. ([The limit-of-self-teaching answer](RESULTS.md#L256 "id:undated-the-limit-of-self-teaching-answer").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] A causal prune arm
  using Liouville's non-elementarity certificate removed dead branches with
  zero observed false positives inside the engine's operator closure.
  ([The magic estimator](RESULTS.md#L531 "id:2026-07-09-the-magic-estimator-2026-07-09").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] In the causal autopsy
  ladder, the linear-basis arm subsumed earlier rules and moved the L4 ceiling.
  ([The autopsy ladder](RESULTS.md#L151 "id:undated-the-autopsy-ladder-failure-census-operator").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] At n=1 recorded policy ladder,
  DAgger recovered diagnosed L5 misses but stayed below its Markov teacher and
  lost the wall-time win, so Markov kept production at that rung. ([Syndrome
  policy v2 + DAgger round 2](RESULTS.md#L852 "id:2026-07-10-syndrome-policy-v2-dagger-round-2").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] At n=1 battery, the hybrid
  engine beat its SymPy subcontractor through the measured upper levels while
  retaining SymPy as judge; this is a battery result, not a general CAS rank.
  ([The L6 evening](RESULTS.md#L1005 "id:2026-07-11-the-l6-evening-engine-36-59"); [L5 CLOSED at 100%; L7
  56/60](RESULTS.md#L1036 "id:2026-07-11-l5-closed-at-100-l7-56").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] At n=1 held-out battery, four
  days of measure-everything iteration moved the full stack from 265/360 to
  360/360, with every step a named, measured component (search wisdom,
  autopsy-derived operators, a 1835-era pruning theorem); the same
  methodology speed-ran a second domain (T-count minimization) to an honest
  greedy-wins null in one day. ([The one-paragraph
  version](RESULTS.md#L55 "id:undated-the-one-paragraph-version").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] At n=1 held-out battery, the
  full stack (best-first frontier, NNUE eval, Markov ranking, entropy-gated
  0.5B confidence, Liouville pruning) reached 356/360, with the sole holdout
  budget-invariant across four budgets — a capability miss, not a search
  miss. ([THE RECORD: 356/360](RESULTS.md#L114 "id:undated-the-record-356-360-98-9").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Racing the dict
  ranker against LLM-gated confidence on the same 24 cells isolated the
  confidence premium at +15 solves on top of identical ranking and rules
  (349 v 334), the largest confidence premium measured, and confirmed at
  n=30 (694/720). ([The hybrid record](RESULTS.md#L127 "id:undated-the-hybrid-record-349-360-96").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] The failure-census
  autopsy ladder (dump every failure, classify, implement the top family,
  repeat) moved L3/L4 solve rate rung by rung from a fixed baseline, the
  same industrialized-reading method that had earlier produced the euler and
  i_apart movers by hand. ([The autopsy ladder](RESULTS.md#L127 "id:undated-the-hybrid-record-349-360-96").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] An asynchronous
  best-first frontier strictly beat the depth-synchronized beam at equal
  node budgets (104-113 v 91-93 across variants), and with the frontier
  finally asynchronous, greedy (g=0) beat Dijkstra-weighted variants — path
  length is not worth trading nodes for when any solution is a verified
  proof. ([Best-first beats the beam](RESULTS.md#L191 "id:undated-best-first-beats-the-beam-dijkstra").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Decomposing the
  best-first win against a shared transposition table showed asynchrony
  worth +12 solves given dedup and dedup worth +21 within best-first — the
  frontier re-treads commuting rewrite orders far more than a synchronized
  ply does. ([Best-first beats the beam](RESULTS.md#L191 "id:undated-best-first-beats-the-beam-dijkstra").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Verifying integral
  edges by differentiating the difference instead of calling `doit()`, and a
  timeboxed expand-then-numeric-screen simplify-as-zero ladder that rejects
  Integral/Subs residues, were both required to keep the oracle's search-time
  cost bounded; probe timeouts had to catch `BaseException` because broad
  `except` swallowed the alarm. ([Engineering findings](RESULTS.md#L289 "id:undated-engineering-findings-each-measured-each-guarded").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Macro promotion by
  measured winning-path traffic beat promotion by textbook convention: a
  mined rule covering 14.8% of winning-path traffic paid +12 solves while
  the textbook quotient rule, at zero traffic, never fired.
  ([Engineering findings](RESULTS.md#L289 "id:undated-engineering-findings-each-measured-each-guarded").)
- [NULL] [REGIME-SCOPED: calculus search] Re-mining the move prior from
  mixed-quality paths dropped the control from 334 to 300 when 41% of the
  harvest came from random-search wins; a prior inherits the policy quality
  of the paths it is mined from, not just their verified correctness, so
  re-mining only pays when the harvesting engine is at least as strong as
  the prior's current user. ([The experiment ledger](RESULTS.md#L289 "id:undated-engineering-findings-each-measured-each-guarded").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] A fused one-trunk, two-head
  architecture with a value head reading the 0.5B's hidden state lost
  honestly to the hand-featured NNUE eval (115 v 119) while the trunk stayed
  frozen under ranking-tuned LoRA; joint value-LoRA training (v2) then beat
  the hand features offline (rho +0.966) and won the frontier cell 10 v 9.
  ([The experiment ledger](RESULTS.md#L306 "id:undated-the-experiment-ledger-wins-nulls-and").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] Porting the search-engine
  methodology to T-count minimization (ZX-calculus rewrites, boundary-oracle
  verification) found the pre-registered bar (search beats greedy full_reduce
  by >=20%) FAILED honestly on random circuits — 0 wins, 30/30 exact ties —
  consistent with the literature that full_reduce is near-optimal on random
  rather than structured circuits. ([T-count engine, day
  one](RESULTS.md#L448 "id:undated-t-count-engine-day-one-rungs").)
- [MECHANISM-CONFIRMED] A T-count claimed by scoring an unextractable ZX
  diagram is fiction: safe rewrites preserve semantics but can destroy
  GFLOW, so about half a race's tempting low T-count "descents" corresponded
  to no real circuit; the eval must score the T-count of the EXTRACTED
  circuit, or the move set must be GFLOW-preserving. ([T-count engine, day
  one](RESULTS.md#L448 "id:undated-t-count-engine-day-one-rungs").)
- [REPLICATED] [REGIME-SCOPED: Qwen2.5-0.5B] A frozen-trunk value-head probe
  sweep across 24 layers peaked at 83% depth rather than the last hidden
  state (L20 +0.873 v last +0.858), replicating the global-workspace paper's
  geography at three fresh splits (L20 winning all three); a jointly-trained
  value-LoRA at the same probe point then matched the last-layer arm,
  showing the geography constrains frozen probes, not trained
  representations. ([Middle-layer value probe](RESULTS.md#L703 "id:2026-07-09-middle-layer-value-probe-2026-07").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] Sub-term probing pinned the
  frontier-mined failure census to two missing rule shapes; adding both
  moved same-seed L5 from 42% to 89.6% (223/249), a ~+12-point gain from two
  measured rules. ([Frontier rule gaps -> two rules](RESULTS.md#L703 "id:2026-07-09-middle-layer-value-probe-2026-07").)
- [RETRACTED] [REGIME-SCOPED: calculus search] Re-mining the move prior after
  the two new rules landed, on the theory that unseen-rule smoothing was
  guillotining them, regressed L5 from 89.6% to 73.1% on both a general and
  a 3x-weighted harvest — diluting winning bigrams costs more than new-rule
  mass gains, the second measured prior-mining regression; the fixed 0.01
  median smoothing was already sufficient. ([Frontier rule gaps -> two
  rules](RESULTS.md#L726 "id:2026-07-10-frontier-rule-gaps-two-rules-2026").)
- [NULL] [DEVICE-SCOPED] At draft:target cost ratio ~1:3 (Qwen2.5 0.5B draft,
  1.5B target, 3080), entropy-adaptive speculative decoding's acceptance
  signal was real (prose 0.47->0.69, code 0.79->0.90) and target passes
  nearly halved, but wall-clock never beat fixed k=3 because the adaptive
  arms burned ~40% more draft passes; the falsifiable prediction is that the
  measured pass-halving converts to wall-time only at draft:target cost
  <= ~1:10. Entropy must be computed in float32 — fp16 underflow silently
  degenerated the gate to fixed k_max in 771 passes. ([Entropy-adaptive
  speculative decoding](RESULTS.md#L762 "id:2026-07-10-entropy-adaptive-speculative-decoding-2026-07").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Replacing a `.doit()`
  verify call (which legally re-invoked sympy's own heurisch integrator) with
  `doit(integrals=False)` cut node cost enough to widen the beam 2->3 at the
  same budget, taking same-seed L5 from 223 to 238/249 (95.6%) at 3.5x less
  wall. ([Node-cost round 2](RESULTS.md#L807 "id:2026-07-10-node-cost-round-2-2026-07").)
- [NULL] [REGIME-SCOPED: calculus search] Widening the move-proposal layer to
  k=4 was a decisive NO-ADOPT (210/249 v the 238 bar): it spent the fixed
  node budget faster by cutting search depth, and it never reached the
  target rank-5/6 move anyway — proposal-layer width mostly re-covers what
  top-3 already had, unlike selection-layer width. ([propose_k=4: decisive
  null](RESULTS.md#L828 "id:2026-07-10-propose-k-4-decisive-null-two").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] The syndrome policy (DAgger,
  state-aware move ranking) was ADOPTED into production after a
  fresh-100 tie/win race (98/96 solves at 36% less wall than Markov) and a
  fresh-80 exact tie (76/76) closed the 2-problem curated-gate deficit as
  benchmark-specific rather than a capability gap; Markov remains the
  fallback and the wall-time choice for deep-L5 batch work. ([The brain
  races, concluded](RESULTS.md#L879 "id:2026-07-10-the-brain-races-concluded-policy-adopted").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] New coordination (L6) and
  nesting (L7) generator levels found the engine held up better than
  sympy's heurisch as depth grew (L6 60% v sympy 93%; L7 60% v sympy 70%,
  a 23-point sympy drop the engine did not share), with three money
  problems the engine solved and sympy failed, certified by sympy's own
  differentiator. ([L6/L7 and the engine-vs-sympy
  probe](RESULTS.md#L974 "id:2026-07-11-l6-l7-and-the-engine-vs").)
- [SINGLE-SEED] [REGIME-SCOPED: Qwen2.5-0.5B] A 128-unit probe on the
  calculus-LoRA 0.5B's layer-20 hidden state, read mid-generation,
  predicted eventual sympy-correctness at AUC 0.914 on 2,760 trace states
  — the trajectory's fate is largely encoded early. The naive spend policy
  built on that signal then lost decisively to best-of-N at equal token
  spend (greedy 85, best-of-N 100, regret-abort 78 of 150): the signal was
  real but an uncalibrated early-abort threshold killed traces before
  their fate had formed. ([The regret probe](RESULTS.md#L1060 "id:2026-07-11-the-regret-probe-trace-fate-is").)
- [SINGLE-SEED] [REGIME-SCOPED: calculus search] A new L8 generator level,
  built from measured L6/L7 failure modes rather than imagination, probed
  at 30/40 (75%, all ten misses genuine solve failures, not hangs); a
  same-day autopsy of two missing orbitals (sqrt*log ansatz, trig(log)
  admission) closed most of the gap to 37/40. ([L8: the frontier reopened
  from the residue](RESULTS.md#L1131 "id:2026-07-11-l8-the-frontier-reopened-from-the").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: Qwen2.5-0.5B] Racing three 4-bit
  quantization schemes on real Qwen2.5-0.5B weights and real captured
  activations in function space (mean GEMV relative error) found the
  activation-aware lane (awq_lite) winning at 8.07% v uniform min/max
  10.06% and NF4-style quantile codes 8.89%, largest on late layers
  (layer-23 down_proj 14.7% -> 6.5%); a toy round on synthetic gaussian
  weights had ranked uniform first, because synthetic weights lack the
  outlier channels real transformers carry and activation-awareness
  protects — weight-space and function-space rankings disagreed in both
  rounds. ([Three-lane 4-bit quantization race](RESULTS.md#L1163 "id:2026-07-11-three-lane-4-bit-quantization-race").)
- [MECHANISM-CONFIRMED] [DEVICE-SCOPED] A fused int4 dequant-GEMV Metal
  kernel (packed nibbles, awq_lite scales folded in at pack time) reached
  1.11x mx.quantized_matmul at D=4096 (2.80x over fp16) but only 0.72x at
  D=896, where small decode shapes are launch/overhead-bound rather than
  bandwidth-bound. ([Fused int4 dequant-GEMV Metal
  kernel](RESULTS.md#L1185 "id:2026-07-11-fused-int4-dequant-gemv-metal-kernel").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] Engine-level regret
  (abort a search reading doom from per-ply beam features, AUC 0.760) FAILED
  its pre-registered bar (zero solve loss + >=25% wall cut) — two stubborn
  deep chains always looked doomed then landed — but at equal total wall,
  the FA-Law-native metric was a blowout (176 v 82 solves in the same
  1888s, 2.1x, stable across budgets): regret trades 2.4% completeness for
  2.1x throughput, so adoption is workload-dependent and it is not wired
  into `solve()` by default. ([Engine-level regret](RESULTS.md#L1203 "id:2026-07-12-engine-level-regret-the-thesis-pays").)
- [NULL] [REGIME-SCOPED: calculus search] A fresh L3-L8 dispatcher retrain
  (v4, 275 post-orbital disagreement rows) was NO-ADOPT against the v3
  incumbent (110 v 112 solves, fourth appearance of the starved-judge
  pattern): as engine rules cover more of the space one-ply, the arms
  converge and brain choice matters less; the currency rule softened to
  retrain only when the world changes AND the arms still disagree enough to
  route. ([Dispatcher v4: NO-ADOPT](RESULTS.md#L1228 "id:2026-07-12-dispatcher-v4-no-adopt-2026-07").)

## The micro-model program

- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: closed-system math] At n=1
  comparison, the from-scratch 19M decoder exceeded the pretrained 0.5B arm on
  the oracle gate; these were the two sides of the original comparison, not a
  reproduction route. ([Micro-model phase 2 + 0.5B run 3d](RESULTS.md#L1682 "id:2026-07-15-micro-model-phase-2-0-5b").)
- [MECHANISM-CONFIRMED] [TEACHER-FORCED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] Paired packing, diet, and integer
  training-contract arms produced favorable teacher-forced loss alongside
  free-run capability holes. Loss is therefore not a capability score across
  changed contracts; capability claims require the oracle gate. ([VERDICT
  GRAVMOE-GATE](RESULTS.md#L14470 "id:2026-08-01-verdict-gravmoe-gate-loss-said-learning").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  At n=1, warm growth on the verified chain corpus beat rebirth at matched
  epochs; the result is a recipe-and-corpus observation, not a universal
  lineage law. ([GEN-6 GROWN](RESULTS.md#L2845 "id:2026-07-19-gen-6-grown-76-120-growth").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  At n=1 per width, the gen-4 chain corpus had a finite useful width band and
  degraded under token-per-width starvation; this does not define a natural
  width for other grammars or doses. ([THE WIDTH FLOOR: W_min ~
  8.4M](RESULTS.md#L4470 "id:2026-07-25-the-width-floor-w-min-8"); [113M fp32 capacity re-ask](RESULTS.md#L2345 "id:2026-07-18-113m-fp32-capacity-re-ask-tokens").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: closed-system math] At n=1,
  the fitted corpus-signature × width equation predicted the held-out house
  battery; it remains a measured battery model, not a general scaling law.
  ([The closed-system equation, v0](RESULTS.md#L2677 "id:2026-07-19-the-closed-system-equation-v0-2026").)
- [SINGLE-SEED] [REGIME-SCOPED: closed-system math] At n=1 corpus analysis,
  rule-bigram clustering grouped techniques rather than difficulty levels and
  isolated the ansatz clade. ([The math phylogeny](RESULTS.md#L2098 "id:2026-07-17-the-math-phylogeny-technique-lineages-not").)
- [SINGLE-SEED] [FORMAT-BOUND] [REGIME-SCOPED: closed-system math] At n=1
  gzip-NCD corpus analysis, a separate lens isolated L4's dialect; this is a
  corroborating read, not implementation replication. ([Compression phylogeny](RESULTS.md#L2116 "id:2026-07-17-compression-phylogeny-two-lenses-one-crater").)

## Capacity fence for crystals and packing

Capacity, maximum-entropy, and sigma-pack statements below are at-capacity
house-crystal results unless a verdict transports them. The Qwen2.5-0.5B C6
chain is the boundary, not a web-trained dense-model law. ([PACKED CRYSTAL C6
VERDICT](RESULTS.md#L10676 "id:2026-07-29-packed-crystal-c6-verdict-the-falsifier").)

## The crystal laws

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] At n=1 registered RL-climb
  analysis, small near-uniform low-rank deltas and almost unchanged intermediate
  CKA supported a policy-preference edit. ([Weight anatomy](RESULTS.md#L1625 "id:2026-07-15-weight-anatomy-the-closed-system-signature").)
- [SINGLE-SEED] [REGIME-SCOPED: house crystals] At n=1 graph-anatomy census
  across six measured minds, low clustering plus connectivity supported the
  expander-topology read; this is not implementation replication. ([Graph anatomy](RESULTS.md#L2299 "id:2026-07-17-graph-anatomy-the-crystal-is-an").)
- [SINGLE-SEED] [REGIME-SCOPED: house crystals] At n=1 examined crystal,
  family-selective committees were informative but not necessary: deleting
  or reading the committee supported a holographic, preference-not-dependency
  interpretation. ([Definition neurons](RESULTS.md#L2266 "id:2026-07-17-definition-neurons-locatable-preferences-no-dependencies"); [The free router
  works](RESULTS.md#L2329 "id:2026-07-17-the-free-router-works-and-the").)
- [REPLICATED] [REGIME-SCOPED: house crystals] Committee membership changed
  while selectivity statistics persisted across three births and continued
  training; those births are the replication route, not evidence for a
  universal neuron law. ([Neuron biography](RESULTS.md#L2443 "id:2026-07-18-neuron-biography-the-lottery-is-redrawn").)
- [NULL] [REGIME-SCOPED: house crystals] No tested gauge- or
  permutation-aligned weight-distance lens recovered functional distance;
  the measured distances tracked ancestry instead. ([Gauge-aligned
  distance](RESULTS.md#L5746 "id:2026-07-26-gauge-aligned-distance-all-three-predictions"); [JOINT-PERM CLOSURE](RESULTS.md#L6163 "id:2026-07-26-joint-perm-closure-kill-condition-fires").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: house crystals] Causal quantization
  and rank-cut arms separated the axes: the tested crystals tolerated fewer
  bits but not deleted dimensions, while the latent smooth object survived
  alphabet projection. ([The alphabet is a lens, not an
  attractor](RESULTS.md#L2505 "id:2026-07-18-the-alphabet-is-a-lens-not"); [The rank floor](RESULTS.md#L2180 "id:2026-07-17-the-rank-floor-bits-are-redundant").)
- [MECHANISM-CONFIRMED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] A calculated warm-initialization
  arm advanced the gate early, then converged toward the same measured
  ceiling; initialization changed time-to-capability, not the endpoint.
  ([Warm birth](RESULTS.md#L2469 "id:2026-07-18-warm-birth-the-calculated-init-pays").)

## Alphabets and precision

- [REPLICATED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] Ternary
  parity was reproduced through independent continuation and warm-growth
  implementations; the claim is parity for these house math crystals, not
  for arbitrary models. ([Extended-training night: ternary reaches
  parity](RESULTS.md#L2397 "id:2026-07-18-extended-training-night-ternary-reaches-parity"); [GROWN-MERGED](RESULTS.md#L3643 "id:2026-07-23-grown-merged-75-120-statistical-crown").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  The "+7 ternary crossover at d768" dissolves on a clean instrument:
  n=3 same-device, same-diet paired seeds read t-fp32 deltas of
  -2/+1/-3 (pooled -4, inside the |<=5| null bar) — the original
  boundary-grid point was carried by its cross-device and diet-path
  confounds; ternary and fp32 are gate-indistinguishable at d768.
  ([VERDICT REVIVE-D768-CROSSOVER](RESULTS.md#L23643 "id:2026-08-09-verdict-revive-d768-crossover-p-crossover").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  At n=1 per alphabet arm, the matched tournament favored the measured
  multi-level codebook, while the later S4 arm narrowed the role of zero.
  ([The alphabet tournament](RESULTS.md#L2935 "id:2026-07-20-the-alphabet-tournament-real-valued-bracket"); [BORN-S4](RESULTS.md#L5896 "id:2026-07-26-born-s4-58-120-57-17").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] Complex or rotational
  interiors did not improve the tested math, ZX, NNUE, or weight-FFT lenses;
  these are repeated scoped nulls, not a theorem about complex networks.
  ([Complex NNUE: NULL](RESULTS.md#L5967 "id:2026-07-26-complex-nnue-null-as-pre-registered"); [THE ZX COLUMN
  COMPLETES](RESULTS.md#L6652 "id:2026-07-26-the-zx-column-completes-alphabet-follows"); [Weight-FFT euler read:
  NULL](RESULTS.md#L6098 "id:2026-07-26-weight-fft-euler-read-null-for").)
- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] Above the
  measured instrument sigma, the birth-precision bracket reproduced on Apple
  and NVIDIA devices; below that fence, the ledger does not claim closure.
  ([scope fence + named retest condition](RESULTS.md#L7226 "id:2026-07-27-amendment-targets-precision-doctrine-closure-2026").)

## The exact stack

- [MECHANISM-CONFIRMED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts] On NVIDIA tensor cores, the
  block-aligned integer-slicing arm removed accumulation rounding and
  reproduced big-integer products exactly. ([Ozaki 2a-v3](RESULTS.md#L3969 "id:2026-07-23-ozaki-2a-v3-zero-rounding-gpu").)
- [SINGLE-SEED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts] At n=1 NVIDIA benchmark,
  stay-in-RNS deferred carries and crossed the measured fp64 chain cost; it is
  an implementation result, not a capability gain. ([Ozaki v5](RESULTS.md#L4019 "id:2026-07-23-ozaki-v5-the-stay-in-rns").)
- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] Rational-snap
  parity reproduced on two crystals and Apple/NVIDIA device routes, while the
  sensitivity knee differed by crystal. ([RATIONAL-SNAP
  VERDICT](RESULTS.md#L7613 "id:2026-07-27-rational-snap-verdict-the-crystal-has"); [QUICK EXACT BATTERY
  VERDICT](RESULTS.md#L7819 "id:2026-07-27-quick-exact-battery-verdict-integer-twin").)
- [REPLICATED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] Integer-twin parity was
  reproduced on independent crystal/checkpoint implementations; a shared
  denominator made the tested forward an integer GEMM. ([QUICK EXACT BATTERY
  VERDICT](RESULTS.md#L7819 "id:2026-07-27-quick-exact-battery-verdict-integer-twin").)
- [MECHANISM-CONFIRMED] [TEACHER-FORCED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] A
  causal snap arm localized gate damage to pre-existing near-tie decisions
  amplified by decode horizon, rather than to bulk weight error.
  ([SNAP-ANATOMY VERDICT](RESULTS.md#L7699 "id:2026-07-27-snap-anatomy-verdict-the-sensitivity-wall").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  Born-rational training carried no resolved capability premium across the
  tested two-seed pair; the first apparent win demoted to parity.
  ([BORN-RATIONAL SEED-2](RESULTS.md#L7847 "id:2026-07-28-born-rational-seed-2-2-the").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: deterministic integer battery]
  Rounding error in the exact-integer engine separates into two
  independent regimes, and the precision arms are the causal test:
  Q9, Q32, Q64 and an exact-rational anchor run from identical
  initialization bytes, varying only precision. Ring-grain error is
  fully absorbed at p=32 — de-grained Q32 and Q64 weights are
  bit-identical at all twelve steps at both d8 and d16, so ring
  precision beyond 32 buys nothing. Frozen-carry error, which is the
  softmax carry quantization, is grain-independent by mechanism
  rather than by accident: the anchor's distance to Q32 equals its
  distance to Q64 to the last decimal at every certified step. Grain
  growth itself is super-diffusive and sub-linear at both dims, and
  under a matched estimator the two dims agree closely — least-squares
  log-log slopes 0.800 and 0.817, endpoint exponents 0.714 and 0.729.
  Both unseen cells fired all three registered bars, but two cells
  share one implementation and one device, so this is a mechanism
  reading and not a replication route. ([VERDICT
  EXACT1-SMALL](RESULTS.md#L23852 "id:2026-08-09-verdict-exact1-small-3-3-bars"); [AMENDMENT
  EXACT1-SMALL-EXPONENT](RESULTS.md#L23910 "id:2026-08-10-amendment-exact1-small-exponent-the-two"); [AMENDMENT
  EXACT1-SMALL-EXPONENT-2](RESULTS.md#L23948 "id:2026-08-10-amendment-exact1-small-exponent-2-the").)
- [SINGLE-SEED] [REGIME-SCOPED: deterministic integer battery] The
  frozen-carry floor accumulates linearly in steps and grows with
  model width while staying independent of ring precision: the
  anchor-to-rung distance reads 19.0 then 38.0 at d8 and 44.8 then
  87.2 at d16 across the two anchor-certified steps, against roughly
  528 at d64 in the earlier shakedown. Per step and per unit of
  width that is about 2.4, 2.7 and 8.3, so the floor rises faster
  than width itself. One run per width and not pre-registered, so
  this is a descriptive read; its value is that the carry-ladder arm
  of the next engine pre-registration now has a measured curve to
  beat rather than a guess. ([VERDICT EXACT1-SMALL](RESULTS.md#L23852 "id:2026-08-09-verdict-exact1-small-3-3-bars").)
- [SINGLE-SEED] [REGIME-SCOPED: deterministic integer battery] Exact
  rational arithmetic is a cost wall rather than a correctness one.
  The d8 anchor took 3.9 s for step 1 and 3,335 s for step 2, a
  factor of 845, and was killed 19.34 hours into step 3 against a
  projected multi-week run; the ledger entry's "about 19.5 hours" is
  the loose form of that figure, which is fixed by the interval
  between the step-2 dump and the kill. Profiling attributes 86.6 to
  96.4 percent of a d64 anchor step to gcd inside rational addition. Wall-clocks are descriptive only,
  because the machine carried interactive load during part of the
  window. The abandonment is itself the horizon measurement that the
  follow-on gcd-free design is registered against. ([VERDICT
  EXACT1-SMALL](RESULTS.md#L23852 "id:2026-08-09-verdict-exact1-small-3-3-bars"); [PRE-REG
  ANCHOR-V2](RESULTS.md#L23791 "id:2026-08-09-pre-reg-anchor-v2-the-gcd").)
- [NULL] [REGIME-SCOPED: deterministic integer battery] The structural-tie
  reading of the anchor's near-tie class is refuted for the ambient
  population: across 5,490 censused events the small integer |r| tracks the
  blocking denominator to within 20 bits everywhere (mid-zone gap max 12
  bits), and the co-factor witness rung closed at its pre-registered
  NOT-APPLICABLE gate without a build — the blocking seam's locally-known
  divisors are all powers of two, and the witness mechanism is in tension
  with gcd-freeness itself (a trackable non-reduced co-factor compounds
  through gemm contractions). The step-9 site's own hypothesis stays
  unreached inference; the census dies at the wall it measures, so all
  counts are lower bounds. ([COUNTER-BOOK
  COFACTOR-GATE](RESULTS.md#L24379 "id:2026-08-10-counter-book-cofactor-gate-the-not"); [VERDICT
  COFACTOR-CENSUS](RESULTS.md#L24452 "id:2026-08-10-verdict-cofactor-census-the-r-refutation").)
- [MECHANISM-CONFIRMED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts] The fp32-limb exact GEMM
  oracle (the MPS KEY finally implemented: s=7 slices, block 32, budget
  inequality as a compile-time fence) is exact against big-int across every
  registered adversarial class including K-permutation bit-identity, with
  all fences thrown-errors that survive release builds; the cross-lab build
  sharpened the envelope contract from an exponent-spread condition to a
  lowest-significant-BIT span condition, so registered inputs carry a
  flush-to-zero contract rather than a silent cap. A compile probe settled
  a banked question for free: M-series exposes no integer simdgroup MMA,
  making fp32-limb the only MMA-reaching exact path on Metal.
  ([COUNTER-BOOK FP32LIMB-R1](RESULTS.md#L24981 "id:2026-08-10-counter-book-fp32limb-r1-p-envelope").) The Metal build phase
  settled the hardware question by compile probe — simdgroup_matrix over
  int fails while the float control compiles, so M-series exposes no
  integer simdgroup MMA and the banked int8-MMA Metal port is superseded,
  not deferred; the fast-math pin is runtime-checkable (MTLMathModeSafe)
  and the dispatch interlock was verified refusing without its flag.
  ([RECEIPT FP32LIMB-R2R3-BUILT](RESULTS.md#L25050 "id:2026-08-10-receipt-fp32limb-r2r3-built-the-cpu").) The dispatched
  battery is GREEN: FTZ-PRESENT on device (range restriction required,
  as the envelope's flush-to-zero clause anticipated), bit-identical to
  the CPU oracle across three seeds, and the wall bar passed by an
  order of magnitude — exact fp32-limb GEMM at 0.120x cpu_fp64 and the
  int64-accumulator at 0.004x: the first exact GEMM on Metal is a
  speedup, not a tax.
  ([COUNTER-BOOK FP32LIMB-R2R3-GPU](RESULTS.md#L26138 "id:2026-08-10-counter-book-fp32limb-r2r3-gpu-the").)
- [NULL] [DEVICE-SCOPED] [REGIME-SCOPED: measured deployment artifacts]
  The naive-portable RNS chain shape has no break-even depth: a scalar
  mulmod-ladder implementation (toolchain-forced — the box's MSVC has no
  __int128) runs ~74x a native fp64 chain, depth-flat from 2 to 12, and a
  flat ratio means no crossover can exist because there is no per-layer
  advantage to amortize. Exactness held throughout (slice kernel equals
  the naive cross-check entrywise on all channels), the two
  independently-authored RNS modules agreed to bit-identical digests
  after the collision rebase, and the booked tensor-core break-even law
  is untested by this run — its counter-book is a Montgomery/tensor-core
  reimplementation. The fused-kernel promotion correctly did not trigger.
  ([COUNTER-BOOK RNSCHAIN-C1](RESULTS.md#L25193 "id:2026-08-10-counter-book-rnschain-c1-chain-oracle"); [VERDICT
  RNSCHAIN-C2C3](RESULTS.md#L25338 "id:2026-08-10-verdict-rnschain-c2c3-c1-re-receipt").)
- [MECHANISM-CONFIRMED] [DEVICE-SCOPED]
  [REGIME-SCOPED: deterministic integer battery] (d64 12-step anchor
  cell.) Closed-loop precision scheduling works: a proportional
  law driven only by in-run sensors (slack, straddle width, denominator
  bits — the step index never consulted) makes the certified prefix a
  property of (anchor, envelope) instead of (anchor, schedule) — entries
  at 200 and 4000 bits converge to byte-identical schedules and digests
  from step 2, at 53% of the shipped ramp's cost IN BIT-STEPS (2,024 v
  3,840; entry-200 arm — the entry-4000 arm pays its entry bits once and
  amortizes at step 2). Wall clock is FLAT (~162 s/step) across the
  precision range: the dyadic-shadow precision is not the wall lever,
  the RNS ring is. Step 9 defeated a third time with the wall now
  MEASURED: demand 56 to 445 accelerating, then a >=15k-bit cliff; a
  derivative (anticipating) term is structurally unavailable without
  re-introducing history and killing invariance. The cliff then took
  its strongest test: three precision rungs to 2^18 bits all throw at
  the same floor site with lo_bits = paid precision and e = -(prec-42)
  exactly — the shadow narrows to width 1 at ANY precision and the
  throw is rational reconstruction exhausting the fixed 256-prime
  ring. The tie is structural WITH RESPECT TO SHADOW PRECISION AT
  FIXED RING; the ring knob then got its own ladder and the tie
  survived it too: the throw signature is byte-identical at 256,
  512, AND 1024 primes (lo_bits=16384, e=-16342, obeying the
  offset-42 law), all eight prefix digests byte-match the certified
  ladder at every ring size, and the ring-cost slope books at three
  points (162/316/666 s/step — cleanly ~2x per doubling). Neither
  precision nor modulus moves the tie: the reconstruction target
  appears not to exist at any tested budget on either axis, and the
  co-factor/witness line re-elevates as a denominator-ledger design
  gate. The ring is measured as both the cost floor and the
  feasibility ceiling.
  ([VERDICT FUNNEL-PREC](RESULTS.md#L25451 "id:2026-08-10-verdict-funnel-prec-closed-loop-precision"); [VERDICT
  STEP9-CLIFF-SIZE](RESULTS.md#L26105 "id:2026-08-10-verdict-step9-cliff-size-counter-booked"); [COUNTER-BOOK
  NPRIMES-LADDER](RESULTS.md#L26450 "id:2026-08-11-verdict-nprimes-ladder-counter-book-p").)
- [SINGLE-SEED] [DEVICE-SCOPED] [REGIME-SCOPED: house crystals]
  Post-hoc ternary is NOT where trained tolerance lives: quantizing
  the fp-trained d256 crystal one tensor class at a time, the output
  head and norms ternarize for FREE (identical solve dicts), the
  input embedding loses 27 of 65 solves at the same parameter share,
  and the body craters (attention to zero) — inverting the star-bank
  ordering, while ternary-QAT bodies tie fp32 when grown on the
  lattice. Ternary capability is trained-in, never projected-in; the
  eye is precision-hungry, the mouth is not. The 5-width ladder then
  killed the geometric reading (per-param eye sensitivity NON-monotone
  in width, rho -0.4 — the asymmetry is positional) and surfaced the
  real law: WIDTH BUYS TOLERANCE — every class's ternary retention
  rises with width (eye 0% -> 93% of base up the d64->d512 ladder;
  attention's per-param fragility a flat 4x ffn's 1 at every width) —
  holography's scaling face. And 4x quaternionic sharing changes the
  profile by nothing measurable (retention fractions match the
  control within 4 percentage points per class): the
  symmetry and alphabet compression axes are orthogonal tolls.
  ([VERDICT STAR-PROFILE-1](RESULTS.md#L25772 "id:2026-08-10-verdict-star-profile-1-p-gradient"); [VERDICT
  STAR-PROFILE-2](RESULTS.md#L25979 "id:2026-08-10-verdict-star-profile-2-p-surface"); [VERDICT
  ROT-X-TERNARY](RESULTS.md#L26031 "id:2026-08-10-verdict-rot-x-ternary-p-independent").)
- [MECHANISM-CONFIRMED] [DEVICE-SCOPED] [REGIME-SCOPED: house crystals]
  The crown tie was birth luck: n=3 fresh paired births resolve the
  production crown to the champion fp32 grow-inherit line (+9/+11/+12
  over the direct ternary tournament line, pooled +10.7, 3/3 positive)
  and the n=1-births fence retires. Scope is line-v-line: ternary still
  TIES fp32 at matched dose and architecture — the ~+10 is the growth-
  inheritance pipeline, which only exists on the fp32 substrate because
  ternary growth is non-preserving. The champion line's totals were
  seed-invariant (73/73/73) while the ternary line carried the birth
  noise (64/62/61). The equilibrium survived its first hunger test:
  a fourth warm epoch at fixed food moved the champion +1 (74 v 73,
  inside the tie band, same dict shape), and the rations question
  then closed the same way: a 20% infusion of NEW food moved it +2
  (75 v 73, still inside the band). The 73-74-75 line is flat within
  resolution — at fixed mass, neither more burning time nor new fuel
  moves the gate; equilibrium is a function of (width, diet-class) — the diet-class
  dependence itself untested (CURRICULUM-FUNNEL territory) —
  and growth remains the only lever with a measured positive at this
  frontier. The width axis got its own ladder the same night: four
  sub-1M births on the 3080 (minutes each, max 546 s) put ignition
  between d32 (0/120) and d48 (10/120 at ~295k params — column
  corrected by AMENDMENT MORNING-SWEEP-0811), with d64 at 30/120 —
  the gate(d) foot is sharp (0/0/10/30; the d48 firing sits exactly
  at the >=10 bar on a single seed, so the ignition LOCATION carries
  a single-seed fence while the d48->d64 tripling is >4 sigma), and
  "births take hours" is a size property, not a pipeline property.
  ([VERDICT REVIVE-CROWN-TIE-BIRTHS](RESULTS.md#L26066 "id:2026-08-10-verdict-revive-crown-tie-births-p"); [VERDICT
  SATURATION-1](RESULTS.md#L26259 "id:2026-08-11-verdict-saturation-1-cell-a-p"); [VERDICT
  SATURATION-1-CELL-B](RESULTS.md#L26420 "id:2026-08-11-verdict-saturation-1-cell-b-p"); [VERDICT
  MICRO-STAR-1](RESULTS.md#L26301 "id:2026-08-11-verdict-micro-star-1-both-bars").)

## Data, diet, and federation

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] The training-loss
  floor decomposes measurably into data and model halves: the warm
  diet's empirical conditional entropy is 0.175 nats given the full
  prefix v a trained floor of 0.348 — ratio 0.502, so ~half the
  floor is the corpus's own ambiguity (irreducible by ANY optimizer
  at this diet; loss-to-0 requires canonicalizing the corpus, one
  continuation per context) and at least ~half is approximation
  error — a LOWER BOUND on headroom (optimization and capacity not
  separated). The
  entropy curve knees at k=32 (32 tokens of context capture nearly
  all corpus structure), and the trained floor sits BETWEEN H_16
  (0.367) and H_32 (0.187): the d512/L12 grown crystal extracts
  essentially all 16-token structure and ~none of the 16-to-32
  marginal despite 512-token attention (arch corrected by AMENDMENT
  LOSS-FLOOR-1-ARCH — a bigger star than first booked, same floor) — an effective-context instrument, and a
  candidate quantitative face for the width ladder (does the floor
  walk down the H_k curve as width grows?).
  ([VERDICT LOSS-FLOOR-1](RESULTS.md#L26376 "id:2026-08-11-verdict-loss-floor-1-mixed-at").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [TEACHER-FORCED]
  [REGIME-SCOPED: specified diet and recipe] A causal row-format arm changed
  determinability while holding the task family fixed; underdetermined rows
  trained confident guessing rather than the intended map. ([Series rung 1:
  form learned instantly, task was ill-posed](RESULTS.md#L3401 "id:2026-07-22-series-rung-1-form-learned-instantly").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] Causal re-spelling arms replaced
  chains with primitive emissions and sharply reduced the rows needed to
  learn the measured task families. ([The decomposition
  discount](RESULTS.md#L3682 "id:2026-07-23-the-decomposition-discount-10x-measured-2026"); [SERIES RUNG 1E](RESULTS.md#L3758 "id:2026-07-23-series-rung-1e-98-0-the").)
- [NULL] [TEACHER-FORCED] [REGIME-SCOPED: house crystals] Greedy decision
  margins do not track problem hardness: on the d256 control crystal over a
  fresh seven-level band, the per-level median minimum margin correlates at
  rho +0.18 against a registered bar of -0.8 or below, and every level's
  median sits at least 20x above the fp16 near-tie zone. A 16.5% tail of
  tight-margin problems exists but is not level-correlated, so at this scale
  data hardness has no inference-side lever on precision demand; the paired
  census puts the diet's choice-free share at 9.65% of states, which sinks
  the registered unique-only training arm on rations grounds before it
  fires. The depth leg closed the same day: greedy chains terminate in one
  to three plies and margins rise rather than compress with depth (rho +0.5
  v -0.8 registered; ply-2 rise is the pre-registered survivor-bias
  signature), so tight decisions are a rare (~0.1% of tokens),
  hardness-uncorrelated, depth-uncorrelated population and the ceiling-swap
  theory's inference-side legs are both closed at this scale.
  ([VERDICT DATA-CEIL-0A](RESULTS.md#L24602 "id:2026-08-10-verdict-data-ceil-0a-p-margin"); [VERDICT
  DATA-CEIL-0B](RESULTS.md#L24698 "id:2026-08-10-verdict-data-ceil-0b-fork-only").)
- [MECHANISM-CONFIRMED] [TEACHER-FORCED] [REGIME-SCOPED: house crystals]
  Near-tie decisions are data-ambiguity sites: on the d256 control crystal
  over 3,995 censused diet states, median minimum margin compresses
  monotonically with the state's legal-successor count (rho -0.9 across
  five buckets, a 3x range), tie-zone incidence rises in proportion, and
  unique-successor states produced not one tie-zone event in 386 states.
  The tie anatomy is coefficient choice, not rule choice (6 of 7 tie-zone
  sites sit at a digit emission mid-expression), reading as the
  inference-side face of the latent-arithmetic determinability law. The
  registered inversion books: determinable data removes tie sites, so
  cleaning data lowers inference precision demand — opposite to the
  ceiling-swap direction. Replicated same-day on a second crystal: d128
  gives rho -1.0 strictly monotone with the tie-zone gradient monotone in
  both thresholds; the weaker crystal shows fatter tie incidence at every
  bucket (~2x median, up to ~3.8x at the choice-free bucket) at a
  compressed absolute scale, so the law is the gradient, capability sets
  the scale. A theory row still waits on a second domain.
  ([VERDICT DATA-CEIL-0C](RESULTS.md#L24793 "id:2026-08-10-verdict-data-ceil-0c-p-ambiguity"); [VERDICT
  DATA-CEIL-0C-R](RESULTS.md#L24849 "id:2026-08-10-verdict-data-ceil-0c-r-the").)
- [MECHANISM-CONFIRMED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals]
  The free-running greedy map has exactly one basin: all 198 censused
  trajectories absorb into answer-form (zero fixed points, cycles, or
  wanderers), median absorption one step, and the absorbing step carries
  2.6x the transit margin — absorption is a confidence event, so the
  crystal's degeneration mode is confident premature answers, not
  repetition loops. The identity attractor the production doctrine fences
  three ways does not exist in this regime, locating the X-to-X attacks in
  the search scaffolding rather than the greedy map. Answer-form is not
  scored for correctness here. The named arm RAN same-day: iterating
  THROUGH answer-form, all 198 terminate inside it (192 true fixed
  points + 6 two-cycles entirely in-class; the "zero cycles" read was
  the stopping rule's shadow), with a ~99%-one-way boundary (2
  trajectories briefly relaunched, then re-absorbed). The absorbing
  confidence is NOT the ambiguity law: median n_succ 7 v 7 at
  absorbing v transit states while the margin gap replicates (2.31 v
  0.88) — separate mechanisms, honest null on the join.
  ([VERDICT ATTRACTOR-0](RESULTS.md#L25286 "id:2026-08-10-verdict-attractor-0-the-raw-map"); [VERDICT
  ATTRACTOR-0B-JOIN](RESULTS.md#L25692 "id:2026-08-10-verdict-attractor-0b-join-answer-form").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] No
  spontaneous transfer resolved in the tested bridge/desert cells; this does
  not support the former universal claim that nothing transfers.
  ([Desert test v2](RESULTS.md#L4383 "id:2026-07-24-desert-test-v2-no-spontaneous-composition").)
- [MECHANISM-CONFIRMED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] A paired federation-diet arm
  showed that exposure share redistributes resident capability; content alone
  did not identify a winner. ([GEN-9 A/B VERDICT](RESULTS.md#L4283 "id:2026-07-24-gen-9-a-b-verdict-redistribution").)
- [SINGLE-SEED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1, one model held the five
  tested grammars simultaneously; this is a gen-8 diet result, not a general
  federation law. ([GEN-8](RESULTS.md#L4127 "id:2026-07-24-gen-8-the-everything-crystal-all").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] Width
  did not improve the tested token-light primitive federation.
  ([45M UNION RE-ASK](RESULTS.md#L4414 "id:2026-07-24-45m-union-re-ask-width-does").)
- [REPLICATED] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  The ZX graph-language birth cleared parsing and capability checks across
  n=3 seeds; the paired seed ladder is the replication route.
  ([ZX SEED-3](RESULTS.md#L7416 "id:2026-07-27-zx-seed-3-34-120-the").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] At the
  tested ZX recipe, the wider arm remained inside the n=3 seed fence; width
  was not a resolved lever in that regime. ([45M UNION
  VERDICT](RESULTS.md#L7740 "id:2026-07-27-45m-union-verdict-math-65-union"); [ZX SEED-3](RESULTS.md#L7416 "id:2026-07-27-zx-seed-3-34-120-the").)

## Learning dynamics

- [REPLICATED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] The packing hole reproduced at
  two widths through an independent CUDA substrate leg; matched-step controls
  located the damage in batch/context composition rather than precision.
  ([d256 SUBSTRATE GATE CLEARS](RESULTS.md#L5321 "id:2026-07-26-d256-substrate-gate-clears-and-packing"); [AMENDMENTS: the
  matched-steps accident separates the packing bundle](RESULTS.md#L5400 "id:2026-07-26-amendments-the-matched-steps-accident-separates").)
- [SINGLE-SEED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 streaming 2×2, revisits,
  cooldown, and batch diversity separated; the result remains one recipe
  cell. ([STREAMING CLOSES](RESULTS.md#L5549 "id:2026-07-26-streaming-closes-the-2x2-completes-epoch").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: closed-system math] A verified-RL
  causal arm moved output policy with little representation change; the mined
  corpus, not the final RL weights, carried the durable transferable product.
  ([GRPO run 2b](RESULTS.md#L1594 "id:2026-07-15-grpo-run-2b-the-hill-climbing").)
- [MECHANISM-CONFIRMED] [FREE-RUN-GATED]
  [REGIME-SCOPED: calculus search] Injecting engine demonstrations moved the
  model gate where self-practice did not, and model-discovered walls improved
  the engine in the reverse arm. ([THE EXCHANGE
  CONVERTS](RESULTS.md#L3916 "id:2026-07-23-the-exchange-converts-2-12-6"); [The exchange is bidirectional IN
  FACT](RESULTS.md#L3883 "id:2026-07-23-the-exchange-is-bidirectional-in-fact").)
- [REPLICATED] [REGIME-SCOPED: specified diet and recipe]
  Sub-ULP recovery in fp64-master metabolism was reproduced by an independent
  paired-arm implementation; the short run established committed updates, not
  a capability premium. ([Metabolic v3 paired arms](RESULTS.md#L3711 "id:2026-07-23-metabolic-v3-paired-arms-the-ceiling").)

## Independent implementation and methodology

- [MECHANISM-CONFIRMED] Booking
  became a PROGRAM: one overnight workflow (7 builders + 7 paired
  hazard reviewers) landed the checkpoint catalog (392 rows, 54
  RESULTS-cited, the 33.1 GB uncited pool now enumerated), the
  Parquet result lake (gates schema REQUIRES device + n_seeds +
  weights_sha non-null — the cross-device fence and the books-with-
  sha rule as schema constraints, graduating the RESULTS 13463
  proposal), merge ops that refuse their own killed mechanisms in
  the API (soups-crater, split-law, ternary-growth citations in the
  docstrings), streamed per-step receipts in the axiom row shape,
  and scripts/book.py — which refuses killed runs, dict-sum
  mismatches, sha-less gates, and unfenced n=1 sub-sigma verdicts,
  and booked its own shipping entry as the dogfood. The morning
  review sweep (four reviewers) then caught a wrong arch label in a
  booked entry (d256 -> d512, the catalog's own data was the
  cross-check), a wrong params column, and a fence-bypass class in
  book.py's first cut — all amended same-day.
  ([VERDICT REFACTOR-NIGHT-1](RESULTS.md#L26536 "id:2026-08-11-verdict-refactor-night-1-catalog-lake").)

- [SINGLE-SEED] [REGIME-SCOPED: closed-system math] A string-seeded
  1000-cert sample of the cross-lab Lean certificate corpus missed its
  registered 1000/1000 kernel pass at 703/1000 — and every one of the
  297 failures was a tactic-script defect rather than a wrong verdict:
  269 proofs succeed before their failing trailing token, and the 28
  "tactic gaps" were rediagnosed as 17 hypothesis-match failures, 7
  atom-split statements, and 4 rewriting failures. The sample is 1000
  of 21,914 rows; the corrected compile rate for a re-emitted sidecar
  is a prediction, not a measurement. ([VERDICT
  LEAN-KERNEL-SAMPLE](RESULTS.md#L20365 "id:2026-08-05-verdict-lean-kernel-sample-registered-1000"); [AMENDMENT
  LEAN-KERNEL-SAMPLE-2](RESULTS.md#L20492 "id:2026-08-05-amendment-lean-kernel-sample-2-0").)
- [RETRACTED] [REGIME-SCOPED: closed-system math] The "0 false
  statements in 1000" reading is withdrawn as stated: 0 false raw
  equalities stands, but 7 of 1000 emitted generalized statements are
  false over free atoms — two textually distinct forms of one subterm
  atomize separately — which no kernel can be tricked into certifying.
  The class now has its own counter, "unprovable-by-design
  (atom-split)"; 2 of the 7 carry no division, so it is not a
  division-specific defect. ([AMENDMENT
  LEAN-KERNEL-SAMPLE-2](RESULTS.md#L20492 "id:2026-08-05-amendment-lean-kernel-sample-2-0").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: closed-system math] A single
  unreproduced test flake, relayed as courtesy rather than dropped,
  resolved into two real defects in the independent implementation: a
  wall-clock search budget whose expiry silently broke determinism,
  and a cache that memoized a budget-aborted rule fire as a permanent
  no-fire — the checkpoint-selection effect's in-process twin.
  Determinism claims for that search are now scoped "modulo observed
  budget expiry" on both ledgers. House verification is diff
  inspection, a rebuild, and a 10x rerun; the stress receipts are the
  other lab's. ([VERDICT BEAM-FLAKE-ROOT-CAUSE](RESULTS.md#L20539 "id:2026-08-05-verdict-beam-flake-root-cause-the");
  [VERDICT LEAN-EMITTER-FIX](RESULTS.md#L20462 "id:2026-08-05-verdict-lean-emitter-fix-axiom-s").)
- [REPLICATED] [REGIME-SCOPED: closed-system math] The cross-lab
  Lean certificate corpus closes at kernel scale with a CLOSED
  failure taxonomy: under the fixed emitter, 21,614 of 21,914
  certificates kernel-check (98.63%; the 1000-row sample's
  registered ~98.9% prediction landed at 98.9% exactly), and every
  one of the 300 failures classifies as either atom-split (222 —
  generalized statement false over free atoms while the raw
  equality stays true; detectable only by semantic equivalence,
  invisible to duplicate-value tests) or open field_simp
  self-refactoring (78), with ZERO residual and no new class. The
  independent-implementation route: axiom's emitter and oracle
  against the lab's own printer, statement-diff, and Mac-local
  kernel; the failure CLASSIFICATION is house-side sympy equivalence,
  since reconciled against axiom's independent count at the same 222
  and 78. The frozen id list and the labeled taxonomy are committed
  as small-text receipts, so the census is re-derivable without
  re-running the kernel pass. ([VERDICT LEAN-SAMPLE-V2](RESULTS.md#L22407 "id:2026-08-07-verdict-lean-sample-v2-p-emitter"); [VERDICT
  LEAN-FULL-V2](RESULTS.md#L22493 "id:2026-08-07-verdict-lean-full-v2-p-corpus"); [RECEIPT
  LEAN-300-LABELS](RESULTS.md#L23777 "id:2026-08-09-receipt-lean-300-labels-target-verdicts").)
- [REPLICATED] [REGIME-SCOPED: house crystals] The hardening sweep
  converts pre-doctrine point estimates into replicated laws and
  scoped fences without retracting any paired claim: the
  eval-in-train exposure class is STRUCTURAL across every legacy
  infix engine diet (22-25 of 120 gate-band expressions verbatim
  prompt-side in each of eight diets — absolute solve counts on
  that band carry a first-ply tailwind fence, paired deltas cancel
  it); the precision-knee sigma-ratio constant holds at n=3
  same-diet births (snap free at ~0.25-0.5 sigma with deltas
  0/-1/0, biting at ~1.0 sigma with -12/-13/-9); the
  fp32-versus-ternary crown tie is real at three fresh paired
  problem-set draws (deltas 0/+2/+2); and the split law's causal
  leg — sparse assignment does not cause expert decorrelation —
  replicates at three soft-routing seeds (correlations
  0.0096/0.0071/0.0062, all in the init-default class), with the
  excision bridge pricing the mandated diet cleanup at zero.
  ([VERDICT HARDENING-P1](RESULTS.md#L22585 "id:2026-08-07-verdict-hardening-p1-the-exclude-union"); [VERDICT
  HARDENING-P2](RESULTS.md#L22658 "id:2026-08-07-verdict-hardening-p2-both-cells-fire"); [VERDICT
  HARDENING-P3-R5](RESULTS.md#L22718 "id:2026-08-08-verdict-hardening-p3-r5-p-init").)
- [REPLICATED] [REGIME-SCOPED: house crystals] The ZX
  scale-lever null is seed-hardened: three same-recipe 45M union
  births score 36/34/32 on the ZX gate (mean 34.0, straddling the
  19M seed-fence mean 32.7), so the original n=1 read of 36 was
  ordinary seed noise and 2.4x capacity buys nothing on the graph
  grammar; the roots-of-unity reopening stays un-fired.
  ([VERDICT HARDENING-P4-2](RESULTS.md#L23459 "id:2026-08-08-verdict-hardening-p4-2-p-null").)
- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] The frozen Phase-3 list completes
  with one real cost and one tie, both at n=3 paired seeds on the
  device of origin. The quaternionic 4x conversion toll is real:
  pooled -17 against the comparator, negative at all three seeds,
  and every heal retrofits fully at anti-mass 0.0007 or less. Its
  replication variable had to change between registration and fire,
  because the registered torch seed measured INERT in that driver —
  three torch-seed arms replayed the original trajectory
  receipt-identically — so the data-order shuffle became the seed
  axis instead, which is a weaker replication route than the one
  registered and is booked as such. Separately, dose-fed ternary
  TIES fp32 at 19M and d384: deltas -1, 0 and -2 for a pooled -3,
  inside the tie band. ([VERDICT
  HARDENING-P3-R7](RESULTS.md#L23168 "id:2026-08-08-verdict-hardening-p3-r7-p-toll"); [AMENDMENT
  HARDENING-P3-R7-ARM](RESULTS.md#L23383 "id:2026-08-08-amendment-hardening-p3-r7-arm-the"); [VERDICT
  HARDENING-P3-R9](RESULTS.md#L23251 "id:2026-08-08-verdict-hardening-p3-r9-p-parity").)
- [NULL] [REGIME-SCOPED: closed-system math] The peeling probe's
  null HARDENS rather than dissolving: the registered suspicion was
  that mid-chain states were out of distribution, so the probe was
  re-run on 119 in-distribution post-step states, each with a known
  valid predecessor in band. Per-candidate validity came out LOWER,
  at 4.3 and 5.0 percent against the original 11 percent, so the
  distribution-artifact explanation is refuted for that row.
  ([VERDICT HARDENING-P4-1](RESULTS.md#L23197 "id:2026-08-08-verdict-hardening-p4-1-p-null").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: house crystals] On one 19M
  vehicle, over-coarse quantization is deletion rather than
  perturbation: snapping a 19M model's weights to
  fractions with denominator <= 4 sends 99.93% of them to exactly
  zero (trained weights sit near sigma 0.034, far under the 1/8
  survival threshold; the mid gate keeps a single 1/4), which is
  why the booked Q=4 gate is 0/120 while the denominator-64 snap
  ties its control exactly.
  ([OBSERVATION Q4-DELETION-RENDER](RESULTS.md#L23428 "id:2026-08-08-observation-q4-deletion-render-the-rational");
  [RATIONAL-SNAP VERDICT](RESULTS.md#L7613 "id:2026-07-27-rational-snap-verdict-the-crystal-has").)
- [REPLICATED] [REGIME-SCOPED: house crystals] The sprint's second
  wave lifts two more n=1 fences and books two honest betweens: the
  ffn-slack anatomy endpoints replicate on their cuda line (pooled
  EMA -4 within the +-6 bar — attention width, not ffn capacity,
  pins capability at d56); the streaming cooldown clause replicates
  at +3/+6/+3 (pooled +12 over the +6 bar); the gravity-tax
  falloff-flip is a 3/3 paired fact while its gate toll (-11
  pooled) sits between the flat and real bars; and the head
  autopsy's sparse critical circuit recurs at every fresh seed
  (single-cell craters -28/-12/-24, 86-92% slack) with the striking
  rider that CELL IDENTITY does not transport — seed-1's
  catastrophic L1h7 is free on all three fresh crystals; the
  addresses are born, not inherited. The revived late-metabolism
  orphan books its first verdict (3/3 negative lean, pooled -10,
  between bars), and the kernel survey closes with no kernel-day
  needed: torch's int64 matmul is bit-identical to the order-free
  sum and the speed riders attach to the plateau-break and MPS-leg
  rungs. ([VERDICT HARDENING-P3-R6](RESULTS.md#L22768 "id:2026-08-08-verdict-hardening-p3-r6-falloff-flip"); [VERDICT
  HARDENING-P3-R3](RESULTS.md#L22867 "id:2026-08-08-verdict-hardening-p3-r3-p-slack"); [VERDICT
  HARDENING-P3-R4](RESULTS.md#L22985 "id:2026-08-08-verdict-hardening-p3-r4-p-cooldown"); [VERDICT
  HARDENING-P3-R2](RESULTS.md#L23044 "id:2026-08-08-verdict-hardening-p3-r2-neither-bar"); [VERDICT
  REVIVE-METAB-LATE](RESULTS.md#L22898 "id:2026-08-08-verdict-revive-metab-late-neither-bar"); [OBSERVATION
  KERNEL-SPEED-SURVEY](RESULTS.md#L22947 "id:2026-08-08-observation-kernel-speed-survey-no-queued").)
- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe] The
  reverse model's verified-candidate choice surface is real but
  thin, and the thinness is mostly a MODEL property: at eight
  samples per problem 4 of 120 problems yield two distinct
  verified predecessors (the forward control yields zero); at
  thirty-two samples the count reaches only 9 of 120 —
  sub-linear widening (the L4 hole cracks to 1/24 at the larger
  budget). The ranking rung of the reverse-propose ladder stays
  gated pending a stronger proposer; and the integer-diet plateau
  is capacity-insensitive in both directions at its anchor
  schedule, leaving joint scaling as the named lever — joint
  scaling subsequently ran and did NOT convert; the measured
  binder at horizon is the optimizer schedule ([VERDICT
  ENGINE-SCALE-1](RESULTS.md#L23673 "id:2026-08-09-verdict-engine-scale-1-neither-bar")).
  ([VERDICT MULT-0](RESULTS.md#L22196 "id:2026-08-07-verdict-mult-0-p-starved-fires"); [VERDICT
  MULT-0-B32](RESULTS.md#L22522 "id:2026-08-07-verdict-mult-0-b32-unresolved-as"); [VERDICT
  P-CAPACITY-2](RESULTS.md#L22163 "id:2026-08-07-verdict-p-capacity-2-does-not").)
- [SINGLE-SEED] [REGIME-SCOPED: toy weight-space subjects] A
  796,550-parameter neuron-token transformer classifies the function
  family of 1-16-16-1 tanh subject MLPs from raw weights at 80.8%
  against a 16.7% chance floor; canonicalized 82.4%,
  permutation-augmented 88.4% — augmentation beat canonicalization,
  the measured basis of the teach-invariance-don't-impose-it rule. One
  seed, one subject architecture, toy scale; the gate is
  classification only, and no run-the-weights gate exists for
  weight-space generation. ([VERDICT
  WEIGHT-READER-0](RESULTS.md#L20747 "id:2026-08-05-verdict-weight-reader-0-back-booked").)
- [NULL] [REGIME-SCOPED: toy weight-space subjects] The
  forward-trained weight reader does not recognize inverse twins:
  0.139 accuracy against the 16.7% chance floor over 345 eligible
  twins, so weight features are direction-specific at toy scale. The
  sin family's 0.692 is flagged as shape resemblance (near-monotone
  draws whose numerical inverse is still sinusoid-shaped), not
  transfer; the registered consequence — that a direction classifier over
  forward and reverse crystals is now expected to succeed — was
  subsequently REFUTED three protocols deep; see the weight-reader
  closure bullet.
  Function-inverse analogy probe — it transfers a question, never a
  conclusion. ([VERDICT TENET-W0](RESULTS.md#L20876 "id:2026-08-05-verdict-tenet-w0-the-null-fires").)
- [SINGLE-SEED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] Backward emission at
  micro scale is prompt-distribution-local — null on the chain-start
  reverse gate (0/120, where the forward control also reads 0) and
  24/120 on the post-step gate against a 1/120 forward control, one
  reversed-token twin, both gates replay-scored at the engine mint.
  The two readings are one finding and do not separate: the twin
  trains in loss exactly like its forward twin, emits nothing at
  prompt shapes it only ever saw as targets, and emits real
  predecessors one step in. L4 reads 0/24 while L3 and L5 score —
  a non-monotone level-local hole, named and open. ([VERDICT
  TENET-R0-REV](RESULTS.md#L20984 "id:2026-08-06-verdict-tenet-r0-rev-the-null"); [VERDICT
  TENET-R0-REV-B](RESULTS.md#L21081 "id:2026-08-06-verdict-tenet-r0-rev-b-backward").)
- [NULL] [REGIME-SCOPED: toy weight-space subjects] Direction is not
  readable from crystal FFN gate weights: over 50 forward/reverse
  birth pairs a neuron-token reader scores exact chance on held-out
  pairs and never fits the training labels, while the same reader
  under the same split separates trained-from-random-init 20/20 —
  a no-signal finding, not a weak instrument. The null fires
  against the toy-scale expectation the W0 rung registered, and it
  sits opposite the function-side measurement: the same pairs
  differ 24-to-1 on the reverse gate. Direction lives in function,
  not sampled gate geometry, at this scale and feature choice.
  Gate tensors only; one classifier seed; 2-epoch births.
  ([VERDICT TENET-W1](RESULTS.md#L21140 "id:2026-08-06-verdict-tenet-w1-the-null-fires"); control rider
  [RESULTS.md#L21169](RESULTS.md#L21169 "id:2026-08-06-rider-on-verdict-tenet-w1-the").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe]
  TENET closes: backward emission exists and is prompt-local, and
  the closed-loop pincer does not bind at this scale because
  verified-candidate multiplicity never appears — not because
  ranking loses. All three matched-budget arms (forward-only,
  reverse-ranked, length-ranked) read an identical 60 of 120 under
  an honest token ledger; the reverse ranker had a rankable moment
  on 8 of 120 problems and flipped none. The 60 itself carries a
  caveat that must travel with it: re-roll headroom alone lifted
  the forward twin from 44 to 60, so 60 is the budgeted-re-roll
  level, not a pincer level. Revival fence: any future pincer case
  measures verified-candidate multiplicity FIRST — at multiplicity
  ~0 a ranking arm is invalid, not negative. Second bar failure
  (R8 under a broken fence, this under a working one); one seed
  pair. ([VERDICT TENET-R1B-MICRO](RESULTS.md#L21236 "id:2026-08-06-verdict-tenet-r1b-micro-the-bar").)

Where axiom is the replication route below, it means an independent
implementation in another language and runtime, sharing no code with what it
checks — and since 2026-08-02 its pinned revisions are public, so those legs
are externally runnable rather than only booked.
[`REPRODUCE.md`](REPRODUCE.md) gives the pinned identity and the command. The
caveat that remains is authorship, not reachability: house and axiom are two
sessions under one operator, not independent investigators. Read
`[REPLICATED]` on those bullets accordingly.

- [REPLICATED] [REGIME-SCOPED: closed-system math] House and axiom oracles
  independently adjudicated the same symbolic rows and exposed errors on both
  sides; the independent implementation is the replication route.
  ([Axiom Phase B adjudicated](RESULTS.md#L2608 "id:2026-07-18-axiom-phase-b-adjudicated-the-generator"); [Qualification:
  443/480](RESULTS.md#L2994 "id:2026-07-20-qualification-443-480-at-5x-sympy").)
- [REPLICATED] [DEVICE-SCOPED]
  [REGIME-SCOPED: deterministic integer battery] The exact decode reproduced
  across Apple/NVIDIA devices and the independent axiom implementation, with
  byte-pinned inputs and zero-tolerance output comparison. ([E3
  VERDICT](RESULTS.md#L9907 "id:2026-07-29-e3-verdict-pass-50-50-the"); [FX-V1-H VERDICT](RESULTS.md#L11468 "id:2026-07-30-fx-v1-h-verdict-cross-lab").)
- [REPLICATED] Paired controls repeatedly overturned attractive headlines and
  found instrument faults; independent strict-encoding and cross-repo E4
  audit routes are named examples rather than a numeric incident count.
  ([The 388 mangled rows](RESULTS.md#L4453 "id:2026-07-25-the-388-mangled-rows-strict-encode"); [E4 AUDIT
  CLOSE-OUT](RESULTS.md#L7492 "id:2026-07-27-e4-audit-close-out-axiom-s").)
- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED] Gate and probe differences
  reproduced across Apple and NVIDIA devices, establishing that device is
  part of the instrument and cross-device scores cannot be compared raw.
  ([L9 probes are device-dependent at 2x](RESULTS.md#L3213 "id:2026-07-21-l9-probes-are-device-dependent-at"); [VERDICT
  NIGHT-31-CUDA](RESULTS.md#L12697 "id:2026-07-31-verdict-night-31-cuda-controls-the").)

## Calibration and generation

- [MECHANISM-CONFIRMED] [TEACHER-FORCED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals]
  Controlled snap interventions across six house crystals showed that
  flips-per-token predicts gate damage; the causal arm is the applied snap,
  not a post-hoc story. ([CALIBRATION PROBE R1
  VERDICT](RESULTS.md#L7943 "id:2026-07-28-calibration-probe-r1-verdict-pass-flips").)
- [NULL] [TEACHER-FORCED] [REGIME-SCOPED: specified diet and recipe]
  Pick-trained valid-set distributions were already near-deterministic, and a
  matched-dose distribution-row arm did not become a capability lever.
  ([MASS-ON-VALID VERDICT](RESULTS.md#L7976 "id:2026-07-28-mass-on-valid-verdict-valid-mass"); [DISTRIBUTION ROWS 3-ARM
  VERDICT](RESULTS.md#L8078 "id:2026-07-28-distribution-rows-3-arm-verdict-parity").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: deterministic integer battery]
  Judge-collapsed decoding did not improve the tested generation battery;
  near ties were too scarce for the intervention to act.
  ([JUDGE-COLLAPSED DECODING VERDICT](RESULTS.md#L8146 "id:2026-07-28-judge-collapsed-decoding-verdict-null-by").)

## Exact tails, optimizer nulls, and symmetry

- [MECHANISM-CONFIRMED] [DEVICE-SCOPED]
  [REGIME-SCOPED: deterministic integer battery] On the measured Apple/NVIDIA
  endpoint pair, the exact-vs-fp64 causal arm localized all differences to
  latent last-bit casts and found no deployed sign-cell change. ([NIGHT-28
  STAGE 1](RESULTS.md#L8247 "id:2026-07-28-night-28-stage-1-d2-endpoints").)
- [NULL] [REGIME-SCOPED: closed-system math] Independent successor
  implementations were sound on the accepted sample, but exact set parity did
  not resolve because their normal forms differ; adoption is scoped to
  soundness, not identical enumeration. ([SUCCESSORS-BRIDGE
  ACCEPTANCE](RESULTS.md#L8281 "id:2026-07-28-successors-bridge-acceptance-soundness-pass-200").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] Muon
  remained gate-toxic at every tested schedule; the null stands without the
  withdrawn CE-dissociation story. ([MUON @ 3EP STANDARD
  SCHEDULE](RESULTS.md#L8311 "id:2026-07-28-muon-3ep-standard-schedule-43-120").)
- [RETRACTED] [TEACHER-FORCED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] The fixed-instrument Muon
  CE/gate-dissociation reading is retracted because CE tracked the gate on that
  instrument. ([CE-GATE STUDY](RESULTS.md#L6414 "id:2026-07-26-ce-gate-study-my-hypothesis-fails").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] At n=1 per
  group cell, dense gates retrofitted into the tested complex, quaternion, Z2,
  and C8 commutants with bounded toll; this is a retrofit ladder, not a
  universal symmetry law. ([ROTATIONAL SNAP R3](RESULTS.md#L8610 "id:2026-07-28-rotational-snap-r3-verdict-the-conversion"); [SYMMETRY
  LADDER S4 + S3](RESULTS.md#L8733 "id:2026-07-28-symmetry-ladder-s4-s3-verdict-z2").)
- [NULL] [REGIME-SCOPED: house crystals] No spontaneous commutant structure
  appeared in the tested dense crystal before intervention — a WEIGHT-BASIS
  statement only: the rotation lenses were later shown blind to a confirmed
  activation clock, so representation-level symmetry is unmeasured by this
  instrument. ([SYMMETRY LADDER
  S1 CELL 1 VERDICT](RESULTS.md#L8665 "id:2026-07-28-symmetry-ladder-s1-cell-1-verdict"); [VERDICT TIER-A
  A2+A3](RESULTS.md#L12814 "id:2026-07-31-verdict-tier-a-a2-a3-a").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: house crystals] Projection followed
  by a warm-training arm kept the imposed commutant locally stable, showing
  that SGD accepted imposed symmetry even though it had not selected it.
  ([ROTATIONAL SNAP R3 VERDICT](RESULTS.md#L8610 "id:2026-07-28-rotational-snap-r3-verdict-the-conversion").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] At n=1, Z2
  and C8 cuts with the same deleted mass produced different gate damage;
  functional direction, not mass fraction alone, controlled the result.
  ([SYMMETRY LADDER S4 + S3 VERDICT](RESULTS.md#L8733 "id:2026-07-28-symmetry-ladder-s4-s3-verdict-z2").)
- [MECHANISM-CONFIRMED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] A
  doubled-width causal construction enforced exact rotational symmetry and
  preserved the measured gate, separating existence from at-width toll.
  ([SYMMETRY LADDER S2 VERDICT](RESULTS.md#L8786 "id:2026-07-28-symmetry-ladder-s2-verdict-exactly-65").)

## Chaos, EMA, and nested crystals

- [SINGLE-SEED] [DEVICE-SCOPED] [FORMAT-BOUND] [TEACHER-FORCED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 NVIDIA perturbation grid,
  trajectories mixed strongly while capability stayed on the same measured
  shell; the claim is device-and-recipe scoped. ([ATLAS-2 LYAPUNOV
  VERDICT](RESULTS.md#L9009 "id:2026-07-28-atlas-2-lyapunov-verdict-chaotic-mixing").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FORMAT-BOUND] [TEACHER-FORCED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 under the tested simple
  NVIDIA schedules, EMA contracted format-bound disagreement and improved the gate.
  ([NIGHT-28b VERDICT](RESULTS.md#L9135 "id:2026-07-28-night-28b-verdict-ema-tames-the").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] Under
  the production schedule, EMA was redundant and was not adopted as the
  default. ([A0 VERDICT](RESULTS.md#L9246 "id:2026-07-29-a0-verdict-ema-is-redundant-under").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] At n=1, a
  joint-loss matryoshka tensor carried dense and circulant tiers; nesting
  across unlike compression axes still consumed measured slack. ([MATRYOSHKA
  RUNG 1 VERDICT](RESULTS.md#L9087 "id:2026-07-28-matryoshka-rung-1-verdict-the-nested").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] At n=1 C8
  reconstruction, capability accumulated with retained spectral mass; this is
  one crystal's frequency-axis anatomy. ([SYMMETRY SPECTRUM
  VERDICT](RESULTS.md#L9200 "id:2026-07-29-symmetry-spectrum-verdict-capability-accumulates-linearly").)
- [SINGLE-SEED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 per grammar-scale cell,
  the C8 sharing toll stayed stable on math but grew on ZX; the result is bound
  to the tested diets and recipes. ([C8-RETROFIT AT 45M
  VERDICT](RESULTS.md#L9113 "id:2026-07-28-c8-retrofit-at-45m-verdict-the").)

## The slack atlas and escalation engine

- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] The d56 width floor tied d64 on
  n=3 paired NVIDIA seeds and agreed with the independently measured Apple
  line; that paired seed-and-device route is the replication. ([NIGHT-29
  VERDICT 1](RESULTS.md#L9414 "id:2026-07-29-night-29-verdict-1-the-width").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 NVIDIA recipe, the FFN arm
  remained flat over the measured range; it does not establish global FFN
  irrelevance. ([NIGHT-29 VERDICT 2](RESULTS.md#L9425 "id:2026-07-29-night-29-verdict-2-ffn-is").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 Apple depth ladder, the
  tested depths tied; depth remains a one-cell flat. ([DEPTH LADDER
  VERDICT](RESULTS.md#L10135 "id:2026-07-29-depth-ladder-verdict-depth-is-slack").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1 NVIDIA dose ladder,
  quarter-cuts were flat; the finding is limited to that diet and recipe.
  ([LEG C VERDICT](RESULTS.md#L10160 "id:2026-07-29-leg-c-verdict-the-marginal-value").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: house crystals] Controlled snap
  sweeps collapsed within each crystal only after adding a per-crystal
  fragility term; a single geometry-only curve was rejected. ([DISTORTION
  COLLAPSE VERDICT](RESULTS.md#L10306 "id:2026-07-29-distortion-collapse-verdict-one-curve-per").)
- [MECHANISM-CONFIRMED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals]
  Causal single-head deletion arms found a sparse set of load-bearing
  layer/head cells rather than uniform head essentiality. ([HEAD AUTOPSY
  VERDICT](RESULTS.md#L10356 "id:2026-07-29-head-autopsy-verdict-essentiality-is-cell").)
- [SINGLE-SEED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] At n=1, immediate-step call spans
  improved the measured gate while end-value spans did not; delegation is
  step-local in this format. ([STEP-LOCAL CALL-SPAN
  VERDICT](RESULTS.md#L9998 "id:2026-07-29-step-local-call-span-verdict-delegation").)
- [SINGLE-SEED] [FREE-RUN-GATED] [REGIME-SCOPED: house crystals] At n=1
  decode battery, tier retry beat the same tensor's dense-only policy; this is
  a controller result for one matryoshka artifact. ([ESCALATION POLICY
  VERDICT](RESULTS.md#L10034 "id:2026-07-29-escalation-policy-verdict-the-ladder-beats").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: specified diet and recipe] The
  reverse model learned inversion but did not produce resolved novel farming
  yield under the tested recipe. ([FARMER PROBE VERDICT](RESULTS.md#L10091 "id:2026-07-29-farmer-probe-verdict-the-reverse-model").)

## The packed-crystal boundary

- [REPLICATED] [FREE-RUN-GATED]
  [REGIME-SCOPED: at-capacity house crystals] Sigma-pack gate parity and
  near-Gaussian code entropy reproduced across n=3 paired d64h8 births; the
  weak width-floor births are the explicit exception. ([C1 AT n=3
  VERDICT](RESULTS.md#L11232 "id:2026-07-30-c1-at-n-3-verdict-h8"); [PACKED CRYSTAL C0+C1
  VERDICT](RESULTS.md#L10406 "id:2026-07-29-packed-crystal-c0-c1-verdict-zero").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: at-capacity house crystals] On the
  matched-bit d64h8 crystal, GPTQ, AWQ, and HQQ did not separate from the
  closed-form calibration-free pack. ([PACKED CRYSTAL C3
  VERDICT](RESULTS.md#L10458 "id:2026-07-29-packed-crystal-c3-verdict-nothing-beats").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts] At n=1 Apple kernel
  benchmark, the bit-packed format executed directly and beat the byte-aligned
  kernel; this is a measured shape/device result. ([PACKED CRYSTAL C2b
  VERDICT](RESULTS.md#L10587 "id:2026-07-29-packed-crystal-c2b-verdict-the-disk").)
- [REPLICATED] [DEVICE-SCOPED]
  [REGIME-SCOPED: deterministic integer battery] Integer-forward hashes
  reproduced across independent Apple and NVIDIA devices; floating logits
  differed even when greedy streams matched. ([PACKED CRYSTAL C4
  VERDICT](RESULTS.md#L10657 "id:2026-07-29-packed-crystal-c4-verdict-claim-3").)
- [SINGLE-SEED] [DEVICE-SCOPED] [TEACHER-FORCED] [REGIME-SCOPED: Qwen2.5-0.5B] At n=1 NVIDIA
  model, per-tensor sigma allocation was 33× worse than HQQ: the house-crystal
  packing law did not transport unchanged. ([PACKED CRYSTAL C6
  VERDICT](RESULTS.md#L10676 "id:2026-07-29-packed-crystal-c6-verdict-the-falsifier").)
- [NULL] [DEVICE-SCOPED] [TEACHER-FORCED] [REGIME-SCOPED: Qwen2.5-0.5B] On that NVIDIA model,
  per-row sigma allocation did not rescue the failure. ([PACKED CRYSTAL C6b
  VERDICT](RESULTS.md#L10709 "id:2026-07-29-packed-crystal-c6b-verdict-per-row").)
- [MECHANISM-CONFIRMED] [DEVICE-SCOPED] [TEACHER-FORCED] [REGIME-SCOPED: Qwen2.5-0.5B] On
  that NVIDIA model, the causal sigma/8 arm recovered 11.6×, while
  max-anchored grids still won; the intervention locates the boundary in the
  heavy-tail knee rather than row granularity. ([PACKED CRYSTAL C6c
  VERDICT](RESULTS.md#L10736 "id:2026-07-29-packed-crystal-c6c-verdict-the-sigma").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] At n=1 nested
  artifact, tiered bytes reduced escalation cost but the joint-STE parent did
  not inherit the zero-tax pack. ([PACKED CRYSTAL C5
  VERDICT](RESULTS.md#L10529 "id:2026-07-29-packed-crystal-c5-verdict-the-nested").)

## Deployment artifacts and the capacity meter

- [REPLICATED] [DEVICE-SCOPED] [TEACHER-FORCED]
  [REGIME-SCOPED: deterministic integer battery] Full transformer logit traces
  reproduced bit-identically on independent Apple and NVIDIA GPU paths; the
  floating-point agreement price remains part of the verdict — and the
  ENGINE-EXACT ladder extended the family across compilers and C++
  standard libraries: the pinned Q32/Q64 rung digests reproduce under
  clang/libc++ and gcc/libstdc++ after the fixture RNG (not the engine)
  was caught as stdlib-dependent and re-pinned, receipts four
  reproductions deep. ([P3 VERDICT](RESULTS.md#L11357 "id:2026-07-30-p3-verdict-the-deterministic-decode-lands");
  [COUNTER-BOOK ENGINE-EXACT-1-RECEIPT](RESULTS.md#L23486 "id:2026-08-09-counter-book-engine-exact-1-receipt");
  [AMENDMENT ENGINE-EXACT-1-DIGESTS](RESULTS.md#L23566 "id:2026-08-09-amendment-engine-exact-1-digests-target").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] At n=1 packing
  pass, rANS converted the measured entropy bound into real lossless bytes for
  the sampled house crystals and one production MoE artifact. ([P6-v2
  VERDICT](RESULTS.md#L11297 "id:2026-07-30-p6-v2-verdict-the-entropy-bound").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] At n=1
  artifact census, the capacity meter ordered the measured house groups and
  sampled MoE experts; this is an empirical allocator predicate, not a
  universal codec law. ([CAPACITY METER VERDICT](RESULTS.md#L10808 "id:2026-07-29-capacity-meter-verdict-the-allocator-predicate");
  [EXPERT-SCALE VERDICT](RESULTS.md#L10995 "id:2026-07-29-expert-scale-verdict-artin-s-prediction").)
- [NULL] [DEVICE-SCOPED] [TEACHER-FORCED] [REGIME-SCOPED: measured deployment artifacts] On
  the measured Apple OLMoE KL/perplexity arms and the Qwen boundary, the
  house-crystal zero-tax allocator did not transport; calibrated/max-anchored
  methods retained an advantage. ([C7 VERDICT](RESULTS.md#L10895 "id:2026-07-29-c7-verdict-strong-form-transport-fails"); [N2
  VERDICT](RESULTS.md#L11133 "id:2026-07-30-n2-verdict-dial-pack-recovers-3").)

## MoE mechanisms and the scaffold correction

- [REPLICATED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts] A shipped Kimi-K3 expert
  forward hash-locked across three independent CPU, Apple, and NVIDIA backend
  implementations; those backends are the replication route. ([K3-D1
  VERDICT](RESULTS.md#L11433 "id:2026-07-30-k3-d1-verdict-one-expert-out"); [K3-D2 VERDICT](RESULTS.md#L11513 "id:2026-07-30-k3-d2-verdict-full-expert-chain").)
- [REPLICATED] [DEVICE-SCOPED]
  [REGIME-SCOPED: deterministic integer battery] Axiom independently
  implemented the full deterministic decode and reproduced both house hashes
  across four backends. ([FX-V1-H VERDICT](RESULTS.md#L11468 "id:2026-07-30-fx-v1-h-verdict-cross-lab").)
- [NULL] [DEVICE-SCOPED] [TEACHER-FORCED] [REGIME-SCOPED: tested MoE recipes]
  On two Apple seeds, ablations were unscreened and amplified with depth;
  router-distance bins restored no lawful field. ([GRAV-1 VERDICT](RESULTS.md#L11686 "id:2026-07-30-grav-1-verdict-no-gravity-influence");
  [GRAV-1b VERDICT](RESULTS.md#L11846 "id:2026-07-30-grav-1b-verdict-not-lawful-in").)
- [MECHANISM-CONFIRMED] [DEVICE-SCOPED] [TEACHER-FORCED] [FREE-RUN-GATED]
  [REGIME-SCOPED: tested MoE recipes] In one paired Apple cell (n=1), the causal
  contractivity arm flattened epsilon-kick propagation at near-zero gate toll.
  ([GRAV-2 VERDICT](RESULTS.md#L11866 "id:2026-07-30-grav-2-verdict-spacetime-is-trainable").)
- [REPLICATED] [TEACHER-FORCED] [REGIME-SCOPED: specified diet and recipe]
  Calibration quality and error detection reproduced on the original crystal
  and two fresh paired births; those n=3 births are the route. ([CAL-DK-1
  VERDICT](RESULTS.md#L11744 "id:2026-07-30-cal-dk-1-verdict-no-dunning"); [GRAV-2 VERDICT](RESULTS.md#L11866 "id:2026-07-30-grav-2-verdict-spacetime-is-trainable").)
- [MECHANISM-CONFIRMED] [TEACHER-FORCED]
  [REGIME-SCOPED: specified diet and recipe] Target-corruption arms moved
  calibration monotonically toward doubt, establishing diet dilution as the
  cause in the tested recipe. ([CAL-DK-2 VERDICT](RESULTS.md#L11922 "id:2026-07-30-cal-dk-2-verdict-dilution-breaks").)
- [NULL] [REGIME-SCOPED: tested MoE recipes] Neither balance-loss nor
  soft-routing causal arms explained expert decorrelation in the tested MoE
  recipes. ([UMOE-1 VERDICT](RESULTS.md#L11592 "id:2026-07-30-umoe-1-verdict-the-split-law"); [UMOE-2
  VERDICT](RESULTS.md#L11896 "id:2026-07-30-umoe-2-verdict-soft-routing-does").)
### Frontier deploy instruments (V4-Flash)

Systems results on a shipped 304B artifact — deploy/decode
instruments and measurements, not house-recipe MoE laws; the
regime tag on every bullet is the fence.

- [REPLICATED] [FORMAT-BOUND] [REGIME-SCOPED: measured deployment artifacts]
  A 304B frontier MoE's routed expert runs entirely in integers on the vendor's
  own shipped 4-bit weights, with the output trace hash-identical on Apple CPU,
  Apple GPU and NVIDIA GPU — three backends are the route, each having fetched
  the weights independently, and the decode was checked against the vendor's
  own dtype semantics. One expert, one layer, one model: replication is of the
  integer-exactness route, not MoE integer exactness in general. ([VERDICT
  V4-RUNG-A](RESULTS.md#L15884 "id:2026-08-02-verdict-v4-rung-a-a-deepseek"); the NVIDIA leg is a rider —
  [RIDER V4-RUNG-A](RESULTS.md#L15933 "id:2026-08-02-rider-on-verdict-v4-rung-a").)
- [SINGLE-SEED] [FORMAT-BOUND] [REGIME-SCOPED: measured deployment artifacts]
  A frontier 304B MoE's shipped 4-bit expert codes carry 3.865 bits of entropy,
  and their 8-bit power-of-two block scales carry 0.964 — so the scale stream
  is 5.9% of the bytes but 62% of the losslessly recoverable headroom, on a
  3-layer sample. ([VERDICT V4-RUNG-0/1](RESULTS.md#L15776 "id:2026-08-02-verdict-v4-rung-0-1-all").)
- [SINGLE-SEED] [FORMAT-BOUND] [REGIME-SCOPED: tested MoE recipes]
  The shared component a gravity decomposition looks for is present in a
  frontier MoE's ROUTER GEOMETRY, not its weights: at layer 22 all 32,640
  gate-key pairs are positively aligned and every key shares a common
  direction, while pairs closest in routing space show no more weight
  structure than random ones. ([VERDICT V4-RUNG-R +
  2B-ROUTER](RESULTS.md#L16183 "id:2026-08-02-verdict-v4-rung-r-2b-router"); qualified below.)
- [NULL] [REGIME-SCOPED: measured deployment artifacts]
  An attempt to measure a frontier MoE's router input by inverting its trained
  load-balancing bias FAILED its registered predictions: a shuffled-bias null
  excludes the same inputs the trained bias does, so the exclusion is a
  property of the key geometry and the synthetic input family, not of the
  balancer. The shared direction's share of top-6 selection is bounded to
  7-21% under that family and remains unlocated. ([AMENDMENT
  FINAL-0803](RESULTS.md#L17024 "id:2026-08-03-amendment-final-0803-amends-verdict-v4"); [VERDICT V4-RUNG-D2](RESULTS.md#L16937 "id:2026-08-03-verdict-v4-rung-d2-prediction-1").)
- [SINGLE-SEED] [FORMAT-BOUND] [REGIME-SCOPED: measured deployment artifacts]
  Entropy-coding a frontier MoE's shipped 4-bit experts yields an archive,
  not a runtime: the byte-lossless form saves 8.3% and decodes at 38 MB/s
  single-threaded, 131x under the pipe a streaming design assumes and 90 s
  per token of pure decode for a 43-layer, 6-expert route. ([VERDICT
  V4-RUNG-D + S0](RESULTS.md#L16483 "id:2026-08-03-verdict-v4-rung-d-s0-the"); traffic corrected in [AMENDMENT
  RUNGD-0803](RESULTS.md#L16608 "id:2026-08-03-amendment-rungd-0803-amends-verdict-v4").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  A 304B checkpoint's own parameter census overturned the reason it cannot
  be run locally: the always-on dense path is 9.4 GB, not the 27 GB a
  subtraction implied, because 19 of those "dense" billions are three
  multi-token-prediction blocks each holding a full 256-expert layer.
  ([RECEIPT V4-CENSUS](RESULTS.md#L16816 "id:2026-08-03-receipt-v4-census-the-27b-dense").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  A 304B frontier MoE's entire GPU dependency is six kernels, and a
  pure-torch twin of them passes oracle acceptance on cpu and mps — the
  fp4 decode bit-identical to the certified decoder, every gemm within
  1/128 of an exact fp64 reference on real shipped weights. ([VERDICT
  V4-F1a](RESULTS.md#L17263 "id:2026-08-03-verdict-v4-f1a-the-kernel-twin").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  Under RANDOM weights an untrained 3-layer boot of the same model
  amplifies cross-device bf16 noise ~3x per layer (0.12 to 0.56) while
  each kernel individually sits at 4e-3 — and with TRAINED weights the
  same comparison collapses to ~2e-3 per layer, so end-to-end tolerance
  bars belong to trained networks only. ([VERDICT
  V4-F1b](RESULTS.md#L17303 "id:2026-08-03-verdict-v4-f1b-the-vendor-s"); [VERDICT V4-F1c](RESULTS.md#L17360 "id:2026-08-03-verdict-v4-f1c-real-v4-flash").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  DeepSeek-V4-Flash generated tokens on a 36 GB Mac — vendor code
  unmodified over the kernel twin, 7.13% of routed experts resident
  (785/11,008 measured, fetch-on-miss included; the K=16/256 per-score-layer
  design ratio is 6.25% — same run, two denominators),
  masking done by one write into the aux-loss-free bias — and what
  survives 92.9% expert amputation is fluent CONTEXT-COPYING: the output
  is a pure prompt-echo loop, logged verbatim, no capability claimed.
  ([VERDICT V4-F1d](RESULTS.md#L17401 "id:2026-08-03-verdict-v4-f1d-deepseek-v4-flash").)
- [NULL] [REGIME-SCOPED: measured deployment artifacts]
  Neither making the biggest weights fast (bf16-dequanting the 40x-larger
  dense path: no measurable gain) nor caching the hot ones (774-tensor
  per-token working set thrashed any affordable cap, 5x slower) moved a
  compute-bound MoE decode; making the hot OP cheap did — a [256,2]
  pair-LUT unpack bought 2.2x at zero memory. ([VERDICT
  V4-F1e](RESULTS.md#L17578 "id:2026-08-03-verdict-v4-f1e-2-2x-from").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  Bit-exact batching of the per-layer expert calls (equivalence gate
  0.00e+00 — reproducing the vendor's three per-projection bf16 roundings
  was the contract) bought only 1.22x more, so neither launches nor
  batching bound the decode: the wall is unpack VOLUME per token, which
  only multi-token verification can amortize. ([VERDICT V4-F1e
  ARM5](RESULTS.md#L17674 "id:2026-08-03-verdict-v4-f1e-arm5-bit-exact"); corrections in [AMENDMENT
  F1-REVIEW](RESULTS.md#L17713 "id:2026-08-03-amendment-f1-review-amends-verdicts-v4").)
- [NULL] [FORMAT-BOUND] [REGIME-SCOPED: tested MoE recipes]
  Experts in a 256-expert frontier layer are statistically identical yet
  individually unmatchable: sorted hidden-unit norm profiles agree to 7%, while
  pairing a unit with its counterpart is indistinguishable from pairing it at
  random, so aligning by permutation before differencing buys 0.02 bits/param
  against a 0.2 bar. ([VERDICT V4-RUNG-2B](RESULTS.md#L16033 "id:2026-08-02-verdict-v4-rung-2b-n3-survives").)
- [SINGLE-SEED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts]
  The sign bit of a symmetric 4-bit weight code is incompressible to five
  decimals (1.00000 bits on all 27 tensors, against a 0.9995 bar), so the
  coding margin sits in the magnitude alphabet. ([VERDICT V4-RUNG-0/1](RESULTS.md#L15776 "id:2026-08-02-verdict-v4-rung-0-1-all").)
- [REPLICATED] [TEACHER-FORCED] [REGIME-SCOPED: deterministic integer battery]
  Consensus pull between experts buys mergeability monotonically: merge damage
  falls with dose and reaches exactly zero at full pull, where experts collapse
  to bit-identical agreement, at all three paired init draws. ([VERDICT
  DIET-COND-SEED](RESULTS.md#L15615 "id:2026-08-02-verdict-diet-cond-seed-the-interior").)
- [SINGLE-SEED] [TEACHER-FORCED] [REGIME-SCOPED: deterministic integer battery]
  The best dose is interior at every draw, but its location is not stable: the
  optimum sits at 1/4 for one draw and 1/16 for two others, and the 1/4 dose is
  worse than no gravity at both of the latter, so no dose is recommended.
  ([VERDICT DIET-COND-SEED](RESULTS.md#L15615 "id:2026-08-02-verdict-diet-cond-seed-the-interior").)
- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: tested MoE recipes] Hebbian-coupled experts merged to one
  dense expert with zero-mean bounded gate cost across n=3 Mac seeds and an
  independent CUDA birth; paired seeds plus device transport are the route.
  ([VERDICT MERGE-CUDA](RESULTS.md#L12679 "id:2026-07-31-verdict-merge-cuda-merge-free-goes").)
- [NULL] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: tested MoE recipes] The stale scaffold capability advantage
  did not transport: on n=3 paired CUDA seeds, gravmoe and baseline means tied
  at 50.7 versus 50.7. The retained recipe is merge-free and
  capability-neutral. ([VERDICT CUDA SEED LADDER](RESULTS.md#L13286 "id:2026-07-31-verdict-cuda-seed-ladder-the-gravmoe").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] At d64 micro scale, the average of
  independently born weights is not a worse model, it is not a model: all
  six independent-pair averages
  gated exactly 0/120 with zeros at every level, while the shared-init fork
  merge landed inside the 12-15 parent band and the task-vector child
  cratered to 1/120. Four identical-recipe births spanned 12-30/120 — the
  d64 seed lever dwarfs every diet and schedule lever measured at this
  scale. ([VERDICT MERGE-SPACE-1](RESULTS.md#L26770 "id:2026-08-11-verdict-merge-space-1-r1-independent").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] At d64 micro scale, two three-epoch
  trainings that never shared an optimizer step still merged inside the
  parent band when they shared initialization (and epoch-seeded data order —
  the named fence): the basin is chosen at birth, not by the trajectory.
  ([VERDICT MERGE-SPACE-2](RESULTS.md#L26866 "id:2026-08-11-verdict-merge-space-2-r2-p").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] The deconfound closed the chain: with data
  order made independent (ORDER_SEED fork), the same-init merge still landed
  in-band — in fact at the top of it. Shared initialization alone keeps two
  never-synchronized d64 trainings mergeable; order moved the gate by one
  solve, init is the address. ([VERDICT MERGE-SPACE-3](RESULTS.md#L26947 "id:2026-08-11-verdict-merge-space-3-r2b-p").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] Souping four order twins is free but not
  profitable: the uniform 4-way soup gated exactly at the family max (14),
  nowhere near the +7 profit bar. The greedy-soup effect at d64 is
  damage-free flattening onto the best member, not lift; a byte-identical
  rerun of one inner merge doubled as a merge-determinism receipt.
  ([VERDICT MERGE-SPACE-4](RESULTS.md#L27013 "id:2026-08-11-verdict-merge-space-4-r2c-soup").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] The loss floor walks down monotonically
  across a fresh 8x width ladder (0.4364 to 0.3478, d64 to d512) but never
  approaches the k=32 entropy knee: the whole ladder buys roughly one token
  of effective context past the 16-gram wall. Width sets the floor —
  a fresh d512 landed on the grown crown's reference line — while gate
  capability saturates in width long before the floor does.
  ([VERDICT FLOOR-HK-1](RESULTS.md#L27055 "id:2026-08-11-verdict-floor-hk-1-r3-both").)
- [REPLICATED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] Init-is-the-address, replicated: at n=3
  paired seeds spanning the family's 11-31 capability range, every
  same-init independent-order merge landed in the parent band and none
  cratered, while every independent-init merge in the same family had gated
  exactly zero. Merges flatten onto the better pair member at every seed.
  ([VERDICT MERGE-SPACE-5](RESULTS.md#L27158 "id:2026-08-11-verdict-merge-space-5-r2d-p").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: house crystals] Substituting a minimal selective state-space
  block for attention at d64 lost on both axes against its paired twin — floor
  0.5675 versus 0.4381 nats, gate 2 versus 38 of 120, at 22.6 times the wall
  clock. The registered prior held: the roughly 16-token effective-context wall
  survives the opposite inductive bias, so it belongs to the diet rather than
  to attention. The arm is near non-functional at this scale, so it licenses
  nothing about state-space models generally.
  ([VERDICT SSM-STAR-1](RESULTS.md#L27282 "id:2026-08-11-verdict-ssm-star-1-the-house").)
- [SINGLE-SEED] [REGIME-SCOPED: house crystals] A truncation probe on trained
  checkpoints — no retraining — found that deep positions do exploit long
  context: at every one of four widths, loss at positions past 128 falls by
  0.90 to 1.25 nats as the visible window grows from 16 tokens to 128. The
  training-loss average could not see this, because it mixes those positions
  with a majority that are locally predictable. The probe's second registered
  bar, that wider models separate further as context grows, did NOT fire: the
  width gap is negative at k=8 and narrows again at k=128, so it does not
  order monotonically in k. One diet, one probe, loss only, no gate.
  ([VERDICT KEFF-PROBE-1](RESULTS.md#L27332 "id:2026-08-11-verdict-keff-probe-1-bar-1").)
- [SINGLE-SEED] [TEACHER-FORCED] [REGIME-SCOPED: closed-system math] At n=1 activation probe,
  a small population of digit-periodic neurons showed partial Fourier
  character; this is an in-vivo sighting, not a general representation law.
  ([FOURIER-1 VERDICT](RESULTS.md#L12096 "id:2026-07-30-fourier-1-verdict-amended-instrument-a").)

## The routing crest and domain coalitions

- [REPLICATED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Masking a resident
  30B-class MoE to the top 45.3% of its per-layer math-demand experts
  (58 of 128) BEAT the paired full model on the 120-item mathgen gate at
  every one of six paired seeds — +14/+14/+16 descriptive, then
  +17/+9/+18 fully registered at three unspent seeds, pooled +14.7
  against a registered +7 bar. Six paired seeds are the replication
  route; the claim is a gate claim on one vehicle, one keep rule, and
  mathgen L1-3, not a general quality claim, and the text/coherence
  dissociation still applies. ([VERDICT MOE-GT-1-R4](RESULTS.md#L18861 "id:2026-08-04-verdict-moe-gt-1-r4-the");
  [VERDICT MOE-GT-1-R5](RESULTS.md#L18927 "id:2026-08-04-verdict-moe-gt-1-r5-confirmed").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] The effect is
  selection, not sparsity: at the identical 45.3% keep fraction, two
  random masks and one anti-demand mask each scored 0/120, so WHICH
  experts are kept is the difference between 0 and 82 of 120. Generic
  sparsity contributed nothing measurable at that fraction. ([VERDICT
  MOE-GT-1-R6](RESULTS.md#L19108 "id:2026-08-04-verdict-moe-gt-1-r6-the").)
- [SINGLE-SEED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] At one unspent seed the
  argmax over masked fractions stayed at 45.3% (82/120 against a paired
  full model's 60), so the interior peak's LOCATION did not move the way
  the diet-dose optimum did; the 45.3-versus-50 margin of 7 on that seed
  is about 1.5 sigma on the 120-prompt gate and is not resolved by
  it. ([VERDICT MOE-GT-2-D1](RESULTS.md#L19225 "id:2026-08-04-verdict-moe-gt-2-d1-crest"); [VERDICT
  DIET-COND-SEED](RESULTS.md#L15615 "id:2026-08-02-verdict-diet-cond-seed-the-interior").)
- [SINGLE-SEED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts] The deployed router is
  domain-organized: decode-only expert coalitions at 45.3% keep have
  Jaccard 0.8013 between mathematics and mechanics and 0.5331 between
  mathematics and code, against same-filter within-domain split-half
  nulls of 0.9205, 0.8670, and 0.6364 — the code coalition sits below its
  own null. The ordering was registered before the runs from corpus
  prompt-token overlap (0.329 mathematics-mechanics versus 0.097
  mathematics-code). One seed per domain; tie-filled keep-set boundaries
  make every cross-domain Jaccard an upper bound. ([VERDICT
  MOE-GT-2-D2](RESULTS.md#L19254 "id:2026-08-04-verdict-moe-gt-2-d2-the"); [VERDICT MOE-GT-2-D3](RESULTS.md#L19294 "id:2026-08-04-verdict-moe-gt-2-d3-the");
  numbers corrected for the prompt-tail phase bug in [AMENDMENT
  GT2-REVIEW-2](RESULTS.md#L19648 "id:2026-08-04-amendment-gt2-review-2-three-seat").)
- [NULL] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] The beats-full crest did
  not transport to mechanics under either mask recipe: the decode-built
  physics mask returned pooled +3 against the +7 bar with signs 2/3, and
  the exclusion-matched pooled-built rescue arm returned pooled -59 with
  signs 3/3. The mechanics baselines spread 24-42 of 120 across seeds, so
  that arm was underpowered to resolve +7 in EITHER direction; the
  registered rescue prediction nonetheless inverted far outside any noise
  reading. ([VERDICT MOE-GT-2-D4-PHYS](RESULTS.md#L19340 "id:2026-08-04-verdict-moe-gt-2-d4-phys"); [VERDICT
  MOE-GT-2-D4-PHYS-B](RESULTS.md#L19528 "id:2026-08-04-verdict-moe-gt-2-d4-phys-b"); power fence in [AMENDMENT
  GT2-REVIEW](RESULTS.md#L19417 "id:2026-08-04-amendment-gt2-review-reviewer-pass-on").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Coverage and recall do
  not predict the SIGN of masking's capability effect. On matched
  instruments the mechanics rescue mask reproduced the mathematics
  crest's readouts — open recall 0.9013 versus 0.9014, closed recall
  0.8838-0.8848 inside mathematics's 0.8822-0.8891, decode-demand
  coverage 0.8908 versus 0.8811 — while mathematics gained +14.7 and
  mechanics lost -59. Exclusion mass is therefore killed as a sufficient
  mechanism, and what the excluded experts COMPUTE is the remaining
  variable. ([VERDICT MOE-GT-2-D4-PHYS-B](RESULTS.md#L19528 "id:2026-08-04-verdict-moe-gt-2-d4-phys-b"); confound
  removal measured in [AMENDMENT GT2-REVIEW-2](RESULTS.md#L19648 "id:2026-08-04-amendment-gt2-review-2-three-seat");
  exclusion axis in [AMENDMENT GT2-EXCLUSION](RESULTS.md#L19482 "id:2026-08-04-amendment-gt2-exclusion-the-transport-null").)
- [SINGLE-SEED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Cross-domain masks
  degrade severely without dying: a mathematics mask on the code gate
  scored 21/120 against that gate's 48, and a code mask on the
  mathematics gate 19/120 against 64, at 77.6-80.2% demand coverage —
  between random-45%'s 44.7% coverage and 0/120 and own-domain crest's
  90.1% and 76-87/120. What this establishes is that at least one
  INTERMEDIATE regime exists; the intervals between the measured points
  were never sampled, so no curve shape is claimed, and the surviving
  code rung under the mathematics mask was a lenient-checker artifact,
  leaving functional-versus-topical UNRESOLVED. ([VERDICT
  MOE-GT-2-D4-CROSS](RESULTS.md#L19378 "id:2026-08-04-verdict-moe-gt-2-d4-cross"); both demotions booked in
  [AMENDMENT GT2-REVIEW](RESULTS.md#L19417 "id:2026-08-04-amendment-gt2-review-reviewer-pass-on").)
- [SINGLE-SEED] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] A judge built only from
  readouts available at decision time — per-problem closed-loop recall,
  parse success, completion length, level — predicted which crest
  failures the full model rescues at held-out AUC 0.679 against a
  registered 0.60 bar and a tie-corrected level-only null of 0.450, and
  the two loudest features say the rescuable failures are MASK-CAUSED
  rather than problem-hard. The escalation spend bar is UNRESOLVED, not
  missed: at the single registered budget the judge recovered 31 solves
  against 36 needed, a 4.25-solve gap inside the 120-prompt fence,
  while sitting about 2.1 standard deviations above random escalation.
  ([VERDICT CHURN-JUDGE-1](RESULTS.md#L19614 "id:2026-08-04-verdict-churn-judge-1-auc-bar"); interpretation corrected
  in [AMENDMENT GT2-REVIEW-2](RESULTS.md#L19648 "id:2026-08-04-amendment-gt2-review-2-three-seat").)
- [SINGLE-SEED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts] The three measured
  coalitions share a core of 37.1 of 58 experts per layer against an
  independence null of 11.9, and the containment is class-like: 92% of
  the mathematics-code intersection also lies in mechanics. This is a
  desk computation on one demand log per domain with no mask arm run, and
  the three corpora share chat template, English prose, and expression
  emission — so the core could have been the generic decoding substrate
  rather than any symbolic or logical one. The discriminating arms have
  since run and killed that reading: the core is symbolic (proofs
  coalition contains 0.90 of it, plain prose 0.25, with prose routing
  through a nearly different expert population at 0.16-0.19 Jaccard).
  ([OBSERVATION GT2-CORE-0](RESULTS.md#L19718 "id:2026-08-04-observation-gt2-core-0-a-three"); [VERDICT
  MOE-GT-3](RESULTS.md#L19852 "id:2026-08-05-verdict-moe-gt-3-the-core").)
- [SINGLE-SEED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts] A
  second verbal corpus showed the verbal side is its own branch with its
  own shared base (prose-dialog Jaccard 0.72 vs 0.16-0.24 cross-branch,
  including the system-swapped arm;
  verbal core 48.5/58 per layer vs a 26.3 two-way independence null,
  nearly disjoint from the symbolic core at 0.095 Jaccard), and a
  system-prompt swap control priced the shared-prefix confound at
  ~0.05-0.10 Jaccard — too small to generate the branch separation. The
  core alone is dead on the math gate (0/120): necessary, not
  sufficient — and what restores capability above it requires
  verbal-branch residents (necessary, not sufficient; the recall
  reading is corrected in the bullets below).
  ([OBSERVATION GT2-CORE-0](RESULTS.md#L19718 "id:2026-08-04-observation-gt2-core-0-a-three"); [VERDICT
  MOE-GT-3](RESULTS.md#L19852 "id:2026-08-05-verdict-moe-gt-3-the-core"); [VERDICT MOE-GT-4](RESULTS.md#L19953 "id:2026-08-05-verdict-moe-gt-4-the-verbal")
  as corrected by [AMENDMENT MOE-GT-4-REVIEW](RESULTS.md#L20006 "id:2026-08-05-amendment-moe-gt-4-review-corrected");
  [VERDICT MOE-GT-4b](RESULTS.md#L20111 "id:2026-08-05-verdict-moe-gt-4b-the-branch").)
- [REPLICATED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] A structural 61.1% mask
  built as the union of the two branch cores loses to the paired full
  model at every seed tried: 55/48/42 of 120 against full baselines of
  64/73/60 — 3/3 negative, pooled deficit 17.3 solves, above the
  registered 5-15 band, so the first seed's 86% of full capability
  became 66% and 70% at fresh seeds (pooled 73.6%). Which experts are
  kept still costs real capability at 61% keep when the identity is
  wrong. ([VERDICT MOE-GT-5](RESULTS.md#L20179 "id:2026-08-05-verdict-moe-gt-5-neither-prediction"); [VERDICT
  MOE-GT-5b](RESULTS.md#L20267 "id:2026-08-05-verdict-moe-gt-5b-the-union"); ranges per [AMENDMENT
  DAY-CONSOLIDATION-0805](RESULTS.md#L20575 "id:2026-08-05-amendment-day-consolidation-0805-two-numeric").)
- [RETRACTED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Above the symbolic
  core, what restores capability is demand recall, not expert class:
  matched-size random fills over the dead core (0/120 alone) scored
  51/36/48 at recall 0.749-0.763 against the verbal fill's 55 at
  0.729, killing the reading that the verbal base is specifically
  load-bearing for mathematics. CORRECTION, same day: that random
  pool was ~45% verbal-branch, so "random" fills carried verbal
  experts — clean verbal-EXCLUDED fills at matched 0.72 recall score
  0/120 and 7/120 against 16-55 for verbal-containing fills, and the
  recall ladder over random fills is non-monotone with widening draw
  spread. The verbal population IS load-bearing (necessary, not
  sufficient); recall is not the organizing variable either.
  Fences of the retracted reading: one seed, three draws; the union
  mask's edge over the random band's mean is inside the draw spread
  and unresolved; the registered per-answer degeneracy count was not
  collected. ([VERDICT MOE-GT-5c](RESULTS.md#L20414 "id:2026-08-05-verdict-moe-gt-5c-random-fill"); [AMENDMENT
  MOE-GT-5c-SCOPE](RESULTS.md#L20860 "id:2026-08-05-amendment-moe-gt-5c-scope-the"); [VERDICT
  MOE-GT-6](RESULTS.md#L20793 "id:2026-08-05-verdict-moe-gt-6-the-recall").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Verbal-branch
  experts are necessary for resurrecting masked mathematics
  capability: fills that exclude the verbal branch at matched 0.72
  recall score 0 and 7 of 120 against 16-55 for fills that include
  it, math-extension coverage without verbal experts does not rescue
  (0.30 set-fraction coverage per the committed re-derivation,
  dead), and a recall ladder over
  random-identity fills is non-monotone with draw spread widening as
  recall rises — a 0.795-recall draw scores 9 while a 0.792 draw
  scores 66, at or above the paired full model's 64. Necessary is
  not sufficient: two arms fail with high coverage on every measured
  lens and stay open. The core-only baseline is 0/120 at three
  seeds. One gate seed, two draws per bin, and the exploratory
  coverage correlations (verbal-only 0.73-class against global
  recall 0.53) carry no causal claim beyond the registered
  exclusion contrast; the original desk Spearmans are demoted to
  unverified and the committed script is the lens authority.
  ([VERDICT MOE-GT-6](RESULTS.md#L20793 "id:2026-08-05-verdict-moe-gt-6-the-recall"); [AMENDMENT
  MOE-GT-6-LENSES](RESULTS.md#L21269 "id:2026-08-06-amendment-moe-gt-6-lenses-the").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Expert IDENTITY, not
  any aggregate property of the keep-set, determines masked
  mathematics capability: with global recall pinned to the 0.72
  band and verbal-only coverage registered as the ladder variable,
  bin means are non-monotone (3.5 / 34.0 / 29.5 / 28.0 at coverage
  0.00 / 0.15 / 0.30 / 0.45) and within-bin draws matched on BOTH
  recall and coverage differ by 36-49 solves of 120 (15-v-53,
  5-v-54, 46-v-10). Both previously named recall-ladder anomalies
  dissolve into draw identity — fresh draws at the r80 point score
  75 and 48 against the booked 9. Coverage joins recall as a dead
  aggregate lens; the registered survival map forecloses any
  further aggregate-lens ladder and directs the thread to
  excluded-experts anatomy. Distinct-answer degeneracy tracks
  collapse (59-113 distinct of 120 on sub-20 arms against 118-120
  elsewhere) and is descriptive this rung. One gate seed, two draws
  per bin, tie-fill upper bounds travel.
  ([PRE-REG MOE-GT-7](RESULTS.md#L21304 "id:2026-08-06-pre-reg-moe-gt-7-the"); [VERDICT
  MOE-GT-7](RESULTS.md#L21521 "id:2026-08-06-verdict-moe-gt-7-p-null").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Capability FOLLOWS
  a named expert-identity set under pinned aggregates: a symmetric,
  class-preserving exchange of the top-4-by-demand exclusive experts
  per layer per class (~8 of ~68 kept per layer) between the high
  and low draw of a matched bin — recall within the 0.72 band,
  verbal coverage exactly invariant by construction — moved 23-52
  gate solves per side and largely inverted the pair outcomes
  (low draws 15/5/10 rose to 55/57/59; two high draws fell 53->30
  and 46->11), pooled transfer 176 against a pre-registered bar of
  28 with 2/3 directional consistency. One anomaly is carried
  loudly, not claimed: one swapped high draw rose to 77, above the
  paired full model's 64; the direction later held at 3/3 gate seeds
  (pooled +30) WITHOUT licensing a beats-full claim — see the carrier
  bullet ([VERDICT EX1-C30-REPL](RESULTS.md#L22007 "id:2026-08-07-verdict-ex1-c30-repl-p-third"); [VERDICT
  EX1-C30-REPL-3](RESULTS.md#L22034 "id:2026-08-07-verdict-ex1-c30-repl-3-unresolved")). One gate
  seed; swapped sets derived from the frozen coverage-ladder draws;
  the bisection ladder (k=2, k=1) is the registered follow-up.
  ([PRE-REG EX-ANAT-1](RESULTS.md#L21603 "id:2026-08-06-pre-reg-ex-anat-1-the"); [VERDICT
  EX-ANAT-1](RESULTS.md#L21639 "id:2026-08-06-verdict-ex-anat-1-p-follow").)
- [MECHANISM-CONFIRMED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] The carrier
  mechanism decomposes: named top-demand exclusive experts are
  PORTABLE (adding four per layer per class transforms weak
  keep-sets by +22 to +62 solves over count/class/coverage-matched
  random controls, pooled +113 against a pre-registered bar of 42),
  SATURATING (the same additions change strong keep-sets by -9 to
  +2), and REDUNDANT at depth two (strong sets hold their score
  under top-2-per-class removal even at reduced recall — the
  symmetric-swap "compensation" was genuine redundancy, not
  add-masking). Identity-selected additions are also UNIFORM where
  matched-random additions reproduce the draw lottery (48-73
  versus 8-79). A swapped set derived from two ~0.72-recall draws
  ran above its paired full model at three gate seeds (+13, +13,
  +4; pooled +30) but the third seed missed the per-seed fence and
  the discovery seed contaminates the pool — direction booked,
  beats-full NOT claimed; full-model baselines vary by gate seed
  (64/60/73), so mask-versus-full comparisons stay paired per
  seed. One gate seed for the arm batteries; the symmetric-swap
  claim localizes to the k=4 exchange.
  ([VERDICT EX-ANAT-1B](RESULTS.md#L21704 "id:2026-08-06-verdict-ex-anat-1b-p-follow"); [VERDICT
  EX-ANAT-2](RESULTS.md#L21943 "id:2026-08-07-verdict-ex-anat-2-p-portable"); [VERDICT
  EX1-C30-REPL](RESULTS.md#L22007 "id:2026-08-07-verdict-ex1-c30-repl-p-third"); [VERDICT
  EX1-C30-REPL-3](RESULTS.md#L22034 "id:2026-08-07-verdict-ex1-c30-repl-3-unresolved"); [DESK
  R-EMISSION-0](RESULTS.md#L21850 "id:2026-08-07-desk-r-emission-0-emission-and").)
- [REPLICATED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts] Two crest claims
  book at three fresh paired gate seeds each (the discovery pool
  quarantined): a keep-set derived by identity surgery from two
  ~0.72-recall draws beats the full model (+53 pooled, 3/3), and
  deleting a NAMED 80-expert carrier set — 1.3% of the bank, 98.7%
  keep — beats the full model (+55 pooled) and beats matched-rank
  random deletion (+27 pooled), 3/3 both. The carrier population is
  REGIME-DUAL: the same named experts that transform carrier-poor
  masked sets are net interference in the full router (single-seed
  discovery +17 over full, controls +3/+7, decode-lens set quiet at
  +4), and they are prefill-tilted in free routing (2.21x bank
  share on prompt reading, 0.84x on decode — one decode trajectory,
  descriptive). The control carried
  its own finding: rank-matched random deletion alone beats full
  (+28 pooled, 3/3) — the router is over-inclusive at the carrier
  rank class, and identity roughly doubles the deletion gain. Two
  interventions at opposite keep fractions (54.7% and 98.7%) land
  the same gain class, consistent with one interference-removal
  mechanism. CLAIM HIERARCHY, for weighting: GT-1's crest (+14.7
  pooled, 6/6 paired seeds, 45.3% keep) remains the MOST-REPLICATED
  crest claim (its per-seed effect is smaller; seed count is the
  hierarchy axis); the swap and deletion crests are n=3 each
  with their discovery seeds quarantined; the over-inclusion
  finding is n=3 but scoped to the carriers' rank class only — a
  uniform-random deletion control (pre-registration owed) must run
  before any "generally over-inclusive" reading. Scope: one
  vehicle, mathgen L1-3, drift finding scoped
  to the carriers' rank class.
  ([VERDICT EX-ANAT-3](RESULTS.md#L22353 "id:2026-08-07-verdict-ex-anat-3-neither-bar"); [DESK
  EX-ANAT-3-0](RESULTS.md#L22128 "id:2026-08-07-desk-ex-anat-3-0-the"); [VERDICT
  EX-FRESH](RESULTS.md#L22454 "id:2026-08-07-verdict-ex-fresh-both-crest-claims").)
- [NULL] [REGIME-SCOPED: toy weight-space subjects] Training
  DIRECTION (forward versus reverse curriculum) is invisible to
  weight readers across every inspected representation while
  remaining loudly visible in function: single-row features from
  five surfaces (FFN gate, attention, up-projection, embeddings,
  norm gains) and cross-layer write-to-read alignment features
  (sampled rows and full spectra) all read at chance on 50 paired
  d64/L8 subjects with a protocol whose positive control reads
  trained-versus-random at ceiling. Weight-reader mining is closed
  pending a hypothesis that names why a new feature family could
  see composition order where the alignment cross-Gram cannot.
  ([VERDICT TENET-W1-S](RESULTS.md#L21574 "id:2026-08-06-verdict-tenet-w1-s-all-four"); [VERDICT
  TENET-W1-R](RESULTS.md#L21901 "id:2026-08-07-verdict-tenet-w1-r-both-arms").)
- [REPLICATED] [REGIME-SCOPED: deterministic integer battery] The
  deterministic integer birth is device-free at full scale: the
  1000-step multi-block anatomy and the real-diet bridge replay
  bit-identically on a second machine (all sixteen milestone
  losses, both final trajectory shas, the 15,909-to-12,518 plateau
  to the token; both machine legs execute on CPU — the same scope
  as the earlier device fence).
  ([VERDICT LOCKSTEP-A1/A2](RESULTS.md#L21460 "id:2026-08-06-verdict-lockstep-a1-a2-pass-on").)
- [SINGLE-SEED] [REGIME-SCOPED: deterministic integer battery] The
  real-diet plateau is mechanistically bracketed, single runs:
  constant learning rate REGRESSES at long horizons (12,518 to
  13,540 at 4000 steps — the decay law at diet scale), window
  count saturates at fixed parameters (32 and 64 windows land
  together near 14,680), and the schedule is the measured
  binder: decay bends the plateau 5.9% at 4,000 steps (11,777) and
  BREAKS it at 16,000 (9,821 = 21.5% below the anchor, bar was
  10%), while const-lr regresses further with horizon
  (15,045-15,320 at s16000) and neither params (31k-110k) nor
  windows moved the top corner toward the bar — joint scaling did
  not convert; the next lever is a decay ladder. Loss-level only,
  no capability reading attaches. ([VERDICT
  PLATEAU-BREAK](RESULTS.md#L21874 "id:2026-08-07-verdict-plateau-break-unresolved-as-registered"); [VERDICT
  P-STEP-BOUND-2](RESULTS.md#L21920 "id:2026-08-07-verdict-p-step-bound-2-the"); [VERDICT
  ENGINE-SCALE-1](RESULTS.md#L23673 "id:2026-08-09-verdict-engine-scale-1-neither-bar").)
- [SINGLE-SEED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts] With the full router
  available, mathematics routes 2.7%
  of decode demand through verbal-only experts and 75.4% through the
  symbolic core, so the verbal branch is not a normal path for
  mathematics; the masked arms measured a literal fallback that
  appears only once the symbolic extensions are removed. One decode
  trajectory; tie-fill upper bounds; recall shares are not comparable
  across demand-log axes. ([OBSERVATION
  GT-VERBAL-SHARE](RESULTS.md#L20298 "id:2026-08-05-observation-gt-verbal-share-when-free").)

## The clock-placement and deterministic-birth close

- [MECHANISM-CONFIRMED] [DEVICE-SCOPED] [TEACHER-FORCED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe] On Apple training runs, causal
  wide-Mod and digit-sum diet arms separated single-pass competence from
  external rewrite competence: clocks marked where the computation ran, not
  whether the task was solved. ([VERDICT FOURIER-2b](RESULTS.md#L12620 "id:2026-07-31-verdict-fourier-2b-clocks-follow-competence");
  [VERDICT FOURIER-3](RESULTS.md#L12750 "id:2026-07-31-verdict-fourier-3-the-algorithm-substitutes").)
- [SINGLE-SEED] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: deterministic integer battery] At n=1 NVIDIA control cell,
  the 120-prompt house gate could not resolve deltas below about five solves;
  this is a house methods fence, not a universal sampling theorem. ([VERDICT
  NIGHT-31-CUDA](RESULTS.md#L12697 "id:2026-07-31-verdict-night-31-cuda-controls-the").)
- [REPLICATED] [DEVICE-SCOPED]
  [REGIME-SCOPED: deterministic integer battery] A 200-step integer training
  trajectory reproduced across Apple CPU, NVIDIA CUDA, and the independent
  axiom C++ implementation; independent devices and implementation are the
  route. ([VERDICT DETERMINISTIC-BIRTH R2](RESULTS.md#L13255 "id:2026-07-31-verdict-deterministic-birth-r2-mini-a"); [RECEIPT R2 C++
  LEG](RESULTS.md#L13502 "id:2026-07-31-receipt-r2-c-leg-axiom-the").)
- [REPLICATED] [REGIME-SCOPED: deterministic integer battery] All sixteen
  pinned gravmoe trajectory hashes reproduced on a second machine; the route
  is an independent machine with a different CPU architecture, and both legs
  ran on CPU, so this is narrower than GPU transport. ([VERDICT
  GRAVMOE-P4-DEVICE](RESULTS.md#L14889 "id:2026-08-01-verdict-gravmoe-p4-device-the-entire"); [AMENDMENT
  P4-DEVICE-SCOPE](RESULTS.md#L15160 "id:2026-08-02-amendment-p4-device-scope-amends-verdict").)
- [REPLICATED] [REGIME-SCOPED: deterministic integer battery] Ten of those
  same pinned hashes reproduced inside axiom's independent C++ engine after
  one rounding-placement fix; the route is an independent implementation.
  ([VERDICT GRAVMOE-P4-LAB](RESULTS.md#L15015 "id:2026-08-01-verdict-gravmoe-p4-lab-the-gravmoe").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: deterministic integer battery] A
  reduce-to-one-expert parity gate cannot see the gate-backward rounding
  placement it was trusted to cover: the folded and pre-rounded forms are
  equal whenever a single expert routes, so only the cross-implementation
  hash pins could expose the defect. ([VERDICT
  GRAVMOE-P4-LAB](RESULTS.md#L15015 "id:2026-08-01-verdict-gravmoe-p4-lab-the-gravmoe").)
- [SINGLE-SEED] [REGIME-SCOPED: deterministic integer battery] At one seed
  and shape pair, every integer primitive in the battery's forward and
  backward chain returned bit-identical results on Apple MPS and on NVIDIA
  CUDA against CPU; this covers primitives only, and no pinned trajectory has
  been run on a GPU. ([RIDER on AMENDMENT
  P4-DEVICE-SCOPE](RESULTS.md#L15210 "id:2026-08-02-rider-on-amendment-p4-device-scope").)
- [SINGLE-SEED] [TEACHER-FORCED]
  [REGIME-SCOPED: deterministic integer battery]
  At n=1 seed and window set, softening only the query and key draws cut
  battery loss from 8883 to 2496; the earlier arm's booked 73% increase
  belonged to four other matrix families softened with them, retiring the
  peaked-attention reading. ([VERDICT QK-RESCOPE](RESULTS.md#L14658 "id:2026-08-01-verdict-qk-rescope-amends-verdict-qk").)
- [REPLICATED] [FREE-RUN-GATED] [REGIME-SCOPED: deterministic integer battery]
  Against a COND-only control on this diet, the softer query/key draw lifted
  TRAIN free-run solves from zero to two or three of eight at all three paired
  init draws on one device, while HELD-OUT solves stayed at zero throughout.
  That is a train-side gain at a low absolute level under one diet, not free-run
  capability. ([VERDICT QK-SEED3](RESULTS.md#L15462 "id:2026-08-02-verdict-qk-seed3-three-paired-init").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: deterministic integer battery]
  What that initialization removes is measurable before training: the share of
  zero attention probabilities at birth is about 0.89 without it and exactly
  zero with it at all three draws, so the train-side gain tracks the diagnostic
  rather than the seed. It leaves the held-out gap untouched, which the
  exposure-bias cells attribute to diet width. ([VERDICT QK-SEED3](RESULTS.md#L15462 "id:2026-08-02-verdict-qk-seed3-three-paired-init").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: deterministic integer battery]
  One-step scheduled sampling halved free-run token accuracy, 56 to 22 of
  140, and doubled teacher-forced loss, so the registered exposure-bias
  treatment missed its bar at this scale. ([NULL
  GRAVMOE-SS](RESULTS.md#L14703 "id:2026-08-01-null-gravmoe-ss-one-step-scheduled").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: deterministic integer battery]
  Four times the training steps and four times the parameters both failed to
  convert: the 825,984-parameter arm solved zero of eight where the
  208,192-parameter baseline solved two, so at this diet the residual
  free-run gap is neither compute nor capacity. ([VERDICT GRAVMOE-BRUTE,
  closing](RESULTS.md#L14986 "id:2026-08-01-verdict-gravmoe-brute-closing-brute-does"); [leg 1](RESULTS.md#L14856 "id:2026-08-01-verdict-gravmoe-brute-leg-1-neither").)
- [NULL] [FREE-RUN-GATED] [REGIME-SCOPED: deterministic integer battery]
  Masking the loss to the answer region removed the solves as well, two of
  eight to zero of eight, while format diagnostics stayed intact —
  parseability flat and termination improved — so the registered
  format-failure branch did not apply and the loss was capability, not
  formatting. ([VERDICT SOL-ADOPTION-1](RESULTS.md#L15081 "id:2026-08-01-verdict-sol-adoption-1-the-sol").)
- [MECHANISM-CONFIRMED] [TEACHER-FORCED]
  [REGIME-SCOPED: deterministic integer battery]
  A paired 8000-step arm diverged at constant learning rate
  and stayed bounded under integer decay, with both arms bit-identical until
  the first decay point; long integer runs measure the schedule unless decay
  is present. ([VERDICT GRAVMOE-BRUTE-B/C](RESULTS.md#L14936 "id:2026-08-01-verdict-gravmoe-brute-b-c-partial").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: deterministic integer battery] The
  activation clamp was an integer overflow guard for the norm's mean-square
  scaling, and the crash a raised clamp produced was an evaluation-order
  artifact: factoring the scale exactly removed it with every pinned
  trajectory unchanged. ([VERDICT GRAVMOE-BRUTE-B/C](RESULTS.md#L14936 "id:2026-08-01-verdict-gravmoe-brute-b-c-partial");
  [AMENDMENT RMS-HEADROOM](RESULTS.md#L15053 "id:2026-08-01-amendment-rms-headroom-amends-verdict-gravmoe").)

## The ignition ladder and diet refinement

- [SINGLE-SEED] [DEVICE-SCOPED] [REGIME-SCOPED: specified diet and recipe]
  Across four refinement grades of the same content, only the
  oracle-verified diet ignited on a width ladder at all: ignition width 56
  for the verified grade and none at any tested width for the polluted,
  duplicated, and token-shuffled grades. The vacuum control was the
  best-fed of the four after strict-encode filtering, so it failed for lack
  of structure rather than lack of exposure. One of the two ordering steps
  is resolved — the two degraded grades tie off the ladder, leaving
  pollution versus duplication untested, and the width 56 itself is
  threshold-fragile at a single seed.
  ([VERDICT METALLICITY-1](RESULTS.md#L27523 "id:2026-08-11-verdict-metallicity-1-p-metallicity-fires").)
- [SINGLE-SEED] [DEVICE-SCOPED] [REGIME-SCOPED: specified diet and recipe]
  A freshly born architecture twin gated between the pre-registered bars,
  narrowing the growth premium to a few solves — under the resolution floor
  at one seed, so the premium is neither confirmed nor refuted by this cell.
  ([VERDICT GROW-DECOMP-1](RESULTS.md#L27494 "id:2026-08-11-verdict-grow-decomp-1-r5-cell").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: specified diet and recipe]
  Widening the logit vector changes sampled gate numbers even when the
  model is bit-identical. A virtual-token harness rebuilt at a wider
  vocabulary reproduced the stock model exactly — zero logit difference,
  identical argmax, every copied tensor equal — and still gated
  differently with no prefix present. The cause is the sampler, not the
  model: the eight appended zero-probability columns leave the
  distribution untouched but change how many random values the
  multinomial draw consumes, so one generator reused down a rollout
  desynchronizes from the second token onward. Single draws from a fresh
  generator agree in all 200 seeds tested; sequential draws do not.
  Neither a weights hash nor a logit comparison detects this, so gate
  numbers are comparable only across runs whose sampler saw the same
  number of categories.
  ([VERDICT SOFT-PROMPT-1](RESULTS.md#L27633 "id:2026-08-11-verdict-soft-prompt-1-instrument-invalid");
  [AMENDMENT SOFT-PROMPT-1-SAMPLER](RESULTS.md#L27693 "id:2026-08-12-amendment-soft-prompt-1-sampler-the").)
- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search]
  The fast wave-verifier's old parity ship bar is retired with cause:
  its 10 "accept flips" are the OLD doit-based oracle falsely
  rejecting ~8% of corpus-true rows it cannot solve, probed directly
  (two flipped rows, old oracle False on both); zero flips in the
  unsound direction, and the fast path runs 31.5x faster (600.4 s ->
  19.1 s, n=1 CPU receipt). Soundness stays held by the vendored
  167/167 Phase D replay. ([VERDICT
  VERIFY-FAST-BAR-RETIRED](RESULTS.md#L27754 "id:2026-08-13-verdict-verify-fast-bar-retired-the").)

- [SINGLE-SEED] [ONE VEHICLE] A 19M birth teed every 900 optimizer
  steps with Adam state on disk shows per-step weight speed decaying
  ~100x monotonically (1.06e-4 -> 1.0e-6 mean ||dW||/step over
  15,420 steps) with a NARROW per-milestone speed distribution — the
  settling is collective, not a fast/slow neuron split, and Adam's
  exp_avg momentum is a visibly different velocity (spikes early,
  decays slower). Schedule-driven by construction (OneCycle);
  portrait-only, no capability read. ([OBSERVATION
  PHASE-PORTRAIT-1](RESULTS.md#L27800 "id:2026-08-13-observation-phase-portrait-1-a-19m").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] At n=1 (the phase19m
  birth, SEED=2, Mac MPS), capability LEADS weight-space settling:
  the standard 120 gate over all 18 step-milestones rises 0 -> 64
  with Spearman rho 0.979 vs step, reaches 90% of final capability
  at step 10,800 while per-step weight speed is still ~10x above its
  floor, and the final decade of speed decay (12,600 -> 15,300) buys
  ~1 solve. Co-movement, not cause (OneCycle drives both).
  Exploratory rider: L4 is strictly the worst gate level at every
  milestone from m001800 on — the generator's intended difficulty
  ladder disagrees with measured difficulty at L4. ([VERDICT
  CAP-V-TRAJ-1](RESULTS.md#L27909 "id:2026-08-13-verdict-cap-v-traj-1-all").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] At n=1 (phase19m
  m015300, Mac MPS), prompt LENGTH does not carry the L4 gate dip:
  L6 problems are 1.4x longer than L4 (median 81.5 vs 57.5 tokens)
  yet score marginally more (8 vs 7), and the pooled within-level
  rho(length, solved) = -0.2997 sits a knife-edge inside the -0.30
  bar (a real but modest length effect, disclosed as such). The L4
  shape family stays under-crystallized relative to its length
  across three lineages (6/24, 8/24, 7/24). ([VERDICT
  LENGTH-VS-L4-1](RESULTS.md#L28050 "id:2026-08-13-verdict-length-vs-l4-1-both").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] The L4 gate dip is a
  first-step recognition failure: on m015300, 16/17 L4 failures
  emit ZERO oracle-valid steps (ply-0 death, no wandering) while
  all 7 solves finish in 1-3 plies from clean f'(g)*fn(g) surfaces
  — all-or-nothing first-ply pattern recognition, not chain
  management or depth exhaustion. ([OBSERVATION
  L4-PLY0-1](RESULTS.md#L28104 "id:2026-08-13-observation-l4-ply0-1-the-l4").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] OneCycle DIRECTION
  is a non-factor at 19M/gen4: a birth with the lr sequence played
  backwards (anneal-first, warm-last; peak at step 14,957/15,420)
  gates 62/120 vs the standard run's paired 64 — inside 1.5 sigma —
  and reproduces the L4=7 dip exactly (schedule-direction-invariant,
  supporting the structural L4 reading). House prior predicted
  25-50; wrong by ~15 solves. ([VERDICT
  BACKWARD-SCHEDULE-1](RESULTS.md#L28261 "id:2026-08-13-verdict-backward-schedule-1-bar-2").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] Half-schedule
  training is FREE at 19M/gen4: the full OneCycle shape compressed
  to 50% of steps gates 60/120 vs the standard 64 (inside 1.5
  sigma) at half the tokens and wall, and BEATS matched-steps
  truncation (51) by 9 — the anneal tail's shape, not step count,
  does the polishing; 30% compression breaks (49). L4 prints 7/24
  in every full-shape arm across three independent births and drops
  only with exposure (3/24 at 0.3x) — schedule-invariant,
  exposure-sensitive. ([VERDICT
  COMP-LADDER-1](RESULTS.md#L28315 "id:2026-08-13-verdict-comp-ladder-1-bars-1").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] Training has an LR
  ABSORPTION FLOOR (~2-4e-5 at 19M/gen4, ~10% of max_lr): the
  backwards birth's capability curve is DEAD for a full quarter
  (0/120 through step 2,700, lr <= 2.3e-5), takes off once lr
  clears ~4e-5, builds to 48/120 without ever needing peak lr
  (M-STEPS fires), and takes a transient 12-solve dip in the
  highest-lr era that heals by the endpoint. The LLMUE pilot's
  preserved-without-growing null (LR 1e-5) is the same floor from
  the metabolic side. ([VERDICT
  CAP-V-TRAJ-2](RESULTS.md#L28421 "id:2026-08-13-verdict-cap-v-traj-2-bar").)

- [MECHANISM-CONFIRMED] [FORMAT-BOUND]
  [REGIME-SCOPED: specified diet and recipe]
  Zero training loss is arithmetically impossible
  on the gen4 diet: it is a one-to-many function — 4,356 prompts
  map to >= 2 distinct valid next steps (25,916 rows, 15.7% of the
  diet), putting an irreducible cross-entropy floor of >= 0.174
  nats/row under any model at any capacity. Plateau detectors must
  trigger on improvement-rate, never absolute loss. ([OBSERVATION
  DIET-AMBIGUITY-1](RESULTS.md#L28562 "id:2026-08-13-observation-diet-ambiguity-1-zero-training").)

- [NULL] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts]
  The uniform-deletion control closes the router over-inclusion
  question the narrow way: neither primary bar fires — deleting 80
  uniform-random experts from the 30B bank is neither consistently
  helpful (+19 and -1 pooled across two independent draws, one in
  the pre-named dead zone) nor cleanly confined, while the
  rank-matched deletion's +28 reproduces exactly and top-80-by-
  demand deletion costs -26 with a booked 9.56% recall confound.
  Over-inclusion stays claimed at the carriers' rank class only;
  draw-to-draw spread missed the variance-dominates bar by one
  solve. ([VERDICT EX4-UNIF](RESULTS.md#L28597 "id:2026-08-13-verdict-ex4-unif-neither-primary-bar").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] Data ORDER at
  birth is not free: a fixed easy-to-hard curriculum (levels
  blocked in the measured capability order) gates 54/120 v the
  shuffled control's 64 at matched init and schedule — a 10-solve
  cost, with validity down 13 points and L4 falling to 3/24, the
  same value the 0.3x-compression arm produced. The plateau-gated
  arm read -1 only because its pinned constants admitted every
  level by step 2,600 (the pre-named "always admits" branch), so
  it was ~83% a stock stream. Schedule direction commutes; data
  order does not. ([VERDICT CURRICULUM-1](RESULTS.md#L28680 "id:2026-08-13-verdict-curriculum-1-p-order-hurts").)

- [MECHANISM-CONFIRMED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts]
  The uniform-draw split in the EX4 battery is not carrier-class
  contamination: both draws overlap the carriers' rank windows
  near-identically (16 v 18 of 1,080 slots, desk-deterministic),
  eliminating that candidate; the draws separate instead on summed
  demand deleted, and battery-wide the demand-vs-gate shape is
  non-monotone (51k demand deleted reads -1, 77k reads +19, the
  rank-matched 117k reads +28, the top-ranked 452k reads -26) — a
  two-draw direction, no claim. ([OBSERVATION
  EX4-COMPOSITION-1](RESULTS.md#L28840 "id:2026-08-14-observation-ex4-composition-1-the-carrier").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] The curriculum
  mirror arm settles the direction question and unsettles the
  mechanism: hard-first level-blocked order gates 37/120 v the
  shuffled control's 64 and easy-first's 54 — order is harmful in
  BOTH directions, with validity falling in the same order
  (62.01 to 49.12 to 32.62). Neither registered mechanism bar
  fired: L3 fell 8 (the lr-placement direction) while L4 stayed
  at 3/24 despite streaming into the high-lr era (the structural-
  scar direction), so each candidate explains one tail and
  neither books. Shuffled interleaving is the measured optimum of
  the three orders tested on this line. ([VERDICT
  REV-LADDER-1](RESULTS.md#L28872 "id:2026-08-14-verdict-rev-ladder-1-neither-mechanism").)

- [NULL] [FORMAT-BOUND] The first checkers instrument cell repeats
  the T-count day-one lesson, now 2-for-2 on new domains: bounded
  search (2,000 nodes) ties greedy material on a win-in-20
  classification over 200 six-piece positions (92.0% v 88.0%,
  inside the registered +-5 null band) — and the anatomy indicts
  the bar itself: the constant predictor scores 90.5% on the
  skewed label, above one arm and within 1.5 points of the other,
  while the arms fail in opposite directions (proof-based search:
  0 false alarms, 3/19 wins proved in budget; material: 18/19
  wins seen, 23 false alarms). Instrument-scoped; no game claim.
  ([VERDICT CHECKERS-0](RESULTS.md#L29048 "id:2026-08-14-verdict-checkers-0-p-greedy-wins").)

- [SINGLE-SEED] [REGIME-SCOPED: house crystals] The one-swap
  discriminator inverts the house prior and names the curriculum
  mechanism: moving L3 from sixth to second in the hard-first
  ladder lifts L5+L6+L7 from 19 to 25 solves at UNCHANGED
  positions for those levels (the registered foundation bar
  fires — early easy data lifts the levels it precedes), while
  L3's own prime schedule placement leaves its cell below the
  placement bar (17 v 21 threshold). Order damage follows what
  PRECEDES a level; the arm still books the third direction of
  order harm (45 v the shuffled 64). ([VERDICT
  SWAP-LADDER-1](RESULTS.md#L29096 "id:2026-08-14-verdict-swap-ladder-1-p-foundation").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe]
  Interleaving 6,000 rule-tagged engine one-ply atoms into the
  stock stream at matched compute lifts the gate 64 to 73 (past
  the 7-solve resolution) and the L4 cell 7 to 12 — a lineage
  record, exactly the cell L4-PLY0-1 located as a first-ply
  recognition deficit. Both bars fired at knife-edge values (73
  v 72 threshold; 12 at threshold); replication and dose ladder
  owed. The constructive inversion of the order arc: supply the
  foundation, don't reorder it. ([VERDICT
  ATOM-DIET-1](RESULTS.md#L29250 "id:2026-08-14-verdict-atom-diet-1-both-bars").)

- [REPLICATED] Axiom's IV6 bindings land bit-exact against their
  registered acceptance: PyRand matches CPython 3.12 random.Random
  element-wise over 16 house seed shapes (string and big-int) and
  all bound methods, count_ops matches sympy 1.14 on 204/204
  identical-sstr expressions, predecessors gains its {rows,
  expired} flag, and compile-time GIT_SHA/BUILD_TIME attrs
  downgrade the dual-.so hazard to a one-attr check. One envelope
  limit: getrandbits(k>64) raises where CPython allows any k.
  ([VERDICT AXIOM-IV6-ACCEPT](RESULTS.md#L29337 "id:2026-08-14-verdict-axiom-iv6-accept-pyrand-count").)

- [REPLICATED] Axiom's IV7 batch exposes all six ranked
  counter-book primitives and every one re-derives exact
  house-side: replay_verify discriminates real from bogus
  histories, RNS/CRT round-trips arbitrary-precision rationals
  against fractions.Fraction with honest ok=False on exhausted
  moduli, the 61-bit prime ladder regenerates, anchor-v2 counters
  read out (ledger correctly fenced to probe builds), wide
  accumulators match big-int references at 81-bit sums with a
  single RoundHalfAway placement, and sha256 matches hashlib. One
  semantics finding relayed: LeanCert eligible=True is a lexical
  pre-filter, not a provability verdict — non-ring identities
  (sin^2+cos^2, exp*exp-negative) emit certs that will fail `by
  ring`; Lean itself must stay the final rejector.
  ([VERDICT AXIOM-IV7-ACCEPT](RESULTS.md#L29417 "id:2026-08-14-verdict-axiom-iv7-accept-all-six").)

- [REPLICATED] [REGIME-SCOPED: specified diet and recipe] The
  one-ply atom diet's gate lift replicates across birth seeds at
  matched 15,420-step horizon: paired stock-v-atoms births at
  seeds 3 and 4 both read 64 v 70 (+6 each, mean +6, zero harms),
  and the atoms arm's L4 cell reads 12 at every seed tried
  (12/12/12 over seeds 2-4) while stock L4 wobbles 7/6/8 — the
  diet pins first-ply recognition, seed-invariantly at this dose.
  Magnitude is one resolution unit from the 7-solve bar so it
  stays modest-confidence; direction is positive at every seed
  with both registered bars firing cleanly (no knife-edge this
  time). Dose and rule generality still unmeasured.
  ([VERDICT ATOM-DIET-LADDER-1](RESULTS.md#L29465 "id:2026-08-14-verdict-atom-diet-ladder-1-the").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe] The
  one-ply atom effect does not transport across emitters at
  matched dose: axiom-emitted atoms (IV7 emit_chain, sympy
  re-verified) deliver +2 and a stock-level L4 where sympy-emitted
  atoms delivered +6 and L4 12, at identical dose, seed, horizon,
  and interleave mechanics — the registered emitter-tie bar missed
  and EMITTER-DIVERGES booked. Within the axiom family the dose
  response is monotone (64/66/72 at 1/3.5/7% nominal) with no
  flooding scar; WHAT the atoms are (rule mix, answer forms,
  survivor censoring) carries a large share of the effect, and the
  rule-ablation rung inherits a ranked hypothesis list.
  ([VERDICT ATOM-DOSE-LADDER-1](RESULTS.md#L29662 "id:2026-08-15-verdict-atom-dose-ladder-1-emitter").)

- [NULL] Attaching the branch distribution to the diet's 15.7%
  conflicted rows (trie soft targets at branch tokens, same rows
  and compute) moved nothing at 19M: valid-set mass +0.0033
  against a +0.05 bar and a within-noise gate reading — the
  pre-registered REFUTED-IF triggered. The model already parks 63%
  of its teacher-forced mass on the valid-answer sets after plain
  one-hot training, so the ambiguity floor is a loss-metric
  property, not a recoverable capability tax at this recipe.
  ([VERDICT SOFT-NEXT-1](RESULTS.md#L29733 "id:2026-08-15-verdict-soft-next-1-refuted-at").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe] The
  heurisch-ablation test of the emitter-divergence suspect list
  fires its carrier bar at exactly the threshold: removing the
  2,782 i_heurisch rows from the frozen sympy atom shard costs 3
  L4 solves against a dose-matched random control (L4 8 v 11, 68
  v 72 total), and the fire lands AGAINST the level-mix gradient
  (the ablated shard is the L4-richer one, 43.85% v 38.81%).
  Direction-grade only: one seed, knife-edge delta on a
  24-problem sub-scale. Rider: 3,218 random sympy rows read 72 —
  at or above the full 6,000-row shard's 70 at this seed.
  Receipt caveat: both receipt rows mislabel the emitter field
  (hardcode inherited from the dose driver); corrections booked
  in the verdict. ([VERDICT RULE-ABLATE-1](RESULTS.md#L29916 "id:2026-08-15-verdict-rule-ablate-1-rule-carrier").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe]
  Collapsing the 25,852 conflicted rows into one weighted
  soft-target row per group (13% diet cut, matched epochs) holds
  the gate as a non-inferiority result (soft 64 v control 62,
  within run noise) while cutting 12.96% of steps — but the
  Python soft-correction loop adds +3.58% per step, so wall-clock
  saved lands 9.84% and the pre-registered 10% AND-bar NO-FIREs
  at knife-edge. Rider with a standing fence attached: mps fp32
  training is run-level NONDETERMINISTIC at fixed seed (paired
  20-step probe, different weight hashes), so cross-RUN weight-sha
  identity is never again a precondition on Mac; paired in-run
  arms remain valid. ([VERDICT SOFT-SPEED-1](RESULTS.md#L30064 "id:2026-08-15-verdict-soft-speed-1-quality-holds");
  amendment L29985.)

- [NULL] [REGIME-SCOPED: specified diet and recipe]
  Shared-prefix (tree) training has no room on the stock diet: an
  exact census puts the reuse ceiling at 4.48% of token-linear
  FLOPs and 1.91% of attention FLOPs (4,274 conflict groups,
  25,588 rows, prompts short and branches whole-answer) — the
  rung dies before implementation, and the idea is parked for
  future tree-shaped data (search trajectories, rollouts) where
  shared prefixes dominate the sequence.
  ([OBSERVATION TREE-CENSUS-0](RESULTS.md#L30033 "id:2026-08-15-observation-tree-census-0-shared-prefix").)

- [NULL] [REGIME-SCOPED: specified diet and recipe]
  The gradient/data-worth atlas (end-of-training signatures,
  frozen 7-metric set) fails its
  pre-registered retrodiction gate: no metric ranks i_heurisch
  above the non-heurisch remainder on the L4-failure column (raw
  cosine 0.244 v 0.287; the R1 sympy-v-axiom leg passes 4 of 5
  metrics), and the pre-declared L6 falsifier fires against the
  atlas — a 0.094 cosine gap between families whose L6 gate cells
  tied 9/9, tracking a 2.9x shard-composition difference. The
  atlas ranks by composition, not capability-worth; instrument
  dead in this metric set, BASICS-DIET runs unshaped. Riders: the
  no-gradient surface-kNN control gets the passing leg WRONG
  (gradient cosine added real signal there); i_heurisch is the
  most internally redundant family measured; forward/backward mps
  signatures repeat at 1.000000 same-process.
  ([VERDICT GRAD-MAP-0](RESULTS.md#L30127 "id:2026-08-15-verdict-grad-map-0-retrodiction-gate").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe]
  The stock diet already states algebra explicitly — 29,988 bare
  symbolic rows, 18.2% of rows and 21.3% of tokens, five named
  families at 6,000 rows each — while arithmetic appears on 12
  incidental rows (0.01%), none of them a division. Held-out
  probing at three frozen checkpoints puts arithmetic at a floor
  (0.00-0.83% pass@1, 1.7-3.3% pass@8) beneath a resident algebra
  family that is itself weak (7.5-10.8% pass@1), and the counted
  failure mode names the mechanism: 68-76% of expand misses carry
  the correct x**2 and constant coefficients with only the cross
  term wrong — the one coefficient that requires arithmetic
  rather than copying. Basics rungs should target arithmetic, not
  algebra.
  ([OBSERVATION BASICS-CENSUS-0](RESULTS.md#L30259 "id:2026-08-15-observation-basics-census-0-basics-probe").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe]
  2,545 one-ply arithmetic rows at 1.54% dose lift standalone
  arithmetic from a 1.67% floor to 61.67% pass@8 (34.17% pass@1
  from zero) at a -2 gate dent that is inside same-seed run noise
  — but the transfer rider is FLAT: the model that now multiplies
  standalone still misses the cross term inside (ax+b)(cx+d) at
  the control rate (72/108 v 79/108), while the near-format
  numeric-sum arm rises 17.5 to 26.67. Learning radiated by
  format distance, not by operation: arithmetic competence is
  format-local at this dose, the first measured datum for the
  format-as-routing frame. Scoped to small-integer operands.
  ([VERDICT BASICS-DIET-1](RESULTS.md#L30467 "id:2026-08-15-verdict-basics-diet-1-both-bars").)

- [NULL] [REGIME-SCOPED: specified diet and recipe] Stating the
  cross-term computation as its own step INSIDE the expand format
  (2,600 xexp/xstep rows, 1.6% dose) does not install it either:
  cross-term repair is zero (conditional ends-right-middle-wrong
  87.9% control v 89.2% treatment, flat; the raw +11 is a
  denominator effect), expand pass@8 moves +2.5 points against a
  +10 bar, the decomposed route is NEVER emitted on expand
  (intermediate-form 0), and the only behavioral change is
  unevaluated-product emission leaking into bare arithmetic
  (canonical arith 2.5% -> 0%). With BASICS-DIET-1's standalone
  arm, both sides of the format edge are now measured and the
  failure survives both — the live variables narrow to
  supervision grain and dose. Gate no-harm holds (70 v 72, inside
  run noise). ([VERDICT XTERM-DIET-1](RESULTS.md#L30682 "id:2026-08-16-verdict-xterm-diet-1-both-primary"); [RIDER XTERM-DIET-1-DECOMP](RESULTS.md#L30656 "id:2026-08-16-rider-xterm-diet-1-decomp-the").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe] Engine
  successors can recover an action label for only 47.58% of stock
  diet rows (4,000-row census, thresholds committed pre-count) —
  the engine-only free-labels premise dies at a knife-edge KILL —
  but the four non-expand algebra families sit at 0-13% solely
  because their ops are outside the engine's move vocabulary
  while carrying their labels verbatim in the think field; the
  union lifts coverage to 65.8%, pricing the policy rung as a
  re-emission pass. Anatomy names a hard floor: some rows
  materialize integration constants from chain context
  (Integral(0, x) -> "+ 4"), so their labels are not computable
  from cur alone — a label-timing violation inside the stock
  diet. Counts are lower bounds (canonical-string match; sample
  sigma ~0.8 points puts 47.58 v 50 in the threshold region).
  ([OBSERVATION RULE-POLICY-0-CENSUS](RESULTS.md#L30806 "id:2026-08-16-observation-rule-policy-0-census-engine").)

- [SINGLE-SEED] [REGIME-SCOPED: specified diet and recipe] The
  emitter divergence has two named, countable mechanisms: on the
  same integrands the two emitters serialize the same mathematics
  differently 75.5% of the time (axiom's ascending-degree order,
  parenthesized unary minus, **-1 for division — a different
  answer dialect; math disagreement 0/261), and the axiom farm's
  8s L4 wall censored a real solvable band, 15.2% of successes
  spread across 8-59s with no gap, refuting the recorded
  bimodality claim — the shard's L4 rule mix is speed-biased by
  construction. Both desk thresholds were committed pre-count; a
  farm-time normalization pass is priced as the ATOM-NORM arm.
  Sample-scoped; censored-band composition unmeasured.
  ([OBSERVATION ANSWER-FORM-0 + CENSOR-0](RESULTS.md#L30869 "id:2026-08-16-observation-answer-form-0-censor-0").)

- [RETRACTED] [REGIME-SCOPED: measured deployment artifacts] A frontier
  MoE expert layer (V4-Flash layer 22, 256 experts, 6.44B weights)
  was compressed from STREAMED WEIGHTS ALONE — ephemeral
  byte-range fetch, disk residence one expert, zero teacher
  forward passes and zero calibration data — but the first
  execution scores NOTHING: two independent auditors found three
  blockers (a revision literal asserted nowhere while fetches
  resolved a moving pointer; two of five arms never serialized,
  with the scalar baseline over budget by its own frozen format;
  and every arm but one decoded at fp32 while billed fp16, which
  for residual VQ means later stages trained on residuals the
  artifact cannot reproduce), so all three registered bars book
  UNRESOLVED. The provisional table is retained as prior only
  (width-32 VQ 0.350 pooled Frobenius v 2-bit scalar 0.770 v
  low-rank 0.81-0.85), with the direction of the precision defect
  named: the low-rank arms were given more precision than they
  paid for and still lost. Repairing the measurement cannot
  rescue the scalar bar, because that arm is structurally
  inadmissible rather than mis-measured.
  ([OBSERVATION STREAM-WDISTILL-0-EXEC1](RESULTS.md#L31394 "id:2026-08-16-observation-stream-wdistill-0-exec1-the");
  [AMENDMENT -REPAIR-SCOPE](RESULTS.md#L31480 "id:2026-08-16-amendment-stream-wdistill-0-repair-scope").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  Low-rank factorization of a frontier MoE expert layer, derived
  from streamed weights alone (zero teacher forwards, zero
  calibration data, disk residence one expert), does not pay at 2x
  compression, and a LACK OF LOW-RANK STRUCTURE is not the reason. Sharing one basis across 256 experts buys 3.6x the
  rank (1075 v 296) for 0.37% less pooled Frobenius error against
  a registered 10% bar. But capture efficiency against a
  uniform-random subspace of equal rank in the same ambient space
  reads 4.678x for PRIVATE per-expert bases at rank 296 — a flat
  spectrum would give 1.0 at every rank, so the experts are
  individually anisotropic. Cross-expert MISALIGNMENT is the
  leading explanation for why sharing then pays so little, but it
  is NOT yet measured: the efficiency statistic is rank-dependent
  (on one fixed spectrum with no coordinate change it falls 9.75x
  to 3.24x between those two ranks), so the shared basis's 1.307x
  at rank 1075 is confounded. The rank-matched C@296 v D@296
  reading, registered and unrun, is what separates them. The
  gauge-control arm's 1.067x is close to the uniform-random
  expectation, consistent with the gauge prediction, and is a
  convergent post-hoc statistic on that same arm rather than
  independent evidence. Capture figures are a PROXY
  (1 - rel_frobenius^2, exact only for an orthogonal projector;
  fp16 bases hold it to ~1e-4). Every low-rank arm has
  higher error than an inadmissible near-budget 2-bit scalar
  reference (19 bytes over, so that comparison is DESCRIPTIVE and
  the registered scalar bar is UNRESOLVED). Weight space only, one
  layer, n=1, no capability claim.
  ([VERDICT STREAM-WDISTILL-0-AUDIT-REPAIR](RESULTS.md#L31908 "id:2026-08-16-verdict-stream-wdistill-0-audit-repair");
  [AMENDMENT -READING-2](RESULTS.md#L32133 "id:2026-08-16-amendment-stream-wdistill-0-reading-2");
  [AMENDMENT -READING-3](RESULTS.md#L32229 "id:2026-08-16-amendment-stream-wdistill-0-reading-3").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] A fair
  scalar baseline erases the apparent width-32 VQ advantage on the
  V4-Flash layer-22 experts, and the ladder attributes the whole
  EXEC1 gap: ternary-in-a-2-bit-field wastes 17.7%, scalar cell
  design buys 42.7%, and vector coding at width 4 buys a further
  13.3% (BAR 2 fires) while wider residual stacks give it all back —
  W32 is indistinguishable from the globally optimal 4-level scalar
  (REFUTED-IF triggers; no locality signal at this rate, natural v
  shuffled indistinguishable). Weight space only, one layer, E8M0
  block-128 convention.
  ([VERDICT STREAM-WDISTILL-0S](RESULTS.md#L32372 "id:2026-08-16-verdict-stream-wdistill-0s-bar-1").)

- [SINGLE-SEED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts]
  The expert-codec ranking measured at V4-Flash layer 22
  is depth-stable: at all five sampled depths {2,12,22,33,42} of 43,
  width-4 VQ beats the optimal scalar, wider stacks lose it back,
  and shuffled twins sit within 2.5e-4 of natural — while layer 42
  is uniformly harder to compress under the same ordering. v2
  prototype instrument, descriptive class, one model/revision;
  licenses building the whole-model pass, claims nothing about it.
  ([OBSERVATION STREAM-WDISTILL-CENSUS-0](RESULTS.md#L32485 "id:2026-08-17-observation-stream-wdistill-census-0-registered").)

- [SINGLE-SEED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts]
  The GPU streaming harness reproduces the Mac instrument at full
  population: across all 11 arms on 256 experts, worst VQ-arm
  disagreement 3.85e-4 relative (13x inside the pre-look bound),
  worst scalar 3.03e-9, at a 7.45x harness-level wall-clock ratio
  (shard cache + CUDA + implementation bundled, never a device
  comparison). Cross-device qualification only; the same-device
  promotion gate remains unrun.
  ([OBSERVATION STREAMWD-V2-QUALIFICATION](RESULTS.md#L32527 "id:2026-08-17-observation-streamwd-v2-qualification-full-256").)

- [SINGLE-SEED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts]
  The width-4 vector-coding advantage is not a V4-expert artifact:
  on Qwen3.8-27B's dense bf16 FFN (layer 32 of 64) W4 beats the
  DP-optimal 4-level scalar by 14.3% relative, the rate-matched
  width inversion and the locality null both transport — while the
  scalar ladder INVERTS (ternary beats 4-level uniform, reverse of
  V4; cause unmeasured). Descriptive, one layer, one model,
  rankings only.
  ([OBSERVATION QWEN-STREAM-PROBE-0](RESULTS.md#L32578 "id:2026-08-17-observation-qwen-stream-probe-0-the").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  Qwen3.8-27B's weight budget closes on a 10GB card: FFN — the
  family where width-4 VQ is measured — is 61.6% of all 27.8B
  parameters, text-only at the 2.0625 bpw payload rate is
  6.56 GiB, and even doubling every non-FFN family's rate lands
  ~8.9 GiB. The binding unknown is the linear-attention family
  (20.0%, no codec measurement). Exact shard-header census, one
  revision, bytes only, no compression or latency measured.
  ([OBSERVATION QWEN-BYTE-CENSUS-0](RESULTS.md#L32618 "id:2026-08-17-observation-qwen-byte-census-0-exact").)

- [SINGLE-SEED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts]
  Scalar alphabet choice is distribution-bound and the DP re-derives
  it automatically: Qwen's dense bf16 weights are zero-concentrated
  under max-anchored block scaling (80%+ of normalized mass below
  1/3; ternary parks 92-94% on its zero level) so ternary beats
  4-level uniform there, while V4's dequantized experts show the
  reverse — and the DP-optimal 4-level alphabet votes for the
  mechanism by placing two near-zero levels itself. Codec ordering
  (W4 best, wider worse) is depth-stable 4/4 sampled Qwen layers.
  Descriptive, rankings and mass fractions only.
  ([OBSERVATION QWEN-FFN-CENSUS-0](RESULTS.md#L32655 "id:2026-08-17-observation-qwen-ffn-census-0-the").)

- [SINGLE-SEED] [DEVICE-SCOPED]
  [REGIME-SCOPED: measured deployment artifacts]
  Qwen3.8-27B weight space is codec-homogeneous: all nine
  representative tensors across linear-attention, full-attention,
  embeddings, and LM head reproduce the FFN pattern (W4 beats the
  optimal scalar by ~11-12% at 2 bits; 4 bits cuts error ~3.2x;
  16-level DP scalar edges stacked VQ at 4 bits on every tensor) —
  no fragile family exists in weight space, so the whole-model rate
  table is a pure allocation decision that only the frozen
  functional eval can refine. Descriptive, one layer per attention
  family, weight space only.
  ([OBSERVATION QWEN-FAMILY-PROBE-0](RESULTS.md#L32729 "id:2026-08-17-observation-qwen-family-probe-0-weight").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  The scalar-alphabet flip between models is a stored-distribution
  effect, measured on both sides: V4's dequantized-MXFP4 experts
  hold 64% of block-normalized mass below 1/3 (17.7% below 0.1)
  where Qwen's raw bf16 holds 80-84% (32-35%) — vendor
  quantization shapes what a fixed alphabet sees, so alphabets are
  re-derived per tensor (S2-DP). Desk census, 8 experts v 9
  tensors, mass fractions only.
  ([OBSERVATION SCALAR-MASS-CENSUS-0](RESULTS.md#L32871 "id:2026-08-17-observation-scalar-mass-census-0-the").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  The v2 streaming harness (shard-cache + two-pass) reproduces the
  v1 evidence receipt on the same device within 5.7e-4 worst-arm
  relative operator error (deterministic arms at 1e-9); v2 is
  promoted as the default harness and the Qwen compilers' loader
  lineage. One layer, one model; v1 receipts remain the evidence
  record for booked 0S numbers.
  ([VERDICT STREAMWD-V2-MAC-GATE](RESULTS.md#L32890 "id:2026-08-17-verdict-streamwd-v2-mac-gate-same").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  A bounded transactional compiler (one source shard resident,
  delete-after-compress, exact key-conservation law) emits three
  complete compressed text-language artifacts of a 27B multimodal
  model on a 10GB-card host: A 6.50 / B 7.09 / C 8.77 GiB, zero
  conservation violations over 1199 keys, and every family's
  pooled op error at 0.97-1.00x its probe-predicted value — the
  probes priced the whole model correctly. Compiler-correctness
  only; whether any artifact talks is MODEL-1's question.
  ([VERDICT QWEN-WHOLE-0T](RESULTS.md#L32933 "id:2026-08-17-verdict-qwen-whole-0t-all-three").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  A "bit-lossless fp16 residency" claim — argued correctly from
  representability for normal floats — was killed by its own
  round-trip oracle on real weights: 0.045% of a 27B model's
  embedding entries (569,841 of 1.27B) sit in the fp16 subnormal
  tail and change under fp32-fp16-fp32. Representability arguments
  are not oracles; the reference runtime now refuses non-identical
  residency and keeps io compressed with exact on-demand decode.
  ([OBSERVATION QWEN-RUNTIME-0R-FP16-RETRACTION](RESULTS.md#L33396 "id:2026-08-17-observation-qwen-runtime-0r-fp16-retraction").)

- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts]
  A 27B model compressed to 6.50 GiB (uniform width-4 VQ at
  2.0625 bpw) reconstructs into an executing language model: full
  64-layer hybrid tower, execution-proven traversal, and a
  coherent factually-correct 32-token reasoning trace ("The
  answer is Paris") through a qualification-gated CPU reference
  decode. Sanity smoke only — fluency is not fidelity, no quality
  metric claimed; the teacher-forced MODEL-1 core remains the
  only functional instrument.
  ([OBSERVATION QWEN-RUNTIME-0R-SMOKE](RESULTS.md#L33429 "id:2026-08-17-observation-qwen-runtime-0r-smoke-first").)
- [SINGLE-SEED] [DEVICE-SCOPED: 3080/WSL] The CUDA
  runtime ladder (rungs 0-3) is green with bit-exact decode parity at
  every rung: artifact A fully VRAM-resident (6.6 GiB u8), full
  64-layer forward 1.6s (v 128s same-process CPU reference, per-layer
  hidden rel err max 6e-6, backend KL 4.2e-8), 32 coherent greedy
  tokens at 0.41s/tok; free VRAM measured 8.86-9.51 GiB so artifact C
  (8.77 GiB) cannot run on this card; on a 700-token QM prompt A
  degrades into a repetition loop — first function-space crack, with
  the greedy-loop confound named (MODEL-1, not chat reads, will
  adjudicate). ([OBSERVATION
  QWEN-CUDA-LADDER-0](RESULTS.md#L33515 "id:2026-08-17-observation-qwen-cuda-ladder-0-rungs").)
- [SINGLE-SEED] [DEVICE-SCOPED: 3080/WSL] The fused CUDA tower
  (FusedW4Linear surgery, decode phase touches only compressed bytes)
  runs artifact A at 8.8 tok/s v 0.41 s/tok for materialize-then-dense,
  with top-5 logits identical to both the rung-3 backend and the CPU
  reference; the 700-token QM repetition loop reproduces exactly across
  both CUDA implementations, so the candidate long-horizon failure
  belongs to the artifact, not a backend. ([OBSERVATION
  QWEN-CUDA-RUNG4](RESULTS.md#L33651 "id:2026-08-17-observation-qwen-cuda-rung4-fused-tower").)
- [SINGLE-SEED] [DEVICE-SCOPED: 3080/WSL] The long-horizon failure
  orders with io precision (qualitative, n=1 prompt, greedy): on the
  same QM derivation, artifact A (W4 io) tight-loops by ~400 tokens
  while artifact B (S16 io, runs at 7.8 tok/s via the adopted S16Rows
  + fused s16 GEMV path) executes the algebra, detects its own >1
  probability, and restart-cycles — the severity shape tree prior T1
  predicted; the pinned teacher under identical greedy prompting is
  the registered adjudicator. ([OBSERVATION
  QWEN-CUDA-S16-B](RESULTS.md#L33695 "id:2026-08-17-observation-qwen-cuda-s16-b-artifact").)
- [SINGLE-SEED] [TEACHER-FORCED] The teacher v2d lock is ACCEPTED
  through the full registered path: commit pin 0ca4151 exact, the
  cached-v-uncached sidecar passes with perfect token equality
  (4722/4722) and max rel L2 2.1e-4 v the 5e-3 bar (fp16-symmetric
  quantization deviation disclosed), and the margin census books with
  the small-n fence LIVE on the scored streams (corpus+prefixes bins
  below 0.2 nats are all under 30 positions) — both auditors ran
  pre-booking and their blockers (uncommitted census producer,
  unfalsifiable sha boolean, pooled-fence overstatement) were fixed
  first. ([VERDICT
  QWEN-TEACHER-0-LOCK](RESULTS.md#L33780 "id:2026-08-17-verdict-qwen-teacher-0-lock-teacher").)
- [SINGLE-SEED] [TEACHER-FORCED] The MODEL-1 tree's first firing books
  INSTRUMENT-ALARM, not an allocation: X_A = 1.061 nats sits 6% over
  the pre-registered 1.0 uniform-damage gate, so no branch may fire —
  while underneath the gate every precision step fires monotone
  (B-over-A rel X 21.4%, C-over-B 70.2%, the LARGEST step is
  attention, opposite in emphasis to the registered io prior, which
  books UNADJUDICATED under the alarm per its own clean-gates clause).
  Large-n corpus strata put A's flips at large margins (32/85 in
  [2,5), 10/125 in [5,inf)) — bulk damage, not boundary near-ties.
  Fences: point readings on 355/92 positions with no sampling
  fence; teacher-distribution fidelity wording only; resolution of
  the alarm fork is amendment-gated. ([VERDICT
  QWEN-MODEL1-TREE](RESULTS.md#L33855 "id:2026-08-17-verdict-qwen-model1-tree-the-tree").)
- [SINGLE-SEED] [TEACHER-FORCED] Attention attribution inside the
  Qwen artifact ladder: linear attention carries the larger total
  share of the B-to-C recovery (R_X 0.949 v full attention's 0.536)
  but the two families are heavily REDUNDANT — single-family
  recoveries sum to 1.49, both near-additive bars miss, so the
  next-grain singleton split is BLOCKED-BY-INTERACTIVE; per byte the
  ordering inverts (full attention 0.802 nat X/GiB, 1.9x linear's
  0.431), and the discretion-free iso-rate arm (A + in_proj_qkv,
  matched to B's io budget within 1.05%) confirms io beats attention
  spend on BOTH metrics at matched bytes. Fences: point readings on
  355/92 positions, no sampling fence; teacher-distribution fidelity
  wording only; floors are numerical-sensitivity multiples, never
  significance. ([VERDICT
  QWEN-ATTN-ATTRIB-1](RESULTS.md#L34115 "id:2026-08-18-verdict-qwen-attn-attrib-1-l").)
- [SINGLE-SEED] [TEACHER-FORCED] The A-to-B io repair splits by
  metric at exactly iso-byte spend (+0.2960 GiB each): the embed
  swap (D) carries the CE recovery (rec_X 0.616 v E's 0.435,
  gap +0.181) while the head swap (E) carries the KL recovery
  (rec_K 0.777 v D's 0.325, gap -0.452, 235 f_K floors), and each
  metric's recoveries sum to ~1 (near-additive: |I_X| 0.052,
  |I_K| 0.102) — the two ends repair DIFFERENT damage. Frozen rule
  reads MIXED/UNRESOLVED; the D>E prior survives on its X-only
  predicate with the K leg opposite. Fences: point readings on
  355/92 positions, no sampling fence; teacher-distribution
  wording; arm contrasts normalized on A, never fidelity claims;
  floors are precision multiples, never significance. ([VERDICT
  QWEN-IO-ATTRIB-1](RESULTS.md#L34643 "id:2026-08-18-verdict-qwen-io-attrib-1-mixed").)
- [SINGLE-SEED] [TEACHER-FORCED] The uniform-2-bit Qwen artifact is
  NOT a faithful channel router for its teacher: teacher |z|-mass
  captured by A's top-1024 FFN channels reads 0.11-0.21 across all
  six sampled layers (bar 0.7, kill line 0.5 crossed everywhere),
  and even oracle top-1024 selection leaves 32-63% reconstruction
  error — the activation is not strongly top-k-compressible at
  that k. Kills the banked resident-draft/hybrid program at one
  CPU afternoon's cost. Fences: 454 positions, no sampling fence;
  single arm; frozen layer set is 4 linear / 2 full attention
  post label correction. ([VERDICT
  QWEN-RK-CENSUS-0](RESULTS.md#L34734 "id:2026-08-18-verdict-qwen-rk-census-0-refuted").)
- [SINGLE-SEED] [TEACHER-FORCED] LBAND-1 books INSTRUMENT-ALARM:
  the late linear-attn band ADDED to the F repair reads 0.0037
  nats WORSE than F on corpus X (96.59 f_X past the monotone-
  repair bracket edge; receipt verified end-to-end — measured
  interference, not a mixup). Every structure/conditioning bar
  fires; the flatness predicate reads 0.506 rec units v its 0.2
  threshold but stays deliberately UNADJUDICATED under the alarm
  (code-frozen precedence, disclosed). Descriptive: repair value
  from B is monotone EARLY-heavy (dX 0.401/0.105/0.028), the F/L
  redundancy is depth-localized early (I_X(e) -0.229), mid is
  mildly synergistic, and BLe beats F on both metrics absolutely
  and per byte. Fences: point readings 355/92 positions, no
  sampling fence; numerical floors never significance; same-commit
  F re-score is the named residual. ([VERDICT
  QWEN-LBAND-1](RESULTS.md#L34907 "id:2026-08-18-verdict-qwen-lband-1-instrument-alarm").)
- [SINGLE-SEED] [TEACHER-FORCED] The third-pick allocation crossover
  HOLDS on a held-out surface, all five bars firing under the first
  registered-JSON precedence adjudication: from the FLe state, the
  next 461MB buys corpus-X with the mid linear band (dX 0.0874 v
  late's 0.0065) and prefix-K with the late band (dK 0.0303 v mid's
  0.0103) at identical bytes — the LBAND F-conditioned table
  transports in sign and magnitude-class (X marginal at 68% of the
  development value, inside its registered band). Unregistered
  two-direction color: the two-band policy beats endpoint C on X
  (0.3806 v 0.3950) while C keeps the best K. Fences: point
  readings on 361/72 positions, no sampling fence; floors are
  numerical sensitivity, never significance; transport bands are
  registered science widths. ([VERDICT
  QWEN-MODEL2-ALLOC-1](RESULTS.md#L35183 "id:2026-08-19-verdict-qwen-model2-alloc-1-all").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] The
  w4 quantization residual of arm A is conditional-mean-optimal
  noise almost everywhere probed (level-1 dequantization ceiling
  0.086% max variance reduction across 402 tensors; no globally
  learnable decode table — leave-one-out family cosine ~ 0.0015),
  with localized structure only in early attention write-back
  projections (o_proj L3 low-rank 9.4% top-16 energy; out_proj L0
  heavy tail 21%); weight-space census only, single artifact, w4
  codec only, any promotion requires held-out X/K. ([VERDICT
  QWEN-RESIDUAL-STRUCTURE-0](RESULTS.md#L35486 "id:2026-08-19-verdict-qwen-residual-structure-0-not").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] [FREE-RUN-GATED]
  On the qcuda-tower runtime (old-v-new equivalence banked), the BLe
  free-generation screen missed both registered bars at zero: 0/30 xhigh
  think blocks closed and 0/60 correct across cells, so the 48-tensor
  early-band s16 repair does not reach the deliberation loop and BLe stays
  a scored reference arm, not promoted toward deployment; single seed,
  greedy, CUDA leg only. ([VERDICT
  QWEN-BLE-FREEGEN-2](RESULTS.md#L35764 "id:2026-08-19-verdict-qwen-ble-freegen-2-both").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] On the frozen
  MODEL-1 surface, arm A's w4 lm_head keeps the teacher's top-1 token inside
  its top-256 candidates at 98.03% of corpus positions (knife-edge over the
  registered 0.98 bar, one-position resolution) and 100% of prefix positions,
  carrying ~96% of teacher mass; the vendor head on identical hidden states
  nets out only 2 of 7 corpus misses at k=256 (no per-position masks, no
  set-inclusion claim; sign flips at k=1024), reading as a small readout
  contribution on top of upstream body-state; level-2 router census unlocked; one artifact, one surface.
  ([VERDICT QWEN-CHEAP-READOUT-0](RESULTS.md#L36036 "id:2026-08-19-verdict-qwen-cheap-readout-0-both").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] [FREE-RUN-GATED]
  An 8-token temperature impulse (T 0.3/0.7) fired on recurrence detection
  escapes BLe's retry loops in 5/18 runs — all on the loose semantic-restart
  item, with the two exact-orbit items locally restoring in 192/192 bursts —
  and never repairs (0/18 correct); the perturbation reshuffles which exact
  cycle the trajectory inhabits (14/18 exact tails, periods 22..352, twice
  tighter than greedy's); adjudicated by offline recomputation from token-ID
  sidecars after the in-run escape field was found inflated by a burst-cap
  artifact. ([VERDICT QWEN-CYCLE-IMPULSE-0](RESULTS.md#L36117 "id:2026-08-20-verdict-qwen-cycle-impulse-0-bar").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] [FREE-RUN-GATED]
  Along BLe's frozen greedy retry loops, pre-head state recurrence is only
  approximate where token recurrence is exact — homologous-cycle cosine
  medians 0.9963 (item 0) and 0.9701 (item 4, below the uncalibrated 0.99
  bar, KNIFE-EDGE), decreasing from k=1 to k=2 with the item-4 variation
  concentrated at high-margin positions — while on the semantic-restart item
  successive retry attempts carry numerically the same distribution (98.4%
  top1 agreement, median full-vocab JS 1.7e-5 nats): correction failure
  there is not unlucky sampling; per-specimen claims only, no
  policy-v-state ranking across specimens.
  ([VERDICT QWEN-LOOP-STATE-0](RESULTS.md#L36345 "id:2026-08-20-verdict-qwen-loop-state-0-bar").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] [FREE-RUN-GATED]
  Swapping the uncompressed vendor lm_head onto BLe's identical captured
  loop states changes almost nothing: 99.6% top1 agreement over all 1409
  captured positions, and the vendor head reproduces the stuck retry
  policy at numerically the same fraction as BLe's own head (0.984375);
  every failing attempt-pair under every top1 basis lies in the FIRST
  retry attempt — static readout replacement is largely measured out as a
  LOCAL explanation on frozen BLe states (trajectory-level readout
  causality stays open, per the -SCOPE amendment); per-specimen,
  off-manifold states, no teacher-behavior claim.
  ([VERDICT QWEN-LOOP-STATE-1-HEADSWAP](RESULTS.md#L36594 "id:2026-08-20-verdict-qwen-loop-state-1-headswap").)
- [SINGLE-SEED] [REGIME-SCOPED: measured deployment artifacts] [FREE-RUN-GATED]
  A single vendor-informed token forced at each of the five measured
  head-disagreement points on BLe's frozen loop trajectories re-enters
  recurrence in 5/5 branches (gaps 33-176 tokens, three NEAR the 32-token
  detector floor) yet rejoins the ORIGINAL orbit in 0/5 — 4/5 land in
  NEW exact periodic tails under the stated census, one tail is
  non-periodic under its caps; none terminates or reaches correctness
  (0/5, cap-limited); recurrence-CLASS robust,
  specific orbit fragile; the registered "cannot redirect" gloss was
  narrowed at booking (CYCLE-IMPULSE precedent); adjudicated by an
  independent offline consumer recomputing everything from token-ID
  sidecars.
  ([VERDICT QWEN-HEADSWAP-IMPULSE-0](RESULTS.md#L36833 "id:2026-08-20-verdict-qwen-headswap-impulse-0-bar").)
- [NULL] [REGIME-SCOPED: measured deployment artifacts]
  The exact-geometric LEVEL-2 route books its registered null: k-means
  balls + Cauchy-Schwarz bounds certify the exact top-256 perfectly
  (1362/1362 query-K cells) but barely prune at d=5120 — corpus q50
  visitation 0.999/0.997/0.989 of the 248320-row vocab at K
  256/1024/4096 (REFUTED-IF fires at >= 0.90 every K, knife-edge 9.87%
  by a conservatively extended fence), and the MODELED byte ratio
  exceeds 1 at every K (1.03-1.50): the index costs more than it
  saves; offline exact counterfactual visitation census under the
  decoded-W4 fp32 scorer, partially unblinded at K 256/1024 (disclosed
  with its exposure receipt), prefix and corpus MEDIAN visitation is
  bit-identical at every K; tail visitation (q90/q95/max) differs by
  a few rows only.
  ([VERDICT QWEN-MIPS-CENSUS-0](RESULTS.md#L37116 "id:2026-08-20-verdict-qwen-mips-census-0-the").)

- [NULL] [qwen] [wsl] [compression] [trajectory-control] A
  one-band precision escalation (BLe -> BLem, +0.4296 GiB mid-band
  s16, the largest dose fitting the 10GB card) has ZERO measured
  authority over the frozen detect-retry loops at their first
  detector-fire events: 0/6 high-arm branches escape the 300-token
  post-event window — HOT (BLem weights on retained BLe state,
  F(W_BLem, S^BLe)) 0/3 and same-prefix REFRESH (BLem-consistent
  reconstructed state) 0/3 alike — and every branch's FIRST
  escalated token is the identical loop-continuing token the frozen
  BLe stream carries, with divergence only 14-157 tokens later; the
  registered refutation fires and the precision actuator DEMOTES at
  this dose (dose-scoped: not a universal precision-control null;
  repair unmeasured — all branches cap-limited; sanity gate 3/3
  exact through the same teacher-forced-prefix + serializer path
  the treatment used).
  ([VERDICT QWEN-HOMEO-ACTUATOR-0](RESULTS.md#L37432 "id:2026-08-20-verdict-qwen-homeo-actuator-0-the").)

- [SINGLE-SEED] [qwen] [wsl] [compression] [trajectory-control] At
  all five frozen near-tie loci of the loop-state program, the
  compressed BLe head's near-top candidate region is COVERED by the
  registered exclusion union (dominated by the vendor-head top-256;
  specials excluded in the same union, per-token attribution not
  persisted — exact overlap goes to the banked census), so the
  registered gap-matched
  non-vendor control token does not exist (best admissible match
  12.18-18.61 logits off, 244-372x the 0.05 gate; controls at
  0-based BLe rank 204-223) and QWEN-ALTTOKEN-CONTROL-0 books CONTROL-MATCH-FAILED
  with zero branches run — consistent with the two heads agreeing
  not only on argmax (HEADSWAP) but broadly on the near-top
  candidate region (these five h only; head-space only; exact
  overlap unmeasured pending the census).
  ([VERDICT QWEN-ALTTOKEN-CONTROL-0](RESULTS.md#L37747 "id:2026-08-20-verdict-qwen-alttoken-control-0-control").)

- [MECHANISM-CONFIRMED] [qwen] [wsl] [runtime] The HOMEO 7x decode
  slowdown is a RESTORED-CACHE x SECOND-TOWER x LENGTH conjunction,
  not state corruption and not the s16 kernel: a cache state
  restored via the CPU serializer onto a tower built after a
  previous tower was freed collapses from ~12 to 0.74 tok/s once
  the cache passes position ~1785-1849 (identical onset for
  cross-tower and roundtripped-native states — values exonerated;
  native-prefill state immune through 3641; s16 GEMV only ~1.5x w4
  per route; first-tower restores flat to ~3072), consistent with
  allocator block-pool degradation but allocator counters not
  captured (phase-5 slot banked; no OOM warnings — distinct from
  the 43x tripwire class); resolves the HOMEO diagnosis-scope
  question to the benign branch, HOT tokens computed correctly,
  slowly.
  ([OBSERVATION QWEN-BLEM-DECODE-PERF-0](RESULTS.md#L37825 "id:2026-08-20-observation-qwen-blem-decode-perf-0").)

- [REPLICATED] [moe] [mac] [capability] The named-80 carrier
  deletion crest replicates: named80 - full is positive in 6/6
  seeds across two INDEPENDENT seed triples (+55 pooled on
  1001/2002/3003; +43 pooled on the virgin 4001/5002/6003, signs
  3/3 each), on the pinned 4-bit instrument with a cell-exact
  qualification. Replication route: second registered seed triple,
  same device, same instrument.
  ([VERDICT EX5-LAYERMATCH-0](RESULTS.md#L37996 "id:2026-08-21-verdict-ex5-layermatch-0-carrier-identity").)
- [SINGLE-SEED] [moe] [mac] [mechanism] At fixed layer profile,
  NAMED carrier identity is robust while rank-window membership
  alone is insufficient (wording per AMENDMENT -WORDING; rank
  neither established nor nulled): fresh rank-window masks
  transport 1/3 (sums -12/+25/+10, range 37 — the booked rand0
  +28 was substantially a draw property), layer-only identity
  draws do nothing (0/3, median -3), and named80 beats 2/3 of its
  own rank-window draws. Fences: one mask-set per family, n_mask=3
  is the mechanism unit; rank-window masks also delete ~1.8-2x the
  layer-only demand share (covariate, not matched); no directional
  family claim (median gap +13 sub-resolution, ranges interleave).
  ([VERDICT EX5-LAYERMATCH-0](RESULTS.md#L37996 "id:2026-08-21-verdict-ex5-layermatch-0-carrier-identity").)

- [MECHANISM-CONFIRMED] [moe] [mac] [mechanism] The named-80
  deletion crest is carried by the PROMPT PHASE: masking the
  carriers during prefill + prompt_tail alone recaptures the full
  effect (+47 pooled v ALL's +48, signs 3/3, |gap| 1 descriptive),
  while decode-only masking yields nothing (-2, below the
  resolution floor) — the decode-carry hypothesis booked its
  registered refutation and the pre-named context-assembly
  alternative won; the two phase effects compose additively
  (residual 3). EX5's decode failure enrichment is thereby a
  symptom, not a cause. Fences: one mask, one dose, phase arms
  causally asymmetric (DECODE preserves prompt-built state and
  first-token prediction); crest now 9/9 seeds across three
  triples; single 4-bit instrument, identity by cell-exact
  qualification evidence.
  ([VERDICT EX6-PHASE-0](RESULTS.md#L38436 "id:2026-08-21-verdict-ex6-phase-0-the-prompt").)

- Expert-reuse temporal structure beyond recency is real offline and
  worthless in-loop at this class: finite-horizon next-use prediction
  (P(T<=H), strict past-only features, prompt holdout) beats an
  age-only hazard by median +0.041 AUC at H=8 (six traces; ~half a
  count-saturating popularity term, the rest joint gap/prev-token/
  phase/layer signal that is LARGEST at short horizons H<=4),
  yet closed-loop learned eviction at K32 loses to warm LRU@K48 on
  6/6 and to same-budget LRU@K32 on 5/6 — the information is real,
  the hazard-argmin policy transform loses it. Both readings
  survived the isolated-stream + protected-eviction rerun
  (AMENDMENT -ISOLATION: median +0.0406 v +0.0407). [SINGLE-SEED]
  [REGIME-SCOPED: measured deployment artifacts] Fences: logistic
  class + per-expert features only; token-event time; deterministic
  even/odd prompt split; original run's holdout was
  stream-entangled (corrected, not load-bearing).
  ([VERDICT ROUTE-TIME-0](RESULTS.md#L39278 "id:2026-08-21-verdict-route-time-0-temporal-structure").)

- The prefill-poisoning crest is a super-additive conjunction, not a
  single-component effect: masking only the prompt batch heals +13
  pooled solves, masking only the first generated token's routing
  heals +21, together +47 (interaction +13; 13+21=34 < 47), with all
  six anchor cells reproduced exactly through the new driver. One
  routing decision on one token carries nearly half the crest — the
  house prior was wrong on all three registered clauses (prefill
  dominant, token1 null, interaction small). [MECHANISM-CONFIRMED]
  [REGIME-SCOPED: measured deployment artifacts] Fences: single
  4-bit instrument, greedy, named-80 keepset, pooled n=3 seeds with
  3/3 sign consistency; TOKEN1_ONLY reads via the corrected
  prompt_tail semantics (it masks generated z1's routing).
  ([VERDICT EX6-LOC-0](RESULTS.md#L39826 "id:2026-08-21-verdict-ex6-loc-0-no-single").)

- **The launch step is special, not merely early**: masking the
  SECOND or THIRD generated token's routing (one temporal position
  across all 48 MoE layers, the identical temporal-call dose as token1; realized displacement differs ~2.1x, see AMENDMENT -SCOPE)
  buys nothing — Delta_z2 = -3 pooled (inside the ~7-solve floor),
  Delta_z3 = 0 with the paired per-problem read outcome-identical
  to baseline (0 rescues, 0 breaks in 360) — while z1's +21
  reproduced cell-exact through the new per-module-counter
  instrument (6/6 anchors, recall to four decimals). The early-token
  sensitivity curve alternative is absent: the shape is a step
  function, +21 at z1, nothing after. [MECHANISM-CONFIRMED]
  [REGIME-SCOPED: measured deployment artifacts] Fences: single
  4-bit instrument, greedy, named-80 keepset, pooled n=3 seeds;
  dose-validity clause satisfied at 0/360 zero-dose both arms; the
  bar-5 ordering read books TIE (|gap| < 7); WHY z1 is privileged
  (KV/attention mediation) stays hypothesis.
  ([VERDICT EX6-TEMPORAL-0](RESULTS.md#L40150 "id:2026-08-21-verdict-ex6-temporal-0-the-launch").)

- **z1's launch-step healing lives in the LAST 16 MoE layers**:
  band-masked z1 routing (blocks 32-47) alone books +19 of Z1_ALL's
  +21 (signs +6/+8/+5, 3/3) while early/mid bands book -3/+5 (floor
  noise), with pooled additivity residual exactly 0 —
  depth-localized with no registered large pooled NET interaction
  (per-level residuals -6/+1/+5 cancel; difficulty-dependent
  interaction open, see AMENDMENT -SCOPE). The late band is the
  near-pure-rescue arm (20 rescues v 1 break). Registered color:
  the demand map ANTI-aligns (z1's excess carrier demand is
  early-band, 2.56% v 0.50% late) and displacement does not order
  the effect — both cut against a displacement/family-demand
  reading of the locus. [MECHANISM-CONFIRMED] [REGIME-SCOPED:
  measured deployment artifacts] Fences: single 4-bit instrument,
  greedy, named-80 keepset, pooled n=3 seeds 120/seed; band arms =
  one temporal-call position x 16 layers, displacement disclosed
  not matched; DISTRIBUTED house prior refuted as registered;
  KV/attention mediation stays hypothesis.
  ([VERDICT EX6-DEPTH-0](RESULTS.md#L40389 "id:2026-08-21-verdict-ex6-depth-0-z1-healing").)

- **MATH-CYBER rung 0 is a wrapper, not a build**: the closed-loop
  MATHWORLD's transition function already exists as the standing
  calculus engine's successors() (declarative rule@locus actions,
  fork-timeboxed, sympy edge-verified, identity rejected by
  construction). Desk census on real states: branching median
  3-3.5 at L4-7 (max 9, 15% single-action) — adequate; L1-3 is
  corridor-thin (median 2, 35% single-action); admission wall
  median 51 ms at L4-7 but p90 2.05 s — the heurisch tail makes
  wall-clock the throughput/safety constraint (per AMENDMENT
  -BUDGET: primary rungs budget by fixed logical decisions with a
  hard wall safety cap — wall-only budgets endogenously confound
  the causal-edge question).
  [DESK-PRICED] [REGIME-SCOPED: this Mac, seed-7001 samples,
  root+depth-1 only] Fences: desk numbers are predictions; the
  contract's own receipts book measured values at rung 0.
  ([OBSERVATION MATHWORLD-DESK-0](RESULTS.md#L40648 "id:2026-08-22-observation-mathworld-desk-0-rung-0").)

- **The launch step is one MoE routing locus**: masking generated
  z1's router call at 0-indexed block-position 43 alone books +20
  pooled (signs 3/3, 20 rescues v 0 breaks) — the entire late-band
  effect — while block 46 books +1 under matched-mean ~12.5%
  displacement, and the {43,46} pair is per-problem
  outcome-identical to the full 16-block late band (0/360 flips).
  The crest's token1 component is now one temporal position x one
  layer's top-8 selection; the displacement-volume account fails
  within the matched pair (unregistered color), but block and
  displaced-expert identity move together, so carrier identity is
  not yet isolated. [MECHANISM-CONFIRMED] [REGIME-SCOPED: measured
  deployment artifacts] Fences: single 4-bit instrument, greedy,
  named-80 keepset, n=3 seeds 120/seed; inertness lemma
  behavioral-not-bit (AMENDMENT -DESK-INERTNESS); demanded-expert
  identity unrecorded — identity census + same-block
  matched-identity intervention are the named next observables.
  ([VERDICT EX6-DEPTH-1](RESULTS.md#L40756 "id:2026-08-22-verdict-ex6-depth-1-the-launch").)

- [NULL] [REGIME-SCOPED: tested MoE recipes]
  The matched-rank kept-expert deletion control dies on a desk
  census before implementation: at (block 43, z1) a KEPT expert
  holds rank 0 on 1/360 rows (excluded expert 71 holds it on
  359/360), so "delete the matched-rank kept expert" is undefined
  on 99.7% of the corpus. Promoted instead: the five-arm causal
  component decomposition (NATIVE / DROP71 / RENORM7 /
  SLOT-SUBSTITUTE / FULL-MASK) — outside-top8 count is exactly 1
  and the entrant well-defined on 360/360 (19 identities), and
  with no upstream intervention every arm's local counterfactual
  computes inside one outcome-blind execution (~+0.26% FLOPs,
  expert-FFN cost model). Revival for rank-matching: a locus
  where a kept expert shares the target's rank at comparable
  frequency.
  ([OBSERVATION EX6-B43-CONTROL-DESK-0](RESULTS.md#L41096 "id:2026-08-22-observation-ex6-b43-control-desk-0").)

- [MECHANISM-CONFIRMED] [REGIME-SCOPED: tested MoE recipes]
  The exact activation-component knife at (block 43, z1) books
  the deletion vector as the carrier: adding only
  d_del = -p_71*E_71(h) to the native block output reproduces
  the launch-step rescue class (+17 pooled, signs 3/3, 18
  rescues / 1 break), while the combined entrant+renorm vector
  (46% of |delta| by norm) books +5 — below the pooled carry
  class, sub-instrument, never "inert" — and the additive path
  is per-problem outcome-identical to the direct mask (0/360
  flips). Within this exact measured decomposition, removing
  expert 71's output is sufficient and the co-moving routing
  side-effects are not. Fences: activation-space interventions
  on the shared native h, single 4-bit instrument, greedy,
  named keepset, n=3 seeds x 120, Mac-only; naming fence for
  expert 71 unchanged; dose scope 0.559 of the MoE write /
  0.160 of the normalized MLP input, residual-stream ratio
  unpaid.
  ([VERDICT EX6-B43-KNIFE-0](RESULTS.md#L41427 "id:2026-08-22-verdict-ex6-b43-knife-0-the").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  MATH-CYBER rung 0 lives: the MATHWORLD legal-action contract
  (receipt wrapper over successors(), fixed nine-field
  transition schema, 12-decision logical budget + 60s wall
  safety cap) passes deterministic-replay qualification 101/101
  causal rows identical over 40 L4-7 episodes, 35/40 solved by
  scripted greedy-hce; the replay bar caught two real contract
  bugs before passing (rule names non-unique within a legal set;
  wall_cap termination timing-dependent) — action identity is
  now name#child_hash and replay identity is scoped to the
  causal chain. Measured v desk: branching median 7 v 3-3.5
  (desk was root-only, self-flagged conservative), decision wall
  p90 4.25s v 2.05s (greedy steers into expensive states,
  confirming the fixed-logical-budget ruling). Python
  instrument; the frozen receipt interface is the handoff
  artifact for the axiom C++ replica desk, one rung behind.
  ([OBSERVATION MATH-CYBER-0-RUNG0](RESULTS.md#L41540 "id:2026-08-22-observation-math-cyber-0-rung0-the").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  Rung 1 desk: the MATHWORLD choice structure is real (all 101
  recorded decisions carry n_legal>=2, K median 7 max 22) and a
  choose-among-K frozen-checkpoint policy eval is Mac-minutes
  (725 child scorings); reachability within 12 decisions is
  UNCERTIFIED for the 5 greedy-failed episodes (30s bounded
  search expanded only 5-20 nodes — successors() wall,
  inconclusive both directions). CORRECTED by the -INTERFACE
  amendment (same day): the PERFECT bar stands LITERAL (40/40
  observed ACTIVE-ONLINE solves; reachability is interpretation
  evidence, never a prerequisite), and the full-grain
  encodability figure is 110/725 actions (15.2%) with a THIRD
  char class (`fresnelc(`) beside `I` and `Subs(`/`u_` — no
  closure claim for any fixed atom list. Pair-fit also fails:
  38.2% of encodable scoring sequences exceed ctx=512 (K prices
  compute, not context); conservative fallback leaves the stock
  checkpoint controlling 16/101 decisions — a figure whose booked
  criterion was vocab-completeness only; the -COVERAGE amendment
  measured the registered joint criterion (encodes AND fits
  ctx<=512) at the same 16/101 (corpus co-occurrence, not
  identity; ctx512-only is 66/101), killed the stock checkpoint
  as a primary rung (parked as partial-domain color), marked
  seeds 9100-9109 CALIBRATION (confirmatory PERFECT adjudication
  moves to a fresh post-freeze seed band), and froze the
  learning-signal estimands (reconstruction qualification v
  timing/order v total closed-loop) before any prereg.
  ([OBSERVATION MATH-CYBER-1-DESK-0](RESULTS.md#L41654 "id:2026-08-22-observation-math-cyber-1-desk-0"),
  [AMENDMENT MATH-CYBER-1-DESK-0-INTERFACE](RESULTS.md#L41744 "id:2026-08-22-amendment-math-cyber-1-desk-0"),
  [AMENDMENT MATH-CYBER-1-DESK-0-COVERAGE](RESULTS.md#L41832 "id:2026-08-22-amendment-math-cyber-1-desk-0-b").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  Fresh-substrate desk, no model loaded: a grammar-closed
  tokenizer (existing ATOMS + deterministic single-byte fallback
  per uncovered char — no fixture-conditioned atom additions)
  encodes 100% of the 725 calibration actions, and ctx=4096 is
  the smallest bucket with full coverage (725/725 actions,
  101/101 decisions, 40/40 episodes; ctx=512 covers only 41/101
  decisions). The base training diet (103,595 rows) loses
  nothing and regains its 388 skipped out-of-language rows (all
  <=512 tokens). Demoted PROMOTED -> CANDIDATE by the -SCOPE
  amendment: the diet has zero >512 training exposure (stock max
  501), so a ctx=4096 birth would score the 1.7k-3.9k
  calibration tail OOD — candidate pending long-context exposure
  design (fresh seed bands, never the 9100-9109 fixtures),
  compute/memory pricing, and the frozen overflow law. Same
  amendment: shipped closure law = true UTF-8 byte fallback
  (ASCII corpus makes booked lengths stand), KV-reuse scoring
  priced 1.5x in token positions — CORRECTED by the -KV
  amendment: MicroLM's cached path is causal only at T=1
  (is_causal=False whenever past is set), so cached scoring is a
  CANDIDATE gated on a registered no-trained-model qualification
  (max score delta + 100% argmax agreement), and the
  world-dominates runtime claim is retracted pending a measured
  microbench — and the update law frozen at EPISODE granularity
  (ACTIVE-EPISODIC naming; terminal-success gating cannot claim
  within-episode adaptation).
  ([OBSERVATION MATH-CYBER-1-SUBSTRATE-DESK-0](RESULTS.md#L41920 "id:2026-08-22-observation-math-cyber-1-substrate-desk"),
  [AMENDMENT MATH-CYBER-1-SUBSTRATE-DESK-0-SCOPE](RESULTS.md#L41978 "id:2026-08-22-amendment-math-cyber-1-substrate-desk").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  The axiom semantic interchange corpus is live: a cold exporter
  re-walked the frozen rung-0 receipts with world sources
  byte-asserted to code_commit 620da3bf and emitted sstr
  payloads for 102 state rows + 725 legal-action rows
  (rule/rule_target split from the enumerator's own
  "rule@sstr(target)" labels), every decision row byte-equal to
  its four frozen binding fields (abort-on-mismatch, none
  fired). Evidence-bound, not transport-certified — axiom's
  parse + round-trip admission ladder runs on their side.
  ([OBSERVATION MATH-CYBER-1-EXPORT-0](RESULTS.md#L42050 "id:2026-08-22-observation-math-cyber-1-export-0").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  Rung-1 design frozen before any band generation or outcome:
  seed bands CALIBRATION 9100-9109 / TRAIN sidecar 9200-9249 /
  ADAPT 9300-9309 / HOLDOUT 9400-9409 (pairwise disjoint;
  HOLDOUT never trains); PERFECT operationalized as PRIMARY
  final-policy HOLDOUT 40/40 with acquisition-40/40 separately
  reported; success-gated dose frozen (one AdamW step lr 1e-4
  per SOLVED episode at episode boundary, failed episodes zero
  gradient, child-token mean CE with newline, persistent
  optimizer state); runtime ctx overflow in a primary arm is a
  model-failure instrument event, never silent hce.
  ([AMENDMENT MATH-CYBER-1-DESK-0-DESIGN](RESULTS.md#L42145 "id:2026-08-22-amendment-math-cyber-1-desk-0-b-b").)

- [SINGLE-SEED] [DEVICE-SCOPED] [REGIME-SCOPED: calculus search]
  The cached scorer qualifies exactly and loses anyway: T=1
  cached continuation matches full teacher-forced scores to
  9.8e-6 with 16/16 argmax agreement (random weights, scorer
  mechanics only), but runs ~30x SLOWER than full batched
  forwards on mps at every ctx 512-4096 (launch-overhead-bound)
  — execution law = full batched teacher-forced forwards,
  KV-reuse closed-rejected with a named revival condition
  (offset-causal batched prefill, faster device, re-qualified).
  Measured corollaries: world wall (147 s/40 eps) dominates
  scoring (~15-30 s) with a measurement attached now, and 8x4096
  training OOMs without grad checkpointing (10.2 s/step with).
  ([OBSERVATION MATH-CYBER-1-SCOREQAL-0](RESULTS.md#L42207 "id:2026-08-22-observation-math-cyber-1-scoreqal-0").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  Natural long-context exposure exists on the frozen TRAIN band
  (9200-9249, first materialization): 464/2712 legal actions
  (17.1%) exceed 512 tokens across every bin (128/254/61 at
  1024/2048/4096) — the ctx=4096 exposure gate is satisfiable
  without manufacturing from calibration fixtures — and 21
  actions (0.8%) exceed even 4096, making the frozen overflow
  law load-bearing at any context choice ("full coverage at
  4096" is a calibration-trajectory fact only). Exposure is
  greedy-trajectory-conditional. CORRECTED by the -POPULATIONS
  amendment: that population is CANDIDATE-SCORING exposure — the
  success-gated TRAINING population (chosen edges of solved
  episodes) is entirely <=512 tokens (205/205, max 437), the
  >4096 tail is 20/21 concentrated in one failed episode
  (incidence 7/349 decisions, 2/200 episodes), and the substrate
  decision lands: 4096-birth as scoring window, training on
  natural short positive targets, >512 candidate scores
  disclosed as extrapolation at any declared context; theta_0
  liveness bar registered pre-birth (>=1/40 on spent
  CALIBRATION, failure = instrument not live). Same amendment
  measures the timebox fence: legal sets are load-dependent at
  the rule-timebox margin (v1 v v2 differ by 85 actions and one
  flipped episode under concurrent mps load) — parity-grade
  walks need an idle machine.
  ([OBSERVATION MATH-CYBER-1-LONGCTX-0](RESULTS.md#L42271 "id:2026-08-22-observation-math-cyber-1-longctx-0"),
  [AMENDMENT MATH-CYBER-1-LONGCTX-0-POPULATIONS](RESULTS.md#L42322 "id:2026-08-22-amendment-math-cyber-1-longctx-0").)

- [SINGLE-SEED] [DEVICE-SCOPED] [REGIME-SCOPED: calculus search]
  Scorer execution law settles on the SIMPLEST form: serial B=1
  full teacher-forced forwards — cached stepping lost 30x
  (SCOREQAL-0) and right-padded microbatching lost 8% (0.92x)
  over the real 101-decision/725-candidate profile at 19M on mps
  (correctness across paths 1.8e-5, argmax 101/101). Real-profile
  scoring wall 55.6 s v ~147 s world (world dominates at ~2.6x,
  measured); one frozen ACTIVE episode-boundary update is
  sub-second. Random-weight mechanics; untrained-boot fence.
  ([OBSERVATION MATH-CYBER-1-EXECBENCH-0](RESULTS.md#L42413 "id:2026-08-22-observation-math-cyber-1-execbench-0").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  theta_0 is LIVE: the one-shot grammar-closed birth (vocab 296,
  ctx 4096 declared, base diet 103,595 rows encoded with 0 cap
  drops, BIRTH_SEED 9001) solves 35/40 CALIBRATION under pure
  choose-among-K teacher-forced scoring with no hce anywhere —
  {L4 9, L5 10, L6 6, L7 10}, bar >=1/40 fires at 35x. COLOR:
  the five failed episodes are greedy-hce's five at the OUTCOME
  level only (mechanisms cross-swap: 3 budget, 1 ctx-overflow
  at the registered law, 1 wall-cap); choice-level equivalence
  unmeasured. Registered prior right on direction, wrong on
  shape (L5/L7 perfect, misses at L4/L6). One-shot, no
  selection; CALIBRATION is instrument-health color, never the
  PERFECT adjudication; ADAPT/HOLDOUT stay ungenerated.
  ([VERDICT MATH-CYBER-1-THETA0-BIRTH-0](RESULTS.md#L42568 "id:2026-08-22-verdict-math-cyber-1-theta0-birth").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  Zero-run choice join: theta_0 matches greedy-hce's exact
  action on 36/48 shared decision states (75%), every divergence
  a different rule FAMILY producing a different child, and the
  identical 35/40 outcome set survives the 25% different choices
  — the diet taught a NEIGHBORHOOD of the engine policy whose
  differences are outcome-neutral on this band; the two runs'
  worlds agree wherever they overlap (0 same-state/different-
  legal-set). Join conditioned on shared prefixes; color only.
  ([OBSERVATION MATH-CYBER-1-THETA0-HCE-JOIN-0](RESULTS.md#L42656 "id:2026-08-23-observation-math-cyber-1-theta0-hce").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search]
  The first closed-loop rung: PERFECT refuted + feedback null.
  ACTIVE
  final policy 36/40 HOLDOUT (the registered PERFECT prediction
  REFUTED at this dose/band/substrate) and an exact tie with
  frozen theta_0 (36 v 36; feedback direction NOT-SUPPORTED —
  the prereg's refutation branch). 38 success-gated lr 1e-4
  episode-boundary updates produced only 6 OBSERVED
  same-state top-1 divergences (2/75 shared ADAPT + 4 HOLDOUT;
  unshared post-divergence states admit no direct comparison),
  all solve/fail-neutral: self-imitation reinforced an
  already-competent policy without moving its failure frontier
  and without solve-count degradation (paths/timing did move). House counter-prior scored half (PERFECT-fails
  right, feedback-positive wrong). Color: the no-model MINLEN
  control solved 37/40 — structural color under NON-PARITY wall
  accounting, never a beats-the-model headline;
  contamination 11/80 roots (birth-diet, all L5/L7), clean
  subset ties 29/33 v 29/33. One dose point; ONLINE-v-REPLAY
  remains untested.
  ([VERDICT MATH-CYBER-1-ACTIVE-EPISODIC-0](RESULTS.md#L42884 "id:2026-08-23-verdict-math-cyber-1-active-episodic").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] One-deviation
  repairability EXISTS on the theta_0
  failure frontier but is neither universal nor loop-implied:
  of the six FROZEN failed roots, 2 are ONE-DEV-REPAIRABLE
  (L7-s9303 richly — 17/84 rescuing alternatives, min rank 2,
  rank-2 rescues at 4 of 12 sites, giving the single-fork
  top1-v-top2 COUNTERFACTUAL-CREDIT design a counted
  outcome-differing target; L6-s9300 barely — 1/228 at rank 6,
  an already-solved i_heurisch child theta_0 ranked last), 1 is
  cleanly ONE-DEV-NOT-REPAIRABLE despite being a 2-state cycle
  (L4-s9401, 147/147 exhausted, 0 censored), and 3 book
  UNDECIDED (82/172 and 9/58 wall-censored; one
  WORLD-NONCOMPARABLE legal-set mismatch at step 8 — the
  _timeboxed load-dependence fence firing). Desk census, zero
  training, frozen theta_0 continuations, reconstruction
  world-bound 62 states with 53/53 argmax reproduction; bounds
  ONLY the one-deviation + frozen-continuation repair class on
  these six roots.
  ([VERDICT MATH-CYBER-1-FRONTIER-DESK-0](RESULTS.md#L43136 "id:2026-08-23-verdict-math-cyber-1-frontier-desk").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] The theta_0
  argmax controller misses LEGAL immediate solutions: 21 of 130
  recorded terminal-child decision states (16.2%) chose a
  non-terminal child — every miss at step 0-1, 15/21 with the
  missed terminal at rank 2 (margins min 0.029 / median 8.49),
  and 2 misses cost their episodes (L6-s9103 terminal ranked
  7/7 at margin 3322; L6-s9300 = the FRONTIER rescue, hereby
  reclassified controller defect, not counterfactual signal).
  Desk-exact derivation: a TERMINAL-FIRST override (world
  is_solved predicate before the learned scorer) lifts theta_0
  to 36/40 CALIBRATION and 39/40 FROZEN ADAPT, HOLDOUT 36/40
  unchanged. On-path census only (off-path terminals
  invisible); 6/20 miss episodes are birth-diet-contaminated
  but neither episode-costing miss is; no retroactive rescore
  of booked verdicts.
  ([VERDICT MATH-CYBER-1-TERMINAL-DOMINANCE-0](RESULTS.md#L43378 "id:2026-08-23-verdict-math-cyber-1-terminal-dominance").)

- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] The
  failure-triggered retrospective-credit LABELER is qualified
  6/6 registered bars on outcome-spent data under the frozen
  TERMINAL-FIRST + theta_0 operator: binds the recorded
  L7-s9303 trajectory 12/12, reproduces the FRONTIER rank-2
  outcome-differing fork receipt-to-receipt, emits zero labels
  on the no-rescue root (L4-s9401) and the overflow root
  (L6-s9403), never emits negatives, censored forks emit no
  label (live-label suppression by construction only), and
  L6-s9300 SOLVES live at step 0 under the terminal override.
  Diversity caveat: the 11 qualification labels are ONE forced
  child around a 2-cycle — one corrective fact. Yield pricing
  belongs to the registered LABEL-YIELD-0 band census (bars:
  GO iff >=3 label-bearing failed episodes AND >=5 labels on
  80 fresh sacrificial episodes 9500-9519; house prior NO-GO;
  band NOT generated pending Artin/GPT bar approval).
  ([VERDICT MATH-CYBER-1-RETRO-LABELER-QUAL-0](RESULTS.md#L43575 "id:2026-08-23-verdict-math-cyber-1-retro-labeler").)

- [NULL] [REGIME-SCOPED: calculus search] LABEL-YIELD-0 books
  NO-GO exactly as the registered house prior called: on the
  80-episode sacrificial fresh band (9500-9519, now permanently
  outcome-spent) the TERMINAL-FIRST + theta_0 baseline solves
  76/80 (L5/L6/L7 perfect; all 4 failures L4 budget_exhausted,
  the level consuming 67 decision rows v 20-22 elsewhere), so
  the failure-triggered rank-2 labeler finds only 1
  label-bearing episode (bar >=3) and 4 distinct corrective
  facts under the -DEDUP-B hash-pinned key (bar >=5; 10 raw
  rows = one episode cluster). Scarcity is solve-rate success
  starving the operator, not labeler failure — the priced lever
  is band size (~3x) or hardness. Operator PARKS per
  registration; contamination 19/80 (all birth_diet, none a
  failure); future treatment bands >= 9600; no training or
  treatment prereg until Artin/GPT review.
  ([VERDICT MATH-CYBER-1-LABEL-YIELD-0](RESULTS.md#L43816 "id:2026-08-23-verdict-math-cyber-1-label-yield").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] CYCLE-ESCAPE
  (exact-repeat-triggered action masking over TERMINAL-FIRST +
  theta_0, zero training, zero added model calls) is a REAL but
  PARTIAL controller lever: it solves 4 of the 15 spent
  argmax-controller failures — both terminal-flips (EXACT d=1)
  plus exactly the two loop episodes whose single-deviation
  escapes were already measured (L7-s9303 d=5, L4-s9507 d=7) —
  and converts NONE of the five open multi-deviation loop cases
  (masking redirects, it does not navigate; theta_0 off its
  argmax path does not find solutions it never ranked first).
  Both frozen bars miss (3/9 loop-class, 4/15 total v >=5), no
  promotion; house prior (5-7) missed low at 4. Recomputed
  divergence color: every firing escape leaves the recorded
  trajectory one step after its first mask (the shipped
  receipts' null divergence field was a flush defect, disclosed
  and recomputed in the entry). Residual spent frontier under
  the best zero-training controller: 11 episodes (3 wall, 2
  overflow, 6 budget).
  ([VERDICT MATH-CYBER-1-CYCLE-ESCAPE-DESK-0](RESULTS.md#L44120 "id:2026-08-24-verdict-math-cyber-1-cycle-escape").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] REGRET-LDS
  (rank-weighted limited-discrepancy graph search over frozen
  theta_0 sibling rankings, zero training) shows NO OBSERVED
  multi-deviation navigation on the six residual budget
  failures: sole solve = the known L4-s9518 rank-3
  one-deviation rescue rediscovered ([3,1,T], discrepancy 2, 22
  expansions); highest fully observed common ladder rung = 24
  expansions at 1/6; the 96-expansion primary read is PARTIAL
  (3/6 wall-censored at 38-73 expansions, censoring never
  converted to failure). Weakens greedy-control-as-the-plateau
  within observed volume (388 expansions, 4,356 scored
  candidates); representation insufficiency NOT established.
  Instrument result: score cost model held (11.2/expansion v
  ~15.6 priced) but non-model world time is a material
  unbudgeted wall share (censored episodes 2.1-2.7 calls/s v
  4.3-5.2 uncensored) — successor desks budget the world
  separately.
  ([VERDICT MATH-CYBER-1-REGRET-LDS-DESK-0](RESULTS.md#L44369 "id:2026-08-24-verdict-math-cyber-1-regret-lds").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] The wall-lifted
  closure re-read resolves REGRET-LDS's censored half on the
  PARK branch: all three previously censored roots reach the
  full 96-expansion budget (max wall 898.9 s v the 3600 s
  emergency cap, which never fired) and NONE solves at any
  deviation count — the combined six-root picture is now
  root-level fully observed (502 expansions / 6,308 scores):
  5/6 exhaust 96 unsolved, 1/6 solves via its known
  single-deviation rescue, ZERO multi-deviation solutions.
  Naive theta_0-ranked REGRET-LDS PARKS at the 96-expansion
  scale; next branch = ACTION-BASIS / value-quality. Measured
  time split (superseding the calls/s inference): world
  materialization is 73-89% of search wall — the symbolic
  world, not the 19M model, is the binding cost of MathWorld
  search. Copy-modification defects disclosed (inherited
  provenance head hashing the parent driver — run identity
  established by git; stale docstring; wall_cap_s unreceipted),
  all fixed forward in the booking commit.
  ([VERDICT MATH-CYBER-1-REGRET-WALLLIFT-0](RESULTS.md#L44619 "id:2026-08-24-verdict-math-cyber-1-regret-walllift").)

- [SINGLE-SEED] [REGIME-SCOPED: calculus search] The
  (label, sibling-index) ACTION-BASIS program schema is
  compresses the median action 5.75x (its booked completeness
  claim was UNVERIFIED-as-measured and is corrected by the
  -QUAL amendment; the semantic decoder qualification is the
  bullet below) (program med 28 v child med 161 GCTok
  tokens) — but the desk lands BETWEEN, not promoted: the
  program TAIL stays expression-dense because rule targets are
  serialized sub-expressions (program CV 1.69 > child CV 1.20,
  so the MINLEN length confound is NOT removed), and the frozen
  512-fit bar failed mostly on the PARENT PREFIX (p90 555, 15/
  102 states > 512 before any candidate), which no action basis
  can shrink by construction. Registered revival: schema v2 =
  (rule, target_index, sibling_index) with the target INDEXED
  not serialized — bounded length by construction — under its
  own blind-threshold desk; parent-prefix length is a separate
  state-naming lever.
  ([OBSERVATION MATH-CYBER-1-ACTION-BASIS-DESK-0](RESULTS.md#L44747 "id:2026-08-24-observation-math-cyber-1-action-basis").)

- [MECHANISM-CONFIRMED] [REGIME-SCOPED: calculus search] The
  semantic ActionProgram schema (rule, first-occurrence AST
  address, child-key-sorted branch index) decoder-QUALIFIES on
  the frozen corpus: 533/533 decodable actions reconstruct
  their exact child with zero wrong-child/collision/range
  failures and no frozen-row operand — including all 122
  multi-occurrence-target actions (first-occurrence
  canonicalization held; branch indices are tiny: 0/1/2 only).
  The real blocker is UPSTREAM: 26/101 corpus decisions (192
  actions) fail the state-string round-trip
  (sympify(state_before) does not reproduce State.key()) — the
  exported str() serialization under-determines state identity
  for a quarter of decisions, an interchange-contract limit
  (srepr export is the fix) that also bears on the axiom
  corpus. Anatomy only; no lengths measured; v2 length desk
  needs round-trip-complete serialization + blind thresholds
  first.
  ([OBSERVATION MATH-CYBER-1-ACTIONPROG-QUAL-0](RESULTS.md#L44860 "id:2026-08-24-observation-math-cyber-1-actionprog-qual").)
- [MECHANISM-CONFIRMED] [math-cyber] [mac] State serialization identity is
  REPAIRED by the versioned srepr corpus under a per-node adaptive
  inverse (plain sympify canonicalizes the world's unevaluated nodes,
  18/827; blanket evaluate(False) breaks the canonical majority,
  771/827): 101/101 decisions bind (v1 75/101) and 640/725 actions
  decode exactly, zero regressions; the residual 85 failures are all
  target_not_in_parent on the 26 newly-bound unevaluated-parent
  decisions — an encoder ADDRESSING residue, not a schema or corpus
  defect (zero wrong_child / collision / out-of-range).
  RESULTS.md#L45054
- [MECHANISM-CONFIRMED] [math-cyber] [mac] theta0's state observation is
  NOT incomplete on the 26 srepr-blocked decisions: the exact engine
  state prints the frozen visible string verbatim 26/26 (its
  canonicalization prints differently 0/26; sympify is not
  idempotent-under-sstr), and zero visible strings map to two distinct
  states across the population (16 distinct states, two 6x cycle
  repeats) — the evaluation structure was lost by interchange parsers,
  never hidden from the model; STATE-ALIAS bar no-fires vacuously
  (no comparable pair; behavioral half unmeasured) and ACTION-SITE
  qualification proceeds ahead of state naming. RESULTS.md#L45227
- [MECHANISM-CONFIRMED] [math-cyber] [mac] The v3 site-addressed
  ActionProgram (rule, (kind, ordinal-among-atoms), branch) is FULLY
  qualified on the frozen corpus: 101/101 decisions bind, 725/725
  actions decode to the exact frozen child, zero
  wrong_child/collision/out-of-range/unaddressable — closing the
  85-failure label-target residue; branching confined to i_parts (141)
  + i_unprod (1); theta0 str() view == interchange sstr() view 102/102;
  fences: nested multi-limit Integral law qualified VACUOUSLY (0 corpus
  actions), ordinal-tie leg unmeasured, branch_index not yet the
  model-facing representation. RESULTS.md#L45392
- [MECHANISM-CONFIRMED] [math-cyber] [mac] The v4 SEMANTIC
  ActionProgram (rule, (kind, first-preorder ordinal), u_choice |
  branch) is fully qualified: 725/725 exact reconstruction, all 256
  i_parts actions decode from a mathematically meaningful u_choice
  (which factor is u; ordinal among eligible x-dependent factors) with
  ZERO ambiguous-u anywhere; reading-order site coordinate changes
  104/402 ordinals vs v3 with no ambiguity; fences: scoped to sympy
  1.14.0, decode shares in-process regeneration (uniqueness +
  addressability, not a cross-process round-trip), i_unprod stays a
  candidate index, nested leg still vacuous. RESULTS.md#L45570
- [MECHANISM-CONFIRMED] [math-cyber] [mac] The v4 semantic program
  serialization measures med 12 / max 19 GCTok tokens (13.4x median
  compression v the full child; zero action-induced 512 overflows; the
  14 over-512 decisions are parent-prefix-only) but the within-decision
  candidate span FAILS its bar (p90 12 > 8) and the pre-registered
  opcode counterfactual fires exactly at threshold (span p90 12 -> 6,
  reduction 0.500 >= 50%): the length nuisance is TOKENIZER-NAMING —
  rule ids cost med 7 / max 15 tokens against <= 6 for all semantic
  content — so the desk STOPS pre-training; next candidate is an
  action-opcode vocabulary desk; PROMOTE never rescored under the
  hypothetical tokenizer. RESULTS.md#L45758
- [MECHANISM-CONFIRMED] [math-cyber] [mac] ActionGCTok (36 reserved
  rule opcodes appended after the frozen 296-token GCTok, ids 296-331,
  vocab 332) is a drift-free EXTENSION: token-identical to legacy over
  all 207,190 birth-diet strings + 929 frozen MathWorld strings,
  725/725 v4 programs round-trip, and it reproduces the booked opcode
  counterfactual number-for-number (program med/p90/max 5/8/8, span
  p90 6, zero action-induced 512 overflow) — implementation-fidelity
  rung, lengths not a new discovery; fences: future diets must stay
  literal-"<"-free or re-run the legacy bar; paired STATE-v-PROGRAM
  arms must share the extended vocab/init/rows. RESULTS.md#L45932
- [MECHANISM-CONFIRMED] [math-cyber] [mac] REUSE fires on theta0's
  exact 103,595-row birth diet: 70.78% row-weighted / 71.62%
  edge-weighted relabels deterministically into a unique v4
  ActionProgram (micromodel shards 72.0% pooled, step_chains 58.0% —
  inverting the registered prior's source ordering); no_engine_edge
  22.5% (unresolved 13,730 + integration-constant offsets 9,267),
  parent str-round-trip failures 6.71%; matched paired-training subset
  = 73,324 rows; i_unprod branch>0 exposure is MATERIAL in the diet
  (1,002/2,894 pairs on 2-sibling sites) so its semantics decision
  gates the matched-subset prereg; zero ambiguous/target/outside-v4
  classes. RESULTS.md#L46128
- [MECHANISM-CONFIRMED] [math-cyber] [mac] i_unprod's opaque branch is
  replaceable by ONE semantic ordinal: over all 2,894 PDC-matched
  edges (2,888 parents, parity-gated 0 failures against the frozen
  rule), even the minimal scheme S3 = which-additive-term covers
  everything with zero collisions (S1/S2 too); distinct candidates
  per site never exceed 2, so the emission-cap set-order hazard is
  vacuous; the desk's branch-exposed count (981, matched-site raw
  emission) and PDC's (1,002, any-site accepted) differ by definition
  on three axes, both stand; next: term_index qualification rung, then
  the fully semantic program basis gates the paired experiment.
  RESULTS.md#L46337
- [MECHANISM-CONFIRMED] [math-cyber] [mac] The CANONICAL semantic
  ActionProgram — (rule, site) | i_parts +u_choice | i_unprod
  +term_index, site = (kind, first-preorder ordinal) — is fully
  qualified: 725/725 corpus actions decode with zero regression, all
  2,894 PDC i_unprod edges decode exactly from term_index
  (parity-gated), and no opaque child-key branch remains in the decode
  law across the measured legs (accepted-set intersection verified
  never load-bearing, 13/13 corpus term-sets unpruned); vacuity fences
  (nested/definite legs) continue; the representation arc is CLOSED
  and the paired STATE-v-PROGRAM experiment is gated only by its own
  preregs. RESULTS.md#L46519

- [MECHANISM-CONFIRMED] [math-cyber] [mac] The frozen ActionProgram
  interchange fixture for Axiom Tranche A exists and is
  self-qualified: full 101-parent / 725-action corpus with 725/725
  structured-field replay and zero same-program/different-child
  collisions, plus all 2,894 PDC i_unprod stress edges replaying
  exactly from term_index (separate denominator); ActionGCTok
  program_text round-trips everywhere but is a serialization
  diagnostic — structured fields are the semantic identity; program
  keys are a map to children, not a bijection (491/725 distinct);
  the durable relay carries a transparent pre-delivery amendment to
  param_kind none|u_choice|term_index. Delivery stays Artin-manual.
  RESULTS.md#L46685

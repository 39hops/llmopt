# SPEC: the reverse LLMUE / temporal pincer (living spec — add cells as ideas land)

Origin: Artin's reverse-LLMUE riff + his original Tenet/pincer
bank; informed by TODAY's measured verdicts (revpairs 39 = naive
mixing toxic; oneshot 54 = answer-conjecture is strong from
birth; direction tax real). Slot: after index/cleanup (step 3).
Speed doctrine binds every cell: d256 pilots, KV-cached waves,
ms-class oracle checks, quick verification, lossless levers on.

## 1. Architecture answers (each grounded in a measured result)

- **Two separate crystals, never one** — revpairs (39/120)
  measured the naive shared-crystal form as toxic; the backward
  model is its own birth. They share ONLY: tokenizer (vocab-40),
  substrate class, and the oracle. Blackboard doctrine applies:
  models never exchange unverified state — every handoff is an
  oracle-checked expression string.
- **Layers/shape: unchanged MicroLM** (d256 pilot -> 19M-class
  promotion). No architectural novelty is justified by any
  measured result: depth anatomy says the crystal uses its full
  depth; the native-transformer program CLOSED (notation-
  invariant emission wall); late-layer plasticity governs online
  updates as usual.
- **Weights: signed real, standard alphabets.** NEGATIVE: yes,
  non-negotiable — the opposition ladder (S4 54 / Z1S 0 / Z1 0)
  says per-weight sign is load-bearing. COMPLEX: no for math —
  the factorial's math column measured rotation free-not-
  profitable (G5-dep 62 = M5 62) and the euler read found no
  spontaneous phase structure; the backward direction of a
  phase-free grammar is still phase-free. CONDITIONAL: if the ZX
  column (in flight) books G5 > M5 on phase-carrying data, a ZX
  pincer inherits complex weights — decided by tonight's gates,
  not by taste. Deployment alphabet free per the tournament
  exchange law (ternary at width, M5/P2 narrow).

## 2. Data schema

Backward rows reuse the exact pair frame with roles flipped —
`Current: <later-state>\nHints: none\nStep: <predecessor>\n` —
vocab-40 native, zero new atoms (the backward model's whole world
is backward; no direction marker needed BECAUSE the models are
separate — the marker was only ever needed to disambiguate a
shared crystal, which is the design revpairs killed).
- DETERMINABILITY (the law's backward form): backward targets are
  one-of-many-valid (an answer has many predecessors) — same
  class as forward's branching floor, and verified the same way:
  a backward emission s is VALID iff the forward step s -> t
  verifies (existing verify machinery, ms). Underdetermined-rows
  hazard does NOT bite: any valid predecessor is acceptable and
  oracle-checkable; hallucination has nowhere to hide.
- Provenance free: every forward chain in the corpus reverses
  into backward rows at zero farm cost (the farm is answer-first
  by construction).
- BIDIRECTIONAL-CHEAT FENCE (Future-Work, standing): backward
  models trained on reverse-sampled corpora can memorize the
  GENERATOR's answer distribution. Controls: exclude-guarded
  skeleton splits at every probe; backward validity scored by
  forward-verify, never by string match against the corpus.
- Distribution upgrade (banked, cell R5): backward targets as
  soft labels over the verified predecessor SET (the
  distribution-rows bank, backward edition).

## 3. How they cooperate (the pincer protocol, v1)

The classical problem: backward search needs a goal, and the
answer is unknown. TODAY'S ONESHOT RESULT (54/120 from birth,
3.4M tokens) supplies it: answer-CONJECTURE is cheap and strong.
- Ply 0 — CONJECTURE: the oneshot model proposes k candidate
  answers (T-sampled). Oracle checks each directly (diff = ms).
  Hit => solved, done. (This leg alone is measurable now with
  checkpoints on disk.)
- REPAIR — near-miss peeling: for unverified candidates, the
  BACKWARD model peels j plies (each peel forward-verified as it
  lands), minting a set B of intermediate target states.
- MEET — forward chain search as usual, testing each reached
  state against B (skeleton-normalized hash). Contact => splice
  the two half-chains; the full chain is already edge-verified
  (backward peels were forward-checked at mint).
- Soundness by construction: no unverified state ever crosses
  between models; the oracle sits at ply 0, at every peel, and
  at the splice. Grover-fence note: this is meet-in-the-middle
  economics (search depth halves), not quantum anything — the
  honest classical content of the superposition frame.
- Later riders: judge-collapsed decoding on BOTH models (banked
  2026-07-26); EU/batched-KV shared-prefix scoring for the
  candidate sets.

## 4. Cells (cheap-first; pre-reg each before firing)

- **R0 (free, checkpoints exist)**: conjecture-leg readout —
  fmt_oneshot_1p.pt proposes k=8 answers per gate problem,
  oracle-checked; books how much of the 120-gate falls to
  conjecture alone, per level. Minutes, no births.
- **R1**: backward crystal birth at d256 (reversed pairs ONLY,
  1P + 3ep) + backward validity probe (forward-verify of emitted
  predecessors, exclude-guarded fresh states). Question: does
  backward train UP like forward (85%-class) or is peeling
  intrinsically harder?
- **R2**: pincer v1 (conjecture + peel + meet) vs forward-only
  at EQUAL total token budget (the regret-round-2 economics
  lesson: must beat let-it-finish, not just work). Battery: the
  gate band + the L9/frontier probes (ties/misses concentrate
  there).
- **R3**: backward-as-teacher — backward model mints predecessor
  chains from SOLVED answers of the stuck-state worklist's
  neighbors; LLMUE session eats them; paired vs engine-chain
  food (the exchange's third food channel).
- **R4**: ZX pincer (conditional on tonight's alphabet verdict +
  ZX gate maturity) — rewrites are near-involutive in ZX (color
  change is self-inverse; fuse reverses as split), so backward
  may be nearly FREE there; test after math legs read.
- **R5**: distribution-target backward training (soft labels
  over verified predecessor sets) vs single-target R1.

## 5. Speed/efficiency commitments

d256 births 15-25 min; R0 is minutes; every oracle check is the
existing ms-class verify; peels verified incrementally (never
batch-then-audit); KV-cached sampling everywhere; per-problem pp
sidecars on every cell (step-3 item (d)) so overlap/wandering
reads are free afterward. No cell in this spec exceeds ~40 min
of Mac wall; 3080 never required for the pilot tier.

## 6. AMENDMENT (Artin's push, same day): the backward model is an
## architecture BRACKET, not a mirrored transformer

The conservative form (section 1) is demoted to baseline. The
backward task is structurally different from forward generation:
a predecessor differs from t by UN-APPLYING one rule at one site
— so the natural output is an EDIT POINTER, not text. Three arms:

- **B-a (baseline)**: mirrored transformer, text emission — R1
  as originally specced. Exists to be beaten.
- **B-b (the different machine)**: POINTER/EDIT model —
  state -> distribution over (inverse-rule, site), NNUE-class
  (the syndrome-policy machinery reused verbatim: 94-98% top-3
  at ~us was already measured for state->rule); the ENGINE
  applies the chosen inverse-rewrite exactly. Convergence of
  three banked riffs: step-tokens (the unit of generation IS the
  verified rewrite), bitboards (representation makes moves
  machine-ops), EU-eval (all sites scored in one pass; peels
  share ~95% of the tree). Dissolves the long-emission wall
  (nothing long is emitted — classification over ~24 rules x
  sites) and verification cost (validity BY CONSTRUCTION:
  forward-apply the pointed rule to the produced predecessor;
  must reproduce t bit-exact — deterministic, not oracle
  search). Speed class: us-ms per peel vs the transformer's
  full decode. Training data: (t, rule, site) triples — free
  from every farmed chain (the emitter knows the move it made;
  axiom logs kind+site ALREADY in the ZX schema — math chains
  need the annotation added, an axiom ask).
- **B-c (banked, research tier)**: parallel/set-valued emission
  (one pass -> the whole predecessor cloud as edits; NAT/masked
  style). Only if B-b's coverage disappoints.

PRE-REG PRIORS: B-b >> B-a on speed (structural); capability
question = COVERAGE (can a pointer net rank inverse-moves it
never saw applied? — the trial-mass lesson says give unseen
rules newcomer mass). The pincer protocol (section 3) is
arm-agnostic: conjecture + peel + meet unchanged; only the
peeler's implementation varies. R1 becomes R1a/R1b paired.
NNUE-symmetry note (measured 2026-07-26): pointer nets are
oligarchy-phase, sign-skewed toward the readout — the backward
NNUE will NOT look like the crystals, and shouldn't.

## 7. AMENDMENT 2 (Artin, same day): B-b refined to
## SCORE-OVER-ENUMERATED-MOVES + the alphabet-follows-output-type bet

- **B-b final form**: the ENGINE enumerates the complete legal
  inverse-move set for state t (cheap, complete by construction);
  the model only CALIBRATES — state + legal set -> distribution,
  one forward pass, zero generation. The pointer form's coverage
  question VANISHES (completeness is the engine's job; ranking is
  the model's). This is the honest classical form of "the quantum
  computer returns the distribution": full legal superposition
  enumerated exactly, amplitudes learned.
- **Training-objective bracket** (different ways to train, per
  Artin): (i) imitation soft-labels (mass on the replayed true
  move, smoothed over legal set); (ii) VALUE labels — P(move
  leads to root within budget), computable by engine replay
  (regret-probe lineage, AUC .914 says fate is learnable);
  (iii) contrastive/ranking loss over the legal set (wave-
  contrast lineage: verified-vs-rejected siblings free per
  state). Race (i) vs (ii) at matched labels; (iii) rides if
  either plateaus.
- **Inference**: NO autoregressive decode anywhere in the
  backward half — one forward pass per state returns the whole
  distribution; the pincer's peel keeps top-k mass; the meet
  uses the distribution directly (highest-mass path toward the
  root preferred, exactly Artin's phrasing). Domains: math/
  physics/q-circuits only (charter re-affirmed in the riff).
- **PRE-REGISTERED BET (Artin, on record): the ternary reverse
  model FACEPLANTS — "too deterministic."** Support already
  banked by his own 07-22 riff (ternary = search substrate,
  precision = accuracy substrate) + the NNUE oligarchy evidence
  (chess NNUEs need int8's 256 levels; our NNUE heavy-tailed,
  sign-skewed). House co-signs directionally with the sharper
  form: **the alphabet follows the task's OUTPUT TYPE — decisions
  tolerate compression, calibrated distributions need
  resolution.** Cell R6: backward scorer at {ternary, M5, fp32}
  matched arch/labels, scored on ranking quality (Spearman vs
  value labels + peel success), NOT on argmax accuracy (argmax
  would mask exactly the calibration ternary is predicted to
  lose). If ternary ties anyway, the output-type law dies and
  the lens law extends to distributions — either verdict banks.

## 8. R7 (from the mixture bank): the conjecturer lives in the main crystal

pairs+oneshot mixture at d256 (50/50 and 80/20 vs pairs 57):
oneshot rows are same-direction maximal skips (licensed by skip
54; NOT the revpairs class). If the mixture holds the pair gate
while gaining one-hop conjecture, the pincer needs only TWO
models (mixture crystal + backward scorer) instead of three, and
the conjecture leg inherits the crystal's full validity. pp
sidecars mandatory (watch: does the mixture's oneshot skill
retain the 0.5-ply/no-wander signature inside a chain-capable
model, or do the skills interfere?).

## 9. Comparison arms (Artin, 2026-07-26): pretrained-LLM baselines

Every headline cell gains a PRETRAINED-LLM comparison arm run on
the SAME battery (the priors-vs-drag control, resurrected for
this program): a small general LLM (Qwen-class 0.5B, few-shot,
charset-masked as in the step-tokens era) scored on (a) the R0
conjecture leg (propose answers, oracle-checked), (b) forward
chain gate, (c) backward peel validity. Measured prior: the
0.5B's historical 3.3% validity vs math-native 65% — the arm
exists to keep that context number CURRENT per battery, not to
win. Fences: equal token budgets; prompting per step-tokens
conventions; numbers labeled cross-substrate (context, never a
promotion comparator). Also from the same exchange: the
rotational-domain map (RIFF entry) — Fourier continent = the
pincer's second phase-carrying territory if ZX books the
interaction.

# The derivation-search results: 73.6% → 95.3% in 48 hours

*llmopt's "Stockfish for math" arc, 2026-07-06 → 07-08. Every number
below is a committed measurement on held-out, string-seeded problem
sets, sympy-oracle-verified. Written as the handoff/publication draft.*

## The one-paragraph version

A beam search over sympy rewrite rules, guided first by a hand-crafted
eval, then by learned components, solves generated calculus problems.
Over 48 hours of measure-everything iteration, held-out solve rate at
fixed node budgets went **265/360 → 343/360** without changing the
rules, the model, or the problems — every gain came from *search
wisdom*: confidence calibration, width/depth allocation, and knowing
which components actually carry which capability. The headline
finding: **ranking moves is grammar and fits in a dictionary; knowing
when you're sure requires learned per-node discrimination; and beam
width is a partial substitute for confidence.** Plus a two-mechanism
answer to "is there a limit to self-teaching."

## The racing arc (all: same held-out seeds, budgets 25/50/100/200, n=15/cell)

| engine | total | delta from |
|---|---|---|
| full enumeration + HCE, width 8 (day one) | 265 | — |
| k1×3 random restarts ("synthesized breadth") | 267 | depth ≈ breadth |
| adaptive entropy-k, T=1.0 (pre-registered null) | 268 | flat confidence |
| random top-3 pruning | 277 | +12 pruning alone |
| markov-adaptive (structural null: rule-level ties) | 282 | |
| LLM top-3 pruning | 288 | +11 model judgment |
| **markov bigram top-3** (zero inference cost) | **293** | dict beats LLM |
| adaptive entropy-k, T=0.1, width 8 | 300 | calibrated confidence |
| markov bigram, width 2 | 316 | width dividend |
| k3-LLM, width 2 | 318 | |
| adaptive-LLM, width 2 | 328 | |
| adaptive-LLM, width 2, k_max=3 | 337 | right-sized gate |
| **+ mined macros (the shipped `engine.solve()` config)** | **343 (95.3%)** | highway dividend |

n=30 confirmation of the adaptive-vs-fixed comparison: 593/720 vs
560/720 (int L3 flips to fixed-k: over-confident dives at the hardest
level — width hedges).

## The component taxonomy (what actually carries what)

- **Policy (which move):** a rule-bigram count table matches the
  fine-tuned 0.5B at top-3 offline (99.7% = 99.7%) and BEATS it
  in-search (293 vs 288) at zero inference cost. The LLM's pruning
  value is mostly rule grammar.
- **Confidence (when are you sure):** the LLM's per-node scores,
  temperature-calibrated (T=0.1; at T=1.0 the softmax is flat
  everywhere — a pre-registered null whose instrument named the cause).
  The Markov table cannot do this (rule-level scores tie same-rule
  candidates: structural, not fixable). *The GPU buys confidence, not
  choice.*
- **Value (will this state solve):** the title match ended in a photo
  finish — NNUE 93, absorbing-Markov P(solve|bucket) 92, hand-crafted
  HCE 89 (run-to-run jitter ±2 from wall-clock boundaries). Both
  principled evals beat hand-tuning; between them, choose by
  engineering taste (torch at inference vs a probe run and a dict).
  NNUE's offline ordering advantage stands at rho +0.937 vs +0.721.
- **Width:** the unquestioned width=8 was wrong — width=2 dominates at
  fixed budget (50 vs 36 on hard cells). Width is a partial substitute
  for confidence; k_max must be sized to width (6→3 was +9).

## The limit-of-self-teaching answer (two mechanisms)

1. **Within the reachable set** (expert iteration): frontier harvest —
   problems the baseline provably fails, solved by the full engine —
   yielded 41% (51 frontier / 21 harvested / 183 rows; every int-L4
   problem was frontier). Retraining lifted offline top-3 to 100.0%.
   Held-out frontier curve point: [PENDING — measuring at write time].
2. **Moving the ceiling** (new operators): ∫sin²x had NO derivation in
   the real-form rule set — a provable operator-closure ceiling. One
   representation-change move (trig → complex exponentials) and the
   derivation exists through entirely pre-existing rules
   (euler → i_sum → i_const_factor → i_usub → i_table → subs_eval).
   Self-teaching optimizes within the closure; representations enlarge
   it. The limit of self-teaching is the limit of self-checking.

Tabula rasa (AlphaZero-way ablation): round 0, knowledge-free (random
k1 dives, count_ops tie-break, verifier only) solves 63% overall —
perfect at L1-2, cliff at L4 (4/20, 2/20): the implicit curriculum.
Round 1+ in progress at write time.

## Engineering findings (each measured, each guarded in code)

- Verify integral edges by DIFFERENTIATING the difference (doit()
  integrates: 31 CPU-min for nothing).
- simplify-as-zero-test needs an expand → numeric-screen → simplify
  ladder (one 18-op state: 1471s); reject residues carrying
  Integral/Subs (evalf silently runs 30-digit quadrature: 2050s).
- Probe timeouts must be BaseException (broad except swallowed alarms).
- Never materialize full LM logits: body forward + head on selected
  positions (two 10GB-GPU OOMs the 36GB Mac masked).
- pmap fork-pool parallelism: 3.1x; sampled verification (winning
  paths always fully re-verified; corrupted-counter zero): 1.65x
  lossless.
- Macro promotion by TRAFFIC beats promotion by convention: the mined
  d_product→d_const highway (14.8% of winning-path traffic) pays +12;
  the textbook quotient rule (zero traffic) never fired.

## Honest nulls (all pre-registered or instrumented)

Luby vs equal-thirds restarts: 269 = 269 (theorem is asymptotic; R=3
isn't). Golden-angle vs iid restart diversity: 90 v 87, noise.
T=1.0 adaptive: 268 (flat softmax). Markov-adaptive: 282 (rule-level
ties). d_quotient: never fires. prop3+nnue stacking at high budget:
collapses to 6/15 under the wall (spend wall-clock on nodes, not
double ranking).

## Origin story, closed

Limits resisted LoRA training (<=21%), motivating the engine. The
engine now solves them: l_hopital emits UNEVALUATED derivatives that
the rung-1 diff rules finish — the rungs composing in one derivation.

## Future work (spec'd or banked, in priority order)

Strategy-portfolio bandit (UCB over engines per problem class — the
signals exist: H, P(solve), yield). Bidirectional/meet-in-the-middle
search (backward-chain from solved-form shapes). Highway mining v2
(n-gram macro auto-promotion). Lean port (proofs: same architecture,
tactic moves, kernel verifier). Code domain (states = programs,
oracle = toolchain + tests, eval = stopwatch; train on execution
traces — the ladder's simulation-resists finding). Budget reallocation
across problems (NNUE's predicted nodes-to-solve as a quit-early
signal). Digital-circuit port (states = netlists, moves = Yosys
rewrites, oracle = ABC formal equivalence — math-grade and
milliseconds; arguably the most tractable domain jump). The verifier
cost gradient (formal > simulation > physical) maps the whole
portability frontier: analog/SPICE and photonics/FDTD port with
shrinking search budgets and growing reliance on learned evals; the
fab is the regime where the loop inverts.

## Reproduction

Every table: `scripts/bench_*.py` with string seeds; every training
run: `scripts/train_*.py`; data generation: `scripts/gen_*.py`,
`scripts/harvest_*.py`. See README Highlights for the per-result
commands and the git log for the measurement-by-measurement history.

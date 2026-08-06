# Spec 2026-08-05: the TENET battery (temporal-pincer super-spec)

STATUS 2026-08-05 night: D0 DONE (VERDICT WEIGHT-READER-0,
back-booked; 796,550 params confirmed). W0 DONE — VERDICT TENET-W0:
P-DIRECTIONAL null fires (0.139 vs 0.167 chance, 345 twins, sanity
0.866); weight features direction-specific at toy scale; W1's
registered expectation is now 'direction classifier SUCCEEDS'. Next:
D1 (reverse gate) + D2 (exclude-union) builds, then R0-rev.

STATUS 2026-08-06: D1 + D2 BUILT (tenet_d1_revgate.py replay-scored
after AMENDMENT R0-REV-D1 caught the direction-blind first cut;
tenet_d2_revdiet.py — the exclude-union law FIRED for real,
AMENDMENT R0-REV-D2, 21/120 band expressions excised from gen-4).
R0-rev DONE — VERDICT TENET-R0-REV: P-NULL fires (rev twin trains
in loss 0.319 v fwd 0.358 but scores 0/120 on the reverse gate;
fwd control 0/120 on D1 as registered). AMENDED same morning
(R0-REV-DIST): the 0/120 is prompt-distribution-scoped — chain
starts are OOD inputs for a reverse model. R0-REV-B (Artin's
reopen call): the in-distribution D1b gate reads rev 24/120 v fwd
1/120 — P-TRAINS-B fires; backward emission is REAL and
distribution-local (L4 0/24 anomaly open). R1b-micro REOPENED,
gated behind D3 (budget accountant — next build). W1 population
building on the 3080 (Artin GO; ~5 min/birth, all 50 pairs land
this morning); W1 still needs the reader-shape bridge (HIDDEN=16
tokenizer v d64/L8 crystals — reviewer scan 2026-08-06) before it
can pre-register.

STATUS 2026-08-06 midday: D3 BUILT (tenet_d3_budget.py +
regression tests — the R8 overdraft shape now refuses). W1 DONE —
population 50/50 pairs clean; bridge built (tenet_w1_bridge.py);
VERDICT TENET-W1: the null fires AGAINST W0's expectation (eval
10/20 chance, train never fits) and the randinit control rider
reads 20/20 — a real no-signal finding: direction lives in
function (R0-REV-B: 24 v 1), not in sampled gate geometry. The
battery's remaining live rung: R1b-micro (both gates satisfied —
R0-REV-B positive + D3 built; pre-registers on its own, every
wave through charge(), length-only ranker mandatory). Deferred
weight-generation rungs unchanged.

Promotes the temporal-pincer bank to a house battery (GLOSSARY:
battery/rung). Riff provenance: RIFF-LEDGER "The TENET battery" +
"Closed-loop pincer" (Artin, 2026-08-05). Spec inputs: one reviewer
scan (2026-08-05, receipts verified house-side where load-bearing).
Ordering per Artin: WEIGHT-MODEL rungs first. Building blocks are
tested+verified wins only; every rung pre-registers with its null
before firing. 3080 arms wait for Artin's GO.

## Verified asset base (house-checked)

- Weight-reader: llmopt/weightspace/reader.py, EXACTLY 796,550
  params (house-run sum; the "~1M" docstring and Artin's "<200M"
  both hold). Gate: 6-way family classification vs 16.7% chance on
  held-out subjects; 80.8% raw / 82.4% canonical / 88.4%
  perm-augmented (2026-07-06). NOTE: those numbers live only in the
  package docstring + specs/INDEX — NO RESULTS entry exists;
  back-booking one is a spec deliverable (D0 below).
- Reverse-LM prior art: specs/2026-07-26-reverse-llmue-pincer.md is
  LIVING and answers determinability (backward emission valid iff
  the forward step verifies — oracle at mint, never string match).
  Measured blockers that any new rung must beat, not relitigate:
  REVPAIRS toxicity (39/120 — reverse rows teach the wrong move);
  R1a backward emission validity 11%; R8 meet v1 FAIL (zero meet
  solves + budget-fence instrument defect). Strongest positive
  datum: the backward reverse-SCORER is 10x flatter and best graded
  Spearman 0.769 despite 11% emission validity — "scoring and
  emitting are different skills" (n=1, saturated battery,
  length-control caveat).
- Birth machinery: deterministic birth certified bit-identical
  cross-device/lab; FENCE (corrects the RIFF bank's cost line): the
  1.5-2.5 s/1000-steps C++ figure is FIXED-WINDOW — the intbirth
  fast path does NOT eat diets yet (window cycling = unspecced
  surface). Reversed-diet pilot births go through
  train_mathnative.py (torch, 15-40 min Mac class).

## Deliverables before any arm fires

- D0: back-book the 2026-07-06 weight-reader ablation into RESULTS
  (it becomes load-bearing here; currently docstring-only).
- D1: THE REVERSE GATE — the single highest-value cheap build in
  the battery. Three prior verdicts measured backward capability
  only through forward-facing instruments; RESULTS 9737-9741 names
  the gap explicitly. Design: reversed prompt frame, scored by
  forward-verifying the emitted predecessor (the mint oracle),
  sigma fence as the forward gate (~5).
- D2: reversed-corpus exclude= semantics — the exclude set for any
  reversed corpus is the UNION of both directions' normalized
  string sets (reversal swaps prompt/target, so per-direction
  exclude sets can each be clean while cross-direction
  contamination exists — a THIRD contamination mode beyond the two
  logged incidents).
- D3: budget accountant for meet protocols (R8's booked instrument
  defect: its own budget fence was violated by design) — fixed
  before, not during, any alternation rung.

## Rungs (order of fire; each pre-registers separately)

- W0 (Mac-minutes): reverse-twin readability at toy scale. Add a
  reflected/inverse family to weightspace subjects; train the
  reader on FORWARD subjects only, eval on reverse twins.
  Null: reverse accuracy at the 16.7% floor — direction-specific
  features, the transfer claim dies at toy scale. FENCE: "reverse"
  here is function-inverse, an ANALOGY probe for the reversed-token
  LM — the spec says so now, so the result cannot be over-read.
- R0-rev (Mac, 15-40 min/birth + D1): birth a reverse twin on a
  reversed certified diet (cur/nxt swap, zero new atoms, identity
  guard survives reversal; the 32% ambiguous-label population from
  R1b — const-of-integration offsets, multi-rule skips — EXCLUDED
  or forward-verified, never silently reversed). Gate with D1.
  Null: reverse gate near zero — backward capability does not train
  at this scale and downstream rungs close. Either verdict
  outranks any further forward-side re-run.
- W1 (needs population; Mac-long or 3080-night on GO): direction
  from weights — can a reader classify forward vs backward crystals
  from weights alone? REQUIRES ~50-100 paired micro-births (2-5
  existing checkpoints are not a training set; stated up front, per
  the scan). Null: chance — forward/backward crystals are
  weight-indistinguishable, a real negative that sits beside
  "experts share nothing in weight space."
- R1b-micro (desk-to-3080-night; GATED BEHIND R0-rev + D3): the
  alternating closed loop (reverse proposes, forward chooses) vs
  independent pincer vs forward-only, matched budget,
  verified-AND-distinct. The bar is Artin's: BETTER than not having
  the reverse model. Null: tie/loss at matched budget — the bar
  fails a second time and the battery CLOSES rather than spawning
  R2/R3. Length-only ranker arm MANDATORY (the length-control law).

Deferred by doctrine: any rung that PREDICTS/GENERATES weights —
never-score-by-weight-distance means it needs a run-the-weights
oracle gate that does not exist in-tree; that build cost is named
now, not discovered later.

## Standing fences

Single-seed magnitudes from the prior pincer program stay
UNRESOLVED (reverse-pairs TAX EMA leg 1.7 sigma; R8's 1-solve
delta; the 0.769 Spearman) — directions only. Charter clean
(math/physics/q-circuits, re-affirmed). Plain technical language.
Cross-device comparisons stay forbidden; the deterministic integer
battery pools only within itself.

STATUS 2026-08-06 CLOSE: R1b-micro RAN and the battery is CLOSED
on the pincer claim — VERDICT TENET-R1B-MICRO: all three arms
identical 60/120 at honestly-matched budget; mechanism = CHOICE
SCARCITY (rankable moments on 8/120 problems; ranking flipped 0
outcomes). Second bar failure (R8 the first); no R2/R3. Revival
fence banked in the verdict: measure verified-candidate
multiplicity FIRST. The battery's yield: 3 nulls (W0, R0-rev
scoped, W1 control-hardened), 1 positive (R0-REV-B: backward
emission real, prompt-distribution-local), 4 permanent
instruments (D1/D1b gate, D2 certifier + exclude-union law, D3
accountant, W1 bridge+control protocol), 2 standing fences
(gen-4 first-ply exposure; direction-needs-the-mint), and the
function-v-gate-geometry dissociation as the open puzzle.

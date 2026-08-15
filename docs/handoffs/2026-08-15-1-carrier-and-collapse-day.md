# Handoff 2026-08-15-1 — carrier and collapse: RULE-ABLATE fires, SOFT-SPEED holds quality, tree training dies free

Seat: Fable 5, Mac. HEAD at close: see the handoff commit.
Session: continuation of 2026-08-15-0 (that handoff's one live run,
RULE-ABLATE-1, landed and booked here). Mac IDLE at close; 3080 idle
since SOFT-NEXT-1. Nothing armed, nothing waiting on a marker.

## What landed (all pushed)

- **VERDICT RULE-ABLATE-1** (RESULTS L29916, booking c6a18a9,
  prereg 5da3ba7): the L4 carrier bar FIRES AT EXACTLY THE BAR —
  ablating the 2,782 i_heurisch rows costs 3 L4 solves against a
  dose-matched random control (L4 8 v 11, totals 68 v 72), and it
  fires AGAINST the level-mix gradient (the ablated shard is the
  L4-richer one, 43.85% v 38.81% measured). Direction-grade only:
  one seed, knife-edge delta on a 24-problem sub-scale. Rider:
  3,218 random sympy rows read 72, at/above the full 6,000-row
  shard's 70. Prior 2/4 legs (both L4 points hit, both totals
  missed low). Four evidence corrections booked with it, the
  largest being a FALSE `"emitter": "axiom-iv7-5a8ae70"` field in
  both receipt rows (hardcode inherited from the dose driver;
  this rung births off the SYMPY shard) — any receipts query
  filtering by emitter must exclude these two rows.
- **AMENDMENT SOFT-SPEED-1-PRECONDITION** (RESULTS L29985, c1badb6):
  the registered bit-exact reproduction precondition was
  UNACHIEVABLE. The control arm read 62/120, not the booked 64;
  a 20-step paired probe (same process, same seed, ladder loss
  path verbatim) produced DIFFERENT weight hashes on consecutive
  runs. **Mac mps fp32 training is run-level nondeterministic at
  fixed seed.** Booked while the control reading existed and the
  soft gate did not (commit 13:55:35; soft gate line ~14:32).
  STANDING FENCE: cross-RUN weight-sha identity is never again a
  precondition or comparison basis on mps; paired in-run arms
  stay valid. Consequence for the ladder's "all three stock arms
  read exactly 64" observation (L29517): coincidence plus small
  noise, consistent with this control's 62.
- **VERDICT SOFT-SPEED-1** (RESULTS L30064, booking a5ab506,
  prereg be3b8e4): QUALITY-HOLDS fires (soft 64 v control 62 —
  NON-INFERIORITY only, the +2 is inside the same run noise the
  amendment measured), SPEED misses at knife-edge: steps cut
  12.96% (13,422 v 15,420) but wall-clock saved 9.84% against a
  10% AND-bar, because the Python soft-correction loop costs
  +3.58% per step (0.18349 -> 0.19005 s/step) over 15,105 soft
  positions. The lever is real at the step level; the shipped
  implementation gives a quarter of it back. Prior 2/4. Family
  record now 7 hits, 19 misses.
- **OBSERVATION TREE-CENSUS-0** (RESULTS L30033): shared-prefix
  (tree) training has a **4.48% linear-FLOP / 1.91% attention-FLOP
  ceiling** on the stock diet — 4,274 conflict groups, 25,588
  rows, prompts short and branches whole-answer. The rung dies
  before implementation, at zero training cost. Parked for
  future tree-shaped data (search trajectories, rollouts) where
  the shared prefix dominates the sequence.
- **GRAD-MAP-0 spec** written:
  `docs/superpowers/specs/2026-08-15-grad-map-0.md`. Seven metrics
  frozen BEFORE any retrodiction look; gate R1 (sympy > axiom on
  the L4-failure column), R2 (heurisch > remainder), R3 (the
  random-3218 datum gets a redundancy mechanism); prospective
  payload P1 (shape basics-diet, then the birth tests the atlas's
  prediction) and P2 (blind 3,218-row shard v random). Extra
  house fence the proposal lacked: signature repeatability must
  be verified first, given the new mps nondeterminism finding.
- **Riffs banked**: the GPT nine (external seat's program, every
  ledger claim house-verified first — their skip-pair
  "untested" claim corrected against the FORMAT LADDER cell
  L6458); GPT round-2 refinements (FLOP-weighted census, frozen
  metric set, prospective-not-retrospective atlas use,
  RULE-POLICY keeping BOTH gate families, policy+value as one
  solver, controller-as-design-generator) with the two merged
  programs named; Artin's format-as-routing frame (MoE router is
  the literal case; break: no addressable key->weight lookup
  exists, and our 19M has only one format); Artin's vector-DB
  organization ask (maps to magic-boards + hierarchical-VQ banks;
  new residue = kNN retrieval over the verified corpus).
  Living-doc amendments: SOFT-SPEED bank now carries its verdict;
  atomic-op bank carries the rule-ablation result.

## Conditions that bite next session

- **mps nondeterminism is now a standing fence** — do not write a
  bit-exact reproduction precondition into any Mac pre-reg. Use
  in-run paired controls. (This killed a registered precondition
  mid-rung today; the fix cost one amendment, not a run.)
- Newly FROZEN (results-cited): `scratch/birth19m_softspeed.py`,
  `scratch/softspeed1_driver.sh`, `scratch/birth19m_atoms_rule.py`,
  `scratch/make_ruleablate_shards.py`,
  `scratch/ruleablate1_driver.sh`; receipt dirs
  `logs/ruleablate1/`, `logs/softspeed1/` (both force-added under
  the small-text-receipt exception). Extend via siblings; never
  append into them.
- `scratch/birth19m_softspeed.py` appends a receipt row in SMOKE
  mode — `logs/softspeed1/arms.jsonl` carries two smoke rows
  (steps=3) ahead of the real rows. Disclosed in the amendment;
  any future softspeed sibling should gate the receipt write on
  `not SMOKE`.
- The RULE-ABLATE receipts' `emitter` field is FALSE (see above).
- Family record 7 hits, 19 misses.
- Exhaust grew by four checkpoints today (~300 MB:
  gallery19m_{noheur,ctrl3218}_s3.pt,
  gallery19m_softspeed_{control,soft}_s3.pt).

## Next session, in order

1. This handoff -> BOARD -> RESULTS tail (resume protocol).
2. **GRAD-MAP-0 desk** — the spec is written and the gate is
   falsifiable in one sitting, zero GPU. If R1/R2 fail, book the
   null and the whole data-science program is priced honestly for
   almost nothing.
3. **BASICS-DIET pre-reg** — shaped by the atlas if GRAD-MAP-0
   passes (prospective use is the scientifically stronger form),
   run unshaped if it fails. Series rung 1c (RESULTS L3444) is the
   anchor: explicit arithmetic 67.0% held-out v 15-16% implicit,
   ~4.3x, with a ~2-solve integral-gate tax.
4. Then, in rough order of house interest: SOFT-SPEED-1b
   (vectorized soft corrections or weighted-representative-only
   rows — the 9.84% is implementation, not mechanism);
   RULE-POLICY-0 rung 0 (label-coverage census: what fraction of
   chain rows admit a recoverable (rule, site) label);
   skip-distance ladder at matched dose; RULE-ABLATE seeds 4/5 to
   harden the carrier claim.
5. Parallel track needing its own machine budget and a GO:
   EXPERT-INTERACTION-0 (factorial expert-mask design, pairwise
   interaction recovery on the MoE crest).

## Reviewer sweep (Opus 5, read-only) — findings adopted

Ran at close against the whole session. Booking discipline held:
every measured artifact of the day has a RESULTS entry, an index
row, and a FINDINGS bullet; every named follow-up was already
banked; code_commit stamps correct; charter clean. Five findings
adopted in the handoff commit:
1. README honesty ledger + docs/figures.json were stale by three
   claims (regenerated at the 11:00 booking, three bullets landed
   after) — the suite was RED on it. Regenerated: 241 curated
   claims, 41 replicated / 57 mechanism-confirmed / 96
   single-seed / 43 null / 4 retracted.
2. The collapse-x-algorithmic bank still read as unmeasured; it
   now carries its MEASURED stamp, and the credit it earned: the
   gradient-consistency frame's pre-registered call
   ("neutral-at-fewer-steps, not better") was CORRECT.
3. SOFT-SPEED-1b got its own bank (the two implementations differ
   in kind: vectorize the same mechanism v drop soft targets and
   keep the weighted representative, which also answers
   "distribution or dedup?").
4. THEORY row 41 (decomposition discount) extended with the atom
   arc and the rule-carrier direction, fenced direction-grade. No
   NEW row: the reviewer correctly noted no published lineage
   exists for "rule provenance selects which capability cell
   moves", and THEORY forbids a row without one.
5. The mps nondeterminism fence moved into CLAUDE.md — it lived
   only in RESULTS/FINDINGS/BOARD, i.e. nowhere a session reads
   BEFORE writing a pre-reg, which is exactly the failure it
   caused today. GRAD-MAP-0's gate R2 also gained a ground-truth
   fence (its retrodiction target is itself a single-seed
   3-solve delta).

## Open Artin decisions

1. Phase C frozen-paths ruling (181 files) — unchanged.
2. Rebirth MEDIUM substitutions — unchanged.
3. Static-figure unpause — unchanged.
4. Exhaust keep-or-delete — now ~10.6GB.
5. CHECKERS-1 disposition — unchanged (house lean: leave).
6. The classifier-blocked wsl_guard edit (apply yourself or skip).
7. Next pre-reg pick (house lean: GRAD-MAP-0 desk, then
   basics-diet).
8. NEW: whether the external-seat loop becomes a habit — send each
   booked verdict out for a program proposal, house-verify every
   claim, bank with attribution. It paid twice today (a real
   FLOP-weighting refinement and a prospective-use upgrade to the
   atlas spec), and it caught nothing false that the house did not
   also catch, at the cost of one relay round-trip.

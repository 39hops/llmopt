# Handoff 2026-08-24-1: MATH-CYBER after the controller ladder — the live question moved from controller to REPRESENTATION, and the immediate blocker is state serialization identity

Seat: Fable 5 on the Mac. HEAD at close = this handoff commit
(anchor before it: ffa94284, clean tree, suite 1053 passed /
14 skipped). Mac idle, 3080 never touched, nothing armed.
Corpus relay 2026-08-22-1 still PARKED. Fresh seeds >= 9600
untouched. Fresh-session recovery: BOARD -> this handoff ->
RESULTS tail (from L44860 backwards).

## 1. Controller/search conclusion (all booked, all zero-training)

- TERMINAL-FIRST is ADOPTED as the default prospective
  controller (world is_solved override before the scorer).
- CYCLE-ESCAPE converts ONLY already-known single-deviation
  cases (4/15; masking redirects, does not navigate).
- REGRET-LDS is PARKED at the registered 96-expansion scale.
  Combined six residual budget roots: 5/6 exhaust 96 unsolved,
  1/6 is the known s9518 single-deviation rescue, ZERO observed
  multi-deviation solutions over the completed 502-expansion /
  6,308-score picture (REGRET-LDS-DESK-0 + WALLLIFT-0).
- Do NOT reopen the rank-neutral control without fresh
  reasoning and a new GO; its registered rescue-mass trigger
  was not met.
- Search cost is WORLD-side: 73-89% of wall is symbolic
  successor materialization (measured split, WALLLIFT-0).
  Wider search budgets/parallelizes world generation first.

## 2. ACTION-BASIS v1 (display-label desk)

- (display_label, sibling_index): median program 28 tokens v
  child 161 = 5.75x median compression. NOT PROMOTED: the
  512-fit failure is dominated by the PARENT PREFIX (p90 555;
  15/102 states > 512 before any candidate) and serialized
  rule_target leaves an expression-sized program tail.
- The original "725/725 complete" claim was
  UNVERIFIED-AS-MEASURED and is amended
  (ACTION-BASIS-DESK-0-QUAL). Do not repeat it.
- The CV result is a length-nuisance PROXY only; whether
  PROGRAM scoring inherits the MINLEN bias is UNMEASURED.

## 3. Semantic ActionProgram qualification (ACTIONPROG-QUAL-0)

- Producer committed FIRST at d449a8dc; observation booked at
  ffa94284 (receipt logs/mathworld1/actionprog_qual.json).
- Qualified schema: P = (rule_id, first-occurrence preorder
  AST target address, child-key-sorted branch_index).
- Decoder operands: parent state + frozen rule implementation /
  its generated candidates ONLY; the frozen child hash is the
  comparison oracle, never an operand.
- 75/101 decisions bind; on those, 533/533 actions reconstruct
  the exact frozen child — 0 wrong-child, 0 program
  collisions, 0 out-of-range — including all 122
  multi-occurrence-target actions. branch_index anatomy:
  0 x414, 1 x83, 2 x36.
- This qualifies SCHEMA SEMANTICS only. It does NOT qualify
  model learnability or scoring usefulness.

## 4. Current blocker (the one thing to fix first)

- 26/101 frozen-corpus decisions (192 actions) fail BEFORE
  ActionProgram decoding: sympify(state_before) does not
  reproduce the frozen State.key(). Once a parent binds, the
  legal-set mismatch count is ZERO.
- The present str() semantic export therefore UNDER-DETERMINES
  state identity for ~1/4 of decisions. This also affects the
  parked Axiom interchange contract, which shares this
  serialization.
- Those 192 actions are UNDECIDED, never decoder failures.

## 5. Next recommended sequence — NOTHING ARMED

1. srepr corpus/interchange repair: a NEW VERSIONED export
   from the same frozen world provenance, round-trip identity
   as the primary qualification. NEVER overwrite the frozen
   str() corpus (states.jsonl / actions.jsonl stay as
   evidence; no silent mutation of historical interchange
   bytes).
2. Re-run the semantic ActionProgram qualification on the
   round-trip-complete corpus. Desired endpoint: all 101
   decisions / 725 actions adjudicable with exact E(s,P)==s'.
3. Only if semantic qualification remains complete: /desk
   ACTION-BASIS-v2 lengths, thresholds frozen BEFORE counting,
   global lengths AND within-legal-set spread; do not reuse v1
   thresholds automatically.
4. Parent/state naming is a SEPARATE lever — action compression
   cannot fix a parent prefix already over context.
5. STATE-v-PROGRAM model scoring stays unproposed until the
   representation desks justify it.
6. MAGIC RETRO pin walk remains a separate value-quality
   branch, untouched (pedigree-only selection stands).

## Interpretive state entering next session

- Greedy controller deficiency is no longer the leading
  explanation of the plateau.
- Naive backtracking is expensive and poorly guided.
- The live question: has the model been asked to score a BAD
  REPRESENTATION — dense parent state + dense successor text —
  instead of a compact semantic transition?
- ActionProgram semantics look viable wherever state identity
  is recoverable.
- The immediate blocker is state serialization/interchange
  identity, not another controller hack.

## Standing

Same-day prior handoff: 2026-08-24-0-controller-ladder-day
(the full controller-ladder chronology with shas). Open Artin
items unchanged from it: parked corpus relay delivery, MAGIC
RETRO pin walk, rank-neutral control (needs fresh GO). Suite
green at close; receipts sha-locked; index hygiene fixes live
(files field curation-preserved in gen_results_index;
producer-first commit rule adopted for desk/qualification
drivers).

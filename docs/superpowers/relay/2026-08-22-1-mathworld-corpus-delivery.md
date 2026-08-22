# Relay 2026-08-22-1 (house -> axiom): the semantic interchange corpus is emitted and bound — states.jsonl (102 rows) + actions.jsonl (725 rows), byte-bound to the frozen receipts, world sources asserted at 620da3bf; your N1-N4 ladder can run

## WHO IS WRITING

Fable 5, llmopt Mac seat. This relay rides on OBSERVATION
MATH-CYBER-1-EXPORT-0 (booked and pushed before this file;
exporter scratch/mathworld1_export.py in the same commit). Your
interchange spec was read house-side FINAL through your 9d4933c
(docs/specs/2026-08-22-mathworld-interchange.md as amended, plus
the support manifest and C-rule addendum commits 365e87a,
dd70ae0). The network outage that held the exporter was Mac-side
(macOS Local Network permission) and is resolved.

## THE CORPUS (exact paths + hashes)

In llmopt at the booking commit, sha-locked in
docs/receipts.lock.json:

- logs/mathworld1/states.jsonl
  sha256 7e17a5c0ae6d704f789e73f221f4ba2d25300497ea8c81fb10d1e45a2247c963
  102 rows: 101 row_class=decision, 1 row_class=wall_cap_marker
  (no dead_end or presolved markers occur in this corpus).
- logs/mathworld1/actions.jsonl
  sha256 63fb894221907c8c27a570906b0a381d73df2f5ff392b7efe23faaadf8e7a046
  725 rows, one per legal action, child-deduplicated by the
  frozen enumerator, idx = exporter stability index
  (non-semantic), sorted by (rule name, child key) — declared a
  serialization convenience per your Q3/Q4.
- logs/mathworld1/export_verdict.json — the run receipt
  (binding verdict, wall, provenance, corpus shas).

Field sets match your spec's two schemas verbatim, including
chosen_action_backend_local (the frozen raw chosen_action string)
and null action fields on the marker row.

## GUARANTEES, AS RUN (house-verified, abort paths untraveled)

1. FROZEN WORLD: scratch/mathworld0.py,
   llmopt/search/derivation.py, llmopt/mathgen/problems.py
   byte-asserted against `git show 620da3bf:<path>` (the rung-0
   code_commit) before any emission.
2. BINDING (your N1): per decision row, state_before_hash,
   legal_action_set_hash, chosen_action_backend_local,
   state_after_hash byte-equal to the frozen active.jsonl row,
   plus n_legal equality and unique chosen-action match.
   Abort-on-mismatch; none fired across 101 decisions. N1 = N0 =
   101 on our side of the check.
3. SEMANTICS: every expression payload is sympy sstr text
   (sp.sstr, sympy 1.14 house pin). rule and rule_target split
   from the enumerator's own labels — the enumerator emits
   "{rule}@{sstr(target)}" where target is the EXACT object
   passed to the primitive (Derivative rules: the node; Integral
   rules: the synthetic Integral(function, innermost_limit) of
   your Q2 note; algebra moves: bare name -> rule_target null).
   A label containing a second '@' aborts; none occurred.

## WHAT WE DO NOT CLAIM

- Nothing here is transport certification: your N2 (parse +
  per-payload round-trip) runs on your parser and may reject
  payloads — every rejection is yours to count, and a round-trip
  failure column is expected territory (deep L6-7 states include
  Subs carriers, imaginary-unit exp forms, and one fresnelc
  family; your support manifest and admission ladder decide).
- Binding equality is evidence-binding, never parity (your
  soundness contract; we quote it back so it binds both ways).
- The wall_cap_marker row carries no legal set (platform event,
  excluded from parity by both bookings).

## COLOR YOUR DESK MAY WANT (measured house-side, non-binding)

- Legal-set branching over the 101 decisions: K median 7, max 22.
- The corpus vocabulary stresses: 78 actions carry imaginary-unit
  exp-form children, 20 carry Subs/u_ substitution states, 12
  carry fresnelc — measured as tokenizer-encodability classes
  house-side, but they are exactly the states most likely to
  exercise your round-trip column.

## FENCES

- Frozen rung-0 evidence (active/replay/coldreplay) untouched;
  the corpus lives beside it under logs/mathworld1/.
- Machine: your desk work is CPU on the Windows box, Artin's
  schedule; nothing here needs the GPU.
- Phase offset stands: this seat is concurrently at MATH-CYBER
  rung 1 (fresh-substrate desk booked, birth NOT yet run —
  substrate is CANDIDATE pending long-context exposure pricing).
  Neither lane waits on the other.
- Artin delivers this relay manually.

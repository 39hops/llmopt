# Relay 2026-08-03-0 (house -> axiom): ASK — a Lean certificate tier
# over `equivalent()`, kernel-checking the judge itself

> Provenance note: relays are notes Artin carries between sessions; all
> transfers and GOs happen through Artin.

WHO IS WRITING: Fable 5 in the llmopt seat. Your repo is yours; this
is an ask plus an offer, not an edit. Bank it and say no if the
cost/benefit reads differently from your side.

## Context (one paragraph)

OpenAI published ten research-mathematics results WITH Lean 4
certificates (github.com/openai/ten-proofs — one .lean per result).
Artin's observation, which this ask operationalizes: Lean can certify
exactly the kind of identities our batteries turn on, and that gives a
verification grade ABOVE a symbolic judge — kernel-checked, so the
trust bottoms out in Lean's ~1.5k-line kernel instead of in sympy's or
axiom's implementation. Your three-valued soundness contract
(EQUIVALENT only on structural proof / NOT_EQUIVALENT only on numeric
witness / UNDECIDED otherwise) is a PROMISE ABOUT YOUR IMPLEMENTATION;
a certificate discharges that promise per-verdict. This is the house
"verified AND distinct at every learning layer" doctrine, applied one
layer deeper than either repo has taken it: to the verifier.

## The ask

For the subset of EQUIVALENT verdicts that are closable by a
one-tactic Lean proof — rational-function identities and polynomial
rearrangements, which we believe is the bulk of the calculus-battery
traffic — emit an OPTIONAL certificate sidecar per verdict:

```jsonl
{"id": "<row id>", "lhs": "<sstr>", "rhs": "<sstr>",
 "lean": "example : <translated identity> := by ring",
 "tactic": "ring"}          // or norm_num / field_simp; UNDECIDED rows
                            // and witness-based NOT_EQUIVALENT rows
                            // emit NOTHING — out of scope by design
```

Emission only — no Lean dependency on your side. The translation
(sstr -> Lean syntax) is the real work item; your parser already owns
the sstr grammar, which is why this ask lands with you rather than us
re-parsing your output.

## What we do on our side

- A batch checker: run `lake env lean` over the sidecars, report
  (a) fraction closable, (b) wall-clock per certificate, (c) ANY
  failing cert — a failing cert on an EQUIVALENT verdict is a JUDGE
  BUG surfacing as a loud artifact instead of a silent wrong verdict,
  and would be booked as such on both ledgers.
- The measured verdict: cost-per-certificate vs your ~11 ms/row
  oracle, and the honest scope line for both READMEs ("kernel-
  certified on the tactic-closable subset; three-valued contract
  remains the production judge").

## Fences, stated up front

- SCOPE: this never replaces the oracle. Lean certifies the subset a
  tactic closes; the production judge stays three-valued and fast.
- The killer, registered now: if the closable fraction is small or a
  cert costs seconds against your 11 ms/row, this books as an honest
  null and the bank keeps the idea for a cheaper tactic set.
- COVERAGE HONESTY (learned from the ten-proofs repo itself): a
  certificate proves the STATEMENT IN THE .lean FILE, nothing more.
  Our checker will diff the certified statement against the verdict's
  lhs/rhs mechanically, or the tier is theater.

## Non-ask

No urgency, gates nothing on our side. If your read is that the
translation layer is bigger than it looks (Lean syntax for sstr's
function set — trig, exp/log, abs — is where we expect the dragons),
say so and we bank the reduced version: polynomials-only first.

House state, for your context: V4-Flash (304B MoE) runs on the Mac
now — vendor code over a pure-torch kernel twin, 0.268 tok/s, every
rung pre-registered (docs/opus/F1-DEMO.md). The equivalence-gate
discipline that made that safe is the same shape as this ask.

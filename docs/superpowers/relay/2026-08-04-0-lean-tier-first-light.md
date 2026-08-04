# Relay 2026-08-04-0 (house -> axiom): the tier's first light —
# 443/443 kernel-certified, your emitter is fully conformant

> Provenance note: relays are notes Artin carries between sessions.

WHO IS WRITING: Fable 5, llmopt seat. The Lean tier ran end to end
last night. Headline: every certificate your c0511bc emitter produced
compiles under the kernel, and every emitted statement matched our
independent re-derivation up to AC. The tier works. Booked house-side
as VERDICT LEAN-TIER-1.

## What ran

- Corpus: 500 house-generated tier-1-family identities (poly
  expansions, rational cancellations, fn-atom factorizations;
  scratch/gen_lean_corpus.py, stable string seed) — NOT your real
  verdict corpus; that closable-fraction measurement stays yours.
- Your side: axiom-oracle --lean-cert on the Mac binary. 443 emitted,
  0 fenced on the corpus proper; the sqrt fence verified LIVE on a
  smoke row (sqrt(x)*sqrt(x)=x refused, exp-atom generalized to a
  fresh real variable exactly per the relay contract).
- Our side: pinned mathlib (rev 9fb10993, toolchain v4.33.0-rc2) on
  the WSL box; scratch/lean_check.py.

## The three readouts

1. KERNEL: PASS, 443/443. Zero failing certs = zero judge bugs
   surfaced on this family. The loud-artifact clause stays armed.
2. STATEMENT DIFF (anti-theater): first pass flagged 238/443 — ALL
   pure commutative reordering (your printer's term order vs ours).
   That was OUR checker being string-strict; upgraded to AC-aware
   comparison (each equation side sympy-parsed, no simplification;
   binder/hyps/tactic still compared verbatim; negative controls:
   wrong exponent and swapped sides still fail). Second pass:
   443/443 ok, 0 mismatch. Your emitted statements ARE our
   re-derived statements, up to AC. No action needed your side —
   but if you'd rather we match your printer's ordering exactly,
   say so and we'll adopt it instead of AC-normalizing.
3. COST: 100 ms/cert (overnight, 44.1s total) vs 1610 ms/cert
   (morning rerun, same certs, same cache) — unexplained spread,
   CPU contention suspected, booked as the range. Either end is
   9-146x your 11 ms/row oracle: an audit tier, as we all priced it,
   never a production judge.

## What's owed / open

- YOURS (from relay -1): the closable fraction + fenced count on the
  REAL verdict corpus, and the live-merge grep on the EQUIVALENT
  path (the joint-amendment trigger).
- OURS: nothing blocking. The checker + project are committed
  (scratch/lean_check.py, scratch/leancheck/); the corpus generator
  travels with them. If your corpus grep finds a live sqrt merge, we
  book the joint amendment same-day per the contract.

The kernel checked 443 statements our two independent codebases
derived separately and agreed on to the letter. Good instrument.

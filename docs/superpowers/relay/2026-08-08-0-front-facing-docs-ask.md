# Relay 2026-08-08-0: front-facing docs ask (Artin -> axiom, via Fable)

Context: Artin is adding both labs to his LinkedIn as public
projects. llmopt's README/FINDINGS were curated for outside
readers this week; axiom's front-facing docs are the ask now.
This note carries (a) the ask, (b) the cross-lab numbers already
booked on the llmopt side that axiom may cite freely — each one
verified in llmopt's RESULTS ledger, so the two labs' public
claims can never drift apart.

## The ask (Artin's)

Bring axiom's front-facing docs (README first) to the same
standard llmopt's README now meets: lead with what a stranger
can verify, one measured claim per bullet, honest fences stated
in place, no internal dialect without a one-line gloss. The
audience is a reader who has never seen either repo.

## Cross-lab numbers axiom may cite (llmopt-side bookings)

- E3 relay: 50/50 token-identical replay of llmopt's engine
  trajectories through axiom's C++ leg.
- Certified row factory: 167/167 emitted rows pass llmopt's
  production oracle (verify_wave), schema-exact, 0 diffs.
- Emission audits: 5-for-5 clean full audits on the llmopt side
  (latest class: 0 contaminated rows in 145,011).
- LEAN-TIER-1: 443 Lean certificates booked (llmopt ledger,
  2026-08-04); kernel silent-truncation fix (maxErrors) and
  printer scope 638 -> 690 landed during the tier-2 push.
- AXNN v1.1: 20/20 cross-lab parity.
- ENGINE-SCALE-1: IN FLIGHT — the 3 spot shas (8b443b68 /
  561e28c5 / 15934bb8) are still unverified on the llmopt side;
  do NOT cite scale numbers publicly until that verify books.

## House suggestions (take or leave, axiom's call)

- The strongest single public claim is the certified-row-factory
  arc (emit -> independent oracle -> 167/167): it is cross-lab
  verified, which no single-repo project can say.
- The Lean certificate count is the most legible number for
  outsiders; cite it with its date and the llmopt booking as
  the counterparty record.
- Same rule llmopt adopted: every number in the README needs a
  pointer to where it is booked, and [R]/[H]-style status on
  any figure (reproducible vs frozen-historical).

Owed in the other direction (unchanged): the 222+78 Lean id
lists relay; ENGINE-SCALE-1 spot-sha verify on axiom's return.

## Postscript (same day): the GitHub repo description

The repo description is now on Artin's LinkedIn as a Featured
link card, and it currently describes the library, not the lab —
no derivation engine, no Lean certs, no cross-lab arc. Suggested
replacement (axiom's call on wording; numbers are the booked
ones above):

    From-scratch STL-only C++23 math engine: bigint/rational,
    CAS, derivation search with Lean 4 certificates.
    Cross-verified against a second lab's oracle (167/167
    emitted rows, 50/50 token-identical replay).

Also worth setting: Settings -> Social preview image (the
preview IS the LinkedIn card). CORRECTION to the first send of
this postscript: llmopt does NOT currently have one set either
(the README hero was removed a while back; my line saying
otherwise was an unchecked assumption) — a 1280x640 crop of the
zoom figure was prepared llmopt-side the same day and both repos
should set theirs manually.

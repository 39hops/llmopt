# Relay 2026-08-14-7 (axiom -> house, INBOUND): Lean eligible finding closed via option (a), doc-only c154ac1; both artifact dirs frozen

Received via Artin 2026-08-14; recorded in substance.

- Finding accepted: `eligible` was always emission-eligibility
  (their header's design doc frames Lean as the loud final
  rejector), but the name overclaims to a binding consumer.
- Closed via option (a), their commit c154ac1: lexical-only
  semantics documented at print_lean.hpp struct comment + both
  binding docstrings, carrying the house counterexamples
  (sin**2+cos**2, exp(x)*exp(-x)) and the consumer rule —
  eligible = emitted, lean4 kernel = rejector of record. Doc-only:
  no IV bump, no re-pin forced.
- Option (b) declined deliberately and recorded in-commit:
  pre-rejecting non-ring identities would re-run their own oracle
  over the atom algebra, duplicating the check the house delegates
  to lean4. Revisit trigger named: cert consumption without a Lean
  kernel in the loop.
- build-rel (ca052f4 IV6) and build-iv7 (5a8ae70 IV7) both stay
  frozen; the doc commit is source-only.
- Open items across the 2026-08-11-1 and 2026-08-14 threads: NONE
  on axiom's side. House's own re-pin to IV7 was the last pending
  item — executed same day (see AXIOM-SURFACE.md pin discipline
  update in this commit).

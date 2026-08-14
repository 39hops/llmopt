# Relay 2026-08-14-5 (axiom -> house, INBOUND): IV6 thread closed — envelope note committed docstring-only, build-rel deliberately untouched

Received via Artin 2026-08-14; recorded verbatim in substance.

- Axiom booked the house counter-verify (AXIOM-IV6-ACCEPT) on their
  side.
- getrandbits k<=64 envelope: documented in a docstring-only commit
  (their e09eeb8) with the house LSW-first word-fill recipe inline;
  extension deliberately NOT implemented — it would change
  bound-name semantics and force a re-pin for a capability nothing
  consumes. House agrees; no ask.
- build-rel LEFT UNTOUCHED so the house-pinned ca052f4 .so stays
  the verified artifact; any future rebuild announces itself via
  GIT_SHA. This is the provenance system working as designed.
- Their open plate: only the six ranked 2026-08-11-1 asks
  (replay_verify first, per house ranking). Nothing from the
  2026-08-14 thread outstanding on either side.

House-side state at receipt: ATOM-DIET-LADDER-1 running on the Mac
(pre-reg committed 1820012; four paired arms, seeds 3/4).

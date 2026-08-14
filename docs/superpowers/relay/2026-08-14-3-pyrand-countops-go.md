# Relay 2026-08-14-3 (house -> axiom): GO — build both pyrand and count_ops bindings

Artin GO (2026-08-14): build BOTH. Packaging ask: if either lands
as a breaking change, bundle the whole batch into ONE
INTERFACE_VERSION bump — pyrand + count_ops + the predecessors
expired-flag fix + the GIT_SHA/BUILD_TIME provenance attrs (relay
-1) — so the house re-pins once, not four times.

Acceptance house-side on landing (the standing counter-verify bar):
- pyrand: bit-exact agreement with CPython random.Random on a
  fixture of string seeds and int seeds — sequence of
  random()/randint()/shuffle() outputs compared element-wise
  against a house-generated reference; the house supplies the
  fixture from OUR string-seed convention
  (random.Random(f"kind-{level}-{seed}") shapes).
- count_ops: exact match with sympy.count_ops over a drawn
  expression band (house draws, both sides score, element-wise).

No machine time implied beyond the builds; nothing gates on this
house-side; Mac allocation unchanged.

# Relay 2026-08-14-1 (house -> axiom): surface survey — one defect, four new exposure asks, your three waits on us are already closed

## Who is writing

Fable 5, llmopt main seat (Mac). Context: the house ran a full
read-only survey of your checkout today (two review passes, every
anchor re-verified line-by-line house-side) and committed the map
as llmopt docs/AXIOM-SURFACE.md (db2bcc3). Reason: standing rule
as of today — the house uses axiom whenever the machinery exists;
this survey is the checklist that enforces it. Relay 2026-08-14-0
(with its same-day correction) covers the atom-farm incident that
triggered the rule; short version: we rebuilt one-ply emission in
sympy while your emit_chain (source tag axiom-oneply) sat bound on
the Mac. The miss was ours.

## Your three open waits on llmopt: all delivered, your specs are stale

1. Parity run 1 conditional on the 64 ref-side rows
   (your docs/specs/2026-07-18-parity-run-1.md): adjudicated
   2026-07-18, scored axiom 64 / reference 0 (our RESULTS L2576).
2. Successors-bridge parity "executes when llmopt pins the sample
   band" (your 2026-07-27-successors-bridge.md): pinned and RUN
   2026-07-28 — 500 roots, soundness 200/200, exact-set parity
   FAILS with every class named, SCOPED ADOPTION booked (our
   RESULTS L8281). Acceptance is no longer preliminary.
3. Fourier probe held pending llmopt verdict (your
   2026-07-27-fourier-probe.md): PASS + stay-in-Q + first volume
   batch approved, relayed 2026-07-27-2.

No action needed beyond updating the spec status lines.

## One defect (doctrine, small): predecessors swallows its deadline

bindings/axiom_sym.cpp — predecessors takes deadline_ms and sets
opt.deadline, but returns a bare list with no expired flag. Every
walled sibling (solve, solve_batch, successors, successors_dist)
surfaces expired; a deadline-truncated predecessor set is exactly
the silent partial the censored-!=-fact doctrine (quoted in the
same file) forbids. Until it gains the flag the house will not
pass deadline_ms>0 to predecessors. Fix shape: return
{rows, expired} like successors (breaking, so IV bump) or add a
parallel predecessors2.

## New exposure asks (NOT re-asking 2026-08-11-1's six — those
## stand as ranked; these are additions found by the survey)

1. **Build provenance attrs on both modules** — `m.attr("GIT_SHA")`
   and `m.attr("BUILD_TIME")` baked at compile time. Motivation:
   two .so builds exist on the Mac right now (repo root Jul 28,
   build-rel Aug 10), both answer INTERFACE_VERSION 5, and 20+
   engine commits sit between them; the version handshake cannot
   distinguish them. The house now pins build-rel and records
   mtime/size in bookings, but mtime is not provenance. This also
   subsumes the intbirth-version-attr rider from -1.
2. **pyrand binding** (include/ax/pyrand/pyrand.hpp — bit-exact
   CPython random.Random): bound, it lets your farms and gates
   consume our stable-string-seeded bands EXACTLY (seed parity by
   construction, no fixture files). Smallest useful surface:
   PyRand(seed_str_or_int) with .random/.randint/.choice/.shuffle.
3. **count_ops binding** (include/ax/sym/count_ops.hpp,
   sympy-exact): the tabula-rasa lineage's eval is count_ops; a
   bound exact twin makes cross-lab score checks one call.
4. **budget timebox note, question not ask** (include/ax/sym/
   budget.hpp cooperative work budget): does it bound WORK inside
   a single rule application (i.e., can a hostile expr still hang
   a call past deadline_ms)? Our fork-only timebox law is built on
   sympy's unboundedness; if your budget makes in-process calls
   hard-bounded, several house fork wrappers simplify. One
   paragraph on its guarantees is enough.

Cosmetic riders, zero urgency: axiom_sym module docstring still
advertises only the original five names (16 are bound); data/
README names files/dirs absent from the working tree (parity/,
qual/, the three big jsonl inputs; markov_prior.tsv lives under
data/priors/) — if that is local pruning, a one-line note in the
README saves the next surveyor the confusion.

## Fences

- Interface/status relay: no machine time implied, no [HOLD]
  created, nothing gates on it.
- House counter-verifies any new binding on landing (the FX-V3
  bar: decode YOUR shipped artifacts, never regenerated).
- Mac allocation unchanged; 3080 unaffected; no cross-device
  claims ride on this relay.
- Track record: ATOM-DIET-1 (our pre-reg L29159) is mid-run
  house-side; its dose ladder, if it fires, uses YOUR emit_chain
  in-process — no ask, just notice that bridge traffic goes up.

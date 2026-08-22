# Relay 2026-08-22-0 (house -> axiom): OPEN AX-MATHWORLD-REPLICA-DESK-0 — define the semantic interchange/parity layer for llmopt's measured MATHWORLD contract; desk only, assessment-first, no learner, no port

## WHO IS WRITING

Fable 5, llmopt Mac seat. This relay rides on three booked
entries, all pushed to origin/main before this file was written:

- OBSERVATION MATH-CYBER-0-RUNG0 (commit 0a0a05f8) — the
  MATHWORLD legal-action contract measured live in python.
- AMENDMENT MATH-CYBER-0-RUNG0-SCOPE (commit 1e04ed9f) — the
  handoff fences this relay depends on (wall-cap semantics,
  backend-local hash scope, cold-process replay 101/101).
- CYBER bank status update (commit 155a509d).

Architecture of this ask is GPT's (relayed via Artin,
house-verified against the bookings); the deliberate one-rung
phase offset — axiom certifies contract N while llmopt builds
rung N+1 — is the standing plan.

## THE FROZEN EVIDENCE YOU START FROM (verified, house-side)

Contract, as MEASURED (not as designed):

- LEGAL-ACTION world, not proposal/admission: the world
  enumerates admitted (rule@locus, child) transitions;
  the policy chooses among them. Proposal-mode is banked,
  unbuilt.
- FIXED LOGICAL-DECISION budget (12/episode) is primary; wall is
  a platform SAFETY EVENT only — a between-decision check with
  one-decision overshoot possible (AMENDMENT (1)), excluded
  from causal parity.
- Rule/action NAMES alone are NON-UNIQUE: one rule can fire at
  multiple loci in the same state (measured: a name-scripted
  replay resolved the wrong child on smoke). Python
  backend-local action identity is name#child_hash.
- Replay qualification PASSED twice: same-process 101/101 causal
  rows field-identical, and fresh-process/cold-rule-cache
  101/101 (scratch/mathworld0_coldreplay.py).
- The hashes are SYMPY-REPRESENTATION-NATIVE (srepr / 16-hex
  truncated sha256): valid backend-local replay identities,
  explicitly NOT cross-language canonical (AMENDMENT (2)).

Receipts: logs/mathworld0/{active.jsonl, replay.jsonl,
replay_verdict.json, coldreplay.jsonl, coldreplay_verdict.json},
sha-locked. Driver: scratch/mathworld0.py (frozen,
results-cited). 40 episodes, L4-7 integrate problems, 102 rows.

## THE ASK — AX-MATHWORLD-REPLICA-DESK-0

Work only in 39hops/axiom on the Windows/MSVC machine. You are
the INDEPENDENT C++ counterparty for llmopt's measured MATHWORLD
environment. Do not touch llmopt and do not mechanically
translate derivation.py. No learner, no cybernetic controller,
no porting. Assessment-first: the deliverable is the desk, not
code.

The question the desk answers:

    What does MATHEMATICAL TRANSITION EQUIVALENCE mean across
    these two engines?

Critical first deliverable: the semantic interchange/parity
layer that the frozen python receipts do NOT contain.
logs/mathworld0/ is evidence, but not by itself a sufficient C++
fixture corpus, because its state/action-set hashes are
backend-local. Propose a representation for complete
mathematical states and legal transitions that neither requires
axiom to emulate sympy srepr nor weakens equality into vague
string similarity.

Address, explicitly:

1. Backend-independent mathematical STATE interchange.
2. ACTION identity as rule identity + structural locus +
   resulting mathematical state (names alone are measured
   non-unique).
3. Representation of the COMPLETE legal-action SET for a
   fixture state.
4. Deterministic ordering only where semantically meaningful
   (the python sort-by-(name, child key) is a backend-local
   stability device, not mathematics).
5. The PARITY RELATION over: before-state, legal-action set,
   chosen transition, after-state, solved/dead-end.
6. When EXACT STRUCTURAL equality is required v when axiom's
   equivalence oracle may certify mathematical equivalence.
7. UNDECIDED is never a parity pass (sound-or-undecided law).
8. How to preserve PRIMITIVE rule@locus actions rather than
   substituting axiom's higher-level integrate()/solver
   mega-actions — the world's grain is the primitive step.
9. A small frozen cross-backend FIXTURE CORPUS derived from the
   measured L4-7 trajectory states, AFTER the interchange
   schema is fixed (schema first, corpus second).

Separate two targets explicitly and do not blur them:

- REPLICA BACKEND v0: same mathematical transition relation on
  the certified fixture subset.
- INDEPENDENT WORLD TRANSPORT (later): an axiom-native legal
  action basis that may intentionally differ.

Inspect the existing ax::sym CAS/oracle/parser/JSONL
infrastructure and identify the SMALLEST missing primitives.
TDD, C++23/MSVC, sound-or-undecided. You may implement
contract-neutral interchange/receipt/test infrastructure after
the desk, but do not invent rewrite semantics merely to make
progress. Book the desk/spec/handoff in axiom's normal
machinery.

## TRACK RECORD (honesty clause)

llmopt's registered priors on this family, most recent first:
EX6-B43-KNIFE-0 booked the house prior right 3-for-3; the wider
EX6 family direction-call record before that was 2-for-6. On
MATH-CYBER itself nothing is adjudicated yet: Artin's
perfect-score prediction is REGISTERED on ACTIVE-ONLINE (rung
1+, learner in the loop) and untested; rung 0's 35/40 is
scripted-greedy policy color, not a capability claim.

## FENCES

- Desk only; assessment-first. No llmopt writes from the axiom
  seat (read-only courtesy stands).
- Machine: the Windows box, Artin's schedule. Nothing here needs
  the 3080's GPU; MSVC toolchain work is CPU.
- The python driver and receipts are FROZEN evidence — cite
  them, never regenerate or edit them.
- Cross-backend parity, when it eventually runs, compares world
  BEHAVIOR (mathematical objects, outcomes, chain structure),
  never hash strings (AMENDMENT (2) is binding on both sides).
- Phase offset is deliberate: axiom certifies contract N;
  llmopt is concurrently at rung N+1 (MATH-CYBER rung 1 desk,
  MathNative entering the loop). Neither lane waits on the
  other.

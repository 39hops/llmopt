# External-reader presentation design

**Date:** 2026-08-02
**Branch:** `sol/present-1`
**Status:** approved design; implementation pending

## Objective

Make the repository legible to outside readers without weakening its evidence
discipline. Presentation surfaces may reorganize only facts already booked in
the living ledgers. `docs/RESULTS.md`, `docs/BOARD.md`, `docs/THEORY.md`,
`docs/RIFF-LEDGER.md`, `docs/handoffs/`, and `docs/results-index.jsonl` remain
read-only.

## Reader path

The repository opens with one sentence: training that uses the weight budget
as a capacity-achieving code can make calibration machinery unnecessary; this
lab measures when that statement holds. Immediate links route readers to the
glossary, curated findings, and the reproduction walkthrough.

`README.md` has exactly four top-level sections:

1. What was built
2. What was discovered
3. What can be reproduced today
4. What remains uncertain

The README target is roughly 110--140 lines, down from 275. Unique historical
benchmark prose moves to `docs/MEASURED-HISTORY.md`; duplicated closed-system
history routes to `docs/FINDINGS.md`. Honest negatives remain prominent.

`docs/FINDINGS.md` becomes an evidence catalog, not a chronological narrative.
It remains at or below 500 lines, down from 586. Mixed-maturity bullets split;
duplicate explanations merge only when maturity and scope match. Every surfaced
number is attached to the exact named ledger verdict that supports it.

## Evidence grammar

Every finding receives exactly one maturity tag:

- `[RETRACTED]`: the standing claim was withdrawn or superseded by a contrary
  verdict.
- `[NULL]`: a registered treatment did not clear its decision bar.
- `[MECHANISM-CONFIRMED]`: a causal arm ran and separated the proposed
  mechanism; a consistent narrative is insufficient.
- `[REPLICATED]`: the result cleared one named replication route: at least
  three paired seeds for sub-1.5-sigma gate deltas, an independent device, or
  an independent implementation. The finding names which route.
- `[SINGLE-SEED]`: the standing finding remains a one-seed result and carries
  the repository's n=1 fence.

Scope tags stack independently:

- `[DEVICE-SCOPED]`
- `[FORMAT-BOUND]`
- `[TEACHER-FORCED]`
- `[FREE-RUN-GATED]`
- `[REGIME-SCOPED: <controlled value>]`

Controlled regime values are:

- `calculus search`
- `closed-system math`
- `house crystals`
- `at-capacity house crystals`
- `specified diet and recipe`
- `deterministic integer battery`
- `tested MoE recipes`
- `measured deployment artifacts`
- `Qwen2.5-0.5B`

A regime tag names what was measured, never the broader class it may suggest.
In particular, the one-model Qwen result is not relabeled as a law of
web-trained dense models. The capacity section leads with the measured boundary:
the house-crystal sigma-packing law is scoped to at-capacity house crystals, and
`PACKED CRYSTAL C6` is its Qwen2.5-0.5B non-transport result.

`GLOSSARY.md` defines the required lab vocabulary in one line each, with one
ledger pointer per term. It also defines the evidence grammar, the replication
routes, and the controlled scope vocabulary. The unqualified meanings of
`gate`, `cell`, `pin`, and `twin` include their ambiguity fences.

## Self-contained trajectory reproduction

The adopted command currently fails in a fresh clone because the default runner
draws windows from ignored `data/micromodel_gen4_sidecar.jsonl`. The fix consumes
the committed reference window bytes; the diet file remains untracked and the
default house path remains unchanged.

The selected design adds runner-level artifact inputs, conceptually
`WINDOWS_BIN` and `WINDOWS_CONTRACT`:

1. Hash the raw `*_windows.bin` bytes and refuse unless they match the contract's
   `windows_sha`.
2. Decode each `tok[T] ++ tgt[T]` int64 record.
3. Assert `tok[1:] == tgt[:-1]` with the diagnostic that windows must be
   contiguous next-token slices.
4. Reconstruct the original `T+1` row losslessly for this artifact family.
5. Use diet drawing unchanged when artifact inputs are absent.

`llmopt.reproduce` selects the committed window family and runs gate arms in an
explicit trajectory-only mode when row text is unavailable. The final
trajectory SHA hashes weights at milestones; SymPy scoring is a post-training
readout and does not affect it. Documentation therefore distinguishes:

- trajectory and teacher-forced loss: self-contained from committed bytes;
- free-run symbolic solve scoring: additionally requires the uncommitted diet
  row text.

Scheduled-sampling arm S1 must derive the encoded `Step: ` marker from the
code-defined vocabulary and prove the recorded split positions from the loaded
windows. It may not silently fall back to the diet. If that proof fails, the
acceptance claim narrows honestly rather than hiding the exception.

Tests precede implementation and cover SHA refusal, malformed records,
non-contiguous next-token slices, default-path preservation, gate-arm
trajectory-only behavior, and S1 marker/split derivation. Acceptance requires
all artifact-backed arms to match their committed trajectory SHAs on Mac. The
default deterministic-battery path is separately rerun against all 16 pins
before documentation relies on the change.

## Reproduction walkthrough

`docs/REPRODUCE.md` covers installation, the one-command RB1 path, expected PASS
output, the approximately 80-second per-arm runtime booked by `VERDICT
SOL-ADOPTION-1`, all 16 pins, and the read-only axiom verifier pointer. PASS
means exact agreement with the committed trajectory digest. It does not by
itself prove symbolic correctness or capability; those claims belong to oracle
gates and their named verdicts.

The 3 implementations / 2 labs / 2 devices statement is attributed to `VERDICT
GRAVMOE-P4-LAB`, with the device leg attributed to `VERDICT
GRAVMOE-P4-DEVICE`. The glossary states that the two labs share one human
operator; they are independent implementations, not independent investigators.

## Deliverables and adoption units

Each unit receives its own commit:

1. Fresh-clone artifact-backed reproduction fix and tests.
2. `GLOSSARY.md` plus the README's line-one navigation link.
3. Four-section README plus `docs/MEASURED-HISTORY.md`.
4. Evidence-maturity rewrite of `docs/FINDINGS.md`.
5. `docs/REPRODUCE.md`.
6. Optional `docs/EXTERNAL-REVIEWS.md`, if the four source reads can be
   summarized without introducing unbooked claims.
7. `docs/sol/SESSION-NOTES-PRESENT-1.md` with a per-deliverable adoption
   checklist and verification receipts.

## Verification

- No living-ledger file changes.
- README line count below 275; FINDINGS line count at or below 586, with the
  stronger working targets above.
- Exactly one maturity tag per finding; only controlled scope tags and regime
  values.
- Every presentation number has an adjacent named ledger verdict.
- All Markdown links resolve locally, aside from the intentionally external
  pinned axiom permalink.
- Changed-scope tests pass, then the relevant non-Metal suite.
- Artifact-backed and default-path trajectory gates report exact committed
  SHAs before any reproducibility claim is booked.

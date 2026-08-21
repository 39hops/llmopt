# Handoff 2026-08-21-2: the GRF arc — routing tracks content not phrasing, the phase reweighting survives answering, and the horizon rider hands TIME-AS-STRUCTURE its first measured bridge

Seat: Fable 5 on the Mac, Artin home and checking in; both
machines open; HEAD at close = this handoff commit; 3080 idle
(one short TOPSET census earlier today); Mac idle, nothing armed.
Third handoff of 2026-08-21 — read -1 and -0 for the morning and
night blocks.

## What landed since -1 (with shas)

- CHARTER AMENDMENT (Artin ruling, 6528765c): evaluation/engine
  distinction — bio/chem may be MEASURED, never DEVELOPED; all
  hard prohibitions verbatim; anti-Goodhart perturbation fence.
- GENERAL-ROUTING-FACTORIAL arc, complete in one block:
  - Corpus frozen pre-call (200 prompts, 8 topics incl.
    textbook bio/chem, 5 propositions x 5 forms, matched
    wrappers; grf_corpus.py).
  - OBSERVATION GRF-0 (6f6271a3): topic separability SEMANTIC
    (held-out 0.775 v chance 0.125 under template control);
    form-dominant prefill -> topic-dominant decode; matched
    pairs content 0.944 v wrapper 0.702.
  - AMENDMENT GRF-0-CAPTURE (e4d6a4f7): the capture was
    ALL-THINKING (0/200 closed think, 200/200 at ceiling, zero
    answers) — decode reads scoped to thinking-phase routing;
    verbatim worry defused (completion pairs 0.9413); per-topic
    recall supports bio/chem-ordinary (0.88-0.92), factual_qa
    exposed as a non-topic (0.24).
  - PRE-REG GRF-NOTHINK-0 + OBSERVATION (4530e49d): the
    reweighting SURVIVES answering — decode topic contrast
    0.1856 (2.7x form) on a valid capture (median 30 tokens,
    40/40 MCQ letters); E1/E3 hold, E2 wrong in the interesting
    direction (answering MORE topic-organized than thinking);
    registered refutation does not trip.
  - AMENDMENT GRF-NOTHINK-0-RECEIPTS (9e648396): analyzer
    provenance defect disclosed (fingerprinted the wrong
    producer; rider pins the true one + hashes); MCQ readiness
    32/40 under a frozen anchored extractor; "stronger" narrowed
    to the contrast metric (centroid acc fell 0.760->0.700);
    treatment fences (empty-think scaffold treats prefill;
    greedy/96 scope); MATCHED-HORIZON RIDER: thinking decode
    starts TOPIC-FREE (contrast 0.0025 at N=8 v answering
    0.1441) and converges — the difference is maximal at the
    start, not accumulated.
- RIFF TIME-AS-STRUCTURE (915056bd): Artin's frame banked with
  the house inventory (budgets, phase, next-use, LR settling),
  novelty confirmed (time-to-event never a prediction target),
  ROUTE-TIME ranked first (the replay2 Belady machinery is the
  label factory); no race/security work ever, time = partial
  order/horizon/budget. The GRF horizon rider is the bank's
  first measured bridge (cross-linked).

## Conditions that bite

- logs/grf/{traj,rows}{,_nothink}.jsonl are UNTRACKED big
  receipts; their sha256s are pinned in rider.json/rider2.json
  (both committed). Regenerate-don't-download applies.
- grf_capture.py and grf_analyze.py are results-cited/frozen;
  the NOTHINK variants are the live copies.
- MCQ answer-identity leg: data ready (32/40 anchored), scoring
  is a SEPARATE registration, never a side edit.
- ALTTOKEN identity control: parked on current specimens (rank-2
  IS the vendor token; rank-3 0/5 gap-matched) — park v tri-tie
  desk search is Artin's open pick.
- 3080 nightly candidate: NVMe->host->pinned-H2D under
  concurrent decode (the ROUTE-DB portability fence).

## Next session

Start: this handoff -> BOARD -> RESULTS tail. Live picks, priced:
1. GRF depth rung (form->topic handoff by layer; doubly
   motivated, desk on existing captures).
2. ROUTE-TIME label-factory rung (desk; feeds EXPERTDB as a
   fifth policy).
3. Answer-identity registration (32/40 ready).
4. Prefill-poisoning mechanism rung (EX6 residue, still the
   deepest open question).
5. PRECISION-CREST-TRANSPORT / out-of-core preflights / NInfer
   survey (all unblocked, disk now 70GiB free).

## Open decisions for Artin

1. Which of the five picks leads next session.
2. ALTTOKEN: park v tri-tie search.
3. Tonight's 3080 slot: NVMe measurement GO/no.

## Standing

Nothing measured-unbooked; no watchers live; suite green at close
(handoff commit). Today's totals across three handoffs: 2
verdicts, 7 observations, 7 amendments, 2 pre-regs, ~10 riff
banks/updates, 1 charter amendment — and the false-positive rate
of the review loop stayed high-yield all day: every external
audit point was verified in-house before adoption, and several
were wrong in ways the receipts caught.

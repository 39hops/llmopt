# Handoff 2026-08-13-3 — the schedule-mechanics day

Seat: Fable 5, Mac. HEAD at close: `97d8a6e` (plus this commit).
~20 commits since `c9b70f9` (session open). 3080: untouched all
session, free. Mac: free — capvtraj2 was the last job, rc=0, no
live runs, no watchers armed. Supersedes -2 for session state.

## What landed (chronological, all pushed, suite green throughout)

Hygiene (`93e0b9c`): BOARD test count 483 -> 858 passed/11 skipped;
code-quality row gained Phases 6-7; gen_results_index now reads
date-first + in-title dates (null-date 40 -> 24, all genuinely
undated narrative headings; zero curation lost).

The six-rung chain — each pre-registered BEFORE its run fired:

- **CAP-V-TRAJ-1** (pre-reg `c5ee1ce`, verdict `137dc26`): all 18
  phase19m milestones gated (bisection order, resume-safe driver
  `81347b6`). All three bars fire: rho 0.979, last-6 spread 5,
  step_cap90 = 10,800 < speed90 — capability LEADS the settling;
  the OneCycle tail's final decade buys ~1 solve. Honest misses:
  cap90 one milestone above the registered band; "flat within
  sigma" was only flat within 3x-sigma. Prereg-auditor pass before
  booking (8 findings verified + adopted).
- **LENGTH-VS-L4-1** (pre-reg `1585c67`, verdict `04b5eb9`):
  gate_pp on m015300 reproduced the booked dict EXACTLY
  (precondition fired). Both bars NO-FIRE: length does not carry
  the L4 dip (L6 is 1.4x longer yet scores more). KNIFE-EDGE
  disclosed: pooled within-level rho = -0.2997 vs the -0.30 bar.
- **L4-PLY0-1** (observation `c7c0f61`, desk): 16/17 L4 failures
  emit ZERO valid steps (ply-0 death, no wandering); all 7 solves
  are 1-3-ply f'(g)*fn(g) recognitions. The dip is all-or-nothing
  first-step pattern recognition.
- **BACKWARD-SCHEDULE-1** (pre-reg `265af6c`, verdict `3ae2c9a`):
  OneCycle played BACKWARDS gates 62/120 vs the paired 64 — inside
  sigma, COMMUTES fires, arrow dead at n=1. HOUSE PRIOR WRONG BY
  ~15 SOLVES (predicted 25-50), booked plainly. L4=7 reproduced
  exactly. Driver scratch/birth19m_backsched.py (no-op lr-identity
  precondition asserted in-log).
- **COMP-LADDER-1** (pre-reg `de84bcf`, verdict `c9b4f5b`): stock
  shape compressed to 50% of steps gates 60/120 at half the wall —
  FREE; beats matched-steps truncation (m007200's 51) by 9 — the
  anneal tail's SHAPE does the polishing; 30% breaks (49). Prior
  right on all three bars, both point bands contained.
- **CAP-V-TRAJ-2** (pre-reg `e459a71`, verdict `1601697`): the 18
  backwards milestones gated (thin sibling driver
  scratch/gate_backsched.py — gate_phase19m.py is now results-cited
  FROZEN, learned mid-session, edit reverted). M-STEPS fires,
  M-LR-SNAP dead, and the mechanism refines to the day's headline:
  **LR ABSORPTION FLOOR ~2-4e-5 (~10% of max_lr)** — a dead
  quarter (0/120 through step 2,700 at lr <= 2.3e-5), takeoff at
  ~4e-5, 48/120 without peak lr, and a transient 12-solve dip in
  the highest-lr era that HEALS. Cross-thread anchor: the LLMUE
  pilot's 1e-5 preserved-without-growing null is the same floor
  from the metabolic side.

The assembled schedule story (19M/gen4, single-seed fences):
direction free (62), half-length free (60 at 0.5x wall), shape
load-bearing (truncation pays 9), floor at ~10% of max, high-lr
damage transient. The house modeled length well, direction badly —
track record booked in each entry.

Riffs banked (8): length-vs-structure (`e99b4c6`), star-birth
pincer / order-inversion (`e69279d`, cheap arm MEASURED same day,
amended in place), n-log-n compression (`149a71e`, MEASURED same
day, amended in place), engine-trains-the-model / amortized-birth
(`101280f`), random-inputs engine-filter (`ccf6bd1`, two-reading
split, d64 probe residue), concurrent LLMUE pincer +
valuation-routed-metabolism revival (`5b98fde`), loosen-the-weights
repair + plateau-gated curriculum (`97d8a6e`, both floor-aware).

## Conditions that bite next session

- Curation ratchet: 5 FINDINGS bullets added this session, each
  same-commit with its booking — backlog state unchanged
  (cap==backlog); the NEXT booking needs its bullet same-commit.
- Milestone exhaust now ~8GB untracked: checkpoints/phase19m/
  (4.1GB) + checkpoints/backsched19m/ (~3.8GB) + comp50/comp30
  finals + m015300_model_only.pt. ALL cited by booked verdicts —
  logs-doctrine class, deletion is Artin-GO only (open decision).
- scratch/gate_phase19m.py, birth19m_backsched.py, comp_ladder.py,
  gate_backsched.py, gate_pp.py, len_vs_l4.py are now results-cited
  FROZEN (CODEMAP regenerated) — extend, never edit.
- Small receipts force-added under logs/: phase19m_gate/,
  comp_ladder/, backsched_gate/, pp_phase19m_final.jsonl — frozen
  as evidence, never append a new run into them.
- EX4-UNIF still waiting on a free Mac window since 08-09 — the Mac
  is free NOW.

## Next session, in order

1. This handoff -> BOARD -> RESULTS tail (resume protocol).
2. Ready pre-reg candidates, all with banked designs: OVERLAP-BIRTH-1
   (concurrent pincer), CURRICULUM-1 (3 arms incl.
   capability-ordered), REPAIR-FLOOR-1 (3 lr tiers on the English
   0.5B), amortized-birth cheap rung (profile prediction), d64
   random-updates probe. Artin picks; none is armed.
3. Watch-it-think flagship (Artin-endorsed, unstarted) — rung (b)
   of the L4 riff rides its capture hook.
4. Static figures 5-12 + banked repairs (still paused for Artin).
5. Backwards phase portrait (backsched19m milestones have Adam
   state; phase_portrait_precompute pattern applies) — exploratory,
   unregistered.

## Open Artin decisions

1. Phase C frozen-paths ruling (181 files) — unchanged, reorg
   blocker.
2. Rebirth MEDIUM substitutions per family — unchanged.
3. Static-figure queue unpause + palette ruling — unchanged.
4. Milestone exhaust keep-or-delete — now ~8GB across two runs;
   both runs' bookings are complete, so deletion is safe for the
   booked claims (receipts + npz derivatives are committed), but
   the backwards portrait (item 5 above) dies with backsched19m/.
5. Which of the five ready pre-regs (item 2 above) fires next.

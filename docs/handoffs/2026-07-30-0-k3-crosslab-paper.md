# Handoff 2026-07-30-0: K3-D1, FX-V1-H cross-lab PASS, paper writing-complete

State at ~11:30 AM EST. Machines idle; 3080 untouched since
morning, back to Artin at 5 PM EST.

## Booked today (after the 07-29-6 addenda)
1. **K3-D1** (pre-reg + verdict, same morning): one Kimi-K3
   routed expert pulled by safetensors byte-range — 17.5 MB
   out of 2.8T. Discoveries: K3 is a LATENT MoE (experts
   3072x3584, ~33M/expert, half the naive estimate); meter
   M 1.94-2.15 straddles K2's 2.01 (grid-image confound
   fenced — banded claim only); MXFP4 codes carry 3.643
   bits/param (~9% lossless margin; Moonshot packs well);
   deterministic integer GEMV on the SHIPPED format
   sha-identical cpu/mps/cuda, zero requantization.
   Instrument: scratch/k3_expert_demo.py (library calls).
2. **FX-V1-H cross-lab PASS**: axiom reproduced BOTH P3
   digests full-match on their machine (their relay
   2026-07-30-1, commit 70777ea theirs). House added the
   cpu point (P3_DEV knob, axiom's suggestion). Tally: one
   tables file, four backends, two labs, zero tolerance
   columns. Their model-card catch amended (card of record:
   packed d64h8 crystal d64 L8 ffn 256 — relay prose had
   said d256 L4). Receipt relay 2026-07-30-1 pushed; thread
   CLOSED.
3. **Paper writing-complete as a draft**: secs 8 (related
   work, honest deltas) + 9 (fences) drafted; ALL 16 cites
   verified with arXiv IDs pinned in a registry section;
   venue DECIDED (MLSys; efficient-ML workshop fallback);
   K3-D1 + cross-lab folded into secs 3-4. Remaining:
   prose-ification (skeleton -> paper), figures, final
   PDF-level cite re-read.

## Where the program stands
Every build cell of the capacity program and the packed
crystal C-series is DONE and replicated where scoped. The
paper is the deliverable in flight. Banked next-moves (all
in RIFF): packed-expert shelf (router over packed
crystals), micro-MoE conservation 3-arm, matched-operator
flips probe (P5 follow-up), usage-tiered packing,
entangled-experts prefetch, tied-expert ladder, area-law
probe, training lens.

## Resume path
BOARD -> this handoff -> docs/paper-draft-entropy-bound.md
-> RESULTS tail. Nothing in flight, tree clean at commit
time.

## Addendum 1 (~2 PM): K3-D2 + UMOE-1 n=2 booked
- K3-D2: full-expert deterministic chain sha d771796f...
  identical cpu/mps/cuda (SiLU table shipped as bytes,
  f503c814...). Composition fence closed.
- UMOE-1 + seed-2: first house MoE births. Split law =
  sparse assignment, NOT balance loss (n=2). Tied-at-birth
  ~free at the gate. (M,MI) rider killed honestly.
  Soft-routing falsifier banked (kills decorrelation if
  assignment is the mechanism). Checkpoints umoe_*_s{1,2}
  on the 3080 (pull to Mac when convenient).
- ff#8 repeat amended: loader-on-box before launch is now
  the arm-time check.
- Remaining today: paper prose (no GPU); 3080 idle+clean
  for Artin at 5.

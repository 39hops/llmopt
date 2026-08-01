# Handoff 2026-08-01-4 — P4 complete both legs; pre-Sol baseline

## Verdicts this leg

1. **VERDICT GRAVMOE-P4-LAB**: axiom C++ reproduces all 10
   engine-side gravmoe arms bit-identically (their commits end
   8f8376d; verdict relay theirs 2026-08-01-5). One round-trip
   debug: first run FAILED 10/10 — (a) my artifact schema was
   never pinned in the relay (house defect, fixed: windows
   record format, contract sub-dict, pins dict — relay -6), (b)
   their gate backward pre-rounded dout; the house convention
   (dgate exact, /PQ folded into each consumer, dtop_p by Q) was
   pinned as contract text in relay -6 and fixed in their
   a263321. RB1 bisect ladder closed it without E=1 reduction.
2. **Method law (adopted)**: E=1 parity gates are provably blind
   to gate-backward placement (common-factor rdiv identity;
   conventions separate only at E>1). Rounding placement is
   CONTRACT TEXT. Corollary: pin artifact schemas in the relay
   that announces them.
3. Ladder state: deterministic-birth scale program CLOSED at
   3 implementations / 2 labs / 2 devices.

## The Sol experiment (Artin's plan)

GPT 5.6 Sol gets branch sol/review-1 as an ADVERSARIAL IMPROVER:
repo changes + different views, tougher than the Opus review.
Rules encoded in the kickoff prompt (in session notes, Artin
sends): Mac only (no wsl.sh/rjob remote/ssh/3080, no ~/code/axiom
writes), never touches main or its living docs — Sol keeps its
own ledger in docs/sol/ (NOTES.md + RESULTS-SOL.md, same
pre-reg discipline), every claim ships command line + log +
shas. Battery refactors must regression-gate against
scratch/detbwd_gmoe_ref/pins.json. Fable reviews the branch on
Artin's ask (diff against tag pre-sol-baseline + spot-rerun of
claimed shas); nothing merges without that review.

## Repo state at the tag (pre-sol-baseline)

- Both machines idle; no rjob jobs; monitors stopped.
- All 08-01 verdicts booked through GRAVMOE-P4-LAB; index clean
  (0 needs_link).
- Relays: -5 (spec+postscript), -6 (schema fix + placement pins)
  delivered; theirs -4 (multiblock) and -5 (gravmoe verdict)
  received and receipted.
- Fable usage: ~90% weekly, resets Monday 7:00 AM EST — until
  then house work stays verification-shaped (reviews, bookings),
  no big generation legs.

## Queue (post-Sol-review or Monday)

1. COND+QK graduation: seeds/windows ladder + WIDER gate diet
   (the closing brute verdict: data, not compute, binds).
2. rms_fwd headroom rung (spec first) — only if width graduates.
3. Sol branch review on Artin's ask.
4. Carried: diet meter, ckpt forensics [Artin GO], FOURIER-4,
   exact-manipulation diet share.

## Fences

- Sol's numbers live in docs/sol/ and are PROPOSALS until
  house-verified — never cite them from main's ledgers.
- All 08-01 gravmoe capability numbers remain n=1.

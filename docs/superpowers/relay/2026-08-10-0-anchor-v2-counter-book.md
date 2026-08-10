# Relay 2026-08-10-0 (house -> axiom): ANCHOR-V2 counter-booked — P-DIGEST-EQUAL FIRES (byte-identity reproduced here); P-HORIZON misses at 8/12; and the tie-depth numbers measure `sh.e`, not a distance

WHO IS WRITING: Opus 5, llmopt seat (the Fable-only authorship
rule was retired by Artin 2026-08-10 — the session model owns
edits now; same verification bar). Verified from your Mac
checkout at 0c48367, artifacts read directly.

Booked: COUNTER-BOOK ANCHOR-V2 (RESULTS 23990). Congratulations
are warranted first: the gcd wall is gone and your step-1 weights
are byte-identical to the exact anchor. That is the result.

## Reproduced house-side (not taken on trust)

`cmp` clean, anchor2_step1.w9 v probe_d64/anchor_step1.w9, both
sha256 7c9b8f0bfb592185... — and that hash IS the step-1 `digest`
field, so every streamed row self-checks against its own dump.
Wall re-added from rows: 1,286.4 s = 21.44 min, flat ~160 s/step.
Counters consistent: floor_exact cumulative 64..512 (+64/step),
everything else zero. P-DIGEST-EQUAL FIRES on that leg. Your
SCHED-boundary and forced-fallback cells are booked
RELAY-ATTESTED — house did not rebuild your suite, so those two
legs are your receipts, not ours.

P-HORIZON books as a MISS: the bar said 12 steps in <= 4 h, the
run certified 8. You called it partial yourselves; the ledger
records the miss and, separately, the horizon extension, which is
the real news.

## Two corrections, both verified in your source

1. THE TIE-DEPTH SEQUENCE IS NOT A MEASUREMENT. You report the
   blocked site at dist < 2^-358 / 2^-798 / 2^-1585 / 2^-3958 for
   precisions 400 / 840 / 1627 / 4000 and read a super-geometric
   deepening off it. The number is `sh.e`, the SHARED EXPONENT of
   the dyadic interval — dyadic.hpp:19 says "value in [lo*2^e,
   hi*2^e]" — printed next to lo_bits at exact_anchor2.hpp:224-226.
   lo_bits - |e| = 42 at all four precisions, exactly, because
   that difference is log2 of the site's VALUE MAGNITUDE and it is
   one site. The four "distances" are precision minus 42. They say
   nothing about the tie. What IS measured: the site straddles at
   every precision up to 4,000 bits and reconstruction threw.
2. NO floor-near HAS EVER BEEN OBSERVED. floor_near is 0 in every
   shipped row, and it can only increment after a SUCCESSFUL
   reconstruct (`if (r.is_zero()) fb.floor_exact++; else
   fb.floor_near++;`), which step 9 never reached. Your relay
   frames these numbers two ways — per-step at 7/8/9, and
   per-precision retries at one step-9 site. The counters and the
   constant 42 both select the second; the first should go.

## Your hypothesis survives, on better evidence

floor_decl() runs the pin-1 equality test for every integer in a
straddle of width <= 4, and the throw reports w=1 — so the single
candidate WAS tested and was NOT equal. v != k, the site needs a
sign not an equality, and with your measured x37/step
numerator/denominator growth "true distance ~ 1/D" is a
well-motivated inference. Inference, not measurement — that is the
only thing the ledger changes.

## Why that makes the co-factor witness the right next rung

It is not just the fix, it is the missing INSTRUMENT: recovering
t = v*z - k*z = +-r gives you |r|, which IS the tie depth. Build
it and the super-geometric claim becomes testable instead of
inferred. House ask for the amendment: register |r| per site as a
reported observable, and the pin-3 rate question answers itself.

## Small thing

You quote the old anchor's death as 19.5 h; house tightened that
to 19.34 h from the artifact timestamps (AMENDMENT
EXACT1-SMALL-EXPONENT-2) after Artin asked for a recheck that
also caught a mixed-estimator quote of mine. Same class of defect
in both labs today — worth both of us naming estimators and
reading artifacts rather than printed summaries.

Fence unchanged: 3080 stays untouched, one worker, Artin's GO.

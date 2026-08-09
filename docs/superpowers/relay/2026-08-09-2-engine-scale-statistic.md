# Relay 2026-08-09-2 (house -> axiom): ENGINE-SCALE-1 verdict BLOCKED on one statistic — loss_final is last-window, the bars are frozen on final CYCLE-MEAN

WHO IS WRITING: Fable, llmopt seat. House verification of
engine_scale_results_final.jsonl (sha c8895ee0, 30/30 rows,
cell-set exact, provenance fields clean, all three spot traj-shas
MATCH frozen receipts): one blocker.

## The catch (caught by the continuity anchor, as designed)

60k-w8-s1000-const reproduces the certified DIET-BRIDGE
trajectory bit-exactly (8b443b68...) — but its loss_final reads
13,652 while the certified FINAL CYCLE-MEAN is 12,518. Same
trajectory, different number: loss_final is the SINGLE last-window
loss. Confirmed on the sched spot cell too (12,772 v the booked
11,777). The registered bars (P-JOINT <= 11,266; P-DIET-FLOOR's
5%/2% attribution pattern) are frozen on final cycle-mean per the
PLATEAU-BREAK convention (mean of the last NWIN per-step losses).
Near the bar this is not pedantry: 31k-w8-s4000-const sits 137
under the line on the wrong statistic.

## Ask (cheap post-fix)

Re-run the 30 cells on >=7d0f398 with the runner emitting, per
cell: cyc_final (mean of the last NWIN single-window losses,
integer-floored per the house convention in detbwd_diet.py's
cyc computation: sum(losses[-NWIN:]) // NWIN) — and ideally
loss(win)+loss(cyc) per 125-step milestone, matching the house
print convention. Observability-only change: every traj sha must
reproduce identically (the three spot cells re-verify for free,
making it four reproductions per receipt). At post-fix speed the
whole grid is ~6-10 minutes.

## So nothing is lost

The shipped file stays as the wall-clock/defect measurement
record (mixed-provenance annotation is exactly right). The
descriptive SHAPE axiom flagged (sched >> const at s16000 by
2,600-4,900; small-window cells strongest) will almost surely
survive the statistic swap — but bars read only against the
registered number. Verdict books house-side immediately on
receipt of the cyc_final column.

## Also closing two of your open items

The Q32/Q64 cross-stdlib counter-book is DONE (AMENDMENT
ENGINE-EXACT-1-DIGESTS, c8c103a: new pins recorded + WSL
gtest ReferenceDigest counter-run green), and the two-regime
restatement is BOOKED (PRE-REG EXACT1-SMALL, 3b9cdb7; d16 cell
complete, d8 anchor still paying its gcd toll). Only the
ENGINE-SCALE-1 verdict remains open, on the ask above.

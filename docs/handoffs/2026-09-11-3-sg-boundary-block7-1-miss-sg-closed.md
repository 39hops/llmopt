# Handoff 2026-09-11-3: SG-BOUNDARY-BLOCK7-1 SG7-MISS booked; SG credit-writer births closed completely

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run. Nothing armed.

## What landed (after handoff 2026-09-11-2)

- Artin GO (14:34 EDT): seed-28 qualification exactly as sealed.
  Launched under liverun sgbb7q at b067aa89 (armed 18:35 UTC,
  disarmed rc 0 at 20:24 UTC; launch integrity smoke 3 / 3 first).
- VERDICT SG-BOUNDARY-BLOCK7-1 (RESULTS, this commit): SG7-MISS.
  CONTROL 64 / 120 = {3: 23, 4: 7, 5: 16, 6: 7, 7: 11} (ADEQUATE; band
  57 to 71); SG7 37 / 120 = {3: 11, 4: 1, 5: 11, 6: 5, 7: 9}, finite,
  delta -27. Final CE 0.384 v 0.399. The LOCAL predictor tracks block
  7's error (below the zero baseline at 64 of 87 logged steps from
  1,028 on; final MSE ratio 0.58; final cos 0.68) and the cell still
  misses by 27 solves. Prior 4 hits 2 misses (family 23 / 19).
- Consequence, as sealed: SG credit-writer births are CLOSED
  COMPLETELY (no larger, delayed, bootstrapped, bidirectional or
  downstream-weight-conditioned repair). No writer-invariance or
  causal-efficacy claim; the dead-writer disclosure stands.
- Operational disclosure: the run ended at 16:24 EDT; the booking was
  not started until Artin asked at 21:36 EDT (the last Monitor events
  did not wake the seat). No data was affected; the delay is a
  session-hygiene miss, recorded here.
- Prereg-auditor and receipt-auditor run on the draft and receipts
  before booking (findings folded in the verdict text).

## Conditions that bite next session

- Nothing armed. The synthetic-gradient program is closed at the
  arena in all sealed forms (top-four, cross-position, block-7
  boundary). Remaining foreign-writer candidates 3 to 5, or park, are
  Artin decisions with their own pre-regs.
- Checkpoints: checkpoints/sgbb7 (2.6 GB: two cells x 17 snapshots +
  finals + 17 predictor snapshots) retained; checkpoints/sgbb7_smoke
  (439 MB) prunable after a digest inventory (Artin GO). Disk about
  29 GiB free.
- Long-run hygiene: when a Monitor's final events land while the seat
  is idle, the booking can stall; arm a background `until` on the DONE
  marker as well next time.

## Open decisions for Artin

1. Foreign-writer program beyond SG: candidates 3 to 5, or park.
2. Keep set for checkpoints/sgbb7 (propose: per cell steps 463 / 3,084
   / 5,140 / 15,420 + predictor snapshots + control W_0, about 0.8
   GB) and the smoke tree prune.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from the SG-BOUNDARY-BLOCK7-1
verdict, the RIFF bank SYNTHETIC GRADIENTS / DNI (closed in place).

---
name: watch
description: Arm a watcher on a running rung without leaking sealed treatment values — separates always-readable qualification artifacts from rc-gated treatment artifacts
---

# Arming a watcher (the sealed-watcher ritual)

A watcher watches; it does not launch (that half lives in /rung).
This skill governs WHAT A WATCHER IS ALLOWED TO PRINT, because a
watcher's completion dump lands in the session context — reading
it is irreversible.

## The incident that earned this skill

2026-08-21, EX6-MED run 2: the driver exited 3 on its registered
fail-closed qualification, and the watcher ran
`tail cells.log` UNCONDITIONALLY — dumping per-cell treatment
counts into the session. The run could never again produce a
blind verdict-class reading (AMENDMENT EX6-MED-0-QUALFAIL-2,
taint permanent). The qualification machinery worked; the
watcher defeated it.

## Classify artifacts BEFORE arming

Every artifact the run produces gets one visibility class:

- **ALWAYS-READABLE**: qualification receipts, rc files, DONE
  markers, progress heartbeats, provenance blocks, anything the
  pre-reg registers as pre-read (outcome-blind censuses).
- **SEALED-UNTIL-QUALIFICATION**: treatment values — cell
  counts, deltas, per-arm receipts. Readable only after the
  registered qualification adjudicates PASS.
- **SEALED-FOREVER-ON-FAILURE**: what treatment artifacts become
  when qualification fails. Never printed, never read; frozen as
  evidence only.

If the pre-reg has no explicit qualification, everything is
always-readable — say so when arming, so the choice is visible.

## The watcher shape

```bash
until [ -f jobs/<id>.rc ]; do sleep 180; done
rc=$(cat jobs/<id>.rc); echo "RC=$rc"
# ALWAYS-READABLE artifacts: dump unconditionally
grep -E "<qualification-markers>" <driver-log> | tail -20
# SEALED artifacts: MECHANICALLY gated, never hand-checked later
if [ "$rc" -eq 0 ]; then
  tail -20 <treatment-receipt>
else
  echo "qualification failed (rc=$rc): treatment output SEALED"
fi
```

Rules, each one load-bearing:
- The rc gate is IN THE WATCHER COMMAND, not in your intentions.
  "I'll check rc before reading" fails the moment a notification
  dumps the file for you.
- NEVER tail a mixed log that interleaves qualification lines
  with treatment lines. If the driver produces one, fix the
  driver's logging before arming (separate files, or grep-able
  disjoint prefixes — and grep the qualification prefix only).
- A driver that exits nonzero on qualification failure must exit
  BEFORE printing treatment summaries (fail-closed printing, not
  just fail-closed booking).
- The /rung "receipt dump on the same line" instruction applies
  to ALWAYS-READABLE artifacts only.

## Banked generalization

A machine-readable pre-reg extension for artifact visibility
classes (always-readable / sealed-until-qualification /
sealed-forever-on-failure) is banked in RIFF-LEDGER — when it
lands, /watch enforces blinding from the pre-reg instead of from
this checklist.

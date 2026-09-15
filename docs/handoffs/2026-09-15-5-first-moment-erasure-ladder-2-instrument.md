# Handoff 2026-09-15-5: FIRST-MOMENT-ERASURE-LADDER-2 implemented (instrument committed, smoked, reviewed); nothing armed

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file (codemap regenerated in the same commit). 3080 untouched. No live
registered run. Nothing armed.

## What landed (after handoff 2026-09-15-4)

- Artin GO 11:09 EDT: FMEL2 implementation only.
- AMENDMENT -INSTRUMENT (RESULTS L76155): scratch/first_moment_erasure_ladder2.py
  (thin sibling of the FMEL1 instrument, importing its leg mechanics;
  pins extended by that source and the locked FMEL1 receipt),
  scratch/fmel2_launch.sh (liverun fmel2), 11 tests (pins; reference
  digests from locked receipts with torch.load forbidden; independent
  alpha_global + decade slopes + the log-uniform identity; zero norm /
  missing decade / non-finite / refusal; preflight identities v
  informative checks; regime order with both local slopes; locus incl.
  h_curve; disposition law; real-mode constants in a subprocess;
  refuse-if-exists; no-B AST). Smokes: full mechanism (smokel2a,
  smokel2e identical), tampered reference abort (smokel2c), tampered
  preflight digest NOT-RUN (smokel2d). Opus review folded. Full suite
  passed.

## Wall-cap fold (Artin HOLD 11:41 EDT; AMENDMENT -WALL-CAP L76267)

- The sealed 7 h stop law is now mechanical: MAX_WALL_S = 25200 fixed
  in real mode (no env var can change it; SMOKE_MAX_WALL_S is SMOKE
  only), SIGALRM in the main thread, NOT-RUN receipt with the
  interrupted phase and the reason, exit 3, DONE never written, no
  further arm, partial stream and completed-arm snapshots kept; a
  120 s grace hard-exit backstop. Timeout smokes smokel2f / smokel2j
  fired during the long legs; smokel2h ran the DONE path on the folded
  main. Tests 12; review folded; full suite passed.

## Conditions that bite next session

- Nothing armed. Target run = `bash scratch/fmel2_launch.sh` (liverun
  fmel2) at the committed HEAD, only on Artin GO. About 3.3 h, 7 h
  kill; about 4.2 GB under checkpoints/fmel2; refuse below 16 GiB.
- Registered expectation for the preflight (from the FMEL1 receipt):
  R 1.0000132 / 1.00000033 / 1, cos 0.99998559 / 0.99999986 / 1,
  n_1(1) 0.894284643; the fresh preflight digests must equal the
  locked FMEL1 preflight digests, else NOT-RUN.
- Live-run law for the whole wall; checkpoints/fme1 and
  checkpoints/oma1/A/C kept through this rung.
- Any change to a pinned source (now including both ladder
  instruments) refuses the whole family until re-pinned by amendment.
- Deferred cleanup: the detached-third-worktree locator-test blind
  spot.

## Open decisions for Artin

1. GO FIRST-MOMENT-ERASURE-LADDER-2 TARGET RUN.
2. SCHEDULE-PHASE-SENSITIVITY-CENSUS-0 stays banked until FMEL2 books.
3. Checkpoint disposition after FMEL2 books.

## Next session: where to start

This handoff, BOARD line 5 (tail), RESULTS L75924 and L76155,
docs/preregs/first-moment-erasure-ladder-2.json,
scratch/first_moment_erasure_ladder2.py.

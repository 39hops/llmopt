# Handoff 2026-08-17-1: the Qwen night — bridge complete, three-artifact compiler in flight, entropy became an instrument

Seat: Opus 5, Mac; 3080 running the WHOLE-0T compile at close.
Read AFTER 2026-08-17-0 (the 0S verdict + census night).

## What landed since -0 (all pushed)

- v2 FULL cross-device qualification PASSES (13x inside pre-look
  bound; ratio corrected 7.7x -> 7.45x harness-level); CENSUS-0
  readings narrowed; refuted_if now MACHINE-SCORED (schema
  predicate; 0S clause reproduces REFUTED); TARGET-REBASED prior
  marker booked.
- QWEN pivot (Artin GO, GPT-seat program adopted + banked): probe
  L32 (codec ranking TRANSPORTS to dense: W4 +14.3%), BYTE-CENSUS
  (FFN 61.6%, text-only 6.56 GiB @2.0625), FFN depth census
  (stable 4/4 layers, stop-rule fired), FAMILY rate probe (9
  tensors: weight space codec-HOMOGENEOUS, no fragile family,
  vector advantage is a 2-BIT phenomenon — S16-DP edges stacked VQ
  at 4 bits everywhere), scalar-inversion DIAGNOSED (central-gap +
  zero-concentration; wording narrowed by amendment), and
  SCALAR-MASS-CENSUS closed it both sides (V4 64% v Qwen 80-84%
  below 1/3: stored-distribution effect).
- PRE-REG QWEN-WHOLE-0T + -ARMS amendment (Artin: "try all of
  them"): rate tables A/B/C compile as three arms in ONE pass;
  bars conservation / per-arm budget / fidelity tripwires, all
  machine-adjudicated incl. refutation predicate.
- MODEL-1 functional eval FROZEN pre-artifact (evals/qwen_model1/:
  teacher-forced core, fixed prompts/prefixes/corpus, teacher
  logits via ONE streamed vendor CPU pass then locked,
  no-retuning rule). Runtime program banked: RUNTIME-0R CPU
  reference -> Metal direct-W4 (Mac leg) -> CUDA direct-W4 (3080);
  precision-as-escalation (Ozaki closed; Metal exact leg is the
  one open build, only when the runtime consumes it).
- Receipt-lock hardening rounds: prereg-declared paths pend even
  when registration prose names them; sha freezes at booking.
- /doctor pass: 11 unused/redundant plugins disabled (~4-4.5k
  tokens/session), backup at ~/.claude/settings.json.doctor-backup.

## IN FLIGHT at handoff
- 3080: qwen_whole0t.py FULL 18-shard compile (launched ~04:25,
  ~35-45 min; smoke PASSED viol=0, 87 s/shard; receipts land at
  logs/qwenwhole/compile.jsonl + .log; artifacts at
  ~/qwen_whole0t/{A,B,C}/ on WSL, delete-after-compress mode).
  BOOKING PATH: pull receipt -> write the -0T adapter
  (obs_from_receipt style: conservation counts, arm bytes, family
  errors -> measurements 1/2:A/2:C/3:*/refuted:artifact_bytes) ->
  scripts/adjudicate.py docs/preregs/qwen-whole-0t.json -> both
  auditors -> book verdict.
- Mac: v2 same-device promotion gate (v2 CPU L22 v the Mac v1
  receipt; jobs/v2gate.rc watcher armed, ~208/256 at last check).
  On completion: tolerance-compare same-device, book, then the
  promotion gate needs the determinism contract before v2 becomes
  the instrument (shared codec module — qwen drivers import it).

## Gotchas that bit tonight (do not repeat)
- wsl.sh launch with a missing log DIRECTORY dies instantly and
  silently (the /rung gotcha, verbatim). mkdir first.
- A watcher whose pgrep pattern contains the driver name matches
  ITSELF (friendly-fire); bracket one character.
- Receipt paths cited with brace-shorthand (L{8,48}.jsonl) are
  invisible to the citation lock — explicit paths only.
- pytest_gate_guard + the hookify rc rule are firing correctly on
  pipeline shapes; use the redirect + if [ $rc -eq 0 ] pattern.

## Next session
Start: this handoff -> BOARD -> RESULTS tail (L32776+). First
batch: (1) book WHOLE-0T from the compile receipt via the closed
path; (2) book the Mac gate result; (3) build RUNTIME-0R (slow CPU
reference decode) + the vendor teacher-logit streaming pass (the
frozen baseline); (4) then MODEL-1 scoring A v B v C. Open Artin
decisions: none new — table B/C/A all compiling per his GO; MTP
excluded by stated assumption (flag if he wants it in).

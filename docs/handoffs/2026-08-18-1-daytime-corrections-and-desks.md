# Handoff 2026-08-18-1: the corrections-and-desks afternoon (compact point, work IN FLIGHT)

Seat: Fable (main session model), Mac. Continues -0 (same day).
Compact requested mid-window; THIS FILE + the in-flight table
below is the resume state. 3080 window open until ~17:00 EST;
overnight GO not yet given.

## IN FLIGHT at compact (check these FIRST on resume)

1. D/E score chain on the Mac (background task): pulls
   ~/qwen_whole0t/{D,E} from the 3080, scores each with
   ARM=<X> .venv_teacher, deletes after receipt. Watch
   logs/qwenmodel1/score_D.json + score_E.json. When both exist:
   run scratch/qwen_ioattrib_adjudicate.py (committed, 4
   fixtures green), spawn prereg-auditor + receipt-auditor on
   QWEN-IO-ATTRIB-1, verify findings, book. Compose receipts +
   chains already pulled (logs/qwenattrib/compose_{D,E}.json,
   logs/qwenwhole/artifact_digest_{D,E}.txt — untracked, force-add
   at booking).
2. RK-CENSUS vendor capture on the Mac (fp32 + fail-closed
   provenance sidecars, r2 driver): watch
   logs/qwenrouter_vendor.log for the sidecar line. Then:
   MODE=arm ART_DIR=~/qwen_whole0t/A .venv_teacher (A is resident
   on the Mac), then MODE=analyze with .venv. Book ONLY with the
   sidecar protocol closed. Layer-32 family label correction is
   in the driver docstring (disclose at booking); prior rationale
   correction (rung-3 KL was backend parity) already booked in
   the amendment.
3. LBAND-1 six-arm run WAITS for Artin's overnight GO (order he
   set: D/E first, then LBAND). Recipes committed
   (BLe/BLm/BLl/FLe/FLm/FLl), scorer allow-list extended, JSON
   has the F-side K bar + floor semantics pinned. An LBAND
   adjudicator does NOT exist yet — write it mirroring
   qwen_attrib_adjudicate before scoring.

## What landed this afternoon (all committed + pushed)

- OBSERVATION QWEN-EFFORT-QUANT-0 (e7e202f): B 0/60 free-gen
  collapse, xhigh think never terminates 30/30; C residency
  answered NO by OOM; iso-F bank retired.
- Five PRE-REGs (e04fb5c): IO-ATTRIB-1 (iso-byte D/E, +0.2960
  GiB each), LBAND-1, CAPACITY-METER-1, CAL-FEAS-0 (parked),
  RK-CENSUS-0. JSON projections for io-attrib + lband.
- OBSERVATION QWEN-CAPACITY-METER-1 (564cf98) then AMENDMENT
  (fe6acfc): r1 was MTP-contaminated; r2 count-asserted; the
  sampled-M stability rider FIRES (1-2 bit drift = spread size)
  — family M ordering RETRACTED; survives: within-projection
  late-heavy M depth gradient, linear:out kurtosis 14.4 with
  INVERTED (early-heavy 32.4/6.1/4.9) kurtosis depth profile,
  and the diagnostic conclusion a fortiori.
- OBSERVATION QWEN-SCORER-WALL-0 + METAL-INT-MMA-0 (e2550cf):
  scorer decode-bound >=75%; M3 Pro has NO integer simdgroup MMA
  (float/half/bfloat only) — Ozaki-for-Qwen closed with data;
  Mac speed lever = decode kernel (the registered Metal W4 leg).
- CLAUDE.md doctrine correction: exact int8-sliced is 1.07-1.35x
  fp64 WALL (accuracy-per-wall lever), only the approximate
  triangular arm is faster.
- EFFORT-QUANT amendments: comparator q4/4.501 (not "q6-class"),
  think_terminated encoding in both drivers, claim narrowed to
  B-v-external-comparator, tokenizer parity rider 60/60.
- Opus prompting-review adoptions (dbfeb7a): report-all +
  confidence on all reviewers/auditors, progress-audit +
  assessment-first blocks, skill-conflict rule now
  log-then-deliberate.
- mlx q6 resident reference built (~/qwen_mlx_q6, 6.501 bpw,
  4.8 tok/s); q4 deleted (regenerable).

## Conditions that bite

- zsh glob trap bit once (unmatched glob aborts the whole rm
  command line) — the fp16 z-captures survived a "cleaned" echo;
  deleted explicitly after. Watch for it in cleanup one-liners.
- Artifacts D/E exist on the 3080 + (transiently) Mac; F/L/Q on
  3080 only; A resident on Mac for the RK arm leg.
- EFFORT-0/EFFORT-QUANT frozen rows carry the OLD think_tokens=0
  encoding (amendment discloses); new rows use think_terminated.
- Overnight GO checklist Artin set: D/E first (running), LBAND
  second, RK books only with provenance closed, CAL-FEAS parked,
  no fresh long GPU jobs without his explicit GO.

## Next session opens with

1. In-flight table above (items 1-2 likely finished — book them).
2. On Artin's overnight GO: LBAND-1 (write its adjudicator first,
   compose 6 arms on 3080, score on Mac one-resident-at-a-time).
3. Then: MODEL-2-class questions (two-gate tree, rate table with
   the linear:out kurtosis lead), Metal W4 decode kernel as the
   scorer/label-factory speed lever.

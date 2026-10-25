# Handoff 2026-10-25-0: QWEN-GDN-PROMOTE-0 (the Metal recurrence kernel is the serving default, all eight gates fired); the Muon and transfer banks; the August 2026 mathematics study (no leverage)

Session: Fable 5.1 on the Mac. Opus 5.5: a methodology audit of the
registration, driver and adapter BEFORE registration (two blockers,
seven should-fix items, all applied), a receipt audit (no blocker, two
receipt-field items disclosed) and a pre-registration audit (one
blocker, six should-fix wording items, all applied) before booking.
HEAD at close: the handoff commit (git log -1), on origin/main. 3080:
fast-forwarded to it (ff-only, HEAD asserted) at the end of this
session; nothing here depends on it. Previous handoff: 2026-10-24-1
(the kernel qualification). Read BOARD first.

## What happened

1. Artin's GO (2026-10-25) for the promotion rung with requirements:
   QWEN-GDN-KERNEL-0 preserved exactly (bar 9 UNRESOLVED, not
   re-registered); the pinned model artifact and configuration confirmed
   in the promotion receipts; explicit vendor-guarded dispatch, never
   model.eval(); no leak across model instances or sessions; registered
   gates; the default conditional on them. Separately: bank the Muon
   assessment and the transfer candidates (no training authorized) and a
   read-only study of the August 2026 mathematics.
2. Mechanism (0360253a): llmopt/lab/qtower.select_recurrence re-classes
   each GatedDeltaNet of one tower to SelectedGatedDeltaNet, whose call
   is the vendor body with the exact-l2norm edit and a dispatch argument
   that reads the instance's own `recurrence` attribute; the vendor
   class, its module dictionary and the training flag are untouched; the
   kernel is refused under any mlx-lm qwen3_5.py or gated_delta.py sha
   other than the qualified pair (QUALIFIED_VENDOR) or without the Metal
   GPU; the Engine refuses the kernel at a chunk other than 256;
   qserve.DEFAULT_RECURRENCE "kernel", Engine(recurrence=...),
   Engine.select_recurrence; scripts/qwen_chat.py --recurrence
   {kernel,loop} with the selection in the receipt; 7 selector tests
   (tests/test_qtower_recurrence.py), default and chunk pins in
   tests/test_qdefaults.py. llmopt/lab/qgdn.py stays the qualification
   instrument (counting mode only in the promotion driver).
3. PRE-REG QWEN-GDN-PROMOTE-0 (RESULTS L86098, docs/preregs/qwen-gdn-
   promote-0.json, eight gates) at 0360253a with the smoke chain
   force-added; the fresh CLI collection scratch/evalsets/
   qwen_battery_v4_cli.json (16 tasks, first-turn exclusion against v1 to
   v3) frozen at registration.
4. The chain ran under liverun qwen-gdn-promote-0 at 65039e50 (67 min).
   The adapter's first pass read the receipts INVALID for three
   adapter-side reasons; AMENDMENT QWEN-GDN-PROMOTE-0-ADAPTER (L86339,
   934c17d2) corrected the floors before adjudication with the pre-read
   disclosure (every measurement value had been seen; the adjudication
   had not). VERDICT QWEN-GDN-PROMOTE-0 (L86397, fceabff5): all eight
   gates FIRE, NOT-REFUTED, the promotion STANDS. Dispatch 717,168
   kernel and 0 loop executions under the default; 92 tests passed, 0
   failed; held-out parity 0 of 36 comparisons differing (in-process
   loop and the frozen QWEN-GDN-KERNEL-0 loop rows); cache 9 of 9 equal
   with 6 of 6 reuses; v3 battery 56 of 60 with all 60 tasks
   token-identical to the frozen loop receipt; the CLI collection 16 of
   16 token-identical between the arms, 15 of 16 pass in both; TTFT
   0.672 / 0.674 / 0.667 at 512 / 1,024 / 2,048 (8.93 / 18.13 / 34.88 s
   against 13.29 / 26.91 / 52.33), peak 1.008, 16-step decode 0.93.
   Not one token differed between the kernel default and the loop
   anywhere (about 16,000 greedy tokens), so the registered near-tie
   rule was never invoked.
5. RIFF banks (fceabff5): the promotion MEASURED bank; the Muon ledger
   re-read against the reference recipe with the L6414 retraction
   carried and a three-arm reproduction priced and NOT authorized; the
   three transfer candidates (oracle-bootstrap, weight-only invariances,
   offline KD on hold); the August 2026 mathematics study
   (docs/superpowers/specs/2026-10-25-sphere-packing-codes-nonsofic-
   relevance.md, 9bb9c804): the packing and code results are asymptotic
   upper bounds and exact LP exponents with no bearing on a d = 4
   fixed-rate codebook or on representations at 0.0035 bits per
   dimension; the non-sofic group shares no object with weight-space
   symmetry (finite or compact groups, hence sofic); the one item kept is
   Chapter 7's CVP hardness as the shape constraint on any future
   codebook (structured lattices with fast decoders). No experiment
   proposed, quantization stays CLOSED.

## Conditions that bite next session

- The serving default is now the kernel. Every receipt before
  QWEN-GDN-KERNEL-0 ran the loop; --recurrence loop reproduces that path
  (token-identical where measured). An mlx-lm upgrade will make
  select_recurrence REFUSE the kernel (sha pin) until a re-qualification
  updates QUALIFIED_VENDOR in llmopt/lab/qtower.py; the loop stays
  selectable.
- A second Engine cannot be built in one process (build_metal_model
  rewrites the vendor class from its own source; pre-existing). The
  forcing instrument qgdn.recurrence_path("loop"/"kernel") overrides every
  instance's selection by design; never use it in serving. Never
  re-select under a live Session's carried state.
- Two receipt-field defects in the frozen promotion driver, disclosed in
  the verdict: dispatch blocks' "path" field reads "vendor" (the arm is
  proven by the other fields); the cache stage's turn rows carry no ids.
  A future driver copying it should fix both and run the adapter's
  VALID branch on synthetic receipts before the run (three adapter-only
  floor defects would have booked a false HOLD here).
- The prefill wall is now about 70 percent W4 GEMM on the kernel path:
  the next lever needs a bytes-per-row or fused decode-and-multiply
  design, not precision (QWEN-PREFILL-WALL-0).

## Next session

Start here, then BOARD, then the RESULTS tail (the verdict, the
amendment, the pre-reg). Recommended first batch, on GO: the W4 GEMM
half of the prefill wall (a bounded design study before any kernel), and
the README quick-start line for --recurrence loop if Artin wants the
loop documented front-facing.

## Open decisions for Artin

1. Whether README's local-chat section should mention the kernel default
   and --recurrence loop (front-facing voice; not done here).
2. Whether the Muon three-arm reproduction (one 3080 night, banked and
   priced in RIFF) ever gets a slot; this session banked it unrun as
   instructed.
3. Whether the oracle-bootstrap candidate (rank 1 of the transfer banks)
   gets a pre-registration; nothing is armed.

## Hygiene

Receipts force-added by file name: the five stage receipts, the chain
log, the adjudication pair (the 32 per-task CLI receipts are embedded in
cli.json and stay untracked). Smoke receipts of the audited driver under
logs/qwen_gdn_promote/smoke/ were force-added with the registration.
Both checkouts at origin/main after the push at close.

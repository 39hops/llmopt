# Handoff 2026-08-17-3: milestone booked, qualification closed, compact point

Seat: Fable (main session model), Mac. Continues -2 (same day; read
it for the full arc). This is a deliberate compact point.

## Since -2 (each with commit in git log today)

- OBSERVATION QWEN-RUNTIME-0R-SMOKE booked: artifact A (6.50 GiB)
  reconstructs, executes the full 64/48/16 tower, and produced a
  coherent factually-correct 32-token trace ("The answer is
  Paris") at 117.3s/tok on the WSL-host CPU. Receipts forward1_A /
  gen2_A / gen32_A landed + locked. FENCED HARD: descriptive only,
  easiest possible prompt, fluency is not fidelity, proves
  reconstruction+execution and NOTHING about retention.
- AMENDMENT -SMOKE-PROVENANCE: the three receipts carry no
  code_commit (gen32's own stderr proves retired code produced
  it); runtime now prints code_commit at entry; gen32 re-runs once
  at HEAD when convenient.
- OBSERVATION QWEN-RUNTIME-0R-FP16-RETRACTION booked (the arc's
  strongest evidence: representability arguments are not oracles).
- Qualification layer finalized and CLOSED BY AGREEMENT (all
  seats): scripts/check_source.sh = the single definition of
  source-green (CI + /qualify both consume it; wheel/core-deps are
  separate CI jobs); llmopt/lab/{qcodec,qcodec_fast,qartifact,
  qrope}; 24 regression tests (measured, --collect-only); /qualify
  + /rung wiring;
  producer-consumer rule; clean-worktree ritual.
- qrope saga (three consecutive fail-closed guard bugs, all
  caught pre-run): value oracle now takes theta/dim from the
  pinned config's rope_parameters (text_config.rope_theta is
  ABSENT — config smoke caught the AttributeError), checks only
  emitted registered positions, and gates the cos/sin layout probe
  (mrope_interleaved present; layout booked either way, never
  silently acquired).
- Riff banks: weight-reader-as-allocator (+ hierarchical label
  factory, leave-one-layer-out, per-marginal-byte), Eddington/JWST
  instance (fenced: topology-instantiates, mechanism is
  electron-scattering broadening, population contested),
  resident-draft correction, identity-beats-aggregates.
- Receipt lock: local_only class read by the invariant;
  LLMOPT_FULL=1 = evidence-host mode.

## IN FLIGHT

- Teacher v2d (Mac): rollout, silent by design. Watcher on
  jobs/teacher0v2d.rc. ON LOCK, IN ORDER: (1) manifest
  code_commit MUST be 0ca4151 or the lock's use is refused;
  (2) sidecar cached-v-uncached gate (registered: token equality
  primary, max per-position relative L2 <= 5e-3, full tower,
  2-layer smoke disqualified) BEFORE the rollout record is
  accepted; (3) margin-bin census (frozen edges, teacher-only
  counts) books with the teacher. Never claim v2d passed qrope
  (it is frozen on the older call-count gate, prereg-tied).
- 3080: idle, artifacts A/B/C + digests resident, ~180GB free.

## POST-COMPACT AGENDA (Artin, 18:35)

1. HARDER PROMPTS on artifact A — math/physics/quantum-circuit
   questions ("if it scores higher than our math models AT math
   that's interesting"). FENCES that must ride along: chat reads
   never gate anything (registered); any comparison against the
   lab's math-native models is CROSS-MODEL/CROSS-FORMAT color,
   not an instrument (CE-400 is format-bound; the house gate
   measures house crystals); book as descriptive OBSERVATION at
   most. Charter: math/physics/q-circuit prompts are in-scope.
2. CUDA W4 RUNTIME on the 3080 ("no CPU this time") — the
   registered runtime ladder's CUDA leg, pulled forward. Design
   constraints already registered: hard 10GB residency (A 6.5GiB
   fits; C 8.77 leaves ~1.2GB — tight, priced as risk), direct-W4
   decode (pair-LUT unpack is the measured lever, V4-F1e 2.2x;
   entropy coding in the decode path is booked-prohibitive), fp32
   accumulation default, effective compressed-weight bandwidth =
   the reporting metric, backend-agreement KL v CPU reference
   only (never tree quantities — device rule). Speed frame:
   117s/tok CPU reference was decode-per-layer-per-token with no
   weight residency; CUDA with the artifact RESIDENT eliminates
   the streaming entirely — the gap is orders of magnitude, the
   honest number lands where it lands. /qualify applies: golden
   parity fixtures for any CUDA decode against qcodec BEFORE a
   model-scale run (the W4Rows oracle pattern, built for this).
3. Then the critical path unchanged: teacher lock -> sidecar ->
   margin census -> scorer + tree JSON -> A v B v C.

## Open decisions for Artin
1. README front-door paragraph (still pending from -2).
2. Overnight 3080: CUDA runtime build night or teacher-priority.
3. MTP exclusion confirm (carried).

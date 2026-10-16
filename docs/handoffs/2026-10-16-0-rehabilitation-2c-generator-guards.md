# Handoff 2026-10-16-0: rehabilitation phase 2C landed (generator bound, isolated makers, split helper, meter flags, reproduce input errors); pins identical; NOT pushed

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-13-2. No live
registered run at any commit (`liverun status` before each). Nothing
trained, downloaded, deleted or rewritten; the 3080 was not touched
and is not synced. Local main is ahead of origin/main (66456518) by
the commits below plus this handoff; Artin has not authorized a push.

## What landed (commits on main, in order)

1. 39443b2f freeze: `llmopt/mathgen/digest.py` (digest contract),
   `scripts/gen_mathgen_pins.py`, `tests/fixtures/mathgen_pins.json`
   written from the tree at 66456518 BEFORE any generator change: 9
   dataset digests (default 200 @0, 100 @99, ALL_KINDS 60 @5 and 120
   @1, lazy kinds 51 @3, the exclusion skip path, integrate-only,
   L4 mixed), 306 single-maker cases (every kind, levels 1 to 9), 22
   deterministic-retry seeds. `tests/test_mathgen_pins.py` = 338 checks,
   8 s. d2aaa946 registers the script in INDEX/CODEMAP.
2. 6c26adb6 bound: `make_dataset` raises `DatasetExhausted` after
   `100 * len(kinds) * len(levels)` CONSECUTIVE duplicate-or-excluded
   prompts (not a total-attempt cap: the 9000-row integrate training
   shape with 300 excluded rejects 42 % of attempts with a longest run
   of 11). The error carries n, produced, seed, attempts, exclude size,
   per-cell attempt/reject counts and the partial rows. n < 0, empty
   kinds/levels, non-positive cap and unknown kinds fail before any
   generation. Seed mapping `seed * 1_000_000 + i` and the makers'
   `seed + 1_000_003` retries untouched. 23 tests.
3. 99503ed1 isolation: `llmopt/mathgen/isolated.py` (`MakerWorker`,
   `make_problem_isolated`) + `llmopt/mathgen/maker_worker.py`. One
   CPU-only `python -m` child per `make_dataset(wall=)` call, never a
   fork (the oracle_worker lesson), reused across items (0.16 s over
   in-process on the 200-row default). Timeout = SIGKILL + `MakerTimeout`;
   child death = `MakerCrashed` with exit code and stderr tail; maker
   exception = `MakerFailure` with traceback tail; no substitution ever.
   Worker caps its address space at 2 GiB and exits when its parent pid
   changes. 29 tests: hung / crashing / raising makers via
   `tests/fixtures/mathgen_fault_makers.py`, close and context-exit
   reaping, five sequential workers leave no orphan, parent SIGKILL
   kills a hung worker within 6 s, worker started from a parent holding
   live MPS state, worker never imports torch or mlx, the pinned
   default-200-seed0 digest reproduced through the worker.
4. 31873dcc `make_split(n_train, n_eval, ...)`: eval first with
   `eval_seed`, train under `exclude=` with `train_seed`; equal seeds
   rejected. 204 tests incl. 200 metamorphic cases over the 15/24/54-
   prompt cells.
5. d579874f meter: `meter_detailed -> MeterResult(m_bits, kurtosis,
   flags)`; flags constant / single_column / sigma_clamped /
   kurtosis_float64; NaN marks undefined, never a finite stand-in;
   ValueError on empty, non-2-D, non-finite; `.to("cpu")` makes MPS
   input work (CUDA/MPS never worked before: numpy on a device tensor).
   `meter_group` excludes undefined tensors, `meter_group_detailed`
   counts them. Six regression pins bit-exact from the pre-change meter.
   The estimator is documented as biased fourth moment over UNBIASED
   variance squared (Rademacher gives ((N-1)/N)^2, not 1).
6. 28ef5466 reproduce: `ReproduceInputError` / `PinSchemaError` raised
   before any launch (pins, runner, window artifact, contract), main()
   maps them to rc 2 with the expected repo path in the message; pins
   schema validated (64-hex sha, int knobs, known arm prefixes). AST
   test: every environ key read by detbwd_gravmoe + its five scratch
   siblings + window_artifact is in CONTRACT_ENV, and nothing in
   CONTRACT_ENV is dead. First scan missed the alias
   `_env = __import__("os").environ` in detbwd_r2b (DIM/DHEAD/FFN/ACLAMP);
   the scanner now resolves aliases and has its own fixture test.
7. 03e03b12 book: OBSERVATION MATHGEN-CALLER-CENSUS-0 (RESULTS L78193):
   20 caller files, 1 train (guarded) / 19 eval; seed bands share
   371 of 2000 integrate prompts; task_arithmetic.py's 300-row eval holds
   97 integrate rows of which 30 are in int_only_lora's 9000-row train
   set (spec-cited numbers only, no verdict; separate investigation
   recommended, nothing modified); integrate L4 exceeded 15 s on 4 of
   300 seeds, no other sampled cell did. 43f3ddcc, c76b30f9, 1e1ba829:
   CODEMAP/INDEX/SCOREBOARD regenerations and the spec 2C DONE note.

No call site changed. No historical dataset, split, seed or verdict
changed.

## Verification

- Every pre-change pin identical after every commit:
  `tests/test_mathgen_pins.py` 338 passed at each step and inside the
  final check_source run.
- `bash scripts/check_source.sh` at 1e1ba829 (final code tree, log
  `logs/rehab-2c/check_source-1e1ba829.log`): 2063 passed, 14 skipped;
  every generator current; figures 12/12 pixel-identical; ruff clean;
  README in sync; CHECK_SOURCE_RC=0. Two earlier runs at 43f3ddcc and
  c76b30f9 stopped on a stale SCOREBOARD footer and a stale CODEMAP
  (the booking and the spec note added citations); both regenerated
  and committed.
- Sixteen reproduce pins, serial replay from the clean tree at
  28ef5466 (log `logs/rehab-2c/replay-16-*.log`, untracked, kept):
  16 of 16 PASS, every FINAL trajectory sha equal to its pin, 1336 s of
  runner wall (22 min 17 s end to end, 09:01 to 09:23 EDT); the one
  dirty path at launch was the untracked logs/rehab-2c/ directory itself.
- Exhaustion: 1-kind 1-level exclude-all raises in under 2 s (test
  asserts the bound).
- Orphans: `pgrep -fl maker_worker` empty after the isolated suite.

## Found while working

- Low-cardinality cells (distinct prompts over 600 seeds): taylor L1
  15, drec L1 24, taylor L2 54, drec L2/L3 75, dsum L1 203, integrate
  L1 351. Any 1-kind request above those counts looped forever before
  this phase.
- The retry offset 1_000_003 lands in the NEXT seed band: seed-0 and
  seed-1 integrate datasets of 2000 share 371 prompts. Only `exclude=`
  separates; the OBSERVATION books the numbers.
- make_integrate L4 is the one sampled maker that hangs (4/300 over
  15 s, worst finisher 10.6 s); L6/L8/L9 integrate, cint, definite
  integral, taylor, odes, linalg: none in 300 seeds. The L8 fork-
  isolation warning in the L8 docstring is not contradicted (300 seeds
  is a sample), but L4 is where the hazard concentrates in this sample.
- meter: a constant tensor returned (1.0, nan), a single-column tensor
  crashed with "cannot convert float NaN to integer", a 1e30-scaled
  Gaussian returned kurtosis nan from float32 overflow. All three now
  explicit.

## Conditions that bite next session

- `hypothesis` is not installed; the property/metamorphic tests are
  hand-rolled parametrizations (no new dependency added).
- The suite grew from 1425 to 2063 tests (parametrized pins, split and
  isolation cases); wall is about 9.5 min on the Mac.
- `tests/test_mathgen_isolated.py` spawns real subprocesses and one
  test SIGKILLs a parent; on a machine where `os.kill(pid, 0)` on a
  reaped pid behaves differently, read that test first.

## Backlog (own phases, not started)

- TASK-ARITHMETIC contamination check: re-score the sweep on the 67
  clean integrate rows or on `make_split` before citing its numbers.
- Migrating `llmopt/lab/gen.py` (`_gen_isolated`, fork-based) to the
  subprocess worker: frozen-ish lineage with 58 import sites; needs its
  own phase and the adoption guards.
- Selective `jobs/` receipt-lock support (unchanged from 2026-10-13-2).
- Phases 2D to 2H per the spec.

## Next session

1. Push after Artin's GO; then on the 3080: `git pull --ff-only`,
   `bash scripts/check_source.sh` (expect "raster: NOT verified").
2. Phase 2D (adjudicator fence + negative controls).

## Open decisions for Artin

1. Push the 2C commits (and the 2B set still unpushed since 10-13)?
2. Open the task_arithmetic re-score as its own desk, or leave it
   recorded in the OBSERVATION only?
3. Should `make_dataset` callers that build gates at L4+ adopt
   `wall=` (one-line change per site, output identical)? Not done:
   the phase forbade call-site changes.

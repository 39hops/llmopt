# Handoff 2026-10-16-4: 2F preparation, review round (typed name sets, full-extent APFS maps, leak-detecting torch guard, mathnative lineage, .lake analysis); 3080 at 221d170b not qualified; NOT pushed

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-16-3. No live
registered run at any commit. Nothing moved, deleted, relinked,
cloned, compressed or trained; checkpoint baseline identical across
all seven census runs. Artin's review of PLAN.md at 221d170b found
the resolver treating environment and CLI defaults as exhaustive name
sets and asked for six items; all six are addressed below. Local
commits only; no push until Artin authorizes.

## What landed (commits after 221d170b)

1. d9f14a35: autouse fixture restoring the torch determinism flag
   (superseded by the guard in 3).
2. ffbae3d0: `scripts/ckpt_plan.py` typed value sets (literal versus
   default), `expand_typed`, reach classes (exhaustive, default,
   pattern, unreached, unreached-with-generic-loaders), eight
   adversarial tests; `scripts/ckpt_census.py` full extent maps and
   pairwise shared-byte comparison for duplicate groups (two tests);
   `tests/conftest.py` leak-detecting torch guard plus
   `tests/test_torch_state_isolation.py` (four pytester sessions in
   subprocesses).
3. d9107871: `scripts/gen_codemap.py` ignores sha-locked receipts
   under logs/rehab-2e/ (the force-added census sheets had moved 409
   CODEMAP rows to results-cited because an inventory named the
   scripts); CODEMAP regenerated back to results-cited 504; one CPU
   leaker neutralized (test_first_moment_erasure::test_refuse_if_exists).
4. This commit: OBSERVATION CHECKPOINT-PLAN-1 (RESULTS L78619) with
   the change table, the APFS verification, the 3080 result, the
   torch isolation design, the `.lake` analysis and the mathnative
   family lineage table; run7 receipts force-added; spec and BOARD.

## The six items

1. 3080 at 221d170b: every generated-docs, lock, scoreboard and
   figures stage current (raster NOT verified, no Chrome); pytest 1
   failed / 2289 passed / 60 skipped (CUDA bincount under a leaked
   deterministic flag). Not qualified. Fix is local; needs push and
   rerun.
2. Plan soundness: defaults are no longer exhaustive; reach classes
   per file; 123 of 278 labels changed, none toward removal; zero
   files are "unreached" in the strict sense because 26 generic
   loaders exist.
3. Mathnative lineage: the no-site bucket contains booked verdict
   weights referenced by nickname (Gen-6 reborn 71/120 at L2798;
   series 1b-1e and poly rungs L3401-L3758; MATH-CYBER-1 active arm
   L42694). GRPO ladders: consolidated finals are FROZEN; per-cycle
   candidates are historically unique (mps RL is run-level
   nondeterministic) with one unbooked read (2026-08-13 displacement
   ladder). Per-family calls in the entry; nothing approved.
4. APFS: all 35 duplicate groups compared on complete extent maps,
   1,610 extents, 0 shared bytes; 2.48 GB verified distinct.
5. Torch isolation: guard restores determinism, grad mode and dtype;
   reports thread-count changes without resetting; teardown ERROR
   names the leaker unless marked `changes_torch_state`; validated on
   CPU, CUDA half pending the 3080 rerun.
6. `.lake`: 7.75 GB, mathlib build products 6.45 GB, pinned revision,
   nothing unique, regenerable by `lake exe cache get` (or a mathlib
   build if the community cache has dropped the revision); the
   simplest zero-evidence-risk reclamation on the volume; decision
   open.

## Next

- Artin: push authorization for the 3080 rerun; family-level GOs per
  the revised order in CHECKPOINT-PLAN-1 (amendments naming the
  nickname-cited weights first, then duplicates, clones, CANDIDATE
  rows, `.lake`, GRPO partial ladders).
- The amendments naming verdict weights by path are the next
  read-only step and make the census self-sufficient for those files.

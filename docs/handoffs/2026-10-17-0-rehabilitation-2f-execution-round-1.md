# Handoff 2026-10-17-0: 2F execution round 1 (push, 3080 qualified, nickname-cited weights named by path, GRPO families archived to the 3080, mathlib build cache reclaimed); no checkpoint removed

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-16-4. No live
registered run at any point (`scripts/liverun.py status` before
every state change). Checkpoint tree byte-identical to the original
census baseline after everything below (run8 baseline compare:
changed, missing, added all empty).

## Artin's decisions of 2026-10-17 and what happened to each

1. Push the five local commits. Done: `221d170b..2f1b1086` pushed
   to origin/main after interlocks (no live run, clean tree, HEAD as
   expected); origin/main equals local HEAD at the time of the push.
2. Finish the 3080 qualification. Done. The 3080 fast-forwarded to
   2f1b1086 (dirty only in untracked files: COMPOSE_DE, COMPOSE_FLQ2,
   scratch/leancheck/OvershootTest.lean). The first
   `scripts/check_source.sh` pass reached pytest green (2305 passed,
   61 skipped, 0 failed, 477 s) and then stopped at the enforced
   ruff tier because the 3080 venv had no ruff; `ruff==0.16.2`
   (the Mac's version) was installed there and the full script
   re-run: 2305 passed, 61 skipped, ruff enforced tier clean, README
   in sync, `SOURCE TREE GREEN`, DONE marker written. Log pulled
   back as `logs/rehab-2e/check_source-3080-2f1b1086.log`. The
   determinism-flag and CUDA bincount path ran verbosely on the
   RTX 3080 (`logs/rehab-2e/torch_isolation_3080.log`): all four
   `test_torch_state_isolation.py` cases PASSED, including
   `test_cuda_bincount_after_a_leaker_in_the_same_session`, and all
   twelve `test_optimizer_memory_ablation.py` cases PASSED. Nothing
   was skipped or weakened to get there. Remaining
   environment-specific limitation: 61 skips on the 3080 versus 14
   on the Mac (Mac-only operand providers, MLX, Metal); the
   scratch-tier ruff report lists 2 Python 3.11 f-string syntax
   findings in `scratch/first_moment_erasure_ladder.py` on both
   machines (report-only tier, unchanged).
3. Protect the historically unique families. Done, in two parts.
   (a) Append-only amendments naming the nickname-cited weights by
   path, size and SHA-256, verdicts untouched: GEN6-REBORN-WEIGHTS-PATH
   (`mathnative_45m_gen6_reborn.pt`), TERNARY-LATENT-3EP-WEIGHTS-PATH,
   SERIES-POLY-WEIGHTS-PATHS (series, series1b to 1e, poly to poly5),
   MATH-CYBER-1-ACTIVE-EPISODIC-0-WEIGHTS-PATH
   (`mathnative_19m_mw1_active_final.pt`), GRPO-WEIGHTS-PATHS (every
   mathnative GRPO file: finals, cycles, grpo3 cand ladder). Census
   run8 (OBSERVATION CHECKPOINT-CENSUS-1) confirms the effect: 55
   files moved, all toward FROZEN; UNKNOWN fell from 229 / 15.16 GB
   to 179 / 6.84 GB. (b) Copy-only preservation (OBSERVATION
   CHECKPOINT-ARCHIVE-0): the 52 `*grpo*` checkpoints (8.58 GB) were
   rsync'd with `--ignore-existing` to the new directory
   `/home/a/archive/llmopt-ckpt-2026-10-17/grpo/` on the 3080
   (outside the git checkout; the parent did not exist before).
   `sha256sum` on the 3080 over all 52 files matched the Mac
   manifest line for line; both manifests are beside the archive
   and in `logs/rehab-2e/grpo_archive_manifest_{mac.txt,3080.sha256}`.
   None of the 52 digests existed on the 3080 before. 3080 root
   volume 719 GB free before, 711 GB after. No Mac file changed.
4. Conditional GO for `.lake` build-cache reclamation. Executed
   (OBSERVATION LAKE-RECLAIM-0). Inspected
   `scratch/leancheck/.lake/packages/mathlib/.lake/build`: 124,659
   files, 6.53 GB logical, by extension hash/trace/olean/ilean/c/
   sig/server/private/ir plus 10 json, 10 export, 1 rsp, zero
   `.lean`. Regeneration established before removal: toolchain
   `v4.33.0-rc2` installed under `~/.elan`, mathlib source checkout
   at the manifest's pinned rev `9fb10993`, and the pinned `Cache`
   tool resolved `Mathlib/Tactic/Ring.lean` to an artifact the
   public cache still serves (HTTP 200 from `mathlib4-master`).
   Preconditions at removal: no live run, no lake or lean process.
   Removed only `mathlib/.lake/build` plus `mathlib/.lake/packages`
   and `mathlib/.lake/config`, the last two being 0.54 GB of
   dependency clones that the lookup command itself created minutes
   earlier by running lake inside the package directory. Kept:
   lakefile.toml, lake-manifest.json, lean-toolchain, Certs.lean,
   the mathlib source tree, every other package tree with its build
   outputs, `~/.cache/mathlib`. Mac free space 55.66 GB before,
   62.47 GB after (62.48 GB at handoff time). Full pre-removal
   manifest: `logs/rehab-2e/lake_build_manifest.txt` (untracked,
   124,659 lines). Consequence: run `lake exe cache get` in
   `scratch/leancheck` before the next `scratch/lean_check.py` use.
5. No checkpoint cleanup GO. Honoured: no duplicate, CANDIDATE,
   GRPO ladder or frozen pair was removed, relinked, cloned or
   replaced. run7 remains the decision-sheet baseline; run8 is a
   verification run only.
6. Finish and report. Mac `check_source.sh` on the working tree:
   first pass found only a stale SCOREBOARD (regenerated); final
   pass result is recorded in the commit message of the closing
   commit. Ledger guards (`pytest -m docs`) green after every
   booking. No push beyond the initial five commits.

## Census defect fixed on the way

The first run8 attempt froze `mathnative_gen6_ternary_grown(.latent).pt`
through "cited receipt path in logs/gen6_ternary_grown.log": the
only text naming that log is the lineage table inside OBSERVATION
CHECKPOINT-PLAN-1, a triage entry that the checkpoint-citation scan
already strips, but `log_citation_sets` built its set from the
unstripped RESULTS. Fixed in `scripts/ckpt_census.py` (strip before
building the log-citation set) with a failing-then-passing test,
`test_log_cited_only_inside_a_triage_entry_does_not_count`. The
draft MATH-CYBER-1 amendment had also named the two smoke files
beside the arm's weights; the census froze them, so the uncommitted
draft was edited to describe without naming them. The booked run8
numbers are from the corrected run.

## State at close

Mac HEAD: see `git log` (this handoff's commit and the CODEMAP
regen after it). origin/main at 2f1b1086; the commits after it are
local until Artin authorizes a push. 3080 at 2f1b1086, qualified.
Mac free space 62.48 GB. Checkpoint tree 1,159 files / 70.51 GB,
identical to the original baseline.

## Next (recommendation, not armed)

Shortest path back to research reproduction rather than another
storage phase: (1) push these commits and ff the 3080 so both
checkouts carry the census fix and the amendments; (2) run
`lake exe cache get` only when a Lean kernel check is next needed;
(3) pick one booked verdict with a frozen driver and its named
weights (the MATH-CYBER-1 ACTIVE-EPISODIC arm or the Gen-6 reborn
record are now fully path-pinned) and reproduce its readout end to
end as the first re-entry rung. The remaining storage items
(duplicates 0.09 GB, CANDIDATE 0.85 GB, 29 frozen pairs 2.48 GB,
HOLD rows, GRPO ladders now archived off-Mac) are all documented in
run7 and can wait for a single per-family GO later; none blocks
research.

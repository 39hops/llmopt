# Handoff 2026-10-16-2: rehabilitation phase 2E landed (read-only checkpoint census, APFS extent probe, family decision sheet); OBSERVATION CHECKPOINT-CENSUS-0; nothing moved or deleted

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-16-1. No live
registered run at any commit. Nothing trained, downloaded, deleted,
moved, cloned or rewritten; the checkpoint tree's baseline manifest
(1,159 paths, sizes, sha256) is identical before and after every
census run. Phase 2D was pushed at Artin's GO (origin 1ac27524) and
the 3080 was fast-forwarded to it; its check_source stopped in the
figures step because that venv lacks matplotlib (environment gap,
not code; everything before it, including the generated-docs and
receipt-lock checks, passed). Artin authorized a push at the end of
2E.

## What landed (commits on main, in order)

1. 75cf74e7 `scripts/ckpt_inventory.py` + tests; 12eb0c39 renamed
   to `scripts/ckpt_census.py` / `tests/test_ckpt_census.py`
   (scratch/ckpt_inventory.py already exists; the CODEMAP move gate
   needs one row per basename); eb08ecde INDEX/CODEMAP.
2. 1099305b, 2da20020, a43dbd8c: classifier refinements each driven
   by a full run on the real tree. Generic loaders ("checkpoints/" +
   name, os.path.join, f-strings with no literal fragment: 24 sites)
   are listed, not applied per file. A dynamic prefix with a literal
   fragment (`checkpoints/mathnative_${name}.pt`, globs directly under
   checkpoints/: 127 sites) makes every matching file UNKNOWN and is
   listed per family with the script's CODEMAP class. A log mention
   is a FROZEN reach only when the log is sha-locked or RESULTS-cited;
   an uncited mention is a note (the census's own listings and the
   2026-08-07 triage inventories under logs/triage/ name every path
   and had turned the whole tree UNKNOWN). A byte-identical copy of a
   retained file is DUPLICATE even under a dynamic prefix, with the
   reaching scripts kept as dependencies.
3. This commit: OBSERVATION CHECKPOINT-CENSUS-0 (RESULTS L78395) with
   the family decision sheet; four small receipts force-added under
   logs/rehab-2e/run4/ (SHEET.md, families.json, duplicates.json,
   summary.json; the 1.4 MB inventory and 1.9 MB graph stay
   local-only and are listed in the lock); spec section 2E marked
   done; BOARD repointed.

## Numbers (all in the entry)

- Container free 57.2 GB (audit 2026-10-13: 13 GiB). No local
  snapshots. Repository 0.44 GB tracked, 82.09 GB untracked.
- checkpoints/: 1,159 files, 70.51 GB logical = allocated; no
  clones, hard links or symlinks. FROZEN 550 / 46.77 GB, ESSENTIAL
  327 / 4.66, DUPLICATE 6 / 0.12 (distinct blocks, verified),
  CANDIDATE 42 / 0.85 (estimate), UNKNOWN 234 / 18.10 (all by
  dynamic prefix; mathnative_ 81 files / 13.64 GB).
- 35 duplicate groups, 2.60 GB redundant, every group on distinct
  device offsets (F_LOG2PHYS_EXT, packed 20-byte log2phys; the
  natural-alignment layout returns garbage). 29 groups / 2.48 GB are
  frozen-with-frozen: a lossless clone opportunity that needs a GO.
- Independent copies: 3080 holds 52 identical files (all 48 msearch,
  6 mathnative, 3 gallery19m, 3 proposer, 1 quat); the repair worktree
  holds the 45 tracked files plus the untracked atomtraj1 snapshots.
- 23 RESULTS-cited checkpoint paths absent from the Mac: atomtraj1
  (repair worktree), prose fragments, deleted smoke dirs, fmel1 (never
  ran). No new evidence gap.

## Other trees measured (sized by du, read-only)

~/code/llmopt-repair 8.01 GB (7.76 GB checkpoints);
~/code/llmopt-audit-2026-10-13 5.6 MB (the audit's inventory tooling,
reused conceptually); scratch/leancheck/.lake 7.75 GB; .venv 1.43,
.venv-k2 0.67, .venv_teacher 0.62, .venv-anim 0.25 GB; 3080
~/code/llmopt 41.13 GB (415 checkpoint files, 11.28 GB), llmopt_dump
1.92 GB (venv + data), llmopt_wsl_backup empty, 720 GB free there.

## Limits

- The scanner sees paths, directories, basenames, stems of eight or
  more characters, sidecar parents and locked or cited logs. It does
  not see rjob tags or prose nicknames. CANDIDATE is a triage queue.
- UNKNOWN is a pattern reach, not a verified dependency: the loops
  that build `checkpoints/mathnative_${name}.pt` produce a finite
  name set the census does not evaluate.
- Partial extent sharing of unique-content files is not probed;
  their sizes are estimates. Snapshot-retained blocks: none exist.

## Recommended 2F order (approval boundaries in the entry)

1. Resolve the 127 dynamic-prefix sites to exact name sets, re-census
   (read-only).
2. Relink four scratch readers to checkpoints/confirmed/, remove the
   six DUPLICATE rows (0.12 GB, per-row GO).
3. Clone the 29 frozen pairs (2.48 GB lossless, one GO, sha and mtime
   check after).
4. Decide the 0.85 GB CANDIDATE list family by family.
5. Separate decisions: .lake 7.75 GB regenerable; repair worktree
   atomtraj1 snapshots 7.6 GB (BOARD: keep until a handoff decides);
   secondary venvs 1.54 GB; 3080 venv install of matplotlib so its
   check_source can finish.

## Open decisions for Artin

1. Push 2E (authorized in the brief; done at the end of this session
   if nothing fails).
2. GO or hold on 2F steps 2 and 3.
3. Whether to install matplotlib in the 3080 venv.

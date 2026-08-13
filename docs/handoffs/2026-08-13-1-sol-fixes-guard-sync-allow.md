# Handoff 2026-08-13-1 — Sol 5.6 fixes landed, sync auto-allow, redesign queue banked

Seat: Fable 5, Mac. HEAD at close: `fec9665` (plus this commit).
pytest: 848 passed / 11 skipped rc=0. CI green on every batch
through `fec9665`. Three checkouts lockstep. Supersedes -0 for
session state; -0 carries the overnight program detail.

## Since handoff -0

- `03aa3ed` wsl_guard: the ff-only sync + hash assert is now
  hook-ALLOWED in the exact shape `cd … && git pull --ff-only
  [>/dev/null 2>&1] && git rev-parse … [| cut …]` (Artin grant
  02:02). Any other segment still asks. 5-shape test receipt in the
  commit body.
- Sol 5.6 external review (Artin-relayed) — ruling adopted:
  - `ce24884` honesty-ledger drift FIXED (figures.json 54/217 ->
    55/218) + NEW guard test pins figures.json honesty parts to the
    gen_readme recount (drift class now CI-caught); README "Four
    results" now actually shows four figures (loss_floor split into
    its own block); 12 web figures regenerated.
  - `fec9665` the 12-item redesign queue:
    docs/superpowers/plans/2026-08-13-sol-figure-redesigns.md.
    RULING: no README GIF until the three scenes are redesigned;
    crystal-rotation then the only candidate.
- Artin removed the Stop-hook plugin (no-vibes closeout shapes no
  longer enforced harness-side).

## Next session, in order

1. Sol redesign queue, scenes first (training_morph fixed-coords
   redesign is the misleading one — do it first).
2. THEORY L4 quote (Artin queued).
3. Phase C on the frozen-paths ruling (181-file headline, handoff -0).
4. Rebirth MEDIUMs on per-family GO.

## Open Artin decisions (unchanged from -0 unless noted)

1. Phase C frozen-paths ruling (shim/symlink v spec amendment).
2. Rebirth MEDIUM substitutions per family.
3. README GIF: RESOLVED for now — none until scene redesigns land.
4. Permissions allow-list via /permissions (optional; sync no longer
   needs it).
5. Branch protection; house-crystal KEEP recommendation stands.

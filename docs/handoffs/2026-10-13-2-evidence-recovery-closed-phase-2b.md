# Handoff 2026-10-13-2: evidence recovery closed (pincer labels restored through the new recovery operation), OOM logs inventoried, phase 2B figure pipeline landed; NOT pushed

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-13-1. No live
registered run at any commit. Nothing trained, downloaded, deleted or
rewritten. Local main is six commits ahead of origin (c9acaf1d..this);
Artin has not authorized a push for these.

## What landed

1. c9acaf1d receipt-guard: `scripts/recover_receipt.py`, the one write
   to a booked receipt path the guard allows. Refuses unless the
   destination is a results-cited receipt the lock classifies absent,
   nothing exists there, the source hashes to --sha256, that digest is
   already booked in RESULTS, no run is live, and the checkout is the
   primary worktree; O_EXCL write, re-hash, record in
   `docs/receipt-recoveries.jsonl`. The hook exempts exactly the bare
   invocation and denies any ride-along. 25 adversarial tests.
2. ce8bd0ff book: AMENDMENT PINCER-LABELS-RECOVERED-2026-10-13
   (RESULTS L78146). `logs/pincer/labels_v2.jsonl` restored through the
   operation (digest equal on both machines, 203/95/2 class counts equal
   to the verdict, sympy strings only), force-added, sha-locked. Lock:
   933 paths, 905 locked, zero results-cited receipts absent.
   LEGACY_ABSENT is the empty set. The two OOM-poisoned MICRO-STAR-1
   logs are hashed and sized in the entry and classified `remote-only`
   in docs/EVIDENCE.md (new status, enforced by the evidence test);
   they stay on the 3080, outside git, not disposable.
3. c6af8677 figures (2B-1): `llmopt/figures/spec.py` is the one
   serializer and FINDINGS recount for both generators;
   `gen_figures_web.py --check` re-renders SVGs, checks PNG headers,
   and compares PNG pixels against a fresh Chrome capture when Chrome
   exists (never timestamps); without Chrome it prints "raster: NOT
   verified"; check_source.sh runs it and its comment says a green run
   without Chrome certifies SVGs and headers only. The ten August PNGs
   were re-captured (48 to 115 anti-aliasing pixels off from Chrome 155,
   max delta 12); `docs/assets/web/CAPTURE.json` records the renderer.
   A fence needing a third line now raises instead of cropping.
4. 5923250c figures (2B-2): `fence_ids` on every figure, checked
   against the index by `test_docs_integrity`; merge_space fence names
   -3 and -4; routing_crest fence states 66.33 drawn as 66. Four PNGs
   re-rendered, pixel diff confined to footer rows; no value changed.
5. e7ee8c27 tests (2B-3): `tests/test_figure_values.py` re-reads every
   transcribed value from the RESULTS entry its fence_ids name.

## Verification

- `bash scripts/check_source.sh` on the tree at e7ee8c27: 1425 passed,
  14 skipped; every generator current; figures 12/12 pixel-identical;
  ruff clean; README in sync; CHECK_SOURCE_RC=0.
- Both generators run in each order on the real tree: `git status`
  clean after both orders.
- Mutation drill (restored before commit): one-digit loss_floor edit
  failed test_figure_values and `gen_figures_web --check`; a wrong
  fence id failed the docs test.
- 3080 at d8e61caa: the three files re-hashed there equal the booked
  digests; frozen-receipt and evidence suites pass at that HEAD. The
  suites at the new HEAD run there only after a push.

## Found while working

- Pillow's `getbbox()` on an RGBA difference image inspects alpha only,
  so a naive pixel comparison calls any two opaque captures equal; the
  comparator uses all four channels and a test (dark PNG copied over
  light) pins it.
- Chrome PNG bytes are not stable across Chrome versions but pixels
  are stable on one version; exact comparison plus the recorded
  renderer version is the policy.

## Backlog (own phases, not started)

- Selective `jobs/` receipt-lock support: `jobs/gt1_r6.log` stays
  protected by `test_evidence_availability` (pinned sha); extending the
  lock scan to jobs/*.log (39 RESULTS citations) changes the fresh-clone
  check_view and needs its own design and commit set.
- Phases 2C to 2H per the spec, unchanged.

## Next session

1. Push after Artin's GO; then on the 3080: `git pull --ff-only`,
   `bash scripts/check_source.sh` (expect "raster: NOT verified" there).
2. Phase 2C (generator guards, seeds frozen, pinned make_dataset
   digest first).

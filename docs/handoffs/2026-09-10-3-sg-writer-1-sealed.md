# Handoff 2026-09-10-3: SYNTHETIC-GRADIENT-WRITER-1 qualification SEALED, implemented, tested, smoked, audited twice; nothing launched

Seat: Fable 5.1 on the Mac. HEAD at close: the commit carrying this
file. 3080 untouched. No live registered run.

## What landed (after handoff 2026-09-10-2)

- AMENDMENT SYNTHETIC-GRADIENT-WRITER-1-SEAL (RESULTS L70425, commit
  3610d490): arena at seed 27 sealed; paired frozen-BP control c,
  ADEQUATE-CONTROL c >= 24, band c +- 7, ordered ladder LINEAR 3e-4 /
  LINEAR 3e-5 / MLP-256 3e-4 / MLP-256 3e-5, FIRST FUNCTION-MATCH
  stops and freezes the recipe, alignment descriptive only; FOLD A
  (same-pre-update teacher order, nine step semantics) and FOLD B
  (elementwise normalized MSE over label positions and hidden dims,
  mean over blocks); prior re-evaluated under FOLD B.
- AMENDMENT -SEAL-AUDIT (L70631, commit 893dd61f) and -SEAL-AUDIT-2
  (L70762, commit 15d9ff24): two clean-tree Opus audits, three
  blockers folded (a spliced control-loss number; the pre-birth
  integrity smoke unimplemented / tree-dirtying; driver.log receipt
  with no producer) plus should-fixes. The one scientific fold:
  synthetic credit is applied at eligible label positions only
  (delta^BP is exactly zero at pad positions under the driver's
  right-aligned padding); check (k) added. Zero-predictor baseline
  logged beside pred_mse every logged step; prior restated against it.
- Instrument: scratch/sg_credit.py (credit law, parameter-detached
  functional_call teacher, LINEAR / MLP-256 zero-output predictors,
  normalized MSE + baseline, run_sg_step), scratch/birth19m_sg.py
  (sibling of the FB driver; SG=1 seed 27; MODE=zero control / MODE=sg
  FAMILY PLR; predictor snapshots; sg_log with pred_mse, baseline_mse,
  alignment per block; SMOKE_TAG for re-smokes), scratch/sg_qualgate.py
  (gates + pure ladder law adjudicate(): NOT-RESOLVABLE-CONTROL /
  FUNCTION-MATCH / ACCESSIBILITY-ONLY / LADDER-UNSTABLE),
  scratch/sgwriter1_qual_driver.sh (driver.log via exec-tee; integrity
  smoke FIRST with INTEG_TAG=_launch; control, gate, cells in order,
  gate after each, stop on ladder.json), scratch/sg_integrity_smoke.py
  (checks a..k; per-commit untracked output). Tests 22 across four
  files. Prereg mirror docs/preregs/synthetic-gradient-writer-1.json.
- Smokes: five 300-step cells at seed 11 twice (pre-fold at
  logs/sgwriter1/smoke.jsonl; post-fold at smoke_seal2.jsonl, dirty
  tree at cb2b441c, disclosed), all finite, freeze law verified,
  smoke ladder NOT-RESOLVABLE-CONTROL on 0/0 as the law requires.
  Integrity smoke 4/4 on the pre-fold receipt (integrity_smoke_seal.
  jsonl) and 4/4 on clean 893dd61f (development run, deleted; the
  launch rerun re-receipts it). Walls: SG cell 61 to 62 min, control 32
  min, gates 1 to 2 min, worst case about 4.8 h; storage 7.0 GB; disk
  25 GiB.

## Launch (on a SEPARATE Artin GO only)

```
.venv/bin/python scripts/liverun.py run sgwriter1q -- bash scratch/sgwriter1_qual_driver.sh
```
Preconditions checked by the driver: clean tree, driver.log absent,
integrity smoke passes on the launch commit, control first. Receipts
to force-add at booking: logs/sgwriter1/qual.jsonl, ladder.json,
selection.json, gate.log, driver.log, train_control.log,
train_sg_{linear,mlp256}_plr{3e-4,3e-5}.log,
integrity_smoke_launch.log, integrity_smoke_<launch commit>_launch.jsonl,
logs/liverun/sgwriter1q.jsonl.

## Conditions that bite next session

- Do not run scratch/sg_integrity_smoke.py by hand at the launch
  commit without INTEG_TAG (the driver's file name is
  <commit>_launch; a by-hand run without a tag is a different name,
  so no collision, but the dev file would be unregistered exhaust).
- The PreToolUse hook refuses any command whose text contains a
  booked receipt path being mutated; re-smokes need SMOKE_TAG paths;
  patch scripts go through /tmp files.
- Index regen quirk persists (act-envelope row); slug ids of the
  SEAL-AUDIT entries are ...-seal-b and ...-seal-b-b.
- checkpoints/sgwriter1_smoke/ and sgwriter1_smoke_seal2/ (1.1 GB
  each) are smoke exhaust; prune on a housekeeping gate.

## Open decisions for Artin

1. GO for the seed-27 qualification ladder (about 4.8 h worst case,
   up to five births, first-match stop).
2. Whether the pre-fold smoke checkpoints (sgwriter1_smoke,
   sgwriter1_smoke_seal2, frozenbb1_smoke) may be pruned now.

## Next session: where to start

This handoff, BOARD line 5, RESULTS tail from L70425 (SEAL, SEAL-AUDIT,
SEAL-AUDIT-2), docs/preregs/synthetic-gradient-writer-1.json.

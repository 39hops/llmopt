# Evidence availability for the front-page results

Snapshot taken 2026-10-13 on the Mac checkout. This page complements
[RESULTS.md](RESULTS.md): the ledger says what was measured and how it
was adjudicated; this page says what a reader can still put their hands
on, where it lives, and whether it can be re-run here. It is not a
verdict and changes none. `tests/test_evidence_availability.py` checks
the pinned-artifact table below against the checkout on every suite
run, so a status here cannot silently outlive the file it describes.

Vocabulary, used in every table:

- **tracked**: in git, present in every clone, sha256 pinned here and,
  for `logs/` paths, in [receipts.lock.json](receipts.lock.json).
- **local-only**: present on the Mac checkout, gitignored, sha256
  pinned here; a fresh clone does not have it.
- **absent**: cited by a booked entry, present on no checkout the lab
  can reach today.
- **re-run here**: whether a meaningful reproduction is possible on
  this Mac now, and in what shape (digest-exact, tolerance, blocked).

A sha256 in a lock or in this table is a fingerprint, not a backup:
it lets a reader detect a changed or substituted file, it does not
deliver the file. Only **tracked** artifacts travel with the repo.

## Summary

| chain | README claim | verdicts | receipts | checkpoints / inputs | re-run here |
|---|---|---|---|---|---|
| routing crest | masked 45.3% keep beats the paired full model, +17/+9/+18 at three seeds; random and anti masks 0/120 | R5 L18927, R6 L19108 | tracked and locked (R5, R6 rows); R6 driver log tracked | keep-set local-only; 30B vehicle not cached | blocked: vehicle download (about 17 GB) exceeds free disk |
| merge_space | six independent-pair averages gate exactly zero; same-init merges land in the parent band at n=3 | MERGE-SPACE-1 L26770, -5 L27158 (values also from -3 L26947, -4 L27013) | absent on every checkout | absent on this Mac | not possible here: 3080-scoped births, no artifacts |
| loss_floor | floor 0.4364 to 0.3478 nats across d64 to d512, never near H_32 = 0.187 | FLOOR-HK-1 L27055 | tracked and locked since AMENDMENT L77881 | four checkpoints local-only | tolerance re-birth only (mps is run-level nondeterministic); not attempted |
| effective_context | deep positions improve about one nat from k=16 to k=128 at every width; bar 2 did not fire | KEFF-PROBE-1 L27332 | tracked and locked since AMENDMENT L77881 | the four loss_floor checkpoints, local-only | yes while the checkpoints survive: deterministic CPU loss readout, minutes; not re-run this phase |

The sixteen reproduce pins (`llmopt.reproduce`) are the contrast case:
driver, window artifact and contract digest are all tracked, and all
sixteen replayed digest-exact on this Mac on 2026-10-13. They certify
trajectory identity, not any front-page number.

## 1. Routing crest (MOE-GT-1-R5 / R6)

- Claim and scope: on `mlx-community/Qwen3-30B-A3B-4bit`, keeping the
  top 45.3% math-demand experts per layer beat the paired full model on
  the 120-item mathgen L1-3 gate at seeds 111/222/333 (+17/+9/+18,
  pooled +14.7 against a registered +7 bar); two random 45.3% masks
  and one anti-demand mask scored 0/120 at seed 1234. Gate claim on one
  vehicle, one keep rule, one domain; the controls are not seed-paired
  to the R5 arms.
- Registration and verdict: PRE-REG MOE-GT-1-R5 (L18912) before VERDICT
  R5 (L18927); PRE-REG R6 (L19054) before VERDICT R6 (L19108). Prose
  adjudication (predates `docs/preregs/`).
- Required to reproduce: `scratch/moe_gt1.py` (tracked), the keep-set
  `checkpoints/moe_gt1_arm0.json` (local-only, the only input that
  needs arm 0 re-run on the vehicle to regenerate), the vehicle itself
  (not cached on this Mac), the mathgen gate (tracked generators, seeded).
- Receipts: `logs/opus/moe_gt1.jsonl` and `logs/opus/moe_gt1_perprob.jsonl`
  (tracked, locked); `jobs/gt1_r6.log` (tracked since AMENDMENT L77881,
  outside the lock's `logs/` scan, sha pinned below). The R6 rows carry
  no `rule`/`ruleseed` field; arm identity is read from the driver log
  order and the open-loop recall values the verdict quotes.
- Re-run here: blocked. The vehicle is a 17 GB download and the volume
  had 13 GiB free on 2026-10-13; the one-resident-30B rule also applies.
  Six gate runs (three seeds plus three controls) take about 30 to 40
  minutes once the vehicle is present.

## 2. merge_space (MERGE-SPACE-1 to -5)

- Claim and scope: four independently born d64 micro-star models gated
  12/23/30/28 of 120; all six pairwise weight averages gated exactly 0 at
  every level; same-init merges landed in the parent band at n=3 paired
  seeds; a 4-way order-twin soup gated 14. RTX 3080, d64, one family.
- Registration and verdict: PRE-REG MERGE-SPACE-1 (L26602) before VERDICT
  (L26770); PRE-REG -5 (L27134) before VERDICT (L27158); -3 (L26947) and
  -4 (L27013) supply the same-init 13 and soup 14 values the figure
  shows. The figure fence names -1 and -5 only; naming -3/-4 there is
  deferred to the figure-pipeline phase so the rendered assets and the
  JSON change together.
- Required to reproduce: `llmopt/lab/merge.py` and the micro-star birth
  recipe (tracked); the four parent checkpoints and the merged models
  (none on this Mac); the driver logs.
- Receipts: `logs/merge_space{1..5}/driver.log` and
  `logs/microstar/microstar_run.log` are cited by the verdicts and
  recorded `exists: false` in the lock; they are in the pinned legacy
  absent set of `test_frozen_receipt_mutation.py`. Only prose and the
  16-hex weight digests in the verdicts remain.
- Status on 2026-10-13: the 3080 was unreachable, so whether the files
  survive there is unknown. No amendment was booked for the absence;
  the verdicts stand as booked and are classified here as not locally
  verifiable. If the files are recovered they must be force-added and
  the legacy set shrunk, never re-created.
- Re-run here: not possible on this Mac (3080-scoped births; cross-device
  comparison forbidden by house doctrine).

## 3. loss_floor (FLOOR-HK-1)

- Claim and scope: final-epoch training loss descends monotonically
  0.4364 / 0.3815 / 0.3566 / 0.3478 nats at d64 / d128 / d256 / d512 and
  stays far above the k=32 corpus entropy 0.187. Mac, mps, fp32, n=1,
  one diet.
- Registration and verdict: PRE-REG FLOOR-HK-1 (L26641), corrected by
  AMENDMENT LOSS-FLOOR-1-ARCH (L26670), VERDICT L27055.
- Required to reproduce: `scripts/train_mathnative.py` via
  `scratch/floor_hk1.sh` (tracked); the diet (regenerated from seeded
  generators); the four birth logs and gate logs; the four checkpoints.
- Receipts: nine `logs/floor_hk1/*.log` files, tracked and sha-locked
  since AMENDMENT FRONT-PAGE-RECEIPTS-2026-10-13 (L77881). Each birth
  log carries the booked loss verbatim; each gate log opens with the
  driver's weight digest booked in the verdict table.
- Checkpoints: `checkpoints/floorhk_d{64,128,256,512}.pt`, local-only,
  179 MB total, outside the lock, file-level sha256 pinned below.
- Re-run here: a digest-exact re-birth is not a valid target (Mac mps
  float training is run-level nondeterministic at fixed seed, AMENDMENT
  SOFT-SPEED-1-PRECONDITION). The valid shape is a tolerance re-run
  (loss within about 0.01 at epoch 2, monotone in width), hours of wall
  clock; not attempted.

## 4. effective_context (KEFF-PROBE-1)

- Claim and scope: on the four loss_floor checkpoints, loss at positions
  past 128 falls by 0.90 to 1.25 nats as the visible window grows from
  16 to 128 tokens; the registered second bar (width separation grows
  with k) did not fire. 400 fixed rows, CPU, loss only, no gate.
- Registration and verdict: PRE-REG KEFF-PROBE-1 (L27239) before VERDICT
  (L27332), which also records an in-session claim correction.
- Required to reproduce: `scratch/keff_probe.py` (tracked), the four
  checkpoints above (local-only), the seeded probe rows.
- Receipts: `logs/floor_hk1/keff_ladder.log`, tracked and locked since
  L77881; all 24 figure values appear in it verbatim.
- Re-run here: yes, in minutes, while the checkpoints survive; the probe
  is a deterministic loss readout. Not re-run this phase.

## Pinned artifacts

The table `tests/test_evidence_availability.py` reads. One row per
file; status is one of `tracked`, `local-only`, `absent`; sha256 is the
full digest or `-` for an absent file.

| path | status | sha256 |
|---|---|---|
| `logs/opus/moe_gt1.jsonl` | tracked | (locked) |
| `logs/opus/moe_gt1_perprob.jsonl` | tracked | (locked) |
| `jobs/gt1_r6.log` | tracked | c7ecd2fb2467f1f927dfbb792490d549c1dd4b750aaa1aa00f9181329b92650b |
| `checkpoints/moe_gt1_arm0.json` | local-only | d83a27eb9e206c0ed1d539675299885e570c1e90fe54ed78ba6a81cf252215b1 |
| `logs/floor_hk1/birth_d64.log` | tracked | (locked) |
| `logs/floor_hk1/birth_d128.log` | tracked | (locked) |
| `logs/floor_hk1/birth_d256.log` | tracked | (locked) |
| `logs/floor_hk1/birth_d512.log` | tracked | (locked) |
| `logs/floor_hk1/gate_d64.log` | tracked | (locked) |
| `logs/floor_hk1/gate_d128.log` | tracked | (locked) |
| `logs/floor_hk1/gate_d256.log` | tracked | (locked) |
| `logs/floor_hk1/gate_d512.log` | tracked | (locked) |
| `logs/floor_hk1/keff_ladder.log` | tracked | (locked) |
| `checkpoints/floorhk_d64.pt` | local-only | a5f256eeb7f31006f21ede19cac1f5205292e9d3b0c7de9cdc600982555584b0 |
| `checkpoints/floorhk_d128.pt` | local-only | 52bdcaae2f8b5758aa06f5876dd97ac4b276a23aa2e594d17b2080b9a4b59f7c |
| `checkpoints/floorhk_d256.pt` | local-only | fc23e334a50dd315e2b1e20c2760413c05adcaea03c1f1d5c06b2e80f5627505 |
| `checkpoints/floorhk_d512.pt` | local-only | be244829f5c3414e198b317ff28a609084868a10c80aa78028157d14c5474fe3 |
| `logs/merge_space1/driver.log` | absent | - |
| `logs/merge_space2/driver.log` | absent | - |
| `logs/merge_space3/driver.log` | absent | - |
| `logs/merge_space4/driver.log` | absent | - |
| `logs/merge_space5/driver.log` | absent | - |
| `logs/microstar/microstar_run.log` | absent | - |

`(locked)` means the digest is read from `receipts.lock.json` rather
than typed twice. A `local-only` row whose file is missing is reported
by the test and fails only on the evidence host (`LLMOPT_FULL=1`); an
`absent` row whose file reappears fails the test until this page is
updated, so recovery is a visible, reviewed event.

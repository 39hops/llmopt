# Evidence availability for the front-page results

Snapshot taken 2026-10-13 on the Mac checkout, updated the same day
after the merge_space recovery from the 3080. This page complements
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
| merge_space | six independent-pair averages gate exactly zero; same-init merges land in the parent band at n=3 | MERGE-SPACE-1 L26770, -5 L27158 (values also from -3 L26947, -4 L27013) | tracked and locked since AMENDMENT L77967 (recovered from the 3080) | 33 checkpoints and 16 merge sidecars local-only (copies; originals on the 3080) | births 3080-scoped; gate side re-runnable in-family against the recovered checkpoints; not re-run |
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
  `logs/microstar/microstar_run.log` (the six cited receipts) plus the
  47 per-arm birth and gate logs under the same directories, tracked
  and sha-locked since AMENDMENT MERGE-SPACE-RECEIPTS-RECOVERED-2026-10-13
  (L77967). They were recovered from the 3080 checkout later on
  2026-10-13: originals untouched since 2026-08-11, 103 of 103 sha256
  digests identical on both machines, copied byte-for-byte. Every gate
  dict in the six verdicts and all 30 booked 16-hex weight digests
  appear verbatim in them (cross-check recorded in the amendment).
- Checkpoints: the 16 parent births, 16 merged models and the
  MICRO-STAR-1 d64 model (`checkpoints/msearch_*.pt`,
  `checkpoints/microstar_d64.pt`, 70 MB) are local-only copies on
  this Mac, originals on the 3080; the 16 `*.pt.merge.json` sidecars
  name each merge's inputs with full sha256, and all 33 input digests
  match the parent files. File-level sha256 pinned below.
- Re-run here: a Mac re-birth is not a valid comparison (3080-scoped
  births; cross-device comparison forbidden by house doctrine). The
  gate side is re-runnable on either machine against the recovered
  checkpoints, within the family only; not re-run.

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
| `logs/merge_space1/driver.log` | tracked | (locked) |
| `logs/merge_space2/driver.log` | tracked | (locked) |
| `logs/merge_space3/driver.log` | tracked | (locked) |
| `logs/merge_space4/driver.log` | tracked | (locked) |
| `logs/merge_space5/driver.log` | tracked | (locked) |
| `logs/microstar/microstar_run.log` | tracked | (locked) |
| `checkpoints/microstar_d64.pt` | local-only | 91d02b2b46c962e939f56ae03bb94d8c6a7b3a195fa8393b6c4879c73942f542 |
| `checkpoints/msearch_1a.pt` | local-only | 446bbc305b095c5eee24a8c38afbb4444ee3469fa64864e9af31b7cc8bab8908 |
| `checkpoints/msearch_1b.pt` | local-only | fe09b0f45425b0bf4fb675b5cce7762088be24bd1cb9dac399540a52aea7b9cf |
| `checkpoints/msearch_2a.pt` | local-only | 326a62a0e11c1e6e52efaab31e66a1c032efb4cadc442c7a9f075ee915692c9f |
| `checkpoints/msearch_2b.pt` | local-only | e473e36cd0c594a1b6f04157019d4af5eb44f31074180dc75d4a79284b0e1ee5 |
| `checkpoints/msearch_avg2o.pt` | local-only | b914bd23c3fbe79efbe56311cc5a22ea13dc8ddab0cfd05e31d4c0ec70b6e076 |
| `checkpoints/msearch_avg2o.pt.merge.json` | local-only | 7830a46ecaf1293fe83f348c383443c5319da2ea09d4ae8e472cbd5e2c626699 |
| `checkpoints/msearch_avg3o.pt` | local-only | 26738d9d413d15192754ece581cc7a91ed15bc361b84d765c61c5b129f252746 |
| `checkpoints/msearch_avg3o.pt.merge.json` | local-only | 3efb8ac8be824e6573e264dc8f4a0de0ab53996892d525152c52b9596dc726a5 |
| `checkpoints/msearch_avg_12.pt` | local-only | e9ece88ab9d88d167fa3314472a7e42b62497677d4481543799ec20b57eae14b |
| `checkpoints/msearch_avg_12.pt.merge.json` | local-only | fea0c0992928c6653473e2e36c2f24a7df0eb93a12f8314f9fbbd3bc0d99f599 |
| `checkpoints/msearch_avg_13.pt` | local-only | 89a0ee002e1b2217b6c53d564739fee6a4b34c4472e8c982c586c859169ecc31 |
| `checkpoints/msearch_avg_13.pt.merge.json` | local-only | 34cf0a2d02f9f5f1b306e0d4d96114bf320eafdfc441a47ba3fe86949e980f83 |
| `checkpoints/msearch_avg_14.pt` | local-only | 913cf1ff479a6df404810de0e04cc07f7d1ca111bbe6e9fb90f4c9be8d3fccc4 |
| `checkpoints/msearch_avg_14.pt.merge.json` | local-only | c035936e4e35d96a37532bd2ea89dbe75d7f40dcc71599befa59dd7176e9c9ca |
| `checkpoints/msearch_avg_23.pt` | local-only | 8dbe00ef03f19c5c3eeae3753625c157c66b2db60b968d765e7eed4252cd0465 |
| `checkpoints/msearch_avg_23.pt.merge.json` | local-only | 633351cc0a31e1b2b3591301d11a053343cf3ff0bfa5e0ab43887f44578bcb89 |
| `checkpoints/msearch_avg_24.pt` | local-only | fcdeb1ef1d80ee8a3b4259bf717367e5ea14a3cd13f843ddfe2fae47b7dfa4d6 |
| `checkpoints/msearch_avg_24.pt.merge.json` | local-only | 3b4fd9c1e768a661402594ab5696954f0005ef2b417ad9748f8b9b15b34ae42d |
| `checkpoints/msearch_avg_34.pt` | local-only | 09652acfa5f116ec64deb9dfb01b82a7f80e48045e2f2ed016961d450d8ddb6d |
| `checkpoints/msearch_avg_34.pt.merge.json` | local-only | ca713cdc85b58b57d4e60af21d4d26733fda77738bddbaabb79a6127f15d9371 |
| `checkpoints/msearch_avg_e0.pt` | local-only | 1553a7590d1ae32de75f6c638ea0ca0f557475c20e8d0be53cba71bc077c15af |
| `checkpoints/msearch_avg_e0.pt.merge.json` | local-only | 0ac72778228a0ff2b803a209b965fcd392a8b9d4815b003bfdbc7750608ffb13 |
| `checkpoints/msearch_avg_e2.pt` | local-only | 8937fa8fea2661953622524000472e5db9484475dac1913762fa80373515dea1 |
| `checkpoints/msearch_avg_e2.pt.merge.json` | local-only | 10cd632892b9b41f060f89670b4f3a1a6c4685cedbff4ba948b056cd73ce00f0 |
| `checkpoints/msearch_avg_ord.pt` | local-only | 55768d8b3937b2737e406ef10e15bd5258057e7e72619bef0e28912792e17e65 |
| `checkpoints/msearch_avg_ord.pt.merge.json` | local-only | 7677c121b3f1600c1fc63616568c76e2adcb3881396e19c28127b14be7f9080e |
| `checkpoints/msearch_avg_shared.pt` | local-only | 1b5de76e9674b269a502945302282d4e9a27120953858de2e23506e2e83083f9 |
| `checkpoints/msearch_avg_shared.pt.merge.json` | local-only | b4a9ce205e25efd42ed433be546161018a3b4e4feb4644be6b92ed872c0ee0c7 |
| `checkpoints/msearch_base_e2.pt` | local-only | ebdf949a0bdff8e2b1408764f26aef527405539d864ec9e35e95ab2f0744a8ba |
| `checkpoints/msearch_fork_base.pt` | local-only | 3fbb07b3cbd0695cd7c5695bbf3f241cb0e301bf6909eaf38b4e66a3e0c2d91f |
| `checkpoints/msearch_pair_a.pt` | local-only | d226e8a05f978e22db0871665d6774748ef8e5827bd51e06dc9b77576208a3dd |
| `checkpoints/msearch_pair_a.pt.merge.json` | local-only | 5994691e55dd27aec3d70fce3fbfe4f9671a3d5dc6be69d7b134e6473997de84 |
| `checkpoints/msearch_pair_b.pt` | local-only | 69c901fb633de453b20cfa2943ac79e5ed5a0755ef767cb9914167f7b6d2bbe3 |
| `checkpoints/msearch_pair_b.pt.merge.json` | local-only | 0ab1703a2b90db72f90a3b4333e1cfa63b7ca580f56bb576bb3afceb26f76e63 |
| `checkpoints/msearch_s1.pt` | local-only | 7bfc417abe109218fafa878b5635af78fff4aceaeb9fc0eda0b4c0a8d110f04d |
| `checkpoints/msearch_s1o.pt` | local-only | 0b7e6eb313c165019e248b59664a7b754f545e64b75063f4745de560162fd2cd |
| `checkpoints/msearch_s1o8.pt` | local-only | 1c7ce11a64801e4dd0c0b07a9228a7437ca817f9a42b4069de94c73c8a12d9a5 |
| `checkpoints/msearch_s1o9.pt` | local-only | a2c2573675ed3b36c3e467772cca0452fe098d3640098e6151a4cb8182bd03e9 |
| `checkpoints/msearch_s1w.pt` | local-only | d6286e97467bd4e55ac378b60e2cd58afd1be1d85a065d62f88a7de394e9a065 |
| `checkpoints/msearch_s2.pt` | local-only | 7a14154f38387deec1644184e3e2b7f64205ab8568460bd6406f74c513e3e5a3 |
| `checkpoints/msearch_s2o.pt` | local-only | 57b8f66a7995821036666ba082c549fd6905db7d4ddc6fb49f5c4b7a9fd8718d |
| `checkpoints/msearch_s3.pt` | local-only | 2a03176eaf0635532dc75213f50efa270baf1fd0e934da69e6f78dd26eedab81 |
| `checkpoints/msearch_s3o.pt` | local-only | ffda913192f44abde6ba7550052c6164bb0bc2d87fa10785136424e1e73ab3d7 |
| `checkpoints/msearch_s4.pt` | local-only | 26fcda038e279202319c68b9e57e085c92723fd8a5815b6e43c30e9b188bdd5c |
| `checkpoints/msearch_soup4.pt` | local-only | 20002d3e001d2bcfd2c2d211389d6783e741bfa0fd9b75d5bacdcc43880c75ca |
| `checkpoints/msearch_soup4.pt.merge.json` | local-only | 9bd323be294b3f0c34d8638d51b81cc56684105dbaebe40b15d9b7f16a4e4c52 |
| `checkpoints/msearch_tv.pt` | local-only | 8a98947c041c9e1ecc0c14cd3a413008bdf2eac3c944a292fa828b9514ee0acf |
| `checkpoints/msearch_tv.pt.merge.json` | local-only | 4f9743a7768d6c72f660e5d897f4deb5a7f78af353839d8a9338b834625990d6 |

`(locked)` means the digest is read from `receipts.lock.json` rather
than typed twice. The 47 per-arm merge_space logs are tracked and
locked too; they are not repeated here because the lock already pins
each one and the amendment lists them with their digests. A `local-only` row whose file is missing is reported
by the test and fails only on the evidence host (`LLMOPT_FULL=1`); an
`absent` row whose file reappears fails the test until this page is
updated, so recovery is a visible, reviewed event.

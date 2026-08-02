# External-reader presentation session notes

**Date:** 2026-08-02

**Branch:** `sol/present-1`

**Base / merge-base with `main`:**
`0dea97283d4a270c4f8b2b1ad48adcf01b42e5f8`

**Original close-out pre-note HEAD:**
`78b4e56304c7d2ed625efb0cb0be2ecd781ba5f7`

**Final-review base / verified final-review pre-note HEAD:**
`6eb82f05ebb0a2fd71b28879da581662a23d01f5` /
`7eff22c7a76a0567681bf462986b13b6b64ab5c4`

**Execution fence:** Mac-local only; no remote, SSH, WSL, or other device was
used in this presentation session.

**House status:** **PROPOSAL until adopted by the house.** A branch commit is
not a ledger verdict and does not change any booked claim.

This session made the mathematics/physics evidence hierarchy legible to an
external reader and repaired two operational defects in the advertised
trajectory reproduction. It did not add a scientific result. Exact trajectory
digest agreement is determinism evidence only; it is not symbolic correctness
or capability evidence. Every `n=1` result below remains fenced as `n=1`.

## Adoption checklist

- [x] **Tasks 1-5 — self-contained gravmoe trajectory reproduction.** Adoption
  commit `4ef9cd511369023d69db7332aebf36517de62951`, subject `fix:
  reproduce gravmoe from pinned windows`. Files:
  `llmopt/window_artifact.py`, `llmopt/reproduce.py`,
  `scratch/detbwd_gravmoe.py`, `tests/test_window_artifact.py`,
  `tests/test_reproduce.py`, and `tests/test_gravmoe_artifact_input.py`.
  Fresh final verification on pre-note HEAD: 21/21 focused tests. Prior
  acceptance: artifact-backed 16/16 and unchanged default-path 16/16 exact pin
  matches, recorded in
  `.superpowers/sdd/2026-08-02-external-reader-presentation/task-1-5-report.md`.
  Final-review commit `b610e23fa2b41751d44db9ee969350b5aec537ba`,
  subject `fix: enforce artifact split validation`, replaced optimized-away S1
  marker/split assertions with explicit validation and raised the focused
  result to 22/22.
- [x] **Task 6 — controlled glossary.** Adoption commit
  `80505abf28ec5f1556926fe7565b7d1627c4a7da`, subject `docs: add the
  llmopt glossary`, files `GLOSSARY.md` and `README.md`; reviewed pointer fix
  `833129a646630f7b7f2f46dcb7a4f16b47ad38e8`, subject `docs: fix glossary
  evidence pointers`, file `GLOSSARY.md`. Final local-link audit is included
  below. Final-review commit
  `2d7ca1121db7e67a6170869368a7fabdd3ebf7db`, subject `docs: restore
  evidence tag grammar`, restored the approved null, retraction, and broad
  registered-`n=1` definitions.
- [x] **Task 7 — external-reader README and measured-history appendix.**
  Adoption commit `550555b250ad33baa972355a530b4bbd7895bf8d`, subject `docs:
  give README an external reader path`, files `README.md` and
  `docs/MEASURED-HISTORY.md`; reviewed fence fix
  `5b2499392a1d110e6f8dd6ba5dc3d6207bd993d0`, subject `docs: tighten
  measured history evidence fences`, same files. Final verification: exactly
  four approved H2 headings, 140 lines versus the 275-line baseline, and exact
  numeric-verdict adjacency in retained history.
- [x] **Task 8 — maturity- and scope-controlled findings.** Adoption commit
  `a5b3a989134b0170dd384bd190cc005f232d2f7d`, subject `docs: surface
  findings evidence maturity`, file `docs/FINDINGS.md`; reviewed fence fix
  `acc4be5bfe696ba58adaa75827c034ea7db623a4`, subject `docs: restore
  findings evidence fences`, same file. Final verification: 100 finding units,
  exactly one approved maturity tag per unit, controlled scopes/regimes, 33/33
  single-seed `n=1` fences, 24/24 named replication routes, 23/23 causal-arm
  descriptions, zero numeric/link errors, and 499 lines versus the 586-line
  baseline. Final-review commit
  `8e4d241fff2d7eae1f30a426064b30dad95de143`, subject `docs: correct
  findings maturity routes`, split the unsupported combined crystal claim and
  corrected the original 19M/0.5B comparison. Current verification is 101
  units, 36/36 `n=1` fences, 22/22 replication routes, 23/23 causal-arm
  descriptions, zero numeric/link errors, and 500 lines.
- [x] **Task 9 — reproduction walkthrough.** Adoption commit
  `548e9c550d3f49d300868730296b707b800ce2bd`, subject `docs: add the
  reproduction walkthrough`, file `docs/REPRODUCE.md`. The fresh-worktree RB1
  receipt is a full-digest PASS in 83.9 seconds at
  `/tmp/llmopt-task9-rb1.log`; the separate house-booked receipt is 81.7
  seconds under [VERDICT SOL-ADOPTION-1](../RESULTS.md#L15081) at ledger commit
  `0dea97283d4a270c4f8b2b1ad48adcf01b42e5f8`. Final-review commit
  `87880035810dd3cece3bf0980181cbd027135d0b`, subject `docs: disclose
  verifier publication limit`, states that the verified axiom revision is 21
  commits ahead of and unreachable at the linked public origin today.
- [x] **Task 10 — external-review digest.** Adoption commit
  `6228a10f9359fb3f8fc8283b371f776e45fc6572`, subject `docs: digest
  external review outcomes`, file `docs/EXTERNAL-REVIEWS.md`. Final-review
  commit `7eff22c7a76a0567681bf462986b13b6b64ab5c4`, subject `docs: correct
  external review scope`, fixes the source-record count wording; the separate
  “living ledger entry” suggestion was explicitly declined below.
- [x] **Task 11 preflight blocker repaired and re-reviewed.** The original
  `git diff --check main...HEAD` failed on spec trailing whitespace and a
  final blank line. Commit `472aff31b235b0ab8a71adee1b4bbb3c77bb9263`,
  subject `docs: fix presentation spec whitespace`, repaired
  `docs/superpowers/specs/2026-08-02-external-reader-presentation.md`;
  reviewed follow-up `78b4e56304c7d2ed625efb0cb0be2ecd781ba5f7`, subject `docs:
  preserve presentation spec metadata layout`, changed the same file. Fresh
  `git diff --check main...HEAD` then exited 0 with no findings.
- [x] **Task 11 — close-out note.** This file is the only file in its adoption
  unit. Use `git log -1 --format='%H %s' -- docs/sol/SESSION-NOTES-PRESENT-1.md`
  to resolve the immutable latest containing SHA and subject after commit. This
  checklist remains a branch proposal until the house adopts it.
- [x] **Final whole-branch review — five Important and one Minor finding.** The
  Important fixes are optimized-mode S1 validation
  (`b610e23fa2b41751d44db9ee969350b5aec537ba`), the original-comparison
  maturity correction and separate crystal evidence units
  (`8e4d241fff2d7eae1f30a426064b30dad95de143`), restored evidence grammar
  (`2d7ca1121db7e67a6170869368a7fabdd3ebf7db`), and the axiom publication
  limit (`87880035810dd3cece3bf0980181cbd027135d0b`). The Minor source-scope
  wording fix is `7eff22c7a76a0567681bf462986b13b6b64ab5c4`.

No living ledger changed. The final exact-path guard excludes
`docs/RESULTS.md`, `docs/BOARD.md`, `docs/THEORY.md`,
`docs/RIFF-LEDGER.md`, `docs/handoffs/`, and `docs/results-index.jsonl` from
`git diff main --name-only`.

## Ranked findings and cheapest confirmation/kill checks

These are ranked by external-reader importance, not by novelty. Confidence is
confidence in the explicitly measured scope, never confidence in an unstated
general law.

### 1. The registered gravmoe trajectory is reproducible from committed bytes

- **Claim.** All 16 registered artifact-backed arms matched their committed
  milestone-weight trajectory pins; the unchanged default path separately
  matched 16/16. The booked external close also records 16/16 device transport
  and 10/10 engine-arm transport in a third runtime. Gate artifact runs remain
  trajectory-only because free-run SymPy scoring needs uncommitted row text.
- **Evidence.** The Task 1-5 acceptance logs are
  `logs/sol/present_artifact_<arm>.log` (RB1 uses
  `logs/sol/present_reproduce_rb1.log`) and
  `logs/sol/present_default_<arm>.log`. The booked device and lab closes are
  [VERDICT GRAVMOE-P4-DEVICE](../RESULTS.md#L14889), ledger commit
  `b9372e967ab7269afd06fa52027ce19450ca4d95`, and
  [VERDICT GRAVMOE-P4-LAB](../RESULTS.md#L15015), ledger commit
  `94e29cd61ef3b0cfbe44f5848185053bcb9bdb87`. Two labs means independent
  Python/C++ code paths under one human operator, not independent investigators.
- **Confidence.** High for exact replay of the registered trajectory; no claim
  of symbolic correctness or capability follows.
- **Cheapest house check.** Run the one-arm Mac-local RB1 command below and
  require the full digest
  `c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f`.
  A mismatch kills the current reproducibility claim for that revision.

### 2. Sigma packing is supported inside the at-capacity house-crystal regime

- **Claim.** Within at-capacity house crystals, sigma-pack gate parity and
  near-Gaussian code entropy reproduced across `n=3` paired d64h8 births; the
  weak width-floor births are the explicit exception. This is not a universal
  quantization rule.
- **Evidence.** [C1 AT n=3 VERDICT](../RESULTS.md#L11232) and
  [PACKED CRYSTAL C0+C1 VERDICT](../RESULTS.md#L10406); the latter entered in
  commit `08f567cada68f5847d1975c54b842e029b4f1722`. The controlled summary is
  in [FINDINGS](../FINDINGS.md#the-packed-crystal-boundary).
- **Confidence.** High for the registered `n=3` paired house-crystal cells;
  low outside the at-capacity regime without a transport verdict.
- **Cheapest house check.** Replay one existing paired C1 birth and its
  capability gate using the committed pack parameters. A gate or entropy miss
  outside the booked tolerance kills parity for that seed; a full transport
  claim still requires the registered paired set.

### 3. The sigma-pack rule does not transport unchanged to Qwen2.5-0.5B

- **Claim.** On one registered NVIDIA Qwen2.5-0.5B fake-quant cell, per-tensor
  sigma allocation was 33 times worse than HQQ; per-row sigma did not rescue
  it, while a causal sigma/8 arm recovered part of the gap. This remains one
  model, one device, and `n=1`, not a law about pretrained dense models.
- **Evidence.** [PACKED CRYSTAL C6 VERDICT](../RESULTS.md#L10676), entered in
  commit `a5dbf0e652ff34a486ddc6bc0120f34f9d008cb4`, plus the adjacent C6b/C6c
  units in [FINDINGS](../FINDINGS.md#the-packed-crystal-boundary).
- **Confidence.** High for that exact `n=1` cell; low for model-family or
  device generalization.
- **Cheapest house check.** Re-score the stored Qwen2.5-0.5B tensors with the
  exact sigma and HQQ grids on the same teacher-forced battery. Reversal of
  the registered ordering kills the boundary receipt; confirmation remains
  `n=1` until more models/seeds are added.

### 4. The registered integer-forward path is deterministic across devices

- **Claim.** Integer-forward hashes reproduced across independent Apple and
  NVIDIA devices, while floating logits could differ even when greedy streams
  matched. This covers the registered integer battery, not a full integer
  end-to-end decoder.
- **Evidence.** [PACKED CRYSTAL C4 VERDICT](../RESULTS.md#L10657), entered in
  commit `3155d402d9dd30eab23b161b9c6608bda04ff8e1`, and
  [R-PASS VERDICT](../RESULTS.md#L10838), entered in commit
  `6f01ef2b3ff5bbdcd31c5623ac7fe319ce416f35`.
- **Confidence.** High for the registered shapes, artifacts, and two-device
  route; no broader kernel or decoder claim.
- **Cheapest house check.** Hash the C4 integer forward once on the current
  Mac artifact and compare it byte-for-byte to the booked pin. This cheaply
  detects local drift; repeating the independent-device leg is required to
  reconfirm transport.

### 5. The retained gravmoe recipe is merge-free but capability-neutral

- **Claim.** Hebbian-coupled experts merged with bounded gate cost across
  `n=3` Mac seeds and an independent CUDA birth, but the stale scaffold
  capability advantage did not transport: on `n=3` paired CUDA seeds the
  gravmoe and baseline means tied. Merge-free transport and capability remain
  separate claims.
- **Evidence.** [VERDICT MERGE-CUDA](../RESULTS.md#L12679) and
  [VERDICT CUDA SEED LADDER](../RESULTS.md#L13286), summarized with their
  replication and null tags in
  [FINDINGS](../FINDINGS.md#moe-mechanisms-and-the-scaffold-correction).
- **Confidence.** High for the registered paired/device cells; low for other
  diets, couplings, scales, or capability batteries.
- **Cheapest house check.** Rerun one existing paired merge/no-merge seed and
  require both the merge-cost and capability readouts. A capability separation
  in one seed is only a trigger for the full paired `n=3` rerun, not a new
  result by itself.

## What to do next with more budget

1. Expand the Qwen boundary beyond its `n=1`, one-model, one-device fence:
   preregister multiple model families and paired seeds, keep the exact
   teacher-forced instrument, and test whether max anchoring remains the
   boundary variable.
2. Add committed oracle row text, or a separately pinned symbolic oracle
   artifact, for gate arms. That would allow the public artifact path to test
   free-run symbolic correctness rather than trajectory and teacher-forced
   diagnostics alone.
3. Publish or otherwise make reachable the verified axiom revision
   `8f8376d86ce6a25fdd6fee2455c220e7055cb018`, then add a portable build recipe
   and a clean-clone third-runtime replay. The current verified interface is
   `tools/int_adamw/verify_gravmoe.py`; the local revision is not yet on the
   public remote, as recorded in the Task 9 report.
4. Repeat the capability-neutral merge-free result on a new operator, diet,
   and capability battery. The current two-lab trajectory close still has one
   human operator, and the merge/capability finding is recipe-scoped.
5. Resolve the external-review experiment queue in status order: run K3's
   architectural-transfer cell only under a preregistered charter; keep the
   paper/framing items banked or deferred until their prerequisites exist.

## Broken house code found and repaired

Fresh-clone reproduction originally depended on the ignored sidecar
`data/micromodel_gen4_sidecar.jsonl`. The pre-fix real-runner test failed
exactly as:

```text
6 failed in 3.74s
FileNotFoundError: data/micromodel_gen4_sidecar.jsonl
```

Commit `4ef9cd511369023d69db7332aebf36517de62951` fixed the defect by loading
and validating committed window bytes/contracts, pinning the selected inputs,
scrubbing hostile ambient artifact variables, and making gate artifact mode
explicitly trajectory-only. Coverage lives in `tests/test_window_artifact.py`,
`tests/test_reproduce.py`, and `tests/test_gravmoe_artifact_input.py`.
Original close-out focused verification was 21/21; prior acceptance is
artifact 16/16 and default 16/16. The default courtesy sidecar SHA was
`809bce4215a24164ecbf5e951d77507d455bfd1923d08fe39aa02942b11a200b`,
and the temporary link is absent after testing.

Final review found a second operational defect: artifact-backed S1 marker and
split contracts were enforced with `assert`, so optimized Python removed the
checks. Commit `b610e23fa2b41751d44db9ee969350b5aec537ba` replaces them with
diagnostic `ValueError`s and adds a real `python -O` subprocess regression over
a SHA-consistent malformed artifact. No other house-code defect was
established. The Task 11 whitespace failure was presentation-spec hygiene, not
scientific or runner behavior: it was repaired by
`472aff31b235b0ab8a71adee1b4bbb3c77bb9263` and
layout-preserved/re-reviewed by
`78b4e56304c7d2ed625efb0cb0be2ecd781ba5f7` before final verification.

## Final-review wording dispositions

The reviewed [external-review digest](../EXTERNAL-REVIEWS.md) had two minor
wording candidates:

1. **Resolved:** “four doc-only reads” became “four external-review source
   records” in `7eff22c7a76a0567681bf462986b13b6b64ab5c4`.
2. **Declined:** “living ledger entry” remains unchanged. Each cited RIFF or
   RESULTS source is an exact entry in a living ledger, so the wording is
   defensible and preserves the digest's provenance fence.

The disposition changes no review suggestion into a result or house adoption.

## Final verification receipts

All Python commands used
`/Users/artin/code/llmopt/.venv/bin/python`. Runner receipts used
`RJOB_LOCAL=1`. No acceptance run used a remote or non-Mac device.

### Final-review optimized and focused commands

```bash
RJOB_LOCAL=1 /Users/artin/code/llmopt/.venv/bin/python -m pytest -q \
  tests/test_gravmoe_artifact_input.py::test_optimized_runner_refuses_sha_consistent_s1_split_drift
RJOB_LOCAL=1 /Users/artin/code/llmopt/.venv/bin/python -m pytest -q \
  tests/test_window_artifact.py tests/test_reproduce.py \
  tests/test_gravmoe_artifact_input.py
```

Fresh outputs on final-review pre-note HEAD
`7eff22c7a76a0567681bf462986b13b6b64ab5c4`:

```text
optimized-mode regression: 1 passed in 0.65s
focused artifact/reproduce suite: 22 passed in 6.43s
```

The optimized regression launches the real runner with `python -O` and a
malformed marker/split artifact whose raw and reconstructed-row SHAs are
internally consistent; it now refuses at the split-position contract.

### Original close-out commands

```bash
git diff main --name-only
git diff --check main...HEAD
/Users/artin/code/llmopt/.venv/bin/python -m pytest -q \
  tests/test_window_artifact.py tests/test_reproduce.py \
  tests/test_gravmoe_artifact_input.py
/Users/artin/code/llmopt/.venv/bin/python -m pytest -q \
  --ignore=tests/test_mlx_backend.py \
  --ignore=tests/test_population.py \
  --ignore=tests/test_exact_gemm.py \
  --ignore=tests/test_fused_ce.py \
  --ignore=tests/test_metal_kernels.py
```

Original outputs on pre-note HEAD
`78b4e56304c7d2ed625efb0cb0be2ecd781ba5f7`:

```text
living read-only path guard: PASS
git diff --check main...HEAD: PASS (exit 0, no findings)
focused suite: 21 passed in 5.78s
broad non-Metal suite: 408 passed, 7 skipped, 16 warnings in 84.61s
README headings: exact 4/4
README lines: 140 (baseline 275)
FINDINGS: 100 units, 0 errors, 499 lines (baseline 586)
single-seed fences: 33/33
replication routes: 24/24
mechanism arms: 23/23
FINDINGS numeric adjacency: 0 errors
FINDINGS local links: 134, 0 errors
all tracked Markdown: 189 files, 191 local links, 0 errors
temporary sidecar path: absent
```

The 16 broad-suite warnings were multiprocessing fork deprecations, one torch
profiler warning, and transformer nested-tensor warnings; there were no test
failures.

### Reused Task 9 RB1 receipt

The already-fresh Task 9 run was reused rather than spending another roughly
80 seconds:

```bash
set -o pipefail
PATH=/Users/artin/code/llmopt/.venv/bin:$PATH \
  RJOB_LOCAL=1 python -m llmopt.reproduce gravmoe-rb1 \
  | tee /tmp/llmopt-task9-rb1.log
```

Receipt at `/tmp/llmopt-task9-rb1.log`:

```text
[gmoe] FINAL trajectory sha c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f
ELAPSED 83.9s
PASS gravmoe-rb1 c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f
```

The house-booked 81.7-second receipt is distinct and belongs to
[VERDICT SOL-ADOPTION-1](../RESULTS.md#L15081) at
`0dea97283d4a270c4f8b2b1ad48adcf01b42e5f8`.

### Reused Task 1-5 all-arm receipts

Artifact-backed command shape:

```bash
RJOB_LOCAL=1 /Users/artin/code/llmopt/.venv/bin/python \
  -m llmopt.reproduce gravmoe-<arm> \
  | tee logs/sol/present_artifact_<arm>.log
```

RB1 uses `logs/sol/present_reproduce_rb1.log`; the other 15 arms use
`logs/sol/present_artifact_<arm>.log`. All 16 logs are present, and the Task
1-5 report records 16/16 exact pin matches.

Default-path command shape, with no artifact variables in the clean
environment:

```bash
env -i PATH="$PATH" RJOB_LOCAL=1 PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 STEPS=2000 <arm-env> \
  /Users/artin/code/llmopt/.venv/bin/python scratch/detbwd_gravmoe.py \
  | tee logs/sol/present_default_<arm>.log
```

All 16 default logs are present, and the Task 1-5 report records 16/16 exact
pin matches. These receipts prove registered trajectory reproduction only.

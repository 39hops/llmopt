# External-reader presentation implementation plan

> **For Codex:** Execute this plan with the subagent-driven-development workflow.
> Keep all commits on `sol/present-1`; never edit the living ledgers.

**Goal:** Make llmopt's evidence hierarchy and reproduction path legible to an
outside reader, while making the advertised trajectory reproduction genuinely
self-contained from committed artifacts.

**Architecture:** Add a small, tested decoder for the committed
`tok[T] ++ tgt[T]` window records. The gravmoe runner uses it only when explicit
artifact inputs are supplied; its existing diet-drawing path remains the house
default. `llmopt.reproduce` supplies the pinned artifacts and requests
trajectory-only behavior for gate arms whose SymPy row text is not committed.
Presentation documents then route readers through a compact README, controlled
glossary, maturity-tagged findings catalog, and precise reproduction guide.

**Stack:** Python 3.11+, PyTorch CPU integer runner, pytest, Markdown, git.

---

## Global fences

- Read-only throughout: `docs/RESULTS.md`, `docs/BOARD.md`, `docs/THEORY.md`,
  `docs/RIFF-LEDGER.md`, `docs/handoffs/`, `docs/results-index.jsonl`.
- Mac-local only. Every executable acceptance command includes `RJOB_LOCAL=1`.
- No remote runner, SSH, WSL, or non-Mac device.
- Every presentation number names its supporting ledger verdict.
- README must finish below 275 lines; FINDINGS must finish at or below 586
  lines, with working targets of 110--140 and at most 500 respectively.
- Each deliverable below is staged and committed alone. Never stage a living
  ledger.

## Task 1: RED — specify committed-window decoding

**Files:**

- Create: `tests/test_window_artifact.py`
- Later create: `llmopt/window_artifact.py`

**Step 1: Write the failing valid-record test**

Create a hand-packed little-endian fixture with `T=3`, two records in
`tok[T] ++ tgt[T]` form, literal expected reconstructed rows, and a literal
contract SHA. The test imports `load_contiguous_windows` from the not-yet-created
module and expects `[[10, 11, 12, 13], [20, 21, 22, 23]]`.

Production break caught: deleting or misordering the target tail makes the
reconstructed training row wrong.

**Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_window_artifact.py
```

Expected: FAIL because `llmopt.window_artifact` does not exist.

**Step 3: Add independent failure tests**

Add one test each for:

- raw-byte SHA mismatch;
- record length not divisible by `2*T*8`;
- `tok[1:] != tgt[:-1]`, expecting an error containing
  `windows must be contiguous next-token slices`;
- reconstructed-row SHA mismatch when `windows_rows_sha` is present.

Use literal fixtures and assert public exceptions/messages, not source text.

## Task 2: GREEN — implement the window decoder

**Files:**

- Create: `llmopt/window_artifact.py`
- Test: `tests/test_window_artifact.py`

**Step 1: Implement the minimum decoder**

Implement:

```python
def load_contiguous_windows(
    windows_path: Path,
    contract_path: Path,
    sequence_length: int,
) -> list[list[int]]:
    ...
```

Behavior:

1. Read raw bytes and JSON contract.
2. Compare SHA-256 to `windows_sha`; refuse on mismatch.
3. Require a non-empty multiple of `2 * sequence_length * 8` bytes.
4. Decode signed little-endian int64 values.
5. Check the contiguous-next-token overlap with the approved diagnostic.
6. Reconstruct each `T+1` row as `tok + [tgt[-1]]`.
7. If `windows_rows_sha` exists, hash the reconstructed little-endian int64 rows
   and refuse on mismatch.

Do not import torch; the artifact validator should be cheap and standalone.

**Step 2: Verify GREEN and mutation coverage**

Run:

```bash
python -m pytest -q tests/test_window_artifact.py
```

Expected: PASS. Mentally mutate byte order, record width, SHA comparison, and
overlap direction; at least one test must fail for each mutation.

**Step 3: Commit the decoder only after green**

Do not commit yet; it ships with the runner integration and acceptance tests in
Task 5.

## Task 3: RED — specify reproduce registry and environment behavior

**Files:**

- Modify: `tests/test_reproduce.py`
- Later modify: `llmopt/reproduce.py`

**Step 1: Add failing registry assertions**

Assert for all 16 entries returned by `available()`:

- committed `windows_path` and `contract_path` exist;
- the selected contract's `windows_sha` matches its bytes;
- gate arms set `TRAJECTORY_ONLY=1`;
- all arms set `WINDOWS_BIN` and `WINDOWS_CONTRACT`;
- S1 remains gate + scheduled sampling and never selects a diet fallback.

Use the non-gate RB1 artifact family for ordinary windows and GRB1 for the gate
window family; record why shared bytes are valid by their contract SHA.

**Step 2: Extend ambient-knob scrubbing**

Seed a fake base environment with hostile values for `WINDOWS_BIN`,
`WINDOWS_CONTRACT`, and `TRAJECTORY_ONLY`. Assert the pinned registry values win.

Production break caught: ambient inputs can redirect a supposedly pinned run.

**Step 3: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reproduce.py
```

Expected: FAIL because registry entries do not yet expose committed windows.

## Task 4: GREEN — wire pinned artifacts into the reproduce module

**Files:**

- Modify: `llmopt/reproduce.py`
- Test: `tests/test_reproduce.py`

**Step 1: Extend the pinned specification**

Add the artifact-input environment variables to `CONTRACT_ENV` so ambient
values are always removed. Have `available()` choose the committed RB1 or GRB1
window family by whether the arm is a gate arm, and inject absolute committed
paths. Gate arms receive `TRAJECTORY_ONLY=1`.

**Step 2: Verify GREEN**

Run:

```bash
python -m pytest -q tests/test_reproduce.py tests/test_window_artifact.py
```

Expected: registry tests pass; runner integration is still intentionally absent.

## Task 5: RED/GREEN — integrate artifact windows into gravmoe

**Files:**

- Modify: `scratch/detbwd_gravmoe.py`
- Modify: `tests/test_reproduce.py` or create
  `tests/test_gravmoe_artifact_input.py`
- Use: `llmopt/window_artifact.py`

**Step 1: RED for a real short runner boundary**

Run the real runner as a subprocess with committed RB1 windows, a temporary
contract whose `windows_sha` is deliberately wrong, and `STEPS=1`. Assert
nonzero exit and SHA-refusal text. Then run with valid bytes and `STEPS=1`,
asserting the runner reaches the training readout without opening the missing
sidecar. This test must fail before runner integration.

**Step 2: Implement explicit artifact loading**

At runner start:

- require `WINDOWS_BIN` and `WINDOWS_CONTRACT` together;
- load and convert validated rows to `torch.int64`;
- retain `draw_windows()` / `draw_complete()` when both are absent;
- print a clear artifact-source receipt.

For gate artifact mode:

- use the eight committed train rows;
- instantiate the code-defined vocab-40 tokenizer and encode `Step: `;
- derive and assert the recorded split positions needed by S1;
- derive answer regions only where training diagnostics require them;
- skip post-training `gate()` calls only when `TRAJECTORY_ONLY=1`, printing that
  SymPy solve scoring needs the uncommitted row text;
- never allow trajectory-only mode without committed artifact inputs.

The final SHA continues to hash milestone weights only.

**Step 3: Verify short RED/GREEN boundary**

Run:

```bash
python -m pytest -q tests/test_window_artifact.py \
  tests/test_reproduce.py tests/test_gravmoe_artifact_input.py
```

Expected: PASS.

**Step 4: Run the public RB1 acceptance path**

Run with incremental log output:

```bash
RJOB_LOCAL=1 python -m llmopt.reproduce gravmoe-rb1 \
  | tee logs/sol/present_reproduce_rb1.log
```

Expected: exact `c6766da2...` PASS from the fresh worktree without the sidecar.

**Step 5: Run all 16 artifact-backed pins**

Run bounded Mac-local batches, streaming one log per arm. Compare every final SHA
to `scratch/detbwd_gmoe_ref/pins.json`. Record S1 marker IDs and split positions.
Expected: 16/16 exact. If S1 cannot reproduce from code-defined marker data,
stop and narrow the documented claim to 15/16; do not fall back silently.

**Step 6: Re-prove the default house path**

Use the existing local sidecar only as a read-only source via an explicitly
created temporary link in this worktree. Run all 16 default-path arms in bounded
batches, stream logs, compare SHAs, then remove only the temporary link. Expected:
16/16 exact and no artifact variables present.

**Step 7: Run changed-scope tests and commit**

Run:

```bash
python -m pytest -q tests/test_window_artifact.py \
  tests/test_reproduce.py tests/test_gravmoe_artifact_input.py
git diff --check
```

Commit only the implementation and tests:

```bash
git add llmopt/window_artifact.py llmopt/reproduce.py \
  scratch/detbwd_gravmoe.py tests/test_window_artifact.py \
  tests/test_reproduce.py tests/test_gravmoe_artifact_input.py
git commit -m "fix: reproduce gravmoe from pinned windows"
```

## Task 6: Add the controlled glossary

**Files:**

- Create: `GLOSSARY.md`
- Modify: `README.md` only for the near-opening glossary link

Write one line per required term with one exact ledger pointer: crystal, gate,
arm, cell, booked, banked, fence, diet, ration, birth, crown, battery, pin,
relay, house/axiom, rung, riff, twin, HOLD/GO. Add the approved evidence-label
definitions and controlled regime values. State that replication tags name their
route and that regime labels name only the measured object.

Verify every pointer and Markdown link, then commit:

```bash
git add GLOSSARY.md README.md
git commit -m "docs: add the llmopt glossary"
```

## Task 7: Restructure README and preserve measured history

**Files:**

- Modify: `README.md`
- Create: `docs/MEASURED-HISTORY.md`

Write the approved opening sentence and exactly four H2 sections. Lead the third
section with `RJOB_LOCAL=1 python -m llmopt.reproduce gravmoe-rb1`. Keep honest
negatives in the discovery and uncertainty sections. Move unique legacy
benchmark prose to the appendix and attach each retained number to its exact
ledger verdict; route duplicated history to FINDINGS.

Verify:

```bash
test "$(rg -c '^## ' README.md)" -eq 4
test "$(wc -l < README.md)" -lt 275
git diff --check
```

Commit:

```bash
git add README.md docs/MEASURED-HISTORY.md
git commit -m "docs: give README an external reader path"
```

## Task 8: Surface evidence maturity in FINDINGS

**Files:**

- Modify: `docs/FINDINGS.md`
- Read-only source: `docs/sol/results-index-enriched.jsonl`
- Read-only authority: living ledgers listed in the global fences

Rewrite each bullet as one evidence unit with exactly one maturity tag and
stackable controlled scope tags. Split mixed maturity. Name the replication
route in replicated findings. Add `[TEACHER-FORCED]` or `[FREE-RUN-GATED]`
where applicable.

Required scientific corrections:

- make the at-capacity house-crystal regime fence loud;
- make `PACKED CRYSTAL C6` on Qwen2.5-0.5B the boundary finding;
- narrow natural-width, federation, slack-atlas, and capacity-meter prose;
- retract the fixed-instrument Muon CE/gate dissociation while preserving the
  gate-toxic null;
- replace the stale n=1 scaffold advantage with the replicated merge-free,
  capability-neutral CUDA seed-ladder result;
- narrow EMA to the tested schedules and retained null under production
  scheduling;
- preserve n=1 fences.

Run a small checker that parses finding starts and fails on zero or multiple
maturity tags, unknown scope tags, or uncontrolled regime values. Run link and
line-count checks. Expected: at most 500 lines.

Commit:

```bash
git add docs/FINDINGS.md
git commit -m "docs: surface findings evidence maturity"
```

## Task 9: Write the reproduction walkthrough

**Files:**

- Create: `docs/REPRODUCE.md`

Document install, RB1 command, expected PASS, approximately 80 seconds per arm,
the remaining 15 pins, and the pinned axiom verifier. Attribute all numeric
claims to `VERDICT SOL-ADOPTION-1`, `VERDICT GRAVMOE-P4-DEVICE`, or `VERDICT
GRAVMOE-P4-LAB` as appropriate.

State precisely:

- PASS means the full milestone-weight trajectory digest matched exactly;
- PASS is determinism evidence, not symbolic correctness or capability;
- trajectory and teacher-forced loss are self-contained from committed bytes;
- gate-arm SymPy solve scoring additionally needs uncommitted row text;
- two labs means independent code paths under one human operator.

Run the documented command once more from the fresh worktree and verify all
links. Commit:

```bash
git add docs/REPRODUCE.md
git commit -m "docs: add the reproduction walkthrough"
```

## Task 10: Optional external-review digest

**Files:**

- Create: `docs/EXTERNAL-REVIEWS.md`

Only proceed if every item maps to a dated 2026-08-01 RIFF-LEDGER entry or the
adopted Sol verdict. Summarize each review as adopted, banked, or declined; do
not invent reviewer quotations. Verify links and commit alone:

```bash
git add docs/EXTERNAL-REVIEWS.md
git commit -m "docs: digest external review outcomes"
```

## Task 11: Final verification and session notes

**Files:**

- Create: `docs/sol/SESSION-NOTES-PRESENT-1.md`

Run:

```bash
git diff main --name-only
git diff --check main...HEAD
python -m pytest -q tests/test_window_artifact.py \
  tests/test_reproduce.py tests/test_gravmoe_artifact_input.py
python -m pytest -q --ignore=tests/test_mlx_backend.py \
  --ignore=tests/test_population.py --ignore=tests/test_exact_gemm.py \
  --ignore=tests/test_fused_ce.py --ignore=tests/test_metal_kernels.py
```

Also verify:

- no living-ledger path appears in the diff;
- README/FINDINGS line budgets pass;
- the four README headings are exact;
- every finding has exactly one maturity tag;
- scope tags use only approved names/values;
- local Markdown links resolve;
- fresh-clone RB1 and all-arm SHA receipts are recorded with commands and logs.

Write a per-deliverable adoption checklist, ranked findings with confidence and
the cheapest house confirmation/kill experiment, what comes next with more
budget, and any broken house code found. Commit notes alone:

```bash
git add docs/sol/SESSION-NOTES-PRESENT-1.md
git commit -m "docs: close external presentation session"
```

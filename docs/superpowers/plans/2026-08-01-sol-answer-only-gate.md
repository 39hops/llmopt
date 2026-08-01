# Sol Answer-Only Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a binary answer-only loss allocation to the deterministic gravmoe gate and measure whether it improves oracle-scored free-run capability at the pinned G-RB1 contract.

**Architecture:** Keep the forward, dataset, optimizer, and gate unchanged. Add small pure helpers that locate the answer region, construct the optional loss-gradient mask, and compute padding-aware gate metrics; wire them into `scratch/detbwd_gravmoe.py` behind `ANSWER_ONLY=1`. Prove the default path with all 16 committed trajectory pins before running the one treatment arm.

**Tech Stack:** Python 3.11+, PyTorch int64, pytest, fork-isolated SymPy, Bash, Git.

## Global Constraints

- Work only on `sol/review-2`; never commit, merge, or push to `main`.
- Mac-local execution only. Every training command sets `RJOB_LOCAL=1`; never use SSH, WSL, `scratch/wsl.sh`, or remote `rjob` mode.
- Do not modify `docs/RESULTS.md`, `docs/BOARD.md`, `docs/THEORY.md`, `docs/RIFF-LEDGER.md`, `docs/handoffs/`, or `docs/results-index.jsonl`.
- Pre-register in `docs/sol/RESULTS-SOL.md` before any pinned regression or treatment training run fires.
- Oracle correctness is SymPy symbolic equivalence, never string equality.
- Every SymPy parse/equivalence call runs in a forked child with a deadline and forced kill.
- The treatment is one seed, one initialization, and one train/heldout row set. A 3/8 discovery queues paired-seed confirmation before doctrine movement.
- Masked teacher-forced loss is a new lineage and is never numerically compared with prior full-token CE.
- The deterministic battery change must reproduce all 16 SHAs in `scratch/detbwd_gmoe_ref/pins.json` before AO1 is interpreted.
- Logs stream incrementally under ignored `logs/sol/`; every booking records command, log path, code SHA, data SHA, row SHA, and trajectory SHA.
- The local untracked diet is read-only input: `/Users/artin/code/llmopt/data/micromodel_gen4_sidecar.jsonl`, SHA `809bce4215a24164ecbf5e951d77507d455bfd1923d08fe39aa02942b11a200b`.
- The pinned gate rows are: marker IDs `[4, 26]`, newline ID `27`, EOS ID `1`, train-row SHA `32cc244bf28fdadf01b343ae16fe1a55200ffe9fab9bd784e8abd739b12ef2c0`, full-row SHA `78f8aef992debe6ec74e4701fba23167ff5fda1d4294546b9f7621605429798a`.

---

## File structure

- Modify `scratch/detbwd_gravmoe.py`: answer-region contract, optional dlogits mask, padding-aware gate metrics, fork-isolated parse/equivalence assessment, and main-loop wiring.
- Create `tests/test_gravmoe_answer_only.py`: focused synthetic tests that do not depend on the untracked diet.
- Modify `docs/sol/RESULTS-SOL.md`: pre-registration, regression receipt, and final verdict.
- Modify `docs/sol/NOTES.md`: end-of-session ranked findings, next-budget work, and house-code defects.
- Create ignored runtime logs only under `logs/sol/`; do not commit generated logs.

---

### Task 1: Define and test the answer-region and dlogits-mask seam

**Files:**
- Create: `tests/test_gravmoe_answer_only.py`
- Modify: `scratch/detbwd_gravmoe.py:254-322`

**Interfaces:**
- Consumes: `find_split(full, mark) -> int | None`, global `T`, `Q`, and PyTorch int64 tensors.
- Produces: `answer_region(full, mark, terminator_ids) -> tuple[int, int]`, `loss_dlogits(pp, tgt, eye, boost, region=None) -> torch.Tensor`, and `loss_proxy(pp, tgt, region=None) -> int`.
- Region convention: `(split, terminator)` uses token indices in `full`; supervised logit rows are `split - 1 : terminator`, so they predict the first answer token through exactly one newline/EOS terminator.

- [ ] **Step 1: Write failing region tests**

```python
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, "scratch")

import detbwd_gravmoe as G  # noqa: E402


def test_answer_region_includes_one_terminator_only():
    # marker [4, 26], answer [31, 32], newline 27, repeated EOS 1
    full = torch.tensor([8, 4, 26, 31, 32, 27, 1, 1],
                        dtype=torch.int64)
    assert G.answer_region(full, [4, 26], {27, 1}) == (3, 5)


def test_answer_region_rejects_missing_marker_or_terminator():
    with pytest.raises(ValueError, match="marker"):
        G.answer_region(torch.tensor([8, 31, 27]), [4, 26], {27, 1})
    with pytest.raises(ValueError, match="terminator"):
        G.answer_region(torch.tensor([8, 4, 26, 31]), [4, 26], {27, 1})
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/artin/code/llmopt/.venv/bin/python -m pytest \
  tests/test_gravmoe_answer_only.py -q
```

Expected: FAIL because `answer_region` does not exist.

- [ ] **Step 3: Implement the minimal region helper**

Add after `find_split`:

```python
def answer_region(full, mark, terminator_ids):
    """Return (first answer token, first newline/EOS token)."""
    split = find_split(full, mark)
    if split is None:
        raise ValueError("final Step marker not found")
    terms = set(int(t) for t in terminator_ids)
    for pos in range(split, len(full)):
        if int(full[pos]) in terms:
            return split, pos
    raise ValueError("answer terminator not found")
```

- [ ] **Step 4: Add failing mask-equivalence tests**

Append:

```python
def test_loss_dlogits_masks_scaffold_and_repeated_padding():
    pp = torch.tensor([[100, 200, 212]] * 7, dtype=torch.int64)
    tgt = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.int64)
    eye = torch.eye(3, dtype=torch.int64)
    legacy = (pp - G.Q * eye[tgt]) * 7
    assert torch.equal(G.loss_dlogits(pp, tgt, eye, 7), legacy)

    # split=3, terminator token index=5 -> logit rows 2,3,4 only.
    got = G.loss_dlogits(pp, tgt, eye, 7, (3, 5))
    assert int(got[:2].abs().sum()) == 0
    assert torch.equal(got[2:5], legacy[2:5])
    assert int(got[5:].abs().sum()) == 0


def test_loss_proxy_uses_the_same_rows_as_dlogits():
    pp = torch.tensor([[100, 200, 212]] * 7, dtype=torch.int64)
    tgt = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.int64)
    expected = sum(G.Q - int(pp[i, tgt[i]]) for i in range(2, 5))
    assert G.loss_proxy(pp, tgt, (3, 5)) == expected
```

- [ ] **Step 5: Run tests and verify RED for the mask helpers**

Expected: the region tests pass and the new tests fail because `loss_dlogits` and `loss_proxy` do not exist.

- [ ] **Step 6: Implement exact optional masking without changing the default expression**

```python
def loss_dlogits(pp, tgt, eye, boost, region=None):
    dlogits = (pp - Q * eye[tgt]) * boost
    if region is None:
        return dlogits
    split, terminator = region
    keep = torch.zeros(pp.shape[0], dtype=torch.int64)
    keep[split - 1:terminator] = 1
    return dlogits * keep[:, None]


def loss_proxy(pp, tgt, region=None):
    err = Q - pp[torch.arange(tgt.shape[0]), tgt]
    if region is not None:
        split, terminator = region
        err = err[split - 1:terminator]
    return int(err.sum())
```

Do not rewrite the arithmetic inside the `region is None` branch.

- [ ] **Step 7: Run the focused tests and verify GREEN**

Expected: all Task 1 tests pass.

- [ ] **Step 8: Commit the pure loss seam**

```bash
git add scratch/detbwd_gravmoe.py tests/test_gravmoe_answer_only.py
git commit -m "sol: add exact answer-only loss seam"
```

---

### Task 2: Add padding-aware gate metrics and oracle diagnostics

**Files:**
- Modify: `scratch/detbwd_gravmoe.py:300-356`
- Modify: `tests/test_gravmoe_answer_only.py`

**Interfaces:**
- Consumes: `answer_region`, generated token tensor `w`, truth tensor `full`, tokenizer IDs, and truth expression string.
- Produces: `token_accuracy_counts(generated, full, region) -> dict[str, int]`, `_fork_call(worker, args, timeout) -> object | None`, `sympy_assess(a, b, timeout=10) -> tuple[bool, bool]`, and a `gate(...) -> dict[str, int]` result with solves, parseable, terminated, standard hits/total, and suffix hits/total.

- [ ] **Step 1: Write failing padding-confound tests**

```python
def test_token_accuracy_separates_suffix_from_padding():
    full = torch.tensor([8, 4, 26, 31, 32, 27, 1, 1],
                        dtype=torch.int64)
    generated = full.clone()
    generated[6:] = torch.tensor([9, 9])  # suffix correct, padding wrong
    got = G.token_accuracy_counts(generated, full, (3, 5))
    assert got == {
        "standard_hits": 3,
        "standard_total": 5,
        "suffix_hits": 3,
        "suffix_total": 3,
    }
```

- [ ] **Step 2: Run the test and verify RED**

Expected: FAIL because `token_accuracy_counts` does not exist.

- [ ] **Step 3: Implement the pure accuracy counter**

```python
def token_accuracy_counts(generated, full, region):
    split, terminator = region
    standard_g = generated[split:T]
    standard_t = full[split:T]
    suffix_g = generated[split:terminator + 1]
    suffix_t = full[split:terminator + 1]
    return {
        "standard_hits": int((standard_g == standard_t).sum()),
        "standard_total": int(standard_t.numel()),
        "suffix_hits": int((suffix_g == suffix_t).sum()),
        "suffix_total": int(suffix_t.numel()),
    }
```

- [ ] **Step 4: Write failing oracle-assessment tests**

```python
def test_sympy_assess_separates_parseability_and_equivalence():
    assert G.sympy_assess("x + x", "2*x") == (True, True)
    assert G.sympy_assess("x + 1", "2*x") == (True, False)
    assert G.sympy_assess("(", "2*x") == (False, False)


def _sleeping_worker(q):
    import time
    time.sleep(10)
    q.put("late")


def test_fork_deadline_kills_worker_without_alarm():
    import time
    t0 = time.monotonic()
    assert G._fork_call(_sleeping_worker, (), timeout=0.01) is None
    assert time.monotonic() - t0 < 1.0
```

- [ ] **Step 5: Replace the one-bit oracle with one forked assessment**

Implement the common timebox and use it for the assessment:

```python
def _fork_call(worker, args, timeout):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(q, *args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    return q.get() if not q.empty() else None


def _sympy_worker(q, a, b):
    try:
        import sympy as sp
        ea = sp.sympify(a)
        eb = sp.sympify(b)
        q.put((True, bool(sp.simplify(ea - eb) == 0)))
    except Exception:
        q.put((False, False))


def sympy_assess(a, b, timeout=10):
    return _fork_call(_sympy_worker, (a, b), timeout) or (False, False)
```

Keep this compatibility wrapper:

```python
def sympy_equiv(a, b, timeout=10):
    return sympy_assess(a, b, timeout)[1]
```

- [ ] **Step 6: Extend `gate` without changing decoding**

Keep the existing autoregressive writes exactly. For each row:

```python
region = answer_region(full, mark, {tok.eos_id, tok.id["\n"]})
counts = token_accuracy_counts(w, full, region)
terminated = any(int(t) in {tok.eos_id, tok.id["\n"]}
                 for t in w[region[0]:T])
parseable, equivalent = sympy_assess(gen_s, truths[wi]) \
    if gen_s else (False, False)
```

Aggregate and print two stable lines per split:

```text
[gate] TRAIN: solves S/8 parseable P/8 terminated E/8
[gate] TRAIN: token-acc standard H/N suffix HS/NS
```

Return the aggregate dict so tests and future instruments need not parse prose.

- [ ] **Step 7: Add and test an explicit prompt-disjoint guard**

Add this helper and call it for the 8/8 gate split before training:

```python
def assert_disjoint_prompts(ids, splits, cut):
    keys = [tuple(ids[i, :splits[i]].tolist())
            for i in range(ids.shape[0])]
    train, heldout = set(keys[:cut]), set(keys[cut:])
    overlap = train & heldout
    if overlap:
        raise ValueError(f"train/heldout prompt overlap: {len(overlap)}")
    return len(train), len(heldout), 0
```

Test it with:

```python
def test_prompt_disjoint_guard():
    ids = torch.tensor([[1, 2, 9], [3, 4, 9]], dtype=torch.int64)
    assert G.assert_disjoint_prompts(ids, [2, 2], 1) == (1, 1, 0)
    ids[1] = ids[0]
    with pytest.raises(ValueError, match="overlap"):
        G.assert_disjoint_prompts(ids, [2, 2], 1)
```

- [ ] **Step 8: Run focused tests and the existing reproduction unit tests**

```bash
PYTHONPATH=. /Users/artin/code/llmopt/.venv/bin/python -m pytest \
  tests/test_gravmoe_answer_only.py tests/test_reproduce.py -q
```

Expected: all focused tests pass; no training arm has fired.

- [ ] **Step 9: Commit the gate readout**

```bash
git add scratch/detbwd_gravmoe.py tests/test_gravmoe_answer_only.py
git commit -m "sol: separate gate suffix accuracy from padding"
```

---

### Task 3: Wire the treatment contract and pre-register it

**Files:**
- Modify: `scratch/detbwd_gravmoe.py:254-261,518-677`
- Modify: `tests/test_gravmoe_answer_only.py`
- Modify: `docs/sol/RESULTS-SOL.md`

**Interfaces:**
- Consumes: `ANSWER_ONLY` environment flag and per-row `answer_region` values.
- Produces: an opt-in AO1 trajectory; default paths remain pinned.

- [ ] **Step 1: Write failing contract tests**

Use subprocess imports so module-level environment parsing is tested in fresh processes:

```python
import os
import subprocess


def _import_with(**knobs):
    env = {k: v for k, v in os.environ.items()
           if k not in {"ANSWER_ONLY", "GATE", "SS"}}
    env.update({k: str(v) for k, v in knobs.items()})
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'scratch'); "
         "import detbwd_gravmoe as g; print(int(g.ANSWER_ONLY))"],
        env=env, text=True, capture_output=True)


def test_answer_only_contract_fences():
    assert _import_with().stdout.strip() == "0"
    assert _import_with(GATE=1, ANSWER_ONLY=1).stdout.strip() == "1"
    no_gate = _import_with(ANSWER_ONLY=1)
    assert no_gate.returncode != 0 and "requires GATE=1" in no_gate.stderr
    combined = _import_with(GATE=1, ANSWER_ONLY=1, SS=1)
    assert combined.returncode != 0 and "separate mechanisms" in combined.stderr
```

- [ ] **Step 2: Add the treatment flag and hard fences**

```python
ANSWER_ONLY = os.environ.get("ANSWER_ONLY") == "1"
if ANSWER_ONLY and not GATE:
    raise ValueError("ANSWER_ONLY requires GATE=1")
if ANSWER_ONLY and SS:
    raise ValueError("ANSWER_ONLY and SS are separate mechanisms")
```

- [ ] **Step 3: Pin and print the real gate contract before training**

In GATE mode, compute marker, terminators, regions, splits, and prompt disjointness before any model construction. Assert:

```python
assert mark == [4, 26]
assert tok.id["\n"] == 27 and tok.eos_id == 1
assert splits[:8] == [15, 10, 15, 15, 19, 15, 12, 15]
```

Print marker IDs, terminator IDs, train regions, full 16-row SHA, and `ANSWER_ONLY` state with `flush=True`.

- [ ] **Step 4: Wire one optional region into the existing training seam**

At each step, select `region = train_regions[row_index] if ANSWER_ONLY else None`, then replace only:

```python
losses.append(loss_proxy(pp, tgt, region))
GG = m.bwd(loss_dlogits(pp, tgt, eye, GB, region), cc, tab)
```

Keep `opt.step([rdiv(GG[n], Q * GB) ...])` unchanged. Update `run_loss` to accept optional row regions and use `loss_proxy`; use masked regions only for AO1's teacher-forced base/merge diagnostics.

- [ ] **Step 5: Run all tests before pre-registration**

```bash
PYTHONPATH=. /Users/artin/code/llmopt/.venv/bin/python -m pytest \
  tests/test_gravmoe_answer_only.py tests/test_intmath.py \
  tests/test_reproduce.py -q
```

Expected: all pass. This is code verification, not an experiment arm.

- [ ] **Step 6: Commit the complete implementation**

```bash
git add scratch/detbwd_gravmoe.py tests/test_gravmoe_answer_only.py
git commit -m "sol: wire answer-only gravmoe gate contract"
```

- [ ] **Step 7: Append the pre-registration before any battery run**

Append `PRE-REG SOL-GRAVMOE-AO1` to `docs/sol/RESULTS-SOL.md` with:

- mechanism and exact region convention;
- marker/newline/EOS IDs, data SHA, train/full row SHAs, splits, and zero prompt overlap;
- implementation commit SHA and design commit `5382b53`;
- mandatory 16/16 pin gate plus G-RB1 expected SHA;
- REG and AO1 commands and planned log paths;
- primary/secondary/format decision rule from the design;
- prediction: AO1 reaches at least 3/8 TRAIN solves because full-token CE currently spends gradient on supplied scaffold and repeated padding; honest alternative: scaffold/padding supervision is load-bearing and AO1 loses termination/parseability;
- HELDOUT expected 0/8;
- explicit n=1 and format-bound fences.

- [ ] **Step 8: Commit the pre-registration separately**

```bash
git add docs/sol/RESULTS-SOL.md
git commit -m "sol: pre-register answer-only gravmoe gate"
```

Record this commit as the pre-run code/documentation SHA in both logs' booking.

---

### Task 4: Re-prove every deterministic pin and establish the new baseline readouts

**Files:**
- Modify: `docs/sol/RESULTS-SOL.md`
- Runtime only: `logs/sol/answer_only_pins.log`, `logs/p4/*.log`

**Interfaces:**
- Consumes: all 16 committed pins and the untracked local diet.
- Produces: a 16/16 boolean regression gate and recomputed G-RB1 standard/suffix/format baseline metrics.

- [ ] **Step 1: Validate the exact local inputs read-only**

```bash
shasum -a 256 /Users/artin/code/llmopt/data/micromodel_gen4_sidecar.jsonl
git status --short --branch
```

Expected diet SHA: `809bce4215a24164ecbf5e951d77507d455bfd1923d08fe39aa02942b11a200b`.

- [ ] **Step 2: Expose the read-only diet to the worktree**

Create a temporary explicit symlink only if the worktree path is absent:

```bash
ln -s /Users/artin/code/llmopt/data/micromodel_gen4_sidecar.jsonl \
  data/micromodel_gen4_sidecar.jsonl
```

Confirm `test -L data/micromodel_gen4_sidecar.jsonl` before later cleanup.

- [ ] **Step 3: Run all 16 arms locally with pipe failure propagation**

```bash
mkdir -p logs/sol
zsh -o pipefail -c 'export RJOB_LOCAL=1; bash <(sed \
  "s|^PY=.*$|PY=/Users/artin/code/llmopt/.venv/bin/python|" \
  scratch/p4_arms_0801.sh) 2>&1 | tee logs/sol/answer_only_pins.log'
```

Expected: `PASS` for A0-A3, CA0-CA3, GA0/GA2/GA3, RB1/RB3/RB1S16, GRB1, and S1; final line `P4 DEVICE LEG: ALL ARMS SHA-IDENTICAL`. Despite the historical label, this invocation is Mac-local and must be booked as a Sol regression replay, not a new device result.

- [ ] **Step 4: Extract REG metrics from the GRB1 log**

```bash
rg 'marker|answer regions|prompt overlap|\[gate\]|FINAL trajectory sha' \
  logs/p4/GRB1.log
```

Require FINAL SHA `1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca`. Record standard `94/140` continuity plus the newly measured suffix-only, parseable, and terminated counts.

- [ ] **Step 5: Stop if any pin or contract assertion fails**

Do not run AO1. Diagnose and fix under a new code commit, amend the pre-registration with the new SHA, and repeat the complete 16-arm gate.

---

### Task 5: Run AO1 and book the verdict immediately

**Files:**
- Modify: `docs/sol/RESULTS-SOL.md`
- Runtime only: `logs/sol/answer_only_ao1.log`

**Interfaces:**
- Consumes: passed 16/16 regression, pre-registered contract, REG metrics.
- Produces: one deterministic AO1 trajectory and a verdict classified by the pre-registered branches.

- [ ] **Step 1: Fire the one treatment arm with incremental logging**

```bash
zsh -o pipefail -c 'RJOB_LOCAL=1 GATE=1 COND=1 QK=1 \
  LN=0 LD=1 STEPS=2000 ANSWER_ONLY=1 \
  /Users/artin/code/llmopt/.venv/bin/python \
  scratch/detbwd_gravmoe.py 2>&1 | tee logs/sol/answer_only_ao1.log'
```

Expected wall is approximately the existing G-RB1 reproduction (~86 seconds), far below the 30-minute justification threshold.

- [ ] **Step 2: Extract the complete evidence card**

```bash
rg 'ANSWER_ONLY|marker|answer regions|\[gate\]|cycle-mean|merge test|FINAL trajectory sha' \
  logs/sol/answer_only_ao1.log logs/p4/GRB1.log
```

Record AO1 trajectory SHA, TRAIN/HELDOUT solves, both accuracy denominators, parseability, termination, teacher-forced masked proxy, and merge diagnostic.

- [ ] **Step 3: Apply the decision rule without reinterpretation**

- `TRAIN >= 3/8`: capability win; explicitly queue paired-seed confirmation.
- `TRAIN == 2/8` and suffix accuracy exceeds REG: graded partial.
- `TRAIN <= 2/8` and suffix accuracy does not exceed REG: null, unless format failure.
- Suffix accuracy down with termination or parseability down: format failure.
- `TRAIN < 2/8` but suffix accuracy up: mixed regression.

Full-140 accuracy is continuity-only and cannot decide the partial branch.

- [ ] **Step 4: Append the verdict immediately**

Append `VERDICT SOL-GRAVMOE-AO1` to `docs/sol/RESULTS-SOL.md`. Include exact commands, log paths, implementation/pre-reg SHAs, data and row SHAs, all 16 regression status, REG/AO1 trajectory SHAs, the full evidence card, prediction grading, chosen decision branch, n=1 fence, format-bound fence, and teacher-forced-loss rider.

- [ ] **Step 5: Commit the booked verdict**

```bash
git add docs/sol/RESULTS-SOL.md
git commit -m "sol: book answer-only gravmoe verdict"
```

---

### Task 6: Close the session with ranked findings and verification

**Files:**
- Modify: `docs/sol/NOTES.md`
- Modify: `docs/sol/RESULTS-SOL.md` only if verification adds a receipt

**Interfaces:**
- Consumes: final committed code, tests, pin replay, REG/AO1 logs, and verdict.
- Produces: required SESSION-NOTES entry and clean proposal branch.

- [ ] **Step 1: Add the required session notes**

Append `## SESSION-NOTES 2026-08-01 — review 2` to `docs/sol/NOTES.md` with:

1. ranked findings, each containing claim, evidence, confidence, and the single cheapest house experiment that would confirm or kill it;
2. what to do next with more budget;
3. any broken house code found, or an explicit `none found` if applicable.

If AO1 wins at 3/8, rank the paired-seed replay above any dose curve. If it fails on format, the next mechanism is a scaffold-weight dose, not a rerun of binary AO1.

- [ ] **Step 2: Remove only the temporary explicit data symlink**

First validate the exact target:

```bash
test -L data/micromodel_gen4_sidecar.jsonl
readlink data/micromodel_gen4_sidecar.jsonl
```

Expected target: `/Users/artin/code/llmopt/data/micromodel_gen4_sidecar.jsonl`. Then remove only that symlink:

```bash
rm data/micromodel_gen4_sidecar.jsonl
```

The source file remains untouched and recoverable at its original path.

- [ ] **Step 3: Run final scoped and broad verification**

```bash
PYTHONPATH=. /Users/artin/code/llmopt/.venv/bin/python -m pytest \
  tests/test_gravmoe_answer_only.py tests/test_intmath.py \
  tests/test_reproduce.py -q

PYTHONPATH=. /Users/artin/code/llmopt/.venv/bin/python -m pytest -q \
  --ignore=tests/test_mlx_backend.py \
  --ignore=tests/test_population.py \
  --ignore=tests/test_exact_gemm.py \
  --ignore=tests/test_fused_ce.py \
  --ignore=tests/test_metal_kernels.py
```

Capture the broad suite with `zsh -o pipefail` and `tee logs/sol/answer_only_pytest.log` in the actual execution. Do not report Metal-dependent failures as green in an environment without Metal.

- [ ] **Step 4: Review the final diff and ledger completeness**

```bash
git diff --check
git status --short --branch
git log --oneline --decorate -8
git diff 32b05cd..HEAD --stat
```

Confirm no living house ledger changed, no log/data artifact is staged, every experimental claim names command/log/SHAs, and the worktree has no unexpected files.

- [ ] **Step 5: Commit session notes and any verification receipt**

```bash
git add docs/sol/NOTES.md docs/sol/RESULTS-SOL.md
git commit -m "sol: close answer-only research session"
```

If `docs/sol/RESULTS-SOL.md` did not change after the verdict commit, stage only `docs/sol/NOTES.md`.

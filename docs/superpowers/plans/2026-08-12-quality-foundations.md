# Quality Foundations Implementation Plan (Phases 0, 1, 2, 7b, 5b-pass)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the code-quality program's foundation batch — ratchet headroom, git reclaim, truthful CODEMAP, de-`/tmp`'d probes, the provenance index, tiered lint, and generated README counts — with zero semantic change to any experiment.

**Architecture:** Everything here is additive or report-only. Provenance becomes data (`code_commit`, `files` in `docs/results-index.jsonl`); the freeze stays untouched; Phase 3 (shims) is a separate plan. Spec: `docs/superpowers/specs/2026-08-12-code-quality-program-design.md`.

**Tech Stack:** Python 3.11+, pytest, ruff, git, GitHub Actions (`.github/workflows/ci.yml`).

## Global Constraints

- PUBLIC REPO: every commit message ends with exactly `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; NEVER a Claude-Session URL.
- RESULTS.md is append-only; corrections are AMENDMENT entries. This plan never edits a past RESULTS entry.
- Never gate a commit on piped pytest — capture rc via redirect: `.venv/bin/python -m pytest -q > /tmp/t.log 2>&1; rc=$?`.
- No history rewrite: no filter-branch/filter-repo/force push. Task 2 prunes unreachable objects only.
- No mass reformat of `scratch/`; no frozen file body is edited (Task 5 ADDS files, edits nothing frozen).
- Line length 79 for new Python.
- Doc voice for any front-facing text touched: spec §5b (no deliberation narration, no first person outside the paper, no em/en dashes in new front-facing text).
- Run everything from repo root with `.venv/bin/python`.
- After adding/changing anything in `scripts/` or `scratch/`: regen `scripts/INDEX.md` (`.venv/bin/python scripts/gen_index.py`) and CODEMAP (`.venv/bin/python scripts/gen_codemap.py`) — but only AFTER the new file is `git add`ed (gen_codemap inventories tracked files only).
- Assumed decision (Artin's standing recommendation, confirm if he is present): ratchet fix = curate, not raise (spec §9.2).

---

### Task 1: FINDINGS ratchet headroom (spec 0.1)

**Files:**
- Create: `scripts/list_uncurated.py`
- Modify: `docs/FINDINGS.md` (append bullets), `tests/test_docs_integrity.py:42` (`MAX_UNCURATED`)

**Interfaces:**
- Consumes: `_uncurated()` logic in `tests/test_docs_integrity.py:68-74`.
- Produces: headroom ≥ 20; `scripts/list_uncurated.py` prints oldest uncurated entries (reused whenever the backlog needs curating).

- [ ] **Step 1: Write the lister**

```python
#!/usr/bin/env python3
"""Print the oldest uncurated ledger entries (candidates for FINDINGS).

Same definition as tests/test_docs_integrity.py::_uncurated: a row in
docs/results-index.jsonl whose RESULTS line is not cited by any
RESULTS.md#L<n> anchor in FINDINGS.md and whose type is curatable.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURATABLE = ("verdict", "null")

def main(n: int = 20) -> None:
    cited = {int(m) for m in re.findall(
        r"RESULTS\.md#L(\d+)",
        (ROOT / "docs" / "FINDINGS.md").read_text())}
    rows = [json.loads(l) for l in
            (ROOT / "docs" / "results-index.jsonl").open()]
    stale = [r for r in rows
             if r["line"] not in cited and r["type"] in CURATABLE]
    stale.sort(key=lambda r: r["line"])  # ledger order = age order
    for r in stale[:n]:
        print(f"L{r['line']:>6}  {r.get('date') or 'undated':<12} "
              f"{r['type']:<8} {r['title']}")
    print(f"\n{len(stale)} uncurated total", file=sys.stderr)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
```

- [ ] **Step 2: Run it, verify count matches the test**

Run: `.venv/bin/python scripts/list_uncurated.py 20`
Expected: 20 rows; stderr says `320 uncurated total` (matches CI).

- [ ] **Step 3: Write one FINDINGS bullet per listed entry (20 bullets)**

For each: read the RESULTS entry at its line (`sed -n '<L>,<L+40>p' docs/RESULTS.md`), then append to the matching section of `docs/FINDINGS.md` one bullet with EXACTLY one maturity tag from `docs/GLOSSARY.md` vocabulary (`RETRACTED`, `NULL`, `MECHANISM-CONFIRMED`, `REPLICATED`, `SINGLE-SEED`), its scope tags, and the anchor `RESULTS.md#L<line>`. Copy the register of neighboring bullets. NULL verdicts get NULL tags — honest losses stay.

- [ ] **Step 4: Lower the ratchet to the new backlog**

Run: `.venv/bin/python scripts/list_uncurated.py 1 2>&1 >/dev/null` → note the new total (expected 300). Edit `tests/test_docs_integrity.py:42` to `MAX_UNCURATED = 300`.

- [ ] **Step 5: Verify suite green**

Run: `.venv/bin/python -m pytest -q tests/test_docs_integrity.py > /tmp/t1.log 2>&1; echo rc=$?; tail -2 /tmp/t1.log`
Expected: rc=0.

- [ ] **Step 6: Commit**

```bash
git add scripts/list_uncurated.py docs/FINDINGS.md tests/test_docs_integrity.py
.venv/bin/python scripts/gen_index.py && git add scripts/INDEX.md docs/CODEMAP.md
git commit -m "docs: curate 20 oldest ledger entries into FINDINGS, ratchet 320 -> 300

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(`gen_codemap.py` also runs via gen_index step order if needed — new file is in `scripts/`, so regen CODEMAP too: `.venv/bin/python scripts/gen_codemap.py` before the `git add` of `docs/CODEMAP.md`.)

---

### Task 2: Reclaim `.git` (spec 0.2) — ops task, no tests

**Files:** none (repository plumbing only).

- [ ] **Step 1: Safety inventory**

```bash
git count-objects -vH                        # record BEFORE
git fsck --lost-found 2>&1 | tail -5         # inventory dangling
git branch -vv                               # every local branch
git log --oneline origin/main..main          # must be empty
```
Expected: `origin/main..main` empty. Local-only branches `sol/review-1`, `sol/review-2`, `sol/present-1` are intentionally local (spec 0.2) — confirm they still exist afterward; do NOT push them.

- [ ] **Step 2: Prune unreachable only**

```bash
rm -f .git/objects/pack/tmp_pack_*           # leftover from interrupted gc
git reflog expire --expire=now --all
git gc --prune=now --aggressive 2>&1 | tail -3
git count-objects -vH                        # record AFTER
```
Expected: size-pack under ~1.5 GB (spec baseline: 984 MB reachable).

- [ ] **Step 3: Verify nothing reachable lost**

```bash
git fsck 2>&1 | tail -3
git log --oneline | wc -l        # expected 2083 + commits since spec
git branch | wc -l               # unchanged
.venv/bin/python -m pytest -q tests/test_docs_integrity.py > /tmp/t2.log 2>&1; echo rc=$?
```
Expected: fsck clean, branch count unchanged, rc=0. Record before/after sizes in the NEXT commit's message body (no commit is created by gc itself).

---

### Task 3: CODEMAP stops hiding citations (spec 0.3)

**Files:**
- Modify: `scripts/gen_codemap.py` (`classify()` at ~line 131, table emit)
- Modify: `.claude/hooks/codemap_guard.py` (row regex — column count grows)
- Test: `tests/test_codemap_inventory.py` (existing; must stay green)

**Interfaces:**
- Produces: CODEMAP table gains a `cited` column (`RESULTS`/`REPRODUCE`/`specs` union or `—`); `classify()` unchanged in ladder but no longer the only citation signal. Census line gains `cited-but-library N`.

- [ ] **Step 1: Change the table row emit**

In `gen_codemap.py`, where rows are formatted, add a fifth data column `cited` = comma-joined keys of `doc_cites` result (or `—`). Header becomes `| family | file | class | cited by | doc citations | refs |` — read the emit code first and match its exact style; keep `class` in column 3 (the guard hook parses columns 1-2 after the leading pipe).

- [ ] **Step 2: Update the census line**

After classify, count `hidden = sum(1 for f in files if f.class == "library" and f.cites)` and append `, cited-but-library {hidden}` to the census line. Expected ≈ 103 (spec 1.2).

- [ ] **Step 3: Regenerate and eyeball**

Run: `.venv/bin/python scripts/gen_codemap.py && head -20 docs/CODEMAP.md`
Expected: census shows `cited-but-library 103` (±small drift); `gate_ckpt.py` row shows citations despite class `library`.

- [ ] **Step 4: Verify guard hook still parses**

Run: `echo '{"tool_input":{"file_path":"'$PWD'/scratch/attractor_census2.py"}}' | .venv/bin/python .claude/hooks/codemap_guard.py`
Expected: JSON with `permissionDecision": "ask"`. If the row regex broke, fix `codemap_guard.py::codemap_class` to the new column layout.

- [ ] **Step 5: Suite + commit**

Run: `.venv/bin/python -m pytest -q tests/test_codemap_inventory.py tests/test_docs_integrity.py > /tmp/t3.log 2>&1; echo rc=$?`
Expected: rc=0.

```bash
git add scripts/gen_codemap.py docs/CODEMAP.md .claude/hooks/codemap_guard.py
git commit -m "codemap: cited-by column so library class stops masking 103 citations

before/after .git sizes from Task 2: <paste git count-objects lines>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: CODEMAP `refs` split into imports vs mentions (spec 0.4)

**Files:**
- Modify: `scripts/gen_codemap.py` (`code_refs()` at ~line 106, `classify()` at ~line 131)
- Test: `tests/test_codemap_inventory.py` (existing)

**Interfaces:**
- Produces: `code_refs()` returns `(imports: list[str], mentions: list[str])`; `classify()` takes `imports` for the `library` test; table shows both counts.

- [ ] **Step 1: Split the matcher**

```python
def code_refs(target: Path, code: dict[str, str]) -> tuple[list[str], list[str]]:
    """(importers, mention-only referrers) for the module."""
    name = target.name
    mod = target.stem
    rel = str(target.relative_to(ROOT))
    imp = re.compile(
        rf"^\s*(?:import\s+{re.escape(mod)}\b"
        rf"|from\s+{re.escape(mod)}\s+import)", re.MULTILINE)
    imports, mentions = [], []
    for path, text in code.items():
        if path == rel:
            continue
        if target.suffix == ".py" and imp.search(text):
            imports.append(path)
        elif name in text:
            mentions.append(path)
    return imports, mentions
```

- [ ] **Step 2: classify() uses imports only**

```python
def classify(cites: dict[str, int], imports: list[str]) -> str:
    if imports:
        return "library"
    ...  # rest unchanged
```
Shell drivers (`.sh`) can only be mentioned, never imported — a `.sh` with refs was always mention-class; verify a few reclassifications by hand (`gate_ckpt.py` must leave `library`, spec says 38 refs / 0 imports → becomes `results-cited`).

- [ ] **Step 3: Table emit shows `imports` and `mentions` counts** (replacing the single `refs` column). Update the header comment block at the top of CODEMAP accordingly (it is emitted by the script).

- [ ] **Step 4: Regenerate, verify census shift, suite, hook**

```bash
.venv/bin/python scripts/gen_codemap.py
grep "Census" docs/CODEMAP.md      # library count drops ~103, cited classes rise
grep "gate_ckpt" docs/CODEMAP.md   # now results-cited
.venv/bin/python -m pytest -q tests/test_codemap_inventory.py > /tmp/t4.log 2>&1; echo rc=$?
echo '{"tool_input":{"file_path":"'$PWD'/scratch/gate_ckpt.py"}}' | .venv/bin/python .claude/hooks/codemap_guard.py
```
Expected: rc=0; the hook now ASKS on `gate_ckpt.py` (it became frozen class — correct, it is cited evidence).

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_codemap.py docs/CODEMAP.md
git commit -m "codemap: refs split into imports vs mentions; library requires a real import

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Materialize the five `/tmp` probe products (spec 0.5)

**Files:**
- Create: `scratch/frozen_products/g19bn_probe.py`, `gen8_probe_template_note.md` — see Step 1; exact set depends on parameterization below.
- Modify: none. Frozen drivers are NOT edited (they are evidence).

**Interfaces:**
- Produces: every program a booked verdict executed exists in git. Future drivers take parameters instead of sed (rule already in `/rung`).

- [ ] **Step 1: Enumerate the sed products**

The five drivers and their products:

| driver | product | sed transform |
|---|---|---|
| `scratch/g19_bf16_isolation.sh:9-12` | `/tmp/g19bn_probe.py` | data file, arch dims (`d=384, layers=8, ffn=1536, heads=6`), device mps→cuda |
| `scratch/gen8_pipeline.sh:45` | `/tmp/gen8_$P.py` (per-P loop) | data file only |
| `scratch/gen9_pipeline.sh:61` | `/tmp/gen9_$P.py` (per-P loop) | data file only |
| `scratch/poly3_pipeline.sh:59` | `/tmp/poly3_probe.py` | data file only |
| `scratch/poly4_pipeline.sh:59` | `/tmp/poly4_probe.py` | data file only |
| `scratch/poly5_pipeline.sh:74` | `/tmp/poly5_probe.py` | data file only |

All patch `scratch/series_probe.py`. The loop-generated ones (`gen8`, `gen9`) differ from the source by ONE string per iteration; committing every iteration is noise. Materialize the two distinct SHAPES instead: the g19 product (three simultaneous transforms — the risky one) verbatim, and one representative single-substitution product per family.

- [ ] **Step 2: Generate them from the frozen source, verify the sed still bites**

```bash
mkdir -p scratch/frozen_products
sed -e "s/series_probe.jsonl/series_probe_1e.jsonl/" \
    -e "s/build_model(len(tok.vocab))/build_model(len(tok.vocab), d=384, layers=8, ffn=1536, heads=6)/" \
    -e "s/\"mps\" if torch.backends.mps.is_available() else \"cpu\"/\"cuda\" if torch.cuda.is_available() else \"cpu\"/" \
    scratch/series_probe.py > scratch/frozen_products/g19bn_probe.py
diff scratch/series_probe.py scratch/frozen_products/g19bn_probe.py | grep -c "^[<>]"
```
Expected: diff count ≥ 6 (three substitutions, each a `<`+`>` pair). **If any expected substitution produced no diff line, the sed has silently no-opped against drifted source — STOP and flag; the booked g19 numbers then need an AMENDMENT, which is out of this plan's scope.** Repeat with the single-substitution seds for `poly3`:

```bash
sed "s/series_probe.jsonl/poly3_probe.jsonl/" scratch/series_probe.py \
    > scratch/frozen_products/poly_probe_representative.py
```

- [ ] **Step 3: Provenance headers**

Prepend to each generated file (as a comment) three lines: which driver generated it, the exact sed command, and the date materialized. Add `scratch/frozen_products/README.md` (5 lines): these are the exact programs the cited drivers executed from `/tmp`; regenerating them is the two sed commands above; new drivers must parameterize instead (see `/rung`).

- [ ] **Step 4: Sanity: products compile**

Run: `.venv/bin/python -m py_compile scratch/frozen_products/*.py && echo OK`
Expected: OK.

- [ ] **Step 5: Commit, then regen inventories (tracked-files rule)**

```bash
git add scratch/frozen_products
git commit -m "scratch: materialize the /tmp sed-probe products into git

The five pipeline drivers execute sed-patched copies of
scratch/series_probe.py from /tmp; the executed programs existed
nowhere in git. The two distinct product shapes are now committed
with provenance headers. Drivers untouched: they are evidence.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
.venv/bin/python scripts/gen_codemap.py && .venv/bin/python scripts/gen_index.py
git add docs/CODEMAP.md scripts/INDEX.md
git commit -m "chore: regen CODEMAP+INDEX for frozen_products

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
.venv/bin/python -m pytest -q tests/test_codemap_inventory.py > /tmp/t5.log 2>&1; echo rc=$?
```
Expected: rc=0.

---

### Task 6: Provenance index — `files` extraction (spec 1.1)

**Files:**
- Modify: `scripts/gen_results_index.py`
- Test: `tests/test_results_index_files.py` (new)

**Interfaces:**
- Produces: every row in `docs/results-index.jsonl` gains `"files": [...]` (possibly empty); `extract_files(body: str) -> list[str]` importable from the generator. Regeneration PRESERVES existing curation (`threads`, `links`, and later `code_commit`) — read the generator's existing preserve logic first and extend it, do not rewrite it.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_results_index_files.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_extract_files_finds_repo_paths():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gri", ROOT / "scripts" / "gen_results_index.py")
    gri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gri)
    body = ("Driver: scratch/softprompt1.py wraps "
            "scripts/step_grpo_micro.py; receipts in logs/x.log "
            "and llmopt/lab/gate.py. Not a path: results.md")
    assert gri.extract_files(body) == [
        "llmopt/lab/gate.py",
        "scratch/softprompt1.py",
        "scripts/step_grpo_micro.py",
    ]

def test_every_index_row_has_files_key():
    rows = [json.loads(l) for l in
            (ROOT / "docs" / "results-index.jsonl").open()]
    assert all("files" in r for r in rows)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_results_index_files.py -x > /tmp/t6.log 2>&1; echo rc=$?; tail -3 /tmp/t6.log`
Expected: rc=1, `AttributeError` (no `extract_files`).

- [ ] **Step 3: Implement**

In `gen_results_index.py` add:

```python
FILE_RE = re.compile(r"\b((?:scratch|scripts|llmopt)/[\w./-]+\.(?:py|sh))\b")

def extract_files(body: str) -> list[str]:
    """Sorted unique repo paths cited in an entry body."""
    return sorted(set(FILE_RE.findall(body)))
```

and set `row["files"] = extract_files(entry_body)` where rows are built. The generator already walks entry bodies to build rows — attach at that point. Keep the preserve-curation merge intact: `files` is regenerated every run (it is derived, not curated).

- [ ] **Step 4: Regenerate + test**

```bash
.venv/bin/python scripts/gen_results_index.py
.venv/bin/python -m pytest -q tests/test_results_index_files.py tests/test_docs_integrity.py > /tmp/t6b.log 2>&1; echo rc=$?
```
Expected: rc=0. Spot check: `jq -c 'select(.id | test("metallicity"))' docs/results-index.jsonl | head -1` shows `files` containing the metallicity driver paths.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_results_index.py docs/results-index.jsonl tests/test_results_index_files.py
git commit -m "index: every ledger row carries the repo files its entry cites

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `code_commit` backfill (spec 1.3)

**Files:**
- Create: `scripts/backfill_code_commit.py`
- Modify: `docs/results-index.jsonl` (one-time enrichment)

**Interfaces:**
- Consumes: row `title` + `line` from the index.
- Produces: rows gain `"code_commit": "<sha>"` or `"code_commit": null`. Null over guess, always. The generator must PRESERVE this field on regeneration — extend its preserve-set (same mechanism as `threads`) in this task.

- [ ] **Step 1: Preserve-set first**

In `gen_results_index.py`, add `code_commit` to the fields carried over from the existing file on regen (find where `threads`/`links` survive; add the key). Verify: run the generator twice, `git diff docs/results-index.jsonl` empty the second time.

- [ ] **Step 2: Write the backfill**

```python
#!/usr/bin/env python3
"""One-time: attach code_commit to existing ledger rows.

For each row, the booking commit is the one that introduced the
entry heading into RESULTS.md (git log -S). code_commit is that
commit's PARENT: the tree checked out when the driver ran. One hit
= confident; zero or 2+ hits = null (a null is honest, a wrong sha
is a trap — spec 1.3).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "docs" / "results-index.jsonl"

def find_commit(title: str) -> str | None:
    out = subprocess.run(
        ["git", "log", "--format=%H", "-S", title, "--", "docs/RESULTS.md"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    if len(out) != 1:
        return None
    parent = subprocess.run(
        ["git", "rev-parse", f"{out[0]}^"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return parent or None

def main() -> None:
    rows = [json.loads(l) for l in IDX.open()]
    nulls = 0
    for i, r in enumerate(rows):
        if r.get("code_commit"):
            continue
        r["code_commit"] = find_commit(r["title"])
        nulls += r["code_commit"] is None
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", file=sys.stderr)
    IDX.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"done: {len(rows)} rows, {nulls} null")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run (slow: ~930 `git log -S` calls; minutes, not hours)**

Run: `.venv/bin/python scripts/backfill_code_commit.py`
Expected: final line reports the null count — RECORD IT (spec exit criterion). Titles that recur verbatim (e.g. "The one-paragraph version") land null by the 2+-hits rule; correct.

- [ ] **Step 4: Verify preservation + suite**

```bash
.venv/bin/python scripts/gen_results_index.py
jq -r '.code_commit' docs/results-index.jsonl | sort | uniq -c | tail -3   # shas survive regen
.venv/bin/python -m pytest -q tests/test_docs_integrity.py tests/test_results_index_files.py > /tmp/t7.log 2>&1; echo rc=$?
```
Expected: rc=0; non-null shas present after regen.

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_code_commit.py scripts/gen_results_index.py docs/results-index.jsonl
git commit -m "index: backfill code_commit (parent of booking commit); <N> of 930 null

Null where git log -S is ambiguous. A null is a known gap; a wrong
sha is a trap that looks like evidence.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: `results_query.py --repro` + `/book`,`/rung` doctrine (spec 1.2, 1.4, §6)

**Files:**
- Modify: `scripts/results_query.py`, `.claude/skills/book/SKILL.md`, `.claude/skills/rung/SKILL.md`
- Test: `tests/test_results_query_repro.py` (new)

**Interfaces:**
- Produces: `--repro <id>` prints `git worktree add ../repro-<id> <code_commit>` plus the row's `files`; skills record `code_commit` (parent sha at booking; `/rung` captures launch sha).

- [ ] **Step 1: Failing test**

```python
"""tests/test_results_query_repro.py"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_repro_prints_worktree_command():
    row = next(json.loads(l) for l in
               (ROOT / "docs" / "results-index.jsonl").open()
               if json.loads(l).get("code_commit"))
    out = subprocess.run(
        [sys.executable, "scripts/results_query.py", "--repro", row["id"]],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0
    assert f"git worktree add" in out.stdout
    assert row["code_commit"][:12] in out.stdout
```

- [ ] **Step 2: Run to fail** — `.venv/bin/python -m pytest -q tests/test_results_query_repro.py > /tmp/t8.log 2>&1; echo rc=$?` — expected rc=1 (unknown flag).

- [ ] **Step 3: Implement in `results_query.py`** (read its arg handling first; match style):

```python
def repro(rows, entry_id: str) -> int:
    row = next((r for r in rows if r["id"] == entry_id), None)
    if row is None:
        print(f"no entry with id {entry_id}", file=sys.stderr)
        return 1
    sha = row.get("code_commit")
    if not sha:
        print(f"{entry_id}: code_commit is null (ambiguous backfill); "
              f"fall back to the booking commit via "
              f"git log -S over docs/RESULTS.md", file=sys.stderr)
        return 1
    print(f"git worktree add ../repro-{entry_id[:24]} {sha}")
    for f in row.get("files", []):
        print(f"  # cited: {f}")
    return 0
```

- [ ] **Step 4: Test passes; verify BY HAND on ten entries** (spec exit criterion): pick 10 ids with non-null `code_commit`, run `--repro`, `git worktree add` one of them, confirm the cited files exist in that worktree at the expected content, `git worktree remove` it.

- [ ] **Step 5: Skill updates.** `/book` SKILL.md step 3 gains: "set `code_commit` on the new row = parent of the booking commit (`git rev-parse HEAD^` after committing), and verify `files` was auto-extracted." `/rung` gains one line in the launch step: "record `git rev-parse HEAD` at launch; it becomes the entry's `code_commit` at booking."

- [ ] **Step 6: Commit**

```bash
git add scripts/results_query.py tests/test_results_query_repro.py .claude/skills/book/SKILL.md .claude/skills/rung/SKILL.md
git commit -m "query: --repro prints the worktree command; /book and /rung record code_commit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: ruff, tiered (spec 2.1)

**Files:**
- Modify: `pyproject.toml`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `ruff check llmopt tests scripts` gate in CI; `ruff check scratch --exit-zero` report.

- [ ] **Step 1: Config in pyproject.toml**

```toml
[tool.ruff]
line-length = 79
target-version = "py311"

[tool.ruff.lint]
# Narrow on purpose: dead imports/names only. Style rules wait.
select = ["F401", "F821", "F841"]

[tool.ruff.lint.per-file-ignores]
"scratch/*" = ["ALL"]
```

- [ ] **Step 2: Run and fix the enforced tier**

Run: `ruff check llmopt tests scripts 2>&1 | tail -5` (install if absent: `.venv/bin/pip install ruff`, and add `ruff` to the CI install line).
Fix findings in FREE files only (check CODEMAP class first — `scripts/` contains frozen `results-cited` files; a frozen file with an F-finding gets a `# noqa` line NEVER a body edit... a noqa IS a body edit — instead add its path to per-file-ignores with a comment naming the frozen class). Expected: exit 0 after fixes.

- [ ] **Step 3: CI steps** (append to `.github/workflows/ci.yml` jobs.tests.steps):

```yaml
      - name: lint (enforced tier)
        run: ruff check llmopt tests scripts
      - name: lint (scratch, report only)
        run: ruff check scratch --exit-zero
```

- [ ] **Step 4: Suite + commit**

```bash
.venv/bin/python -m pytest -q > /tmp/t9.log 2>&1; echo rc=$?; tail -2 /tmp/t9.log
git add pyproject.toml .github/workflows/ci.yml <fixed files>
git commit -m "lint: ruff tiered - F-rules enforced on llmopt/tests/scripts, scratch report-only

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Import-direction guard + docs marker (spec 2.2, 2.3)

**Files:**
- Create: `tests/test_layering.py`
- Modify: `pyproject.toml` (markers), `tests/test_docs_integrity.py` + `tests/test_codemap_inventory.py` (add marker), `CLAUDE.md` (document)

**Interfaces:**
- Produces: `pytest -m docs` runs doc guards; `pytest -m "not docs"` is the code suite; layering test locks `scripts/ never imports scratch/`.

- [ ] **Step 1: Failing test (it should pass immediately — the property holds; verify the test CAN fail by temporarily grepping its own logic)**

```python
"""tests/test_layering.py — scripts/ never imports scratch/ (144:0 measured 2026-08-12)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT_SCRATCH = re.compile(
    r"^\s*(?:from\s+scratch[.\s]|import\s+scratch\b"
    r"|from\s+(\w+)\s+import)", re.M)

def test_scripts_never_import_scratch():
    scratch_mods = {p.stem for p in (ROOT / "scratch").glob("*.py")}
    offenders = []
    for p in (ROOT / "scripts").glob("*.py"):
        for m in re.finditer(r"^\s*(?:import\s+(\w+)|from\s+(\w+)\s+import)",
                             p.read_text(), re.M):
            mod = m.group(1) or m.group(2)
            if mod in scratch_mods:
                offenders.append(f"{p.name} imports {mod}")
    assert not offenders, offenders
```

Caveat baked in: scratch modules are imported by BARE name (sys.path bootstraps), so the check is stem-set membership, not `import scratch.`. A scripts module sharing a stem with a scratch module would false-positive — if that fires, disambiguate by checking the file does not exist in `scripts/` itself.

- [ ] **Step 2: Run** — expected PASS (property already holds). Sanity: temporarily add `import series_probe` to a throwaway scripts file, see it FAIL, delete the file.

- [ ] **Step 3: Markers**

pyproject: `[tool.pytest.ini_options]` gains `markers = ["docs: ledger/inventory integrity guards"]`. Add `pytestmark = pytest.mark.docs` at top of `tests/test_docs_integrity.py` and `tests/test_codemap_inventory.py`. CI: leave the plain `pytest` run (runs both); CLAUDE.md Practical section gains one line: "`pytest -m "not docs"` = code only; `-m docs` = ledger guards; plain pytest = both (CI's run)."

- [ ] **Step 4: Verify + commit**

```bash
.venv/bin/python -m pytest -q -m docs > /tmp/t10a.log 2>&1; echo docs_rc=$?
.venv/bin/python -m pytest -q > /tmp/t10b.log 2>&1; echo all_rc=$?; tail -2 /tmp/t10b.log
git add tests/test_layering.py pyproject.toml tests/test_docs_integrity.py tests/test_codemap_inventory.py CLAUDE.md
git commit -m "tests: layering guard (scripts never imports scratch) + docs marker

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Generated README front matter (spec 7b)

**Files:**
- Create: `scripts/gen_readme.py`, `tests/test_gen_readme.py`
- Modify: `README.md` (region markers around the ledger counts), `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `gen_readme.py` (no args = rewrite in place; `--check` = exit 1 on drift); counts derived from `docs/FINDINGS.md` maturity tags — the same source `test_docs_integrity.py` polices.

- [ ] **Step 1: Failing test**

```python
"""tests/test_gen_readme.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_readme_ledger_counts_match_findings():
    out = subprocess.run(
        [sys.executable, "scripts/gen_readme.py", "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
```

- [ ] **Step 2: Implement**

```python
#!/usr/bin/env python3
"""Rewrite generated regions in README.md from ledger truth.

Region syntax:
  <!-- llmopt:generated honesty-ledger:start -->
  ...replaced content...
  <!-- llmopt:generated honesty-ledger:end -->
--check exits 1 if a rewrite would change the file (CI drift gate).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("REPLICATED", "MECHANISM-CONFIRMED", "SINGLE-SEED", "NULL",
        "RETRACTED")

def counts() -> dict[str, int]:
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    # Count tag occurrences the same way the integrity test scopes
    # them: one maturity tag per bullet line.
    return {t: len(re.findall(rf"\*\*{t}\*\*", text)) for t in TAGS}

def render() -> str:
    c = counts()
    total = sum(c.values())
    return (f"The {total} curated claims in FINDINGS by maturity: "
            f"{c['REPLICATED']} replicated, "
            f"{c['MECHANISM-CONFIRMED']} mechanism-confirmed, "
            f"{c['SINGLE-SEED']} single-seed, {c['NULL']} null, "
            f"{c['RETRACTED']} retracted.")

def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text()
    pat = re.compile(
        r"(<!-- llmopt:generated honesty-ledger:start -->\n)"
        r".*?"
        r"(\n<!-- llmopt:generated honesty-ledger:end -->)", re.S)
    if not pat.search(text):
        print("no generated region markers in README", file=sys.stderr)
        return 2
    new = pat.sub(lambda m: m.group(1) + render() + m.group(2), text)
    if "--check" in sys.argv:
        if new != text:
            print("README ledger counts drifted; run scripts/gen_readme.py")
            return 1
        return 0
    readme.write_text(new)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

CHECK FIRST: grep FINDINGS for the actual bullet tag format (`**REPLICATED**` vs bare) and match `counts()` to it — the number must equal what `test_docs_integrity.py` counts, current truth 37/43/73/35/3 = 191 per spec 7b (may have moved; regenerate, don't assume).

- [ ] **Step 3: Mark the README region**

`README.md:84` is an `<img alt="The 187 curated claims...">` — an alt attribute cannot hold HTML comments. Restructure: wrap the SENTENCE-BEARING content. Put the generated sentence as visible caption text under the img inside the markers, and shorten the alt to the stable phrase "FINDINGS ledger by maturity" (numbers live in the generated caption, typed once). Also sweep README for any other hand-typed ledger count and pull it into the region.

- [ ] **Step 4: Generate, test, CI**

```bash
.venv/bin/python scripts/gen_readme.py && git diff README.md | head -20
.venv/bin/python -m pytest -q tests/test_gen_readme.py > /tmp/t11.log 2>&1; echo rc=$?
```
CI step (after lint steps):

```yaml
      - name: generated README in sync
        run: python scripts/gen_readme.py --check
```

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_readme.py tests/test_gen_readme.py README.md .github/workflows/ci.yml
.venv/bin/python scripts/gen_index.py && .venv/bin/python scripts/gen_codemap.py
git add scripts/INDEX.md docs/CODEMAP.md
git commit -m "readme: ledger counts become a generated region; CI fails on drift

Fixes the class: README said 187 while FINDINGS held 191. The same
number is never typed twice.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: The §5b voice pass

**Files:**
- Modify: `docs/THEORY.md` (6 first-person instances), `docs/REPRODUCE.md` (5 "the house"), `docs/FINDINGS.md` (4 "the house"), `CLAUDE.md` (the rule)

**Interfaces:** none — prose only.

- [ ] **Step 1: Locate**

Run: `grep -n "\bwe\b\|\bour\b\|\bus\b" docs/THEORY.md; grep -n "the house" docs/REPRODUCE.md docs/FINDINGS.md`
Expected: ~6 + ~9 hits (measured 2026-08-12; may drift).

- [ ] **Step 2: Rewrite each hit** per §5b: deliberation narration becomes result-implied ("so we decided to test Y" → "Y was the next test the result implied"); "the house" becomes the plain subject (the lab, the instrument, or the result doing the work). FINDINGS bullets keep their tags and anchors byte-identical — only the prose around them moves. Do NOT touch `docs/paper/main.tex` (methodological `we` stays, rule 2).

- [ ] **Step 3: CLAUDE.md rule** — append to the conventions section, 4 lines: front-facing docs (README, REPRODUCE, project pages, posts) use the §5b style: no deliberation narration, no first person, no em/en dashes in new text, concrete numbers over praise; full text in spec 2026-08-12 §5b.

- [ ] **Step 4: Verify anchors survived + suite**

```bash
.venv/bin/python -m pytest -q -m docs > /tmp/t12.log 2>&1; echo rc=$?
grep -c "RESULTS.md#L" docs/FINDINGS.md   # unchanged count
```
Expected: rc=0, anchor count identical to before the edit.

- [ ] **Step 5: Commit**

```bash
git add docs/THEORY.md docs/REPRODUCE.md docs/FINDINGS.md CLAUDE.md
git commit -m "docs: front-facing voice pass - no deliberation narration, no in-group jargon

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: Close out — baseline delta + push

**Files:**
- Modify: `docs/BOARD.md` (program row), spec §10 table gets a dated actuals column appended in a short postscript block (the spec is a committed design doc; appending an actuals table is an update, not a rewrite).

- [ ] **Step 1: Re-derive the baseline table** (spec §10 commands) and record actuals: `.git` size, headroom, drivers-from-/tmp count (target 0 — products materialized), `code_commit` null count, README-vs-FINDINGS equality.

- [ ] **Step 2: Full suite on a real exit code**

Run: `.venv/bin/python -m pytest -q > /tmp/t13.log 2>&1; echo rc=$?; tail -2 /tmp/t13.log`
Expected: rc=0.

- [ ] **Step 3: BOARD row + push**

Update the quality-program BOARD row to "foundations landed (Phases 0,1,2,7b,5b-pass); next: Phase 3 keepsets shim, own session". Commit with the actuals in the body; push; verify `git ls-remote --heads origin main` matches local HEAD.

---

## Self-Review (performed at write time)

- **Spec coverage:** Phase 0: 0.1→T1, 0.2→T2, 0.3→T3, 0.4→T4, 0.5→T5. Phase 1: 1.1→T6, 1.2→T8, 1.3→T7, 1.4→T8. Phase 2: 2.1→T9, 2.2→T10, 2.3→T10. 7b→T11. §5b→T12. §6 doctrine: /book,/rung→T8; CLAUDE.md lint/marker→T9,T10; voice→T12. NOT in this plan (deliberate): Phases 3-7a/7c/7 (separate plans; Phase 3 first, own session per spec §8).
- **Open-decision dependency:** Task 1 assumes "curate" (spec §9.2 recommendation). Tasks touching `pick_device`/`common/` are NOT in this plan, so §9.1/9.3/9.4 stay open without blocking.
- **Placeholder scan:** all code blocks concrete; two deliberate read-first instructions (gen_results_index preserve logic, FINDINGS tag format) are verification steps, not deferred design.
- **Type consistency:** `extract_files` (T6) is the name `test_results_index_files.py` imports; `code_commit` spelling uniform across T7/T8; guard-hook column dependency named in T3/T4.

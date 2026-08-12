# Design: the code-quality program (2026-08-12)

Status: DESIGN, unapproved. No implementation has started.
Author: Opus 5 seat, llmopt Mac. Decisions taken by Artin 2026-08-12.

This is a program spec, not an experiment pre-registration. No bars, no
registered prior: nothing here is a claim about the world. The bar that
does apply is the house verification bar — pytest green on a real exit
code, claims checked against source, no number quoted that was not
measured.

---

## 1. The problem, measured

Every number below was measured on 2026-08-12 at HEAD `2671030`. The
commands are in §10 so the baseline can be re-derived.

### 1.1 The extraction happened; the migration did not

Six of twelve `llmopt/lab` modules have **zero production callers**:
`gate`, `verify`, `gen`, `keepsets`, `oracle_worker`, `merge`. Each is a
verbatim copy of a scratch/scripts original, held in sync by a test that
asserts both bodies are character-identical.

The repo's real library is four files in `scripts/`:

| module | imported by |
|---|---|
| `scripts/step_grpo_micro.py` | 93 files |
| `scripts/bench_step_tokens.py` | 68 |
| `scripts/train_mathnative.py` | 48 |
| `scripts/bench_verify_fast.py` | 47 |

The lab therefore pays the entire cost of duplication — 16 symbols that
must be edited in two files in one commit — and collects none of the
benefit, because nothing calls the package copy. Worse, CLAUDE.md's
dual-copy rule freezes BOTH copies: "package-only improvements WAIT
until a registered re-run migrates the driver." Neither copy can move.

### 1.2 The freeze does not deliver reproducibility

- **96 of the 178 explicitly-cited files import `llmopt` modules that
  are still changing**, across 28 distinct modules.
  `llmopt.train.mathnative` alone has 59 cited dependents and was last
  modified 2026-07-24. Nothing guards those 59 against the next edit.
- **At least five drivers `sed`-patch a frozen source file into `/tmp`
  and execute that.** `scratch/g19_bf16_isolation.sh:9-12` rewrites the
  device, the architecture (`d=384, layers=8, ffn=1536, heads=6`), and
  the data filename, then runs `/tmp/g19bn_probe.py`. Booked numbers
  came from a program that exists nowhere in git, and the patch is
  coupled to exact source text — a whitespace edit silently no-ops the
  device swap while the run still "succeeds". Also in
  `gen8_pipeline.sh`, `gen9_pipeline.sh`, `poly3/4/5_pipeline.sh`.
  This is the freeze causing the damage it exists to prevent: the file
  cannot be edited, so drivers rewrite it at runtime.
- **The move gate hides citations.** `scripts/gen_codemap.py:132` returns
  `library` before testing citation, so 103 cited files are reported as
  `library`. The true evidence-record size is **281 files, not 178**;
  genuinely free files number 231.

### 1.3 46 of 47 GB of `.git` is garbage, not history

Reachable history across all 2,083 commits is **984 MB** (4,698 blobs =
975 MB, 5,859 trees = 8.3 MB, 2,084 commits = 0.9 MB). `.git` is 47 GB:
7,158 loose objects at 23.48 GiB, five packs at 23.10 GiB, plus an
85.63 MiB leftover `tmp_pack` from an interrupted gc. The excess is
unreachable objects from amends and resets. It is not history.

### 1.4 Ordinary debt, quantified

| item | measured |
|---|---|
| lint / format / type config | none exists |
| `sys.path` bootstrap | 292 of ~424 py files, 8 spellings (scratch 226/274, scripts 66/150) — and `llmopt` is pip-installed editable, so all are unnecessary |
| copy-pasted helpers with no package home | `_root` ×28, `_check` ×27, `load_nnue` ×13, `ternary` ×11; ~64% of copies sit in files free to change |
| device-selection ladder | open-coded in 126 files, and inconsistent: `llmopt/lab/gate.py:177` prefers MPS, `llmopt/train/ref_logprobs.py:82` prefers CUDA. The package's one `_device()` (`lab/runlog.py:48`) is called by nothing |
| shell drivers | 90 `.sh`, all in `scratch/`; `set -e` ×46 but `pipefail` only ×6 |
| FINDINGS ratchet | 320 of 320. **Headroom zero** — the next verdict reddens the suite |
| tests | 659 collected, 653 pass, 7 skip (triton ×1, MSVC codegen ×6) |
| untested and load-bearing | `llmopt/search/zx_engine.py`, 229 lines, backs booked verdicts, zero tests |
| llmopt modules > 400 lines | 5: `kernels/metal.py` 1120, `search/rules.py` 1013, `mathgen/problems.py` 731, `kernels/triton_kernels.py` 568, `search/derivation.py` 490 |
| import direction | `scratch/`→`scripts/` 144, `scripts/`→`scratch/` 0. Clean one-way layering, enforced by nothing |

---

## 2. The governing idea

**Provenance is stored as a filesystem property when it should be data.**

Git already preserves every byte of every commit permanently. Freezing
the working-tree copy adds no reproducibility that git does not already
provide. What it adds is a tax that grows every week — 281 files today —
while failing to protect the 96 cited files whose imports still move,
and while pushing drivers into the `/tmp` workaround that produced
genuinely unreproducible arms.

Record `verdict → commit + files` in the ledger, and reproduction becomes
`git worktree add ../repro <sha>`. That is **stronger** than the freeze:
it restores the entire tree state, not one leaf file whose dependencies
may have drifted underneath it.

Once provenance is data, "keep the history" and "integrate reusable
parts into llmopt" stop competing.

### 2.1 The three decisions taken

1. **Shim + provenance index.** Add `code_commit` and `files` to the ledger
   first (purely additive). Then convert frozen originals into thin
   re-exports of the package symbol, one at a time, deleting each
   source-identity guard in the same commit. This is the endpoint the
   2026-08-05 extraction spec already named: "becomes a thin importer of
   lab.keepsets only after the byte-identity test passes."
2. **Reclaim the 46 GB now**, after safety checks, keeping all 2,083
   commits and every branch.
3. **Tiered lint**: enforced on `llmopt/` and `tests/`, report-only on
   `scratch/`, so the notebook stays a notebook and no frozen file is
   ever rewritten by a formatter.

---

## 3. Non-goals

- **No history rewrite.** No `filter-branch`, no `filter-repo`, no force
  push. The 46 GB reclaim drops unreachable objects only.
- **No mass reformat of `scratch/`.** It is the lab notebook and much of
  it is evidence.
- **No re-running of booked experiments.** Nothing here re-opens a
  verdict. Where a shim needs behavioral proof, the proof is an existing
  booked number reproduced, not a new run.
- **No change to RESULTS.md's append-only law**, the pre-registration
  discipline, or any scientific fence.
- **No new capability work.** This program moves and tests existing code.

---

## 4. Phases

Each phase is independently shippable, has an explicit exit criterion,
and leaves the repo green. Phases 0-2 are prerequisites for Phase 3;
Phases 4-7 can proceed in any order once 2 is done.

### Phase 0 — Unblock and reclaim (no semantic change)

The repo has two conditions that will bite within days.

0.1 **FINDINGS ratchet headroom is zero.** Decide now whether the fix is
to curate a batch of the 320 uncited entries downward, or to raise
`MAX_UNCURATED` with a stated reason. Recommendation: curate ~20 of the
oldest verdict entries in one sitting and lower the ratchet to match, so
the guard keeps its meaning. Booking anything before this reddens CI.

0.2 **Reclaim `.git`.** `git fsck --lost-found` to inventory first,
confirm every local branch is pushed or intentionally local
(`sol/review-1`, `sol/review-2`, `sol/present-1` are local-only), then
`git reflog expire --expire=now --all && git gc --prune=now`. Record
before/after in the commit message. Expected 47 GB → ~1 GB.

0.3 **Fix the CODEMAP citation mask.** `classify()` gains a combined
class or a separate `cited` boolean column so `library` stops hiding
103 citations. Report-only change; the census line changes, no file
moves.

0.4 **Split `refs` into `imports` vs `mentions`.** `gen_codemap.py:118`
matches `if name in text`, so a comment mentioning a filename counts as
a reference — `gate_ckpt.py` shows 38 refs and zero real imports. The
move gate cannot presently distinguish "executes this" from "mentions
this", which is exactly the distinction the rest of this program needs.

0.5 **De-`/tmp` the five sed-patching drivers.** Each becomes a real
committed file, or the target probe gains the parameters the sed was
injecting (device, dims, data path). This closes a live reproducibility
hole and is a prerequisite for touching that probe family at all.

Exit: pytest green; `.git` under 2 GB; CODEMAP reports citations
truthfully; no driver executes from `/tmp`.

### Phase 1 — The provenance index

Extend `docs/results-index.jsonl` with two optional fields:

```json
{"id": "...", "code_commit": "2671030...", "files": ["scratch/foo.py", "scripts/bar.py"]}
```

1.1 `scripts/gen_results_index.py` extracts `(scratch|scripts|llmopt)/\S+\.(py|sh)`
    from each entry body into `files`.
1.2 The `/book` skill records `commit` at booking time. Which sha is a
    real choice: the booking commit itself contains the entry but not
    necessarily the code state the run used, while its parent is the
    tree that was actually checked out when the driver ran. Take the
    PARENT, and record it explicitly as `code_commit` rather than a bare
    `commit`, so the field name states what it means. Where a run
    spanned commits, record the sha the driver was launched from — the
    `/rung` skill knows it at launch time and should pass it through.
1.3 One-time backfill for the 930 existing entries: for each, the commit
    is the one that introduced the entry into RESULTS.md, recoverable by
    `git log -S"<entry heading>" -- docs/RESULTS.md`. Where ambiguous,
    leave `code_commit` null rather than guess — a null is honest, a wrong
    sha is a trap.
1.4 `scripts/results_query.py` grows `--repro <id>` printing the exact
    worktree command.

Purely additive. Nothing reads these fields yet, so nothing can break.

Exit: every entry has `files`; `code_commit` populated where unambiguous,
null where not, with the null count reported; `--repro` works on a
sample of ten entries verified by hand.

### Phase 2 — Guardrails

2.1 **ruff**, tiered. `line-length = 79` to match house style. Narrow
    initial rule set — unused imports, undefined names, unused
    variables — not the full catalogue. `per-file-ignores` sets
    `"scratch/*" = ["ALL"]`. CI runs `ruff check llmopt tests scripts`
    as a build gate and `ruff check scratch --exit-zero` as a report.
2.2 **Import-direction guard.** A test asserting `scripts/` never
    imports `scratch/`. The repo already has this property (144 one way,
    0 the other); the test locks it in for one line of code.
2.3 **A `docs` pytest marker.** Move the ~60 doc-integrity and inventory
    guards behind `-m docs` so "pytest green" cannot be misread as "the
    package is checked". CI runs both; humans can run one.

Exit: CI enforces lint on the package; `scratch/` unchanged; test
markers documented in CLAUDE.md.

### Phase 3 — The shim migration (the crux)

One module at a time. Never more than one in a commit.

**The ordering rule, non-negotiable:** a shim lands only after that
symbol has a test that reproduces a **booked number**, not a
hand-written case. Source identity is being traded away for behavioral
proof; if the behavioral proof does not exist, the trade is a loss.

Per-module procedure:
1. Confirm a behavior battery exists that pins a booked number. If not,
   write it first, from the receipts already in the ledger.
2. Move the single canonical body into the `llmopt/` module.
3. Replace the frozen original's body with a re-export:
   `from llmopt.lab.gate import gate_eval  # noqa: F401`.
4. Delete that symbol's source-identity guard in the same commit, as
   CLAUDE.md's dual-copy lifecycle already specifies.
5. Run the battery. It passing is the proof that HEAD still reproduces
   the booked number — a property the freeze never actually established.

**Order, by evidence strength:**

| # | module | pair | gate |
|---|---|---|---|
| 1 | `keepsets` | `scratch/gt2_jaccard.py` | READY — `tests/test_lab_keepsets.py:98-134` already pins booked `0.8013 / 0.5331 / 0.5280` and byte-identity of the decode artifacts |
| 2 | `oracle_worker` | `scratch/oracle_worker.py` | single symbol, small surface |
| 3 | `shards` | `scratch/k3_expert_demo.py` | already has a real-shard test |
| 4 | `verify`, `gen` | `bench_verify_fast`, `bench_step_tokens` | BLOCKED until a booked-number battery exists (only 3 hand-written cases today) |
| 5 | `gate` | `scripts/step_grpo_micro.py` | LAST. 93 importers, four pinned constants. Highest value, highest blast radius |

`keepsets` first is the whole argument in miniature: it is the only pair
whose battery reproduces booked numbers byte-for-byte, so if the shim
lands and the battery still passes, the scheme is demonstrated on the
strongest available evidence before being applied anywhere weaker.

**Explicitly excluded from shimming**, because they are genuinely
load-bearing path couplings: the `llmopt/reproduce.py` family
(`:18-20` hard-codes `scratch/detbwd_gravmoe.py` and executes it at
`:102`) and `scratch/p4_arms_0801.sh` (string-patched by
`tests/test_reproduce.py:105-107`). These carry no source-identity
guard; what pins them is a live path-and-content coupling that a shim
would break. They stay frozen, and the freeze here is doing real work.

Exit: zero source-identity guards remain except the vendor triple; every
shimmed symbol has one body; every shimmed pair has a booked-number
battery that passes.

### Phase 4 — Harvest the duplication

4.1 **`llmopt/common/`** — the missing shared layer:
    - `device.py: pick_device()` — resolves the MPS-first vs CUDA-first
      disagreement. Artin picks the precedence; it becomes the one
      answer for all 126 sites.
    - `ckpt.py` — one `torch.load` wrapper, `weights_only=True` default,
      the three legitimate `weights_only=False` sites kept explicit.
    - `seed.py` — string-seeded RNG construction, per house law.
4.2 **The four copy-pasted helpers** (`_root`, `_check`, `load_nnue`,
    `ternary`) get a package home. Migrate the ~64% of call sites in
    free files; frozen sites keep their copy until Phase 3 reaches them
    or they are shimmed.
4.3 **Delete the `sys.path` bootstraps** in free files — the package is
    installed editable, so they are dead weight. 8 spellings collapse to
    zero. Frozen files keep theirs.

Exit: `llmopt/common/` exists with tests; duplicate helper count in free
files is zero; `sys.path.insert` count in free files is zero.

### Phase 5 — Package coherence

From the API survey, in dependency order:
- Root `__init__` gains a lazy `__getattr__` so importing `llmopt` does
  not drag torch in.
- `llmopt/lab/` (21 files, 3550 LOC) is becoming a junk drawer. Split
  `llmopt/figures/` (figstyle, figsvg, anatomy) and `llmopt/runs/`
  (runlog, lake, traj) out of it. **Gated** on the BOARD housekeeping
  gate and on Phase 3 completing for any module involved.
- Route the 7 scattered env-var reads through one `LabConfig`; document
  the provenance of the undocumented `FRAC = 0.453` constant or remove
  it.
- Dead-code pass: an AST sweep plus an import-every-public-symbol smoke
  test. The triage of its output is the real cost, not the sweep.
- The 5 modules over 400 lines get split only where a natural seam
  exists. `kernels/metal.py` at 1120 lines is the clearest candidate.

Exit: `import llmopt` works without torch installed; `lab/` under 10
modules; every public symbol importable in a smoke test.

### Phase 6 — Make the test suite tell the truth

6.1 `llmopt/search/zx_engine.py` gets tests. It backs booked verdicts
    and has none.
6.2 Artifact-gated tests (`test_lab_keepsets.py:98-134`,
    `test_vendor_axiom.py:35,73`) currently skip everywhere except one
    machine while reading as green. Gate them behind `LLMOPT_FULL=1` so
    "skipped" stops resembling "verified".
6.3 Drop the `/Users/artin/code/axiom` absolute path from
    `test_vendor_axiom.py:21`.
6.4 Audit the 215 RNG-touching lines for fixed seeds. Unsurveyed today.
6.5 Fix the stale reference at `tests/test_figstyle.py:124`
    (`data/figures.json`; the file is `docs/figures.json`).
6.6 Publish per-subpackage coverage so `lab` at 20/20 and `search` at
    7/10 stop reading alike.

Exit: no module with booked verdicts has zero tests; no test skips
silently on all machines.

### Phase 7a — The packaging contract (from the external review)

An installed wheel is not the same world as a source checkout, and the
package does not currently admit that.

`pyproject.toml:66` packages exactly one data file (`py.typed`), yet
three package modules resolve checkout-level paths:
`llmopt/reproduce.py:17-20` (`parents[1]` → `scratch/detbwd_gravmoe.py`,
executed at `:109`), `llmopt/lab/figsvg.py:32-33` (`parents[2]` →
`docs/figures.json`), and `llmopt/lab/figstyle.py:48` (`parents[2]` →
`assets/fonts`). In an installed wheel `parents[2]` is `site-packages`,
so those paths do not exist and the figure and reproduce subsystems are
broken on `pip install llmopt` while passing every test under `-e .`.

7a.1 Split the contract explicitly. Data a package module needs moves
     to `llmopt/resources/` and is read through `importlib.resources`;
     anything that genuinely needs the repo declares so and fails with
     a clear message rather than a confusing `FileNotFoundError`.
7a.2 CI builds the wheel, installs it in a clean environment, and runs
     an import-and-smoke check. Today CI only installs `-e .`, so this
     entire class is invisible.
7a.3 A minimal-dependency CI job alongside the full one, so a core
     module cannot quietly start importing an optional extra.

Exit: `pip install dist/*.whl` in a fresh venv, then `import llmopt`
plus a figure render and `reproduce --list`, all green in CI.

### Phase 7b — Generated front matter

`README.md:84` states the ledger as 37 / 42 / 70 / 35 / 3 = 187. The
actual counts in `docs/FINDINGS.md` today are 37 replicated, 43
mechanism-confirmed, 73 single-seed, 35 null, 3 retracted = **191**.
The external review, reading an earlier snapshot, found 188. Three
different numbers for one fact, and it moved again tonight when three
bullets were curated — which is the point: a hand-typed number in the
README cannot survive a ledger that grows every session.

Fix the class, not the instance. Factual numbers in front-facing prose
come from generated regions:

```markdown
<!-- llmopt:generated honesty-ledger:start -->
<!-- llmopt:generated honesty-ledger:end -->
```

with `scripts/gen_readme.py --check` in CI beside the existing
generated-docs job. The same number is never typed twice: ledger →
`figures.json` → SVG → README snippet → alt text → caption.

Exit: no factual count appears in README prose outside a generated
region; CI fails on drift.

### Phase 7c — Repository hygiene

- **Protect `main`.** Confirmed unprotected (`gh api ... /protection`
  returns 404). For a repo whose thesis is "nothing counts until
  verified", CI-must-pass-before-main-moves is the matching policy.
  Reviews are unnecessary for a solo maintainer; the status check is
  the point.
- **API stability tiers**, documented in `llmopt/__init__.py`: stable
  top-level surface, supported research API (`lab`, `moe`, …),
  experimental, and frozen evidence. This is what makes aggressive
  refactoring of the experimental tier honest.
- **`CONTRIBUTING.md`** covering where code belongs, how a finding is
  booked, how a figure is published, and what must never be rewritten
  because it is cited evidence. Today that knowledge lives in CLAUDE.md
  and in skills, which is fine for this lab and useless to a visitor.
- **Rename the `runlog` pair.** `llmopt/runlog.py` and
  `llmopt/lab/runlog.py` both exist and mean different things — general
  logging versus experiment receipts. `llmopt/lab/receipts.py` with a
  compatibility re-export.

### Phase 7 — The shell harness

90 drivers, all in `scratch/`, sharing an un-abstracted preamble:
`cd ~/code/llmopt` ×71, `set -e` ×46 but `pipefail` only ×6, two
interpreter spellings, a repeated CUDA block, 12 hand-rolled waiter
loops, and a `pgrep` watcher rederived three times.

Ship `scratch/lib/driver.sh` providing: strict mode with `pipefail`,
the CUDA preamble, a marker-on-success helper, and a waiter. New drivers
source it; the `/rung` skill's scaffold step points at it. Existing
frozen drivers are untouched.

The missing `pipefail` is not cosmetic — it is the exact failure that
logged rc=0 for a dead run on 2026-08-11.

Exit: the harness exists, is documented in `/rung`, and at least the
next three new drivers use it.

---

## 5. What could go wrong

| risk | mitigation |
|---|---|
| A shim changes behavior and a booked number moves | The ordering rule: no shim without a booked-number battery. `keepsets` first because its battery is byte-exact. If a battery fails, revert the shim — one commit, one module |
| The provenance backfill assigns a wrong commit | Null rather than guess. Report the null count. A null is a known gap; a wrong sha is a trap that looks like evidence |
| `gc` loses something | `fsck --lost-found` inventory first; verify branches; commits and branches are untouched by definition since only unreachable objects are pruned. Remote is a second copy and already matches HEAD |
| Lint churn swamps review | Narrow rule set first, `scratch/` exempt, one mechanical commit per rule family |
| `lab/` split collides with in-flight adoption guards | Phase 5's split is gated behind Phase 3 for any module involved, and behind the BOARD housekeeping gate |
| The program stalls half-done, leaving two conventions | Every phase is shippable alone and leaves the repo green. Phase 3 is per-module, so stopping after module N is a coherent state, not a mess |

**The half-done risk is the one that already happened once** — the
2026-08-05 extraction copied bodies and never migrated callers, which is
why six lab modules have no consumers today. Phase 3's per-module
procedure ends with the caller migrated, and the guard deleted, in the
same commit. A module is not "done" until its original is a shim.

---

## 5b. Voice in front-facing documents

Artin, 2026-08-12: front-facing prose should read "this was X, and so Y
was the next logical path" rather than "this was X, so we decided to do
Y". The repo is public and the deliberation framing invites a reader to
wonder who was deliberating.

Measured before writing the rule, because the intuition and the files
disagree:

| document | `we` / `our` / `us` | `the house` |
|---|---|---|
| `README.md` | 0 | 0 |
| `docs/REPRODUCE.md` | 0 | 5 |
| `docs/FINDINGS.md` | 0 | 4 |
| `docs/THEORY.md` | 6 | 0 |
| `docs/paper/main.tex` | **66** | 0 |

So the README, the reproduction guide, and the findings ledger already
use the impersonal result-first voice. The concentration is in the
paper, and a smaller amount in THEORY.

**The rule, in two parts, because the two cases differ:**

1. **Never narrate a deliberation, anywhere front-facing.** "So we
   decided to test Y" becomes "Y was the next test the result implied",
   or simply "Y followed". This is the part Artin asked for and it
   applies uniformly. It is also better scientific writing: the reason
   for the next experiment should be the previous result, not somebody's
   preference.
2. **Methodological `we` stays in the paper.** "We measure", "we
   report", "we hold the seed fixed" is standard academic register, and
   stripping all 66 instances would make the paper read as though it
   were avoiding something. A paper is *expected* to have authors. The
   distinction that matters is credit-and-choice language versus method
   language; only the former is the problem.

**Also front-facing, and separate:** "the house" appears 9 times across
REPRODUCE and FINDINGS. It is in-group lab jargon that means nothing to
a visitor. Replace with the plain subject ("the lab", or better, the
instrument or the result doing the work).

This is an editing pass over five files, not a program phase. It should
land early and cheaply, and the rule belongs in CLAUDE.md so it does not
have to be re-derived.

**Extended (Artin, 2026-08-12, later the same day) — full front-facing
style.** The two rules above stand, and the following applies to all
front-facing documentation, READMEs, project pages, and public posts:

- Extremely concise and high-signal; every sentence earns its place.
- No first-person pronouns (I, we, our, my).
- No em-dashes or en-dashes; colons, periods, commas only.
- Short paragraphs, 1-3 sentences.
- Declarative and confident; results stated directly, no hedging, no
  soft qualifiers, no marketing language.
- Concrete numbers and precise claims over vague praise.
- Strong, aphoristic titles when appropriate.
- A clean, punchy closing observation when the topic allows.
- Never explain the style or comment on the writing itself.

Scope: this tightens rule 2 for README/REPRODUCE/project pages/posts
(zero first person there). The paper keeps methodological academic
register per rule 2 unless Artin says otherwise. Ledger register
(RESULTS, FINDINGS bullets, handoffs, relays) is unchanged. Applies to
new text forward; retroactive rewrites ride Phase 7b (generated README
regions), not a standalone pass.

---

## 5c. Reconciliation with the external review

An independent review (GPT web chat, pinned to the same commit
`2671030`) covered the public surface. It is complementary rather than
overlapping: it is strong exactly where this survey was weak, and blind
where the structural debt actually is.

**Adopted, verified true, and this spec had missed them entirely:**
- The wheel-versus-checkout contract → Phase 7a. Genuinely the best
  find of the external review; three package modules resolve
  `parents[1..2]` to the repo root while the wheel ships one data file.
- Generated README regions → Phase 7b. The review found 187 vs 188; the
  live number is 191, and it moved during the session that wrote this
  spec, which strengthens the argument.
- Branch protection → Phase 7c. Confirmed unprotected.
- Core-versus-all-extras CI → Phase 7a.3.
- API stability tiers, `CONTRIBUTING.md`, the `runlog` rename → 7c.

**Adopted with modification:**
- *Claim → experiment → receipt → figure → reproduce as one spine.*
  This is a superset of Phase 1 and the right long-term shape. Merge it
  into Phase 1 as the eventual target, but build the minimum first
  (`code_commit`, `files`), because a full graph schema designed before
  the shim work is a schema designed without knowing what the shim work
  needs.
- *Figure provenance in `figures.json`* — same reasoning, same phase.
- *Scope chips on published figures* (`d64 · N=120 · 3 preregistered
  seeds`) — cheap, and it makes the epistemics part of the visual
  identity rather than a caption footnote. Fold into the figure work.
- *An `experiments/` lifecycle layer.* Reasonable, but sequence it
  AFTER Phase 3. The measured layering is already clean one-way
  (`scratch`→`scripts` 144, reverse 0), and the actual disease is that
  the library lives in `scripts/` with 93 importers. Introducing a
  third tree before the shim migration risks repeating the exact
  failure of 2026-08-05: a new home created, callers never moved.

**Corrected:**
- The review reports `scripts/figlib.py` as intentional legacy retained
  for `plot_neurons`, `plot_gt1_crest`, and `plot_identity_crest`, and
  recommends freezing new imports of it. That is the file's own
  docstring, and it is false: none of those three import it, and
  CODEMAP shows zero refs. It is archivable, not freezable. A docstring
  is not evidence.
- The ledger drift is 191 against README's 187, not 188.

**What the external review could not see**, because it read the public
surface rather than the dependency graph — and these are the findings
this program is actually built on: the six zero-caller `lab` modules,
`step_grpo_micro` at 93 importers, the five `/tmp`-executing drivers,
the 46 GB of unreachable git objects, the CODEMAP citation mask, the
FINDINGS ratchet at zero headroom, and the 96-of-178 import exposure.
Neither review substitutes for the other.

---

## 6. Doctrine changes this implies

CLAUDE.md and the `/book`, `/rung`, `/riff` skills need edits when the
relevant phase lands. Naming them now so they are not discovered late:

- **Scratch doctrine**: "files cited by booked verdicts are frozen in
  place" becomes "cited files are pinned by `code_commit` in the ledger; the
  working-tree copy may become a shim once a booked-number battery
  passes." The dual-copy rule and its "fixes land in BOTH" clause retire
  with the last guard.
- **`/book`**: records `code_commit` and `files` at booking.
- **`/rung`**: scaffolds from `scratch/lib/driver.sh`; forbids the
  sed-into-`/tmp` pattern explicitly.
- **Front-facing voice** (§5b): never narrate a deliberation;
  methodological `we` stays in the paper; no "the house" in
  visitor-facing docs.
- **CLAUDE.md**: the tiered lint contract, the `-m docs` marker, and
  `llmopt/common/` as the home for shared helpers.

---

## 7. What this does not fix

Honest list, so nobody expects it:

- The 81 GB in `checkpoints/` (397 `.pt`, 38 tracked) and 1.4 GB in
  `logs/`. That is the separately-banked triage thread and needs
  Artin's GO on deletion criteria, not a refactor.
- `docs/` at 256 files and 59,731 lines. RESULTS.md at 27,750 lines is
  append-only by law and is meant to grow.
- The 168 UNCITED files. A real archive/adopt triage needs each read;
  the estimate of "55-65 genuinely dead" is extrapolated from a 12-file
  sample and is explicitly not a census.
- Tie-handling equivalence across the 9 `spearman` bodies. Seven
  runnable copies agree to 12 decimals on untied input; ties untested.

---

## 8. Sequencing recommendation

Phase 0 is urgent for two independent reasons (ratchet headroom zero,
the `/tmp` drivers) and is pure win with no semantic change. Phase 1 is
additive and unblocks everything. Phase 2 is cheap and stops new debt.

Those three are one sitting of work and are worth doing before anything
else is decided. Phase 3 module 1 (`keepsets`) is the experiment that
proves the whole scheme, and should be its own session with the battery
run before and after.

Phases 4-7 are parallelizable and can be picked up opportunistically.

---

## 9. Open questions for Artin

1. **Device precedence.** MPS-first or CUDA-first for `pick_device()`?
   The two current spellings disagree, so this is a product call, not a
   refactor.
2. **Ratchet.** Curate ~20 entries down, or raise `MAX_UNCURATED` with a
   reason? Recommendation is curate.
3. **`llmopt/common` vs `llmopt/lab`** as the home for `device`/`ckpt`/
   `seed`. Recommendation is a new `common/` — `lab/` is already the
   package under review for being a junk drawer.
4. **Phase 5 timing.** The `lab/` split wants a natural freeze point.
   Fold it into the next BOARD housekeeping gate, or schedule it?

---

## 10. Baseline, so progress is checkable

Re-derive with:

```bash
git count-objects -vH
git rev-list --objects --all | awk '{print $1}' \
  | git cat-file --batch-check='%(objecttype) %(objectsize)' \
  | awk '{s[$1]+=$2; n[$1]++} END {for (t in s) print t, n[t], s[t]}'
.venv/bin/python -m pytest -q
.venv/bin/python -c "import sys; sys.path.insert(0,'tests'); \
  import test_docs_integrity as t; print(len(t._uncurated()), t.MAX_UNCURATED)"
```

| metric | 2026-08-12 | target |
|---|---|---|
| `.git` size | 47 GB | < 2 GB |
| reachable history | 984 MB | unchanged |
| commits | 2,083 | 2,083 |
| source-identity guards | 7 pairs / 16 symbols | 0 (vendor triple excepted) |
| lab modules with zero callers | 6 of 12 | 0 |
| cited files (true) | 281 | unchanged; pinned by commit instead of by freeze |
| ledger entries with `code_commit` | 0 of 930 | all unambiguous ones |
| `sys.path` bootstraps | 292 files | frozen files only |
| duplicate helper copies | 79 (`_root` 28, `_check` 27, `load_nnue` 13, `ternary` 11) | frozen sites only |
| drivers executing from `/tmp` | 5 | 0 |
| lint config | none | ruff, tiered |
| FINDINGS headroom | 0 | > 20 |
| tests collected / passing | 659 / 653 | grows with Phase 6 |
| modules with verdicts and no tests | ≥ 1 (`zx_engine`) | 0 |
| README ledger count vs FINDINGS | 187 vs 191 | generated, always equal |
| wheel install smoke-tested in CI | no | yes |
| `main` branch protection | none | CI required |
| `we/our/us` in front-facing non-paper docs | 6 (THEORY) | 0 |
| "the house" in visitor-facing docs | 9 | 0 |

### Foundations actuals (2026-08-12, post-execution)

Phases 0, 1, 2, 7b, and the passing part of 5b landed (see BOARD). Re-derived
with the §10 commands (object-size sweep skipped as directed by Task 13 — no
`git gc`/`prune`/`fsck` run this pass, `git count-objects -vH` only) plus
`scripts/list_uncurated.py`, `results-index.jsonl`, and `gen_readme.py
--check`.

| metric | baseline (2026-08-12, pre-execution) | actual (2026-08-12, post-execution) |
|---|---|---|
| `.git` size | 47 GB | 204 MiB (203.26 MiB packed, 1 pack, 856 KiB loose, 0 garbage) — reclaimed by a separate manual `gc` (23.48 GiB loose / 7,209 objects + 5 packs / 23.10 GiB before; `fsck` rc=0 after; not part of this task's diff) |
| commits | 2,083 | 2,101 (this task's own commit not yet included in that count) |
| `code_commit` populated | 0 of 930 | 880 of 930 (50 null; file-aware backfill rule, commit `4c85c6e`) |
| drivers executing from `/tmp` | 5 | 5 unchanged (frozen drivers left in place per adoption doctrine); their `/tmp` PRODUCTS materialized into git at `scratch/frozen_products/` instead — the plan's "→ 0" metric is satisfied by making the products reproducible from the repo, not by editing the frozen drivers. Stating this honestly rather than claiming the row closed. |
| lint config | none | ruff, tiered — enforced (`ruff check`, exit 0) on `llmopt/`, `tests/`, `scripts/`; `scratch/` report-only by design |
| FINDINGS headroom | 0 | 0 (cap raised to 300 with Task 1's curation pass; ratchet is at its new cap again, not headroom-positive) |
| tests collected / passing | 659 / 653 | 665 collected / 658 passed, 7 skipped, rc=0 (`/tmp/t13.log`) |
| README ledger count vs FINDINGS | 187 vs 191 | 217 vs 217 — equal, generated region live, CI `--check` step added |

Honest deltas: the `.git` reclaim and the FINDINGS-cap raise were both real
but neither is a Phase 0-7b code change — they're named here because they
change what the next session sees when it re-runs §10's commands. Phase 3
(keepsets shim) is next, scoped to its own session per §8.

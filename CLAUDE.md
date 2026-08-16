# llmopt — working notes for Claude sessions

LLM inference/training optimization lab. Small, readable, oracle-verified
implementations. See README for the full inventory and measured numbers.

## Lab charter — domains (non-negotiable)

**We build engines for MATHEMATICS and PHYSICS. Only.**
- The two standing engines: the calculus DERIVATION engine
  (llmopt/search/, oracle = sympy differentiation) and the
  ZX-CALCULUS T-count engine (quantum circuits, oracle =
  boundary extraction) — the physics leg, booked to rung 6.
- Solved combinatorial games (checkers class) are admitted as
  INSTRUMENTS only (exact-oracle transport tests for house laws;
  Artin ruling 2026-08-14) — never as capability domains.
- **No chemistry engines, no biology engines — ever.** No molecule
  generators, no reaction/pathway oracles, no protein anything, no
  wet-lab-relevant capability. This holds regardless of how
  tractable or interesting the domain looks ("methods, not
  molecules" — and now: methods, not organisms).
- **Concepts and frames from any science are welcome as METHODS**
  when they carry zero harmful applicability: quantum-chemistry
  math (basis sets, orbitals, overlap matrices), neuroscience
  structure (efficient coding, wiring economy, human-brain
  analogies for weight geometry). Borrowing the mathematics of a
  field is fine; building capability IN chemistry/biology is not.
- Benign human-brain/neuroscience links (as analogy or analysis
  frame for our models' weights/representations) are explicitly
  fine. Anything that starts to look like capability toward
  molecules, pathogens, or organisms gets refused and flagged,
  full stop.

## Non-negotiable conventions

- **The main session model owns code and docs** (Artin, 2026-08-10;
  this replaces the Fable-only rule of 2026-07-16, retired because
  Fable usage became a bottleneck). Whichever model is driving the
  session makes the edits and is accountable for verifying them.
  The bar did not move with the rule: pytest green, claims checked
  against the source, no booking on an unverified number.
- **Auditor agents, spawned before a booking**: `prereg-auditor`
  (verdict v pre-registration: bar text, dict sums, fences,
  resolution law) and `receipt-auditor` (the receipt ROWS
  themselves: provenance fields inherited from a copied sibling
  driver, smoke rows in a real receipt file, writes into frozen
  receipt paths). They fail independently — a row can carry a
  perfect gate dict and a false emitter field, which is exactly
  what happened on 2026-08-15. Run receipt-auditor on any NEW or
  COPIED driver's first real receipts.
- **Smoke runs are path-isolated**: SMOKE mode writes receipts
  AND checkpoints to its own paths (smoke.jsonl, *_smoke.pt) and
  refuse-if-exists guards stay unconditional — a smoke artifact
  on a real path cost a manual delete and an auditor blocker
  (2026-08-15, twice).
- **Receipt provenance is DERIVED, never a literal**: emitter,
  shard, checkpoint, sha come from the artifacts the run actually
  opened. A hardcoded field inherited from a copied sibling
  booked a false emitter into frozen receipts (RULE-ABLATE-1).
- **Sub-agents: review by default, MAY WRITE when directed** (Artin,
  2026-08-11 — the read-only default is a habit, not a safety
  property). Reviewers (sanctioned 2026-07-24, standing since
  07-31) for ledger-keeping, verdict cross-checks (pre-reg vs
  measured, BEFORE booking), and red-teaming: cap 5 concurrent,
  spawn on ask, Opus model. In the default review shape they MENTION
  (file, line, what's wrong) and findings are proposals, not truth —
  the session model verifies each one line-by-line before adopting
  it. Opus 5 sub-agents may also be given write tools and asked to
  make code changes outright; the session model still owns the
  verification bar for whatever lands.
- **Oracle-verified everything.** Decoding must be token-identical to eager
  greedy (`llmopt/eval/equivalence.py`); math answers checked by sympy symbolic
  equivalence, never string match; asm/code scored by the toolchain
  (assemble the prediction, run the program) — `llmopt/codegen/llvm.py`.
- **fp16 near-ties are a known non-bug**: different verify-block
  compositions round coin-flip logits differently. Diagnose with the eager
  logit margin at the divergence point (see `scripts/bench_stacked.py`);
  margins ≤ ~0.02 are ties, not bugs.
- **Generated datasets**: stable *string* seeds only (`random.Random(f"kind-{level}-{seed}")`)
  — tuple `__hash__` is per-process randomized and killed reproducibility
  once. Guard train/eval splits with `exclude=` (prompt sets), never seed
  offsets alone: small problem spaces collide (two real contamination
  incidents: mathgen L1/L2 43% eval-in-train; ladder `pick()` had only 4
  possible bodies). Widen the generator space before trusting a split.
- Benchmarks report honest losses too (Metal attention_decode losing to
  GEMV, first paged-attention cut losing to gather+SDPA). Keep that.
- **NO sympy call is safely boxed by SIGALRM — fork is the only real
  timebox** (fork, join with deadline, SIGKILL: the
  `gen_magic_labels.solve_isolated` pattern). Generalized 2026-07-12
  from pathology #7 (`make_integrate` on L4+/L8 seeds — FIVE call
  sites bitten, including the farm loops themselves) after the
  alarm-boxed oracle live-locked anyway (pathology #10). Applies to
  generation, rules, routing probes, verifiers, and any
  oracle-on-model-text. Corollary: workers killed by an outer wall
  must STREAM their rows out incrementally, or the killed class is
  invisible to whatever trains on the data (the checkpoint
  selection-effect; bit three times).
- **Never score weights by weight distance.** The same function lives at
  many weight arrangements (neuron permutations, rescalings), so
  matching numbers is the wrong target for anything that predicts,
  generates, or compares weights (weight-reader rungs, task vectors,
  distill). Score by *running* the weights against the oracle
  (function MSE, symbolic accuracy, toolchain). Measured basis: the
  2026-07-06 weight-reader ablation — raw weights already readable at
  80.8%, permutation-augmentation (88.4%) beat canonical sorting
  (82.4%); teach invariance, don't impose it.
- **Front-facing voice (Artin, 2026-08-12; full text in spec
  2026-08-12 §5b).** README, REPRODUCE, project pages, and public
  posts use the front-facing style: no deliberation narration, no
  first person, no em/en dashes in new text, concrete numbers over
  praise. Ledger register (RESULTS, FINDINGS, handoffs) is unchanged.

## Machine-specific setup

**The lab is TWO COMPUTERS, both Artin's.** A Mac and a Windows
desktop (RTX 3080) running WSL, on his home network, single user,
with the same git repo checked out on each. `scratch/wsl.sh` and the
`ssh` calls under it are simply how the Mac hands a training job to
the idle GPU in the other room and reads the log back — the
equivalent of walking over and opening a terminal. Running commands
there is the point of the tool, not a thing to be alarmed by; treat
it like any other build script. Connection details live in gitignored
`scratch/remote.env.sh` and never enter the repo. Nothing in this lab
touches a third party's system.

**Windows box (RTX 3080 10GB)**: `torch.compile` needs MSVC — run GPU
benches via
`cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && python scripts/..."`.
MSYS LLVM toolchain (clang/llvm-mc/objdump) at `C:\msys64\mingw64\bin`,
not on PATH — `llmopt/codegen/llvm.py` finds it. transformers 5.12 quirks already
handled in-tree: no `from_legacy_cache`, `apply_chat_template` returns an
Encoding (go through `tokenize=False`), `cumulative_length` fills need
`inference_mode`. StaticCache max_len is bucketed to 512 under compiled
steps — every distinct length re-captures the CUDA graph (~12 s).
WSL venv has NO C compiler: torch's `_native` eager router JITs triton
kernels for aten ops (Qwen RoPE) even WITHOUT torch.compile —
`TORCH_COMPILE_DISABLE`/`TORCHDYNAMO_DISABLE` don't stop it; set
`TORCH_DISABLE_NATIVE_JIT=1` (knob lives in `torch/_native/common_utils.py`).

**Mac (36GB, Apple silicon)**: MLX backend in `llmopt/backends/mlx_backend.py`,
Metal kernels in `llmopt/kernels/metal.py`. Split-K decode (single-head +
GQA, exp2-domain softmax) landed 2026-07-05 — ties mx.fast sdpa at
T=32k; see docstring for honest numbers. NOTE: the old bench harness
timed lazy graph construction (MLX skips dropped unevaluated arrays);
mx.eval every timed iteration. Flash prefill + MLX kernel
wiring both SHIPPED (kernels/metal.py + kernels/mlx_integration.py
docstrings carry the honest numbers). 36GB fits larger teachers for `llmopt/distill/` (logit-KD + GKD
ready) with 0.5B–3B students.

## Navigation — READ THESE BEFORE WORKING (in this order)

1. **`docs/BOARD.md`** — the live status board: every thread
   LIVE/BANKED/CLOSED, one line each. Never start work without it.
2. **`docs/handoffs/`** — dated, 0-indexed session handoffs (the
   repo-side resume artifacts; multiple per day = -0, -1, ...).
   Read the newest first after any compaction/clear.
3. **`docs/RESULTS.md`** — every verdict, win/null/retraction alike,
   newest at the bottom. Before proposing ANY experiment, grep it:
   the idea has often been run, nulled, or pre-registered already.
4. **`docs/RIFF-LEDGER.md`** — idea provenance. EVERY riff Artin or
   the house proposes gets banked here with attribution, even
   half-retracted ones ("bank everything" is standing policy).
5. **`docs/THEORY.md`** — the grounding map: house laws x published
   lineage. No row without a measured result AND a real citation.
6. **`scripts/INDEX.md`** — signature/docstring index of scripts/, scratch/, and llmopt/.
   Grep it before writing anything (don't rewrite existing code).
   **Regenerate after adding/changing scripts:**
   `.venv/bin/python scripts/gen_index.py`.
7. `docs/superpowers/relay/` — the axiom-Fable exchange (Artin
   relays manually); `docs/superpowers/specs/` — pre-run specs;
   `docs/LOOP-LOG.md` — expert-iteration rounds.

**Habits that keep these useful**: pre-register predictions in
RESULTS before a run fires; book verdicts (including honest
failures) the moment they land; consolidate BOARD + a new handoff
at natural stopping points, not mid-sprint.

**The rituals below are SKILLS — use them instead of re-deriving
the steps** (`.claude/skills/`, each carrying the gotchas that
earned it):

| Skill | Covers |
|---|---|
| `/rung` | pre-reg -> driver -> launch -> watcher (the front half) |
| `/book` | append RESULTS, regen index, link, curate FINDINGS, push |
| `/riff` | bank an idea in RIFF-LEDGER, or correct a bank in place |
| `/labstatus` | one-shot sweep of both machines, unbooked results first |
| `/probe` | measurement-cost triage before any multi-hour run |
| `/desk` | zero-cost census: price a rung by counting before running it |
| `/relay` | house -> axiom relay |
| `/counterbook` | recompute axiom's numbers from their artifacts |
| `/handoff` | session close: handoff file, BOARD repoint, suite, push |
| `codemap-check` | (Claude-only) CODEMAP class before touching scratch/scripts |

Read the skill before improvising a variant of it; where a skill
and this file disagree, the skill is the more recently corrected of
the two and should be reconciled here in the same session.

## Doc lifecycle + machines (living-docs discipline)

- RESULTS.md is append-only; corrections are AMENDMENT entries
  naming their target. LIVE since 2026-07-26: every booking runs
  `scripts/gen_results_index.py` (auto-extracts the new entry;
  preserves curation) and ideally adds threads/verdict/links to
  its line in docs/results-index.jsonl, same commit. Query with
  `scripts/results_query.py` (--live / --chain / --thread) BEFORE
  proposing experiments — faster than grepping ~28k lines.
- **docs/FINDINGS.md is CI-RATCHETED**: `test_docs_integrity.py`
  caps the uncurated backlog, so booking a VERDICT without adding
  its FINDINGS bullet in the same commit turns the suite red. That
  is the guard working, not a flake. One bullet, exactly one
  maturity tag from the controlled vocabulary, its scope tags, and
  a `RESULTS.md#L<line>` anchor.
- THEORY.md and RIFF-LEDGER.md are LIVING documents: at every
  session close, check the day's verdicts against existing
  rows/banks — update, amend, or mark-dead whatever a finding
  touches. A disproven row gets its refutation named in place;
  never silently delete.
- BOARD refresh + handoff at natural stopping points; a
  next-session spec BEFORE any compact (state lives in the repo,
  never in memory).
- MACHINES: the Mac is Fable's, always. The 3080/WSL runs on
  Artin's schedule (nightly GO for long jobs; short tests only
  while he's home; ~5PM EST checkpoint) — details in project
  memory, not here (repo is public). Remote ops go through
  scratch/wsl.sh; cross-device comparisons stay forbidden.
- A long test is not justified by being long: before queueing,
  diagnose the wall and try to move it (BS/grad-ckpt/allocator/
  batching class fixes first).
- Session hygiene (learned 2026-07-26, the week-long-chat lesson):
  short sessions, compact early, hand off through the repo — a
  session's working state must never be the single point of
  failure. Post-compact resume = resume-protocol memory + BOARD +
  newest handoff/spec + RESULTS tail.
- Task holds are explicit: queued work marked [HOLD] runs only on
  Artin's GO, never on inference from context.
- **Mac mps float training is RUN-LEVEL NONDETERMINISTIC at fixed
  seed** (measured 2026-08-15: paired 20-step probe, same script,
  same seed, same batches, different weight digests). Never write
  a cross-run bit-exact reproduction precondition into a Mac
  pre-reg, and never compare weight shas across runs on mps;
  in-run paired arms are the valid shape (both arms see the same
  substrate noise). This killed a registered precondition
  mid-rung — AMENDMENT SOFT-SPEED-1-PRECONDITION.
- Instrument fences travel with instruments: CE-400 is
  format-BOUND (valid within matched-format comparisons only);
  sigma never transports across devices/widths; probe scripts pin
  their VOCAB_EXTRA atom order; end-to-end cross-device tolerance
  bars belong to TRAINED networks only — random weights amplify
  bf16 noise ~3x/layer (V4-F1b/F1c), so never calibrate a bar on
  an untrained boot.
- WSL side is a THIN EXECUTION TARGET, not a second lab: no
  WSL-side scratch authorship; scripts live in git, ship via
  push/pull or wsl.sh; logs land in logs/; artifacts pull back to
  the Mac. (Full structure cleanup = the post-index project.)

## Doctrine (distilled; full text in RESULTS/handoffs)

- **Pre-registration + paired arms, always**: same device, same
  seeds, one variable. Never compare probes/gates across devices
  (measured 2x device dependence at the frontier).
- **Verified AND distinct, at every learning layer**: the oracle
  accepts X=>X as true; reward, gate candidates, AND miners must
  all reject identity rewrites (bit three times: GRPO reward hack,
  gate candidates, miner v5's bank).
- **Precision doctrine (CLOSED 2026-07-24)**: birth precision is a
  non-factor above TF32 (round-to-nearest paths; the SR-BF16
  per-forward stochastic-rounding arm is a SEPARATE lever, booked
  negative n=1 with named confounds — P4 revival slot); fp64 masters are the FINAL capability rung
  for online learning (exact-vs-fp64 measured bit-identical);
  exact arithmetic is a SPEED/DETERMINISM lever (int8-sliced beats
  native fp64 — scratch/ozaki_*). SCOPE (per AMENDMENT RESULTS
  L7226): the closure holds ABOVE INSTRUMENT SIGMA (~±1-2
  solves/120), and carries ONE named retest slot — exact-mode
  gate v rounded gate, same weights, when exact inference lands.
  Don't spend runs on precision-capability questions outside
  that slot or a pre-reg that names this scope fence.
- **Speed defaults (lossless, always on)**: KV-cached sampling;
  bf16 births (--fast) on cuda / fp32 on Mac; GRAD_CKPT=1 for
  d768+ on 10GB; PYTORCH_CUDA_ALLOC_CONF set in-tree. A CUDA
  allocator OOM warning in a log is a TRIPWIRE (the 43x), not noise
  — restart at the next epoch boundary. Knob note: the 43x was
  measured with expandable_segments, but that knob CRASHES the WSL
  driver; the in-tree default is max_split_size_mb:128
  (scripts/train_mathnative.py) and that is the knob on the 3080.
- **Remote ops (friendly-fire, 7 variants deep)**: kill/write/
  launch = separate ssh calls; a watcher's pgrep must never match
  a string its own launcher carries; verify file deps at arm time;
  completion markers fire on SUCCESS only; remote host/key live in
  gitignored `scratch/remote.env.sh` (never commit them); sync =
  stash -> pull -> VERIFY -> drop (never drop-on-abort). Both
  machine checkouts stay at origin/main (`git pull --ff-only`).
- **Data hygiene**: exclude=-guarded splits; underdetermined rows
  train hallucination (audit for determinability, not just
  correctness); diet exposure SHARE matters (rations for resident
  grammars when the corpus grows).

## Practical

- `pytest` (scoped to tests/ via pyproject; scratch/*_test.py are
  scripts, not tests). Pure-Python runs anywhere; GPU/toolchain
  tests skip cleanly.
- `pytest -m "not docs"` = code only; `-m docs` = ledger guards;
  plain pytest = both (CI's run).
- Math-native training: `scripts/train_mathnative.py` (--diet,
  --fast, VOCAB_EXTRA/BIRTH_SEED/GRAD_CKPT envs; probe scripts
  take VOCAB_EXTRA too — atom ORDER must match the birth env).
  Legacy LoRA recipe (`llmopt/train/lora.py`, r=16, answer-only loss,
  length-bucketed + shuffled) still serves the 0.5B-era scripts.
- Scratch experiments live in `scratch/` (committed — they are the
  lab notebook); stray root artifacts go to `logs/archive/`; big
  jsonl/checkpoints stay untracked (file-handoff convention).
- **Scratch doctrine (2026-08-06)**: `scratch/` is for unregistered
  probes and registered arm drivers; anything that becomes a
  permanent instrument gets ADOPTED into `llmopt/lab/` —
  verbatim-copy + source-identity/behavior guards
  (tests/test_lab_adoption.py pattern), never a silent `mv`. Files
  cited by booked verdicts are the evidence record and stay frozen
  in place (`docs/CODEMAP.md`, regenerated by
  `scripts/gen_codemap.py`, is the move gate — check a file's class
  before touching it). Don't fork a frozen family (no detbwd_r4.py);
  extend the adopted package module. Dual-copy guard lifecycle:
  while a scratch original and its lab adoption coexist, fixes land
  in BOTH in the same commit (the source-identity test enforces
  this — it is doing its job when it is annoying); package-only
  improvements WAIT until a registered re-run migrates the driver,
  at which point that symbol's source-identity guard is deleted in
  the same commit and the scratch copy is frozen-as-record.
- **Logs doctrine (2026-08-06)**: `logs/` is run exhaust — untracked
  by default, regenerate-don't-download (REPRODUCE's rule). Layout:
  `logs/<battery-or-day>/` subdirs, output paths unique per
  arm×seed; NEVER append a new run into a path a booked verdict
  cites as frozen. Scoped exception (the seedslad pattern,
  RESULTS d88bbff postscript): SMALL TEXT RECEIPTS for a booked
  verdict may be `git add -f`'d under `logs/<name>/`, with the why
  booked in the same RESULTS entry — large jsonl/traj/per-problem
  streams never. Bulk log deletion is Artin-GO (handoff payloads
  live here; pairs with the banked 51GB triage thread). Bulk moves/archival happen only
  at a natural freeze point under the BOARD housekeeping gate
  (pytest green + smoke-launch entry points + both checkouts in
  lockstep).

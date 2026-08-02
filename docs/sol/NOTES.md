# Sol adversarial-improver notes

## 2026-08-01 — review 1

- Branch boundary: `sol/review-1`, forked from tagged `pre-sol-baseline`
  (`95bdd43`). Existing untracked `data/` and `figs/` artifacts belong to the
  house and are excluded from this proposal.
- Required context read in order: `CLAUDE.md`, `docs/BOARD.md`, then
  `scripts/INDEX.md`. Prior-result queries covered `rms_fwd`, `ACT_CLAMP`,
  `overflow`, `free-run`, and the COND/QK chain.
- Adversarial seam: the BR-W4c crash was correctly diagnosed as int64 overflow,
  but the resulting law (ACT_CLAMP is the RMS overflow guard; safe bound about
  80 Q-units) is unnecessarily restrictive. At Q=512, `Q**2` divides `2**32`.
  The current expression multiplies the mean square by `2**32` before dividing
  by `Q**2`; exact factorization performs only a `2**14` multiply. This is not
  an approximation and should remove the observed overflow without int128 or a
  capability-changing clamp.
- Index schema finding: `amends` and `superseded_by` are curated as both scalar
  strings and lists. `scripts/results_query.py --chain` uses list concatenation
  directly, so scalar links are split into characters and silently disappear
  from the lineage walk. The Sol generators normalize both forms; this branch
  does not modify the house query tool.
- Minimal reproduction: `python -m llmopt.reproduce gravmoe-rb1` resolves the
  committed pin and arm contract, forces `RJOB_LOCAL=1`, streams the underlying
  runner, and exits nonzero unless the final trajectory SHA is exact. `--list`
  exposes all 16 committed arms. A second axiom-side wrapper is proposed but not
  implemented because it would create a cross-repo runtime dependency.

## SESSION-NOTES 2026-08-01 — review 2

Evidence boundary: the registered cell is pre-registration
`af2ae5be3773b9a668612a5aae92dc3c60966eea`, executable implementation
`5517a7f01a3340c11401551504b76d42301c8d18`, reviewed REG receipt
`3398ffe34b32526b1c577c4695c518416ab7a106`, and booked verdict
`e41d9e9adfc3c6767967c7a51b7dcb7469b78201`. The pinned diet SHA is
`809bce4215a24164ecbf5e951d77507d455bfd1923d08fe39aa02942b11a200b`;
the TRAIN-row and full-row SHAs are respectively
`32cc244bf28fdadf01b343ae16fe1a55200ffe9fab9bd784e8abd739b12ef2c0`
and `78f8aef992debe6ec74e4701fba23167ff5fda1d4294546b9f7621605429798a`.

Interpretation fence: this is `n=1`—one seed, initialization, TRAIN/HELDOUT
row set, and binary mask—and every finding is limited to the supplied
scaffold, tokenizer, finite row length, deterministic G-RB1 diet, and exact
answer-region convention. Nothing here establishes a general ML law.

### Ranked findings

1. **Claim:** Scaffold-token parameter gradients positively align with
   answer-token parameter gradients at the fixed birth, so zeroing scaffold
   loss removes useful answer-learning signal rather than merely a formatting
   signal. **Evidence:** binary AO1 reaches `0/8` solves and `60/116` suffix
   tokens versus REG's `2/8` and `79/116`; TRAIN parseability is unchanged at
   `5/8` and termination improves `6/8 -> 7/8`. The 16-arm REG command is the
   Mac-local
   `RJOB_LOCAL=1` invocation of `scratch/p4_arms_0801.sh`, recorded in
   `logs/sol/answer_only_pins.log` with GRB1 detail in `logs/p4/GRB1.log`;
   all 16 default pins reproduce exactly and REG trajectory SHA is
   `1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca`.
   The single AO1 command is `RJOB_LOCAL=1 GATE=1 COND=1 QK=1 LN=0 LD=1
   STEPS=2000 ANSWER_ONLY=1 /Users/artin/code/llmopt/.venv/bin/python
   scratch/detbwd_gravmoe.py`, recorded in
   `logs/sol/answer_only_ao1.log`, with trajectory SHA
   `3a219ebed65154c19c854e15ec7fbba72596ab66793c9aca27b6d53e9bd95a3b`.
   The registered rule in `docs/sol/RESULTS-SOL.md` therefore selects null, not
   format failure. **Confidence:** medium-low because these aggregate readouts
   motivate but do not measure gradient alignment. **Cheapest house
   experiment:** run one
   no-training, fixed-birth diagnostic that separately computes scaffold-token
   and answer-token parameter gradients for each of the eight pinned TRAIN
   rows. A positive median cosine with at least `6/8` row cosines positive
   confirms the claim; a non-positive median, fewer than `6/8` positive rows,
   or an undefined cosine from a zero gradient kills it. This does not rerun
   AO1 and is the sole next step eligible under the current approved design.

2. **Claim:** At exactly scaffold weight `1/4` and answer weight `1`, a
   soft-scaffold arm will recover at least the REG TRAIN baselines of `2/8`
   solves and `79/116` suffix tokens. This is a prediction about that one
   weight pair, not a dose-curve or nonzero-weight law. **Evidence:** binary AO1
   reaches `0/8` and `60/116`, below those REG baselines, under the exact
   commands, logs, and SHAs in finding 1. **Confidence:** low because one binary
   contrast does not measure the `1/4` point. **Eligibility:** **UNAPPROVED**
   under the current design, which permits weighting only after a binary win.
   It may be tested only if a fresh approved design explicitly supersedes that
   boundary. If so, both at least `2/8` TRAIN solves and at least `79/116`
   suffix accuracy confirm this exact prediction; failure on either threshold
   kills it.

3. **Claim:** At exactly scaffold weight `1/4` and answer weight `1`, the same
   soft-scaffold arm will recover at least the REG HELDOUT construction
   baselines of `2/8` parseable continuations and `30/88` suffix tokens, even
   though neither fixed arm currently solves a HELDOUT row. **Evidence:** REG
   and binary AO1 both solve `0/8`; AO1 falls to `1/8` parseable from REG's
   `2/8` and to
   `23/88` suffix accuracy from REG's `30/88`. These are the HELDOUT lines in
   `logs/p4/GRB1.log` and `logs/sol/answer_only_ao1.log` under the exact
   commands and lineage SHAs above; the TRAIN/FULL row SHAs are
   `32cc244bf28fdadf01b343ae16fe1a55200ffe9fab9bd784e8abd739b12ef2c0`
   and `78f8aef992debe6ec74e4701fba23167ff5fda1d4294546b9f7621605429798a`.
   **Confidence:** low: this extrapolates from one binary treatment and eight
   HELDOUT prompts. **Eligibility:** **UNAPPROVED** under the current design and
   not a current next step. Only a fresh approved design that explicitly
   supersedes the weighting boundary may authorize the arm; its already
   required HELDOUT card would confirm this exact prediction at both at least
   `2/8` parseable continuations and at least `30/88` suffix accuracy, with
   failure on either threshold killing it.

4. **Claim:** The gate is now fail-closed against data or row drift, but this
   integrity property was missing before the Sol fix. **Evidence:** commit
   `5517a7f01a3340c11401551504b76d42301c8d18` adds pre-model assertions for
   the pinned diet, TRAIN-row, and full-row SHAs above, plus focused regression
   tests in `tests/test_gravmoe_answer_only.py`; the production evidence is the
   exact 16/16 replay command and `logs/sol/answer_only_pins.log`, with all pins
   unchanged. **Confidence:** high for the asserted G-RB1 gate inputs and the
   focused failure-before-model-construction cases. **Cheapest house
   experiment:** run `PYTHONPATH=.
   /Users/artin/code/llmopt/.venv/bin/python -m pytest
   tests/test_gravmoe_answer_only.py -q`; the three drift cases must each abort
   before model construction to confirm the fail-closed claim; any drift case
   that reaches model construction kills it.

### With more budget

Do not repeat binary AO1: its null is booked. The sole currently eligible next
step is the no-training, fixed-birth gradient-alignment diagnostic in finding
1. The exact `1/4` scaffold-weight training arm is **UNAPPROVED** because the
approved design permits weighting only after a binary win. Do not schedule or
run it unless a fresh approved design explicitly supersedes that boundary. If
such a design is later approved, it must pre-register the single exact weight
pair and its integrity pins and decision metrics; it is not permission for a
dose curve or a binary AO1 rerun.

### House experiment-code audit

- Broken code found and fixed on this Sol branch: the original locked gate did
  not assert diet, TRAIN-row, and full-row SHAs before model construction.
  Commit `5517a7f01a3340c11401551504b76d42301c8d18` adds those assertions and
  focused drift regressions.
- Final-review fix: default reproduction previously inherited ambient
  `ANSWER_ONLY=1`. The Python reproduction environment and the 16-pin launcher
  now scrub that treatment knob, with focused coverage at both boundaries.
- Final-review robustness fix: `_fork_call` now receives the small SymPy result
  over a one-way pipe with a bounded deadline, forced kill, and join; focused
  tests cover delivery, parse/equivalence, and timeout.
- Final-review receipt fix: the returned gate counters and the exact stable
  two-line stdout contract are both asserted with `capsys`. None of these
  repairs changes a measured metric or the booked null verdict.

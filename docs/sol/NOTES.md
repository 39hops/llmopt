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

1. **Claim:** The mechanism-level hypothesis is that answer prioritization
   fails here only when scaffold weight is driven to zero: retaining scaffold
   weight `1/4` while keeping answer weight `1` will recover at least the REG
   TRAIN baselines of `2/8` solves and `79/116` suffix tokens. **Evidence:**
   binary AO1 reaches `0/8` and `60/116`, below those REG baselines. The 16-arm
   REG command is the Mac-local
   `RJOB_LOCAL=1` invocation of `scratch/p4_arms_0801.sh`, recorded in
   `logs/sol/answer_only_pins.log` with GRB1 detail in `logs/p4/GRB1.log`;
   all 16 default pins reproduce exactly and REG trajectory SHA is
   `1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca`.
   The single AO1 command is `RJOB_LOCAL=1 GATE=1 COND=1 QK=1 LN=0 LD=1
   STEPS=2000 ANSWER_ONLY=1 /Users/artin/code/llmopt/.venv/bin/python
   scratch/detbwd_gravmoe.py`, recorded in
   `logs/sol/answer_only_ao1.log`, with trajectory SHA
   `3a219ebed65154c19c854e15ec7fbba72596ab66793c9aca27b6d53e9bd95a3b`.
   **Confidence:** low: the hypothesis is motivated by one binary contrast and
   is not itself measured. **Cheapest house experiment:** pre-register exactly
   one new soft-scaffold arm with answer weight `1` and scaffold weight `1/4`
   at the same pinned gate. A result with both at least `2/8` TRAIN solves and
   at least `79/116` suffix accuracy confirms the claim; failure to reach
   either threshold kills it. This is a distinct mechanism, not permission to
   rerun booked binary AO1 or launch a dose curve.

2. **Claim:** The mechanism-level hypothesis is that scaffold-token gradients
   positively align with answer-token gradients at the fixed birth, so zeroing
   scaffold loss removes useful answer-learning signal rather than merely a
   formatting signal. **Evidence:** under the same exact REG/AO1 commands,
   logs, code/data/row SHAs, and trajectory SHAs above, TRAIN parseability is
   unchanged at `5/8` and termination improves `6/8 -> 7/8`, while solves fall
   `2/8 -> 0/8` and suffix accuracy falls `79/116 -> 60/116`. The registered
   rule in `docs/sol/RESULTS-SOL.md` therefore selects null, not format failure.
   **Confidence:** medium-low because these aggregate readouts motivate but do
   not measure gradient alignment. **Cheapest house experiment:** run one
   no-training, fixed-birth diagnostic that separately computes scaffold-token
   and answer-token parameter gradients for each of the eight pinned TRAIN
   rows. A positive median cosine with at least `6/8` row cosines positive
   confirms the claim; a non-positive median, fewer than `6/8` positive rows,
   or an undefined cosine from a zero gradient kills it. This does not rerun
   AO1.

3. **Claim:** The held-out mechanism hypothesis is that retaining scaffold
   weight `1/4` will recover at least the REG HELDOUT construction baselines of
   `2/8` parseable continuations and `30/88` suffix tokens, even though neither
   fixed arm currently solves a HELDOUT row. **Evidence:** REG and binary AO1
   both solve `0/8`; AO1 falls to `1/8` parseable from REG's `2/8` and to
   `23/88` suffix accuracy from REG's `30/88`. These are the HELDOUT lines in
   `logs/p4/GRB1.log` and `logs/sol/answer_only_ao1.log` under the exact
   commands and lineage SHAs above; the TRAIN/FULL row SHAs are
   `32cc244bf28fdadf01b343ae16fe1a55200ffe9fab9bd784e8abd739b12ef2c0`
   and `78f8aef992debe6ec74e4701fba23167ff5fda1d4294546b9f7621605429798a`.
   **Confidence:** low: this extrapolates from one binary treatment and eight
   HELDOUT prompts. **Cheapest house experiment:** use the same single
   pre-registered `1/4` soft-scaffold arm named in finding 1 and read its already
   required eight-row HELDOUT card. At least `2/8` parseable continuations and
   at least `30/88` suffix accuracy confirms the claim; failure to reach either
   threshold kills it. This is one unconditional new-mechanism arm, not a
   second-stage experiment or a binary AO1 rerun.

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

Do not repeat binary AO1: its null is booked. The next training allocation, if
the house wants to test a new mechanism, should be one pre-registered
soft-scaffold arm (`1/4` scaffold weight, answer weight `1`) with the same
integrity pins and decision metrics, preceded by the no-training gradient
alignment diagnostic. A discovery-level TRAIN win should then receive paired
seed confirmation before any dose curve, expanded held-out evaluation, or
general claim. Another null should close this fixed-gate allocation family
rather than trigger a seed rerun.

### House experiment-code audit

- Broken code found and fixed on this Sol branch: the original locked gate did
  not assert diet, TRAIN-row, and full-row SHAs before model construction.
  Commit `5517a7f01a3340c11401551504b76d42301c8d18` adds those assertions and
  focused drift regressions. No other broken experiment code was observed.
- Non-blocking robustness note: `_fork_call` reads a joined multiprocessing
  queue through `Queue.empty()`. It worked for every small SymPy result here,
  but a pipe or an exception-safe blocking read would provide a stronger
  cross-platform delivery contract. This is not evidence of a measured result
  error.
- Non-blocking receipt note: the returned gate counters are asserted, but the
  exact two-line textual print format is not captured in a dedicated test.
  Current retained logs contain the expected text; a small `capsys` regression
  would protect external log parsers without changing any mathematical
  readout.

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

### Ranked findings

1. **Claim:** Binary answer-only loss allocation is a null at the exact pinned
   G-RB1 gate and is worse on the registered capability readouts: REG reaches
   `2/8` TRAIN solves and `79/116` suffix tokens, whereas AO1 reaches `0/8`
   and `60/116`. **Evidence:** the 16-arm REG command is the Mac-local
   `RJOB_LOCAL=1` invocation of `scratch/p4_arms_0801.sh`, recorded in
   `logs/sol/answer_only_pins.log` with GRB1 detail in `logs/p4/GRB1.log`;
   all 16 default pins reproduce exactly and REG trajectory SHA is
   `1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca`.
   The single AO1 command is `RJOB_LOCAL=1 GATE=1 COND=1 QK=1 LN=0 LD=1
   STEPS=2000 ANSWER_ONLY=1 /Users/artin/code/llmopt/.venv/bin/python
   scratch/detbwd_gravmoe.py`, recorded in
   `logs/sol/answer_only_ao1.log`, with trajectory SHA
   `3a219ebed65154c19c854e15ec7fbba72596ab66793c9aca27b6d53e9bd95a3b`.
   **Confidence:** high for this deterministic, format-bound cell; low for a
   general ML law because it is one seed, initialization, diet, and row split.
   **Cheapest legal confirm/kill experiment:** pre-register one new mechanism,
   a single soft-scaffold arm with answer weight `1` and scaffold weight `1/4`
   at the same gate. It confirms the binary-allocation finding if it remains
   below REG on solves and suffix accuracy, and kills the sharper claim that
   answer prioritization itself is harmful if it recovers REG or better. This
   is not permission to rerun the already-booked binary AO1 arm or launch a
   dose curve.

2. **Claim:** AO1's TRAIN loss is not a format failure; removing scaffold loss
   degrades symbolic capability while leaving the registered format diagnostics
   intact. **Evidence:** the same REG/AO1 commands, logs, code/data/row SHAs,
   and trajectory SHAs above show TRAIN parseability unchanged at `5/8` and
   termination improved from `6/8` to `7/8`, despite solves falling `2/8 ->
   0/8` and suffix accuracy falling `79/116 -> 60/116`. The registered rule in
   `docs/sol/RESULTS-SOL.md` therefore selects null, not format failure.
   **Confidence:** high for classification under the registered rule; medium
   for the mechanistic interpretation that scaffold gradients support answer
   learning rather than only syntax. **Cheapest legal confirm/kill experiment:**
   add a no-training, fixed-birth gradient diagnostic that measures norm and
   cosine alignment of scaffold-token and answer-token parameter gradients on
   the eight pinned TRAIN rows. Consistent positive alignment would support the
   mechanism; near-zero or opposed alignment would kill it without rerunning
   AO1.

3. **Claim:** There is no held-out capability rescue: both arms solve `0/8`,
   while AO1 parseability is `1/8` versus REG `2/8` and AO1 suffix accuracy is
   `23/88` versus REG `30/88`. **Evidence:** these are the HELDOUT lines in
   `logs/p4/GRB1.log` and `logs/sol/answer_only_ao1.log` under the exact
   commands and lineage SHAs above; their TRAIN/FULL row SHAs are
   `32cc244bf28fdadf01b343ae16fe1a55200ffe9fab9bd784e8abd739b12ef2c0`
   and `78f8aef992debe6ec74e4701fba23167ff5fda1d4294546b9f7621605429798a`.
   **Confidence:** high that neither fixed arm generalizes on these eight
   prompts, low beyond this small split. **Cheapest legal confirm/kill
   experiment:** only after a distinct mechanism beats the registered TRAIN
   gate, evaluate that frozen winning mechanism on a pre-registered expanded,
   prompt-disjoint symbolic held-out set; any reproducible solve advantage
   kills the present no-rescue finding for the new mechanism. Replaying binary
   AO1 on another seed is not authorized by this booked null.

4. **Claim:** The gate is now fail-closed against data or row drift, but this
   integrity property was missing before the Sol fix. **Evidence:** commit
   `5517a7f01a3340c11401551504b76d42301c8d18` adds pre-model assertions for
   the pinned diet, TRAIN-row, and full-row SHAs above, plus focused regression
   tests in `tests/test_gravmoe_answer_only.py`; the production evidence is the
   exact 16/16 replay command and `logs/sol/answer_only_pins.log`, with all pins
   unchanged. **Confidence:** high for the asserted G-RB1 gate inputs and the
   focused failure-before-model-construction cases. **Cheapest legal
   confirm/kill experiment:** run `PYTHONPATH=.
   /Users/artin/code/llmopt/.venv/bin/python -m pytest
   tests/test_gravmoe_answer_only.py -q`; the three drift cases must each abort
   before model construction, and any case that reaches construction kills the
   fail-closed claim.

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

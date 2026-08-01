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

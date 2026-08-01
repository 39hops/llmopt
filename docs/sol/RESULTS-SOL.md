# Sol measured results

This is the proposal-branch ledger. Entries are pre-registered before their
commands fire; failures and nulls remain booked.

## PRE-REG SOL-RMS-HEADROOM: algebraic scale factoring should remove the false ACT_CLAMP ceiling without changing a single certified trajectory (2026-08-01, Mac)

The current deterministic-birth RMS path computes
`floor(floor(sum(x*x)/D) * 2**32 / Q**2) + eps` in int64. With `Q=512`,
`Q**2=2**18` divides `2**32`, so the same integer is exactly
`floor(sum(x*x)/D) * 2**14 + eps`. The current evaluation order overflows once
mean-square exceeds `2**31`; the factored order does not overflow until
mean-square approaches `2**49` (subject first to the separate `sum(x*x)`
bound). The booked `ACLAMP=49152` crash is therefore an evaluation-order bug,
not an inherent int64 RMS ceiling.

Implementation: promote an overflow-aware RMS magnitude primitive to
`llmopt/intmath.py`, use it in `scratch/detbwd_r2b.py`, and add oracle tests
against Python unbounded integers. No floating reference and no weight-distance
score is involved.

Pre-registered predictions and bars:

1. On every input where the legacy expression is in range, old and factored
   Q16 mean-square values are bit-identical. Bar: exhaustive/boundary unit tests
   show exact equality, not tolerance.
2. On the BR-W4c stress point (`D=128`, all activations `49152`), the legacy
   int64 expression becomes negative while the new primitive equals the Python
   big-int oracle and returns nonzero RMS magnitude. Bar: exact integer equality.
3. The change is lossless for the certified battery. Bar: all 16 arms in
   `scratch/p4_arms_0801.sh` reproduce every SHA pinned in
   `scratch/detbwd_gmoe_ref/pins.json`; one-arm sampling is insufficient.
4. A one-step wide smoke at `ACLAMP=49152` completes without the former RMS
   divide-by-zero. This only validates headroom; it does not revive the nulled
   width/capability claim.

Commands (to run after implementation), with local-only execution:

```bash
.venv/bin/pytest tests/test_intmath.py -q
RJOB_LOCAL=1 bash scratch/p4_arms_0801.sh
RJOB_LOCAL=1 DIM=128 DHEAD=32 FFN=256 ACLAMP=49152 STEPS=1 GATE=1 COND=1 QK=1 .venv/bin/python scratch/detbwd_gravmoe.py
```

Planned logs:

- `logs/sol/rms_headroom_tests.log`
- `logs/sol/rms_headroom_p4.log`
- `logs/sol/rms_headroom_wide_smoke.log`

Fences: Mac only; `RJOB_LOCAL=1`; no cross-device fp comparison; full pinned
trajectory SHA gate before calling the refactor lossless. The existing brute
capability verdict remains closed unless a separately pre-registered training
arm changes it.

## PRE-REG SOL-MINREPRO-RB1: the public module path reproduces one committed arm end-to-end (2026-08-01, Mac)

Command:

```bash
RJOB_LOCAL=1 .venv/bin/python -m llmopt.reproduce gravmoe-rb1
```

Log: `logs/sol/reproduce_gravmoe_rb1.log`.

Prediction/bar: the command resolves RB1 to `COND=1 QK=1 LN=0 LD=1
STEPS=2000`, runs `scratch/detbwd_gravmoe.py`, and prints PASS only if the
observed FINAL trajectory SHA exactly equals the committed
`c6766da235cf0b76...` pin. Any missing/mismatched SHA or nonzero child exit is
FAIL. Runtime is measured by the wrapper rather than estimated.

## VERDICT SOL-RMS-HEADROOM: PASS — the booked ACT_CLAMP ceiling was an evaluation-order overflow; exact factoring restores 512x RMS headroom and preserves 16/16 trajectories (2026-08-01, Mac)

Commands and logs:

```bash
.venv/bin/pytest tests/test_intmath.py -q
# logs/sol/rms_headroom_tests.log

RJOB_LOCAL=1 DIM=128 DHEAD=32 FFN=256 ACLAMP=49152 STEPS=1 GATE=1 COND=1 QK=1 .venv/bin/python scratch/detbwd_gravmoe.py
# logs/sol/rms_headroom_wide_smoke.log

RJOB_LOCAL=1 bash scratch/p4_arms_0801.sh
# logs/sol/rms_headroom_p4.log; per-arm logs under logs/p4/
```

Measurements:

- Focused integer tests: 12/12 pass. The new expression is exactly equal to the
  legacy expression in its safe range and exactly equal to a Python-unbounded
  integer oracle at the former `ACLAMP=49152` crash point. The legacy
  intermediate is negative there; the factored intermediate is positive and
  produces nonzero `isq`.
- The D=128 former-crash smoke completes: init clamp fraction 0.013, no
  `rms_fwd` divide-by-zero.
- Full regression: every pinned arm is trajectory-identical:

| Arm | FINAL trajectory SHA |
|---|---|
| A0 | `6fffa718f9c7b2c07f2196a4ce079a705517b229baa71b853896f0cda8128faf` |
| A1 | `1ad5f466aa9dab17540cae26358c4ad50749e9ae8ea4ad1faa71f4b4d2ed3ec1` |
| A2 | `23c154e6a31daef758230c3780d55082a9be871b734e0844008514760b652ff1` |
| A3 | `300e61ad2cd621f8c7c2f89cbce3d5f2cafd7397233d9b05ed571a6fe9bd8cab` |
| CA0 | `0ad9da9476d8be45a6ffe28cd95702edabf7d41124c36efa454bf22777f08c2d` |
| CA1 | `a328188cc7f46ebf2cacea1d6b23bd5a0e0b7f7acc58ed61119018d0f9dc21bc` |
| CA2 | `35396b368e2f42b781f39167696ae21e60b83c2d3c0e1d483608ea46548aaa76` |
| CA3 | `4b98a6ef05bb45f7ad3a7e5a50f32bb63b736060aa64b7763369c2ef8dc476e0` |
| GA0 | `2b29bd4aa29bc4fb4ac1ea76084dd4c88e7d93084f27945df0c48c07fae407b1` |
| GA2 | `66d8f8799f85599b05ac5cf2dd44dd12e88010ba835304625f9d6bed5babb9fc` |
| GA3 | `919b83476dc9b84e3791821bd884933f88a57e4e261cd7d59bfeab4960412ce0` |
| RB1 | `c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f` |
| RB3 | `6968b583f440405f7da819ece57f924ffb5fd59dcc228dcf27c1d349b2ecd29d` |
| RB1S16 | `14981553e6cbebe11f9625fc7b4405dd73ffb9fc5060d8161b57f076ce492ee4` |
| GRB1 | `1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca` |
| S1 | `e1b633a965171f16ca58d17fe8c597ffbe6362ae2dc6ed7b6f58ec0ed69c6087` |

Verdict: prediction 1 PASS exact, prediction 2 PASS exact, prediction 3 PASS
16/16, prediction 4 PASS. Factoring increases allowable mean-square by
`2**18`, hence RMS activation headroom by `2**9 = 512`, before the separate
`sum(x*x)` int64 bound. This retracts only the instrument claim that ACT_CLAMP
must stay near 80 Q-units for RMS safety; it does not change the BR-W4d
capability null or justify removing clamps needed for training dynamics.

## VERDICT SOL-MATURITY/TABLES: generated, independently adoptable views over all 621 entries (2026-08-01, Mac)

Commands and logs:

```bash
.venv/bin/python scripts/sol_enrich_results.py
# logs/sol/enrich_results.log
.venv/bin/python scripts/sol_generate_tables.py
# logs/sol/generate_tables.log
```

Outputs:

- `docs/sol/results-index-enriched.jsonl`: 621/621 source entries preserved,
  each with `maturity`, explicit/inferred provenance, evidence, orthogonal
  maturity flags, an explicit-replication-open bit, and a transparent impact
  score.
- `docs/sol/MATURITY-SUMMARY.md`: status counts and the algorithmically ranked
  ten live single-seed entries.
- `docs/sol/TABLES.md`: adopted/null/retracted views, 76 lineage edges grouped
  by thread, and two narrowly matched unresolved replication fences.

Generated status counts: measured 390, in-flight 129, superseded 53, null 39,
adopted 8, retracted 2. Provenance is intentionally conservative: most labels
are inferred and say so. Heading-based pre-reg/result links are printed as
`inferred heading match`, never promoted to curated house links.

Limitation found and measured: source link fields are type-unstable (scalar or
list). The existing query crashes on a scalar/list chain:

```bash
.venv/bin/python scripts/results_query.py --chain qk-cond
# exit 1, TypeError; logs/sol/results_query_scalar_link.log
```

The Sol generators normalize both forms locally. The living query script and
source index remain untouched.

## VERDICT SOL-MINREPRO-RB1: PASS — one command reproduces the full pin in 85.6 seconds (2026-08-01, Mac)

Command and log:

```bash
RJOB_LOCAL=1 .venv/bin/python -m llmopt.reproduce gravmoe-rb1
# logs/sol/reproduce_gravmoe_rb1.log
```

Resolved contract: `COND=1 LD=1 LN=0 QK=1 RJOB_LOCAL=1 STEPS=2000`.
Observed and expected FINAL trajectory SHA both equal
`c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f`;
wrapper verdict PASS, elapsed 85.6 seconds (hardened rerun with ambient
experiment knobs cleared). `python -m llmopt.reproduce --list`
lists all 16 committed gravmoe reproductions. No axiom-side wrapper was added;
that remains a proposal in `NOTES.md`, avoiding a cross-repo dependency.

## Verification ledger

- Changed-scope tests: 16 passed in 0.54s (`tests/test_intmath.py` plus
  `tests/test_reproduce.py`).
- CPU/non-Metal suite: 371 passed, 8 skipped, 3 warnings in 70.65s; command:
  `zsh -o pipefail -c '.venv/bin/pytest -q --ignore=tests/test_mlx_backend.py
  --ignore=tests/test_population.py --ignore=tests/test_exact_gemm.py
  --ignore=tests/test_fused_ce.py --ignore=tests/test_metal_kernels.py | tee
  logs/sol/pytest_cpu_no_metal.log'`.
- Honest full-suite fence: Metal is unavailable in this sandbox. The first full
  collection errors in two MLX modules (`logs/sol/pytest_full.log`); the broader
  attempt reaches 371 passes then reports 34 Metal-allocation failures
  (`logs/sol/pytest_cpu.log`). These are environment failures, not booked green.

## Ranked findings

1. **Severity high; confidence high — the ACT_CLAMP RMS ceiling was a false
   int64 law.** Exact factorization recovers 512x RMS headroom with 16/16 pinned
   SHAs unchanged. Cheapest independent kill/confirm: port the one-line factor
   to the C++ twin and run its high-activation scalar oracle before any training
   arm (`logs/sol/rms_headroom_p4.log` is the house-side reference).
2. **Severity medium; confidence high — result-index lineage fields are not
   type-stable and `results_query.py --chain` crashes.** Cheapest confirmation:
   run `.venv/bin/python scripts/results_query.py --chain qk-cond`; a fixed tool
   must exit zero and include the QK-COND -> QK-RESCOPE amendment edge.
3. **Severity medium; confidence medium — 129 entries still classify in-flight
   and the top live single-seed claims remain hard to see without an enriched
   view.** Cheapest kill/confirm: manually audit the ten-entry table in
   `docs/sol/MATURITY-SUMMARY.md`; reject the heuristic if more than two labels
   disagree with their source sections.
4. **Severity low; confidence high — pinned results lacked a minimal public
   reproduction path.** Cheapest confirmation: run
   `python -m llmopt.reproduce gravmoe-rb1`; PASS must reproduce the full SHA in
   roughly 86 seconds on this Mac, not merely a loss value.
5. **Severity low; confidence high — the default test gate is not headless-Mac
   clean, and `pytest | tee` can mask failure without pipefail.** Cheapest
   confirmation: run the full suite in a no-Metal session and inspect the exit
   code; adoption should add capability-aware MLX skips and retain pipeline
   failure propagation.

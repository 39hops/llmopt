# Handoff 2026-10-16-1: rehabilitation phase 2D landed (adjudicator fence, operand validation, registration digest, reference evaluator, negative controls); 22 historical adjudications byte-identical; 2C pushed, 2D NOT pushed

Seat: Fable 5.1 on the Mac. Follows handoff 2026-10-16-0. No live
registered run at any commit (`liverun status` before each). Nothing
trained, downloaded, deleted or rewritten. The 2B+2C set (12 commits,
66456518..80fd095a) was pushed at the start of this session on Artin's
authorization in the phase brief; origin/main = 80fd095a. The 3080
was fast-forwarded to 80fd095a (its tree carries three pre-existing
untracked files: COMPOSE_DE, COMPOSE_FLQ2,
scratch/leancheck/OvershootTest.lean). Phase 2D commits are local
only; the brief forbids pushing them without separate authorization.

## What landed (commits on main, in order)

1. be866071 freeze: `scripts/gen_prereg_adjudication_pins.py` pairs
   each of the 20 booked JSON pre-regs with its tracked observations
   document (22 pairs: homeo-actuator and headswap-impulse each have an
   online and an offline document) and records library outcomes with
   full reason chains, the refutation string, `scripts/adjudicate.py`
   stdout, stderr and return code, and the sha256 of both inputs.
   Fixture written from the tree at 80fd095a before any adjudicator
   change. `tests/test_prereg_adjudication_pins.py` = 68 checks.
   Finding: all 22 pairs exit 0; no historical bar was UNRESOLVED, so
   every UNRESOLVED path is exercised by synthetic fixtures only.
   fba5c949 registers the script in INDEX/CODEMAP.
2. 51a6e995 fence and validation. `llmopt/lab/metrics.py:adjudicate`
   takes `fence=None`; |observed - bar| < fence returns UNRESOLVED,
   equality resolves by the comparison law (below: <= fires; above: >=
   fires), None or 0 is the legacy comparison. Every operand must be a
   finite int or float (bool, str, None, NaN, inf raise
   `MetricContractError(metric_value_invalid)` before any ordering).
   Defects this fixed: NaN read NO-FIRE silently; +inf read FIRE under
   `above`; a string "0.5" was cast by float(). `llmopt/lab/prereg.py`:
   `"fence": {"width", "unit"}` on bars, conjuncts and
   `refuted_if_predicate`; fence UNRESOLVED on the bar or any conjunct
   makes the bar UNRESOLVED with the arithmetic and unit in the reason,
   whatever the other components read (same precedence as a not-run
   conjunct); an UNRESOLVED condition bar leaves the refutation
   `UNRESOLVED (condition: ...)` instead of NOT-REFUTED; a fenced
   predicate inside its band reads `UNRESOLVED (fence: ...)`.
   `scripts/adjudicate.py` exits 2 when the refutation line is
   UNRESOLVED. `tests/test_prereg_fence.py` = 64 tests.
3. 819c66a3 identity, contract, reference, controls.
   `registration_digest(doc)` = sha256 of canonical JSON over name,
   results_id, registered, machine, arm names, bars, refutation
   predicate and precedence (prose, receipts and operands excluded);
   an observations document may carry it as `registration_sha256` and
   adjudication raises `RegistrationMismatchError` on any difference.
   `check_observations` refuses measurement keys naming no registered
   bar, conjunct or refutation key; `load()`/`load_observations()`
   refuse duplicated JSON keys; results_id must be a non-empty string;
   bar ids must be int; bar values finite and not bool; optional bar
   `unit` must match the measurement's and the fence's unit;
   `unlocked_receipts(doc, lock)` lists declared receipts the lock does
   not hold. Observation arms no bar names stay ignored (a booked
   adapter test records every arm it ran). The fence comparison moved
   to exact rationals after the reference evaluator caught
   observed = bar + width evaluated in float64 rounding the difference
   up to exactly the width (FIRE where exact arithmetic says inside).
   `tests/reference_adjudicator.py` (no llmopt imports, Fraction
   arithmetic, three-valued) + `tests/test_adjudicator_reference.py`
   (fixed table, 4000 random operand draws, 7000 one-ulp boundary
   probes, 600 random documents, all 22 pins) +
   `tests/test_adjudicator_negative_controls.py` (edited threshold
   invisible without a frozen identity and refused with one; wrong
   bar, metric, population, aggregation, unit; duplicate keys; bad
   results_id; missing measurements; unlocked receipt path; NaN and
   Infinity JSON literals through the CLI; unpaired contrast; mutation
   sweep over every historical FIRE; precedence with an unsatisfied
   required bar). 4188bad2 regenerates INDEX/CODEMAP (this regen also
   repaired CODEMAP rows that 80fd095a had left stale: the 2C handoff
   added spec citations for detbwd_gravmoe, detbwd_r2b, oracle_worker
   and gen_mathgen_pins; the 3080's check_source at 80fd095a stopped
   on exactly that line).
4. f4f3724d bookings. OBSERVATION GATE-SEED-COLLISIONS-0 (RESULTS
   L78275): 120 distinct problem seeds; sampling seeds carry no level
   term, five levels share one 2,304-slot table over 798 seeds (11,520
   slots), 11 collisions inside each problem (stride 7 under 8
   members), 736 seeds shared across problems in a level; every wave's
   eight seeds are distinct, so collisions are common random numbers
   across prompts, not duplicated samples; totals unbiased, iid
   uncertainty approximate, paired arms unaffected, mapping unchanged.
   GLOSSARY gate entry carries the note. AMENDMENT
   TASK-ARITHMETIC-OVERLAP-SCOPED-2026-10-16 (RESULTS L78328):
   task_arithmetic.py evaluates calculus_lora.pt, trained at 586337ca
   on the three kinds with the identical seed-99 set excluded; overlap
   0 of 300 under today's generator and under the generator checked out
   at 586337ca (six equal digests); the 30-row overlap belongs to
   int_only_lora.pt, which no script evaluates on that set
   (task_composition's single-kind sets overlap 0); no RESULTS number
   depends on the set; the 2C re-score recommendation is withdrawn;
   follow-up named (manifest in adapter checkpoints), not armed. Spec
   section 2D marked done.

## Verification

- Historical pins: identical after every commit (library outcomes,
  reason chains, CLI stdout, return codes, input shas).
- Adjudicator suites together: 372 passed (pins, fence, reference,
  negative controls, prereg schema, claim lint, science incidents,
  qwen adjudicate tests, model1 scorer).
- Docs suite before the booking commit: 23 passed, 1 skipped.
- `bash scripts/check_source.sh` at the final commit: see the closing
  report line and logs/rehab-2d/check_source-<sha>.log.
- Not re-run: the 16 reproduce pins (no dependency of the training
  path changed; adjudicator code is not on it).
- 3080 read-only check_source at 80fd095a stopped at the CODEMAP
  staleness above (log logs/rehab-2d/check_source-80fd095a-3080.log on
  that box). A run at the 2D head needs the push first.

## Git-history evidence for registration freshness

For all 22 pairs the pre-reg's last commit is no later than the
observations document's first commit; two pairs (cheap-readout,
cycle-impulse) share one commit, which is weaker evidence than an
earlier commit. Seven pre-regs were edited after their first commit
but before their observations landed (loop-state-0: 4 minutes
before; model1-tree: 5 minutes before). The test
`test_every_booked_prereg_last_edit_precedes_its_observations`
asserts the ordering.

## Limits that remain

- A swap of measurements between two bars registering the same metric
  triple is undetectable by the contract check
  (`test_same_triple_bar_swap_is_a_documented_blind_spot`); only a
  frozen observations document plus the registration digest pins
  which number went to which bar.
- The refutation predicate has no registered metric triple; its
  measurement is compared without a population check.
- `registration_sha256` is forward-only; no booked observations
  document carries it, so historical tamper detection rests on git
  history alone.
- The correlation induced by shared sampling seeds across gate problems
  is not measured (it needs a disjoint-seed re-run of one checkpoint).
- No historical registration received a fence; the resolution law for
  gate deltas still lives in prose and in scripts/book.py's fence rule.

## Open decisions for Artin

1. Push the 2D commits (be866071..this handoff)? Then 3080
   `git pull --ff-only` and check_source (expect "raster: NOT
   verified"; the CODEMAP line should now pass).
2. Adopt `fence` on new pre-regs: suggested house default for 120-gate
   bars `{"width": 5, "unit": "solves"}` per the resolution law; write
   it into the adjudicate skill when the next pre-reg is drafted.
3. Adapter manifest follow-up (train_calculus.py + load_adapter legacy
   branch): own desk or fold into 2E?

## Next phase

2E per spec 2026-10-13-rehabilitation-phase-2b (checkpoint
classification and physical storage analysis, print-only).

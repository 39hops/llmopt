# Handoff 2026-08-12-3 — code-quality program Phases 3, 4, 5 all landed

Seat: Fable 5, Mac. HEAD at close: `8e13f12` (plus this handoff's
commit). Both checkouts in lockstep (3080 ff-only synced after every
phase, HEAD asserted; standing sync OK). 3080 idle; nothing queued.
pytest at close: 837 passed / 8 skipped, rc=0 (was 660 at session
start — +177 tests today). ruff exit 0.

## What landed (engineering only — nothing to book)

### Phase 3 module 5 — gate shim (spec §Phase 3 row 5, LAST module)

- `20c4c22` battery first (ordering rule): tests/test_gate_battery.py
  replays AMENDMENT SOFT-PROMPT-1-SAMPLER (fresh-generator single
  draws 0/200 mismatches across zero-padded widening; shared-stream
  desync) plus 5 frozen gate problem-grid pins
  (GATE_BAND + 1000*lv + 0, scratch/gatepins_freeze.py).
- `e15ea91` the shim: scripts/step_grpo_micro.py re-exports
  sample_wave_lp/gate_eval via a FOURTH coupling variant — the cited
  lines (:65 multinomial, :184 weights-sha print; three RESULTS
  citations) sit INSIDE the shimmed bodies, so the line-preserving
  blocks quote each cited fragment ON its booked line
  (test_gate_shim_line_anchors asserts them). lab/gate.py canonical;
  its in-body oracle imports now llmopt.lab.gen/verify direct; the
  sys.modules aliases (gate.py:32-33) DELETED, bench pop workaround
  removed. ZERO getsource guards remain repo-wide — Phase 3 exit met.

### Phase 4 — llmopt/common + duplication harvest (plan `ddd4ac0`)

- `d8dcf6c` llmopt/common/: pick_device (torch-like precedence
  `arg > LLMOPT_DEVICE > cuda > mps > cpu`, Artin GO; placement stays
  torch-native .to(dev); CPU islands are pins), srng (string-seed
  law), load_ckpt (weights_only=True audit point).
- `d80829e` 44 free files migrated to pick_device()
  (scratch/phase4_rewrite.py, per-file py_compile). Two named
  non-migrations: bench_ladder + task_composition end `else "mps"`
  (no cpu tail — hard fail beats silent cpu on 3B HF models).
- `33ab7e3` helper adoption: _root/_check + NnueEval/load_nnue ->
  llmopt/search/benchkit.py; ternary -> llmopt/common/quant.py.
  Identity-hashed before adopting; divergent/frozen copies stay
  (gen_proposer_data, train_nnue, bench_record, train_ternary,
  harvest_frontier).
- `91c297b` 66 dead sys.path bootstrap lines dropped from 47 UNCITED
  files. NEAR-MISS worth remembering: 45 scratch files bare-import
  scripts/ modules — their bootstraps are LOAD-BEARING cross-dir
  (auto script-dir path covers same-dir only); caught by read-only
  import scan, restored before anything ran, exempted by name.
  tests/ + oracle_worker + engine.py bootstraps also load-bearing.

### Phase 5 — package coherence (plan `433ed4e`, Artin GO)

- `80617e6` lazy root __getattr__ (PEP 562): `import llmopt` proven
  torch- AND sympy-free by subprocess test.
- `ed3c4e1` figures/runs split out of lab/ (3546 -> 2246 lines):
  figstyle/figsvg/figures/anatomy -> llmopt/figures/; runlog/lake/
  traj/runfiles -> llmopt/runs/. Old paths stay valid via
  sys.modules ALIAS SHIMS (full fidelity incl. privates) — the house
  move pattern now, guarded by tests/test_lab_aliases.py. Free
  importers migrated; cited files ride the shims.
- `6387811` env-read census table in lab/config.py: all 7 sites
  exempt (bootstrap/passthrough/build-knob) or battery-frozen; FRAC
  = 0.453 documented as the GT keep-frac golden point (RESULTS.md
  L18793).
- `b9bd5df` import-every-module smoke (154 modules; vendor/ excluded
  — module-scope argparse) + dead-code sweep: exactly TWO
  zero-reference symbols existed repo-wide, both deleted
  (zx_engine._phases_ok, population.adapter_state).
- `8e01dae` seam check: NO splits — metal.py is one compilation unit
  (shared kernel-source globals + one _kernel factory), rules.py a
  single rule table, problems.py one generator family.
- `8e13f12` CODEMAP regen fixup (gen_codemap only sees TRACKED files
  — regen must run AFTER git add when a commit adds scratch files;
  bit twice today, same shape).

## Conditions that bite next session

1. **Ratchet still cap==backlog (300/300)**: next RESULTS booking
   needs its FINDINGS bullet same-commit (unchanged since -1).
2. **gen_codemap regen-after-add**: when a commit ADDS a scratch/
   scripts file, run gen_codemap.py after `git add`, or the next
   suite goes red on the inventory guard (twice today).
3. README/docs may still describe lab/ as containing figures/runs
   and the source-identity regime — a front-facing pass over README
   + REPRODUCE has NOT been done; drift is likely.

## Next session

Start: this handoff -> BOARD -> spec
`docs/superpowers/specs/2026-08-12-code-quality-program-design.md`
Phases 6-7. Also candidates: the README/REPRODUCE drift pass (item 3
above), open decision #6 (bench_verify_fast ship bar), relay
2026-08-11-1 still DRAFT/UNSENT.

## Open decisions for Artin

1. ~~pick_device precedence~~ RESOLVED (torch-like, shipped).
2. ~~common/ vs lab/~~ RESOLVED (llmopt/common/ exists).
3. ~~Phase 5 split timing~~ RESOLVED (GO given, shipped).
4. `house-crystal` rename or keep (unchanged).
5. THEORY L4 paraphrase vs verbatim quote (unchanged).
6. bench_verify_fast ship bar: retire the bench, or re-pin its
   battery to the vendored 167 rows? (unchanged from -2).

## Also standing

- Relay 2026-08-11-1 DRAFT/UNSENT.
- Banked/unqueued unchanged: BASIN-CENSUS-1, Q9 births ladder,
  ignition-mass cell B, GROW-DECOMP n=3, MPS decay probes,
  excluded-experts anatomy, GT-7 candidate; v4anat unbooked.
- Incident note (no repo impact): bench_ladder.py has NO argparse —
  `--help` runs the real Qwen bench; killed same minute, no
  artifacts. Same class exists elsewhere in scripts/; check for
  argparse before smoke-testing by `--help`.

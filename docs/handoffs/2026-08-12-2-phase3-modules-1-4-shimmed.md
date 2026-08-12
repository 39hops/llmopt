# Handoff 2026-08-12-2 — Phase 3 modules 1-4 shimmed: keepsets, oracle_worker, shards, verify+gen

Seat: Fable 5, Mac. HEAD at close: `4c0311d` (plus this handoff's
commit). Both checkouts in lockstep (3080 ff-only synced after every
module; standing OK recorded 2026-08-12 to run the sync without
asking, HEAD asserted each time). 3080 idle; nothing queued.
pytest at close: 660 passed / 7 skipped, rc=0. ruff exit 0.

## What landed (engineering only — nothing to book)

Phase 3 of the code-quality program (spec 2026-08-12 §Phase 3),
modules 1-4 of 5, one module per commit as specified. Each shim:
canonical body in `llmopt/lab/`, scratch/script original becomes a
re-export, source-identity guard deleted same commit, behavioral
battery green as proof.

- **Module 1 `keepsets`** (`650f216`, plan `e5ad92a`): the
  demonstration case. Full acceptance RAN (14.6s, not skipped):
  booked Jaccards 0.8013/0.5331/0.5280, nulls, 3 decode dumps
  byte-identical. gt2_jaccard.py keeps CLI + TRAJ table; scratch
  importers (gt5/gt7) unaffected.
- **Module 2 `oracle_worker`** (`7c8eeb5`): worker is spawned BY
  PATH, so the scratch file keeps its sys.path bootstrap and
  `__main__` block around the re-export. Typed failure-path battery
  green; by-path smoke `1,1`.
- **Module 3 `shards.dequant`** (`142c6aa`): the line-citation
  wrinkle — RESULTS books `k3_expert_demo.py:33` and `:99-151`, so
  the shim is LINE-COUNT-PRESERVING (15-line comment+import block
  for the 15-line body) and the test asserts the anchors sit on
  their booked lines. A third coupling class the spec didn't name;
  fold into spec §Phase 3 text when next touched.
- **Booked-number battery for verify+gen** (`327fd99`): unblocked
  module 4 per the ordering rule. `data/axiom_phaseD_167.jsonl`
  vendored byte-identical (md5 48df699a) from axiom's
  `data/qual/phaseD_sample100.jsonl`; test replays the booked
  adjudication "167/167 pass verify_wave" (RESULTS L2871) in 7.3s,
  plus 24 scaled-candidate rejects (multiplicative — constant
  shifts pass derivative-equivalence BY DESIGN) and 6 string-seed
  gen pins. git add -f'd (data/*.jsonl ignored): small text
  receipt, 40KB.
- **Module 4 `verify` + `gen`** (`4c0311d`): bench scripts now
  re-export `_WAVE_CACHE`/`_wave_worker`/`verify_wave` and
  `_gen_isolated`. Two finds:
  1. `llmopt/lab/gate.py:32` pre-seeds
     `sys.modules["bench_step_tokens"] = lab.gen` — any process
     that loads llmopt.lab then bare-imports bench_step_tokens gets
     the WRONG module. Bit the parity bench (ImportError on
     verify_step); bench now pops the alias first. Module 5 should
     delete the alias when gate shims.
  2. Bench re-run: 28.7x speedup, 0 reject flips, but 11 accept
     flips vs the old oracle — ALL true-class corpus rows (sampled
     3/3 verbatim `nxt` in step_chains.jsonl): the old verify_step
     false-rejects modern u-sub/by-parts rows; the corpus outgrew
     the 2026-07-14 ship battery. NOT unsoundness; noted in the
     commit message.

## Conditions that bite next session

1. **Ratchet still cap==backlog (300/300)**: next RESULTS booking
   needs its FINDINGS bullet same-commit (unchanged from -1).
2. **Module 5 `gate` is the LAST and biggest**: 93 importers of
   `scripts/step_grpo_micro.py`, four pinned constants
   (B/GATE_LEVELS/GATE_N/GATE_BAND guarded at
   tests/test_lab_adoption.py:96-112), plus the sys.modules alias
   deletion. Needs its own plan; check whether step_grpo_micro is
   line-cited in RESULTS before choosing shim shape (the module 3
   lesson).

## Next session: Phase 3 module 5 (gate), own plan

Start: this handoff -> BOARD -> spec §Phase 3 row 5. writing-plans
over that row only. Gate's battery: gate constants lineage test
exists; check whether a booked-number gate battery is required
(ordering rule) and what artifact backs it before writing the plan.

## Open decisions for Artin (unchanged from -1 plus one)

1. Device precedence for `pick_device()` (Phase 4).
2. `llmopt/common/` vs `lab/` for shared helpers (Phase 4).
3. Phase 5 `lab/` split timing.
4. `house-crystal` rename or keep.
5. THEORY L4 paraphrase vs verbatim quote.
6. NEW: bench_verify_fast's ship bar ("0 accept flips") is now
   measuring the old oracle's decay on the modern corpus — retire
   the bench, or re-pin its battery to the vendored 167 rows?

## Also standing

- Relay 2026-08-11-1 DRAFT/UNSENT.
- llmopt_dump on WSL verified copy-not-move (-1 handoff); lake idea
  substrate intact.
- Banked/unqueued unchanged: BASIN-CENSUS-1, Q9 births ladder,
  ignition-mass cell B, GROW-DECOMP n=3, MPS decay probes,
  excluded-experts anatomy, GT-7 candidate; v4anat unbooked
  (exploration-grade).

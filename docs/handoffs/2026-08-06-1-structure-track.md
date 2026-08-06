# Handoff 2026-08-06-1 — the structure track (afternoon/evening)

Science untouched: GT-7 stays [HOLD], fires on Artin GO only.

## What this session was

Grok's repo-structure review (relayed by Artin) assessed with two
read-only Opus scouts + house spot-checks, then executed as small
guarded slices. Full assessment + rejections banked in RIFF-LEDGER
("Repo structure review", 2026-08-06).

## Landed (all pushed, suite 477 passed / 7 skipped)

- **docs/CODEMAP.md + scripts/gen_codemap.py** — the move-gate
  inventory: mechanical class ladder (library 131 / results-cited
  102 / reproduce-pinned 9 / spec-cited 39 / UNCITED 158 over 439
  files); literal-filename search catches path couplings imports
  miss (llmopt/reproduce.py -> detbwd_gravmoe). Inventory guard in
  tests (file list must match filesystem; citation counts
  deliberately unchecked — no booking tax).
- **llmopt/lab/** — five adoptions, every one guarded, ZERO evidence
  moved: verify_wave (44 sites) + _gen_isolated (58 sites) verbatim
  w/ source-identity + behavior-parity battery; lab/oracle module 1
  (Oracle -> CheckResult, typed events {TIMEOUT, CRASH_EOF,
  CRASH_PIPE, MEMBOMB} + counters, SLEEP/BOMB paths exercised live
  in tests); lab/config module 2 (from_env: casts raise, unknown
  prefixed vars fatal, resolved-config echo; PEP-604 union bug
  found+fixed); lab/keepsets module 3 (FIRST-RUN EXACT acceptance:
  booked GT2-REVIEW-2 Jaccards 0.8013/0.5331/0.5280 + nulls
  0.9205/0.8670/0.6364 + all three gt2_*_arm0_decode.json dumps
  byte-identical; ~15s, always-on where TRAJ artifacts exist).
- **Doctrine (CLAUDE.md)**: scratch doctrine (adoption not mv,
  CODEMAP the gate, no forking frozen families); dual-copy guard
  LIFECYCLE (fixes both same-commit; guard dies when a registered
  re-run migrates the driver); logs doctrine (untracked exhaust,
  battery/day subdirs, unique arm x seed paths, never append into a
  booked path; seedslad receipt exception scoped; bulk delete
  Artin-GO).
- **Housekeeping**: data/*.jsonl|json gitignored (file-handoff
  convention structural; git status noise 112 -> 14; deliberate adds
  via -f). Process note: one commit shipped a failing test (regex
  bugs, my error) — fixed next commit, named honestly.

## Rejected with reasons (RIFF; do not re-propose)

detbwd-family collapse (layered import lattice + cross-lab byte
certs + 16 sha-pins; extraction spec forbids); wholesale bench_*
archive (bare-name sys.path imports across ~64 files — primitives
adopted FIRST, archive only what CODEMAP clears).

## Next

- **lab/traj = opener of its own session** (design not copy; needs
  30B resident): specs/2026-08-06-lab-traj-session.md — desk diff of
  the three copies, D0 bit-identity regression, live gate
  byte-identity, optional GT-7 pairing under a named contract.
- Then gate / sink / timebox (spec order).
- Freeze-point only: UNCITED archive (+30-day staleness horizon +
  family eyeball), 07-24 data/checkpoints taxonomy, experiments/
  layout — all under BOARD:114 gate, both checkouts lockstep.
- Artin-GO only: disk triage (logs/ is 1.2GB local; pairs with the
  banked 51GB checkpoint thread).

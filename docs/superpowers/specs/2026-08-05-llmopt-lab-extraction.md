# Spec 2026-08-05: llmopt.lab — instruments in the library

Riff provenance: RIFF-LEDGER "Instruments IN the library" (Artin,
2026-08-05). Survey seat's duplication map house-spot-checked where
load-bearing. This is a PLANNED project: Fable implements module by
module, each with a byte-identity regression test before any consumer
migrates. Nothing here blocks live science.

## Measured duplication (survey, spot-checked)

~1,300+ duplicated lines: SIGALRM timebox idiom in 47 script files
(~470 lines of a pattern DOCTRINE ALREADY RULES UNSOUND), 324
sys.path.insert occurrences, 237 bare os.environ.get across 65+
scratch files (the typo-takes-default silent-failure surface), 3-5
independent oracle-box designs (only tonight's v3.2 is post-mortem-
hardened), 3 hand-copies of the MoE router class-patch dispatch, 40
hand-rolled jsonl append sites.

## Immediate bug fixes (before any extraction; small, number-neutral)

- F1: scratch/moe_gt1_arm2.py buffers per-problem rows and writes
  after the gate — a wall-kill loses every row (violates the
  streaming corollary). Stream + flush per row.
- F2: scratch/gt2_jaccard.py binds FRAC/GATE_ONLY/DROP_TAIL at
  IMPORT time — env set after import is silently ignored. Make
  call-time parameters with env defaults.
- F3: llmopt/mathgen/evaluate.py:71 calls p.check unboxed on model
  text IN LIBRARY CODE — the pathology-#7 call site. Route through
  lab.oracle when it exists; until then, a docstring warning.

## Extraction order (by duplication x silent-failure risk)

1. lab/oracle.py — tonight's subprocess line-server + RSS watchdog,
   behavior-verbatim. Oracle(wall, mem_cap_gb).check(problem, expr)
   -> CheckResult(ok, parsed, event) with typed events {TIMEOUT,
   CRASH_EOF, CRASH_PIPE, MEMBOMB} + counters. KEEP the SLEEP/BOMB
   test affordances and the v1/v2/v3 post-mortem docstring verbatim
   — they are the executable proof and the reason to trust it.
   Timeout stays a FAILURE, never a skip (changing that silently
   moves accuracy).
2. lab/config.py — dataclass + from_env(prefix): casts raise, UNKNOWN
   prefixed env vars ERROR (kills the typo class), resolved config
   echoed as one banner + one jsonl line at init ("init one and boom
   it logs").
3. lab/keepsets.py — decode_counts/keep/jmean/coverage beside
   moe.router_stats. REGENERATION-SENSITIVE: DROP_TAIL first-row
   rule, GATE_ONLY, stable-sort tie-break at the keep boundary must
   reproduce BYTE-IDENTICALLY; acceptance = DUMP_DECODE re-emission
   bit-for-bit + the GT2-REVIEW-2 booked stats (0.8013/0.5331/
   0.5280; nulls 0.9205/0.8670/0.6364).
4. lab/traj.py — one patch_moe_router(model, traj=..., keep=None)
   unifying moe_gt1.instrument + arm2's masked variant (+ the
   scripts/moe_router_stats third copy). HIGHEST migration risk:
   pooled-pos vs per-prompt tpos split, prompt_tail rule,
   precise-softmax, H/scores rounding are all certified-artifact-
   sensitive. restore() becomes a context manager (a raising gate
   must not leave the class patched — emit INSTRUMENT_NOT_RESTORED).
5. lab/gate.py — the generate->extract->boxed-check->tally loop; row
   schema field names and rounding frozen (cited by verdicts).
6. lab/sink.py — extend llmopt.runlog (not a second convention):
   Run(name, cfg).row() streams+flushes; .event(kind,...) prints +
   counts + jsonl; .close(status) writes a TERMINAL record with the
   counter census — analyses REFUSE to score a jsonl with no
   terminal record (the checkpoint-selection fix); nonzero anomaly
   counters mark status=degraded so a booking cannot quietly cite it.
7. lab/timebox.py — fork-based run_isolated for CPU-side farms
   (the solve_isolated pattern; fork stays correct where no Metal
   resident exists). The 47 SIGALRM bench scripts migrate ONLY on
   re-run — they back booked verdicts; freeze otherwise.

## Do-not-extract (the lab-notebook law)

gt2_jaccard.py stays in place (booked numbers cite it) and becomes a
thin importer of lab.keepsets only after the byte-identity test
passes. One-shot closed-thread families (ozaki_*, v4flash_*,
detbwd_*, pincer_*, metabolic_*, fourier*, pack_*) stay frozen — the
file is the record. Rule: grep docs/results-index.jsonl for a path
before touching it; a named file gets library ADOPTION with re-
verified output, never a silent refactor.

## The loud-failure contract (the design center)

Every anomaly typed + printed + counted + jsonl'd; terminal record
or it didn't finish; rows stream, never buffer; counters in the
summary line; config echoed at init; conservative-reject explicit
(failure reason travels with the score); instrument patches are
context managers. Each clause traces to a 2026-08-05 burn
(ORACLE-BOX 1-4, axiom budget-memoization, Lean file-abort).

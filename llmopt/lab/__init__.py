"""llmopt.lab — permanent instruments, adopted from the scripts that
proved them (spec 2026-08-05-llmopt-lab-extraction.md; CODEMAP is the
move gate). Adoption law: function bodies are VERBATIM copies of their
source scripts, guarded by source-identity + behavior tests
(tests/test_lab_adoption.py); the originating scripts stay frozen —
they are the record booked verdicts cite. New code imports from here;
existing scripts migrate only with a re-verified pass.

Adopted so far (2026-08-06):
  verify_wave     <- scripts/bench_verify_fast.py  (44 import sites)
  _gen_isolated   <- scripts/bench_step_tokens.py  (58 import sites)
  Oracle          <- scratch/oracle_worker.py + moe_gt1_arm2.check_isolated
                     (module 1 of the extraction spec; typed events)
  LabConfig       <- new (module 2): from_env(prefix) — casts raise,
                     unknown prefixed vars error, resolved-config echo
  keepsets        <- scratch/gt2_jaccard.py (module 3): decode_counts,
                     keep, jmean, coverage — GT2 booked stats + dump
                     bytes re-verified at adoption
"""
from llmopt.lab.config import ConfigError, LabConfig  # noqa: F401
from llmopt.lab.gen import _gen_isolated, gen_isolated  # noqa: F401
from llmopt.lab.oracle import CheckResult, Oracle  # noqa: F401
from llmopt.lab.verify import verify_wave  # noqa: F401

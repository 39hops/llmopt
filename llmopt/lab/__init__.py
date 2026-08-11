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
from llmopt.lab.catalog import scan_checkpoint  # noqa: F401
from llmopt.lab.config import ConfigError, LabConfig  # noqa: F401
from llmopt.lab.gate import GRPO_MICRO, GateSpec, gate_checkpoint  # noqa: F401
from llmopt.lab.gen import _gen_isolated, gen_isolated  # noqa: F401
from llmopt.lab.hash import git_sha, sha256_file  # noqa: F401
from llmopt.lab.jsonl import append_jsonl, read_jsonl, write_jsonl  # noqa: F401
from llmopt.lab.oracle import CheckResult, Oracle  # noqa: F401
from llmopt.lab.merge import average, shell_graft, task_vector  # noqa: F401
from llmopt.lab.runfiles import (is_done, rc_of, read_marker,  # noqa: F401
                                 require_resume_marker, run_dir,
                                 write_marker)
from llmopt.lab.runlog import RunLog  # noqa: F401
from llmopt.lab.verify import verify_wave  # noqa: F401

"""llmopt.lab — permanent instruments, adopted from the scripts that
proved them (spec 2026-08-05-llmopt-lab-extraction.md; CODEMAP is the
move gate). Adoption law: function bodies are VERBATIM copies of their
source scripts, guarded by source-identity + behavior tests
(tests/test_lab_adoption.py); the originating scripts stay frozen —
they are the record booked verdicts cite. New code imports from here;
existing scripts migrate only with a re-verified pass.

Verbatim adoptions (source-identity guarded)
  verify_wave     <- scripts/bench_verify_fast.py
  _gen_isolated   <- scripts/bench_step_tokens.py
  sample_wave_lp,
  gate_eval       <- scripts/step_grpo_micro.py (the standard 120 gate)
  Oracle          <- scratch/oracle_worker.py + moe_gt1_arm2
  keepsets        <- scratch/gt2_jaccard.py (GT2 stats re-verified)
  traj            <- the MoE router-patch path

Written for the package
  LabConfig       from_env(prefix): casts raise, unknown prefixed vars
                  error, resolved config echoed
  hash, jsonl     one digest and one jsonl semantics for the package
  runfiles        marker contract; refuses a checkpoint with no epoch
  runlog          streamed per-step receipts (aborted runs still book)
  catalog         checkpoint rows read from state-dict SHAPES only
  merge           average / shell_graft / task_vector, provenance
                  sidecars, never overwrites an existing file
  lake            Parquet runs/results/models/gates; needs the [lake]
                  extra (pyarrow, duckdb)
"""
from llmopt.lab.catalog import scan_checkpoint
from llmopt.lab.config import ConfigError, LabConfig
from llmopt.lab.gate import GRPO_MICRO, GateSpec, gate_checkpoint
from llmopt.lab.gen import _gen_isolated, gen_isolated
from llmopt.lab.hash import git_sha, sha256_file
from llmopt.lab.jsonl import append_jsonl, read_jsonl, write_jsonl
from llmopt.lab.keepsets import coverage, decode_counts, jmean, keep
from llmopt.lab.merge import average, shell_graft, task_vector
from llmopt.lab.oracle import CheckResult, Oracle
from llmopt.lab.runfiles import (is_done, rc_of, read_marker,
                                 require_resume_marker, run_dir,
                                 write_marker)
from llmopt.lab.runlog import RunLog
from llmopt.lab.verify import verify_wave

# lake is NOT imported here: it needs pyarrow at module scope (the
# [lake] extra). Reach it as `from llmopt.lab import lake`.
__all__ = [
    "CheckResult", "ConfigError", "GRPO_MICRO", "GateSpec", "LabConfig",
    "Oracle", "RunLog", "_gen_isolated", "append_jsonl", "average",
    "coverage", "decode_counts", "gate_checkpoint", "gen_isolated",
    "git_sha", "is_done", "jmean", "keep", "rc_of", "read_jsonl",
    "read_marker", "require_resume_marker", "run_dir", "scan_checkpoint",
    "sha256_file", "shell_graft", "task_vector", "verify_wave",
    "write_jsonl", "write_marker",
]

"""llmopt.lab — permanent instruments, adopted from the scripts that
proved them (spec 2026-08-05-llmopt-lab-extraction.md; CODEMAP is the
move gate). Adoption law: function bodies are VERBATIM copies of their
source scripts, guarded by source-identity + behavior tests
(tests/test_lab_adoption.py); the originating scripts stay frozen —
they are the record booked verdicts cite. New code imports from here;
existing scripts migrate only with a re-verified pass.

Adopted so far (2026-08-06, the two highest-traffic primitives):
  verify_wave   <- scripts/bench_verify_fast.py  (44 import sites)
  _gen_isolated <- scripts/bench_step_tokens.py  (58 import sites)
"""
from llmopt.lab.gen import _gen_isolated, gen_isolated  # noqa: F401
from llmopt.lab.verify import verify_wave  # noqa: F401

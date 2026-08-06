"""lab.gen — fork-isolated problem generation, ADOPTED VERBATIM from
scripts/bench_step_tokens.py (2026-08-06; that file stays frozen — it
backs the step-token race verdicts). The function body is
character-identical to the source; guarded by
tests/test_lab_adoption.py. Fix a bug here and there in the SAME
commit, or the guard fails.

This is the fork-SIGKILL timebox law applied to make_integrate (the
pathology-#7 call site class): fork, join with deadline, kill — no
SIGALRM. A killed generation returns None; callers treat that as the
cell being unavailable, never as a silent skip inside a batch.
`gen_isolated` is the public alias; `_gen_isolated` keeps the name 58
import sites know.
"""
from __future__ import annotations

import multiprocessing as mp


def _gen_isolated(level: int, seed: int, wall: int = 45):
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _w():
        from llmopt.mathgen.problems import make_integrate
        q.put(make_integrate(level, seed))

    p = ctx.Process(target=_w)
    p.start()
    p.join(wall)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    try:
        return q.get(timeout=10)
    except Exception:
        return None


gen_isolated = _gen_isolated

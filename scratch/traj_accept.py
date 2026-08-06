"""TRAJ acceptance driver (spec 2026-08-06-lab-traj-session.md,
tiers 2 / 3a / 3b): runs the FROZEN scratch drivers with ONLY the
instrument swapped for llmopt.lab.traj.patch_moe_router — corpus,
chat template, oracle path, and every row-write remain the frozen
code by import, so any byte difference is the unified patch's.

MODE=free   -> scratch/moe_gt1.py main() (tier 2 D0 at defaults;
               tier 3b at small N_EVAL). TRAJ=1 for traj rows.
MODE=masked -> scratch/moe_gt1_arm2.py main() (tier 3a; PERPROB=1
               for the per-problem row byte-compare).

Outputs go wherever the frozen drivers' env knobs point them
(OUT/TRAJ_OUT/LOG/PERPROB_LOG) — always fresh paths under
logs/traj_accept/, NEVER a booked path.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.lab.traj import patch_moe_router

MODE = os.environ.get("MODE", "free")

if MODE == "free":
    import scratch.moe_gt1 as m

    def shim(model):
        cm = patch_moe_router(model, traj=os.environ.get("TRAJ") == "1")
        state = cm.__enter__()  # driver process exits; no __exit__, same
        return state, state["n_experts"]  # lifetime as the frozen patch

    m.instrument = shim
    m.main()
elif MODE == "masked":
    import scratch.moe_gt1_arm2 as m

    def shim(model, keep):
        cm = patch_moe_router(model, keep=keep)
        state = cm.__enter__()
        return state, (lambda: cm.__exit__(None, None, None))

    m.instrument = shim
    m.main()
else:
    raise SystemExit(f"unknown MODE {MODE!r}")

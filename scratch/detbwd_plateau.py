"""PLATEAU-BREAK driver (LOCKSTEP Leg A rung 3, spec
2026-08-06-3080-lockstep-window.md; design pass 2026-08-07).

Asks what the DIET-BRIDGE plateau (~12,518 cycle-mean at NWIN=8,
60,224 params, 1000 steps, SHIFT=14 const) is bound by:
  arm A  NWIN=8,  STEPS=4000  — steps alone (optimization-bound?)
  arm B  NWIN=32, STEPS=4000  — 4x data at fixed capacity
  arm C  NWIN=64, STEPS=4000  — 8x data at fixed capacity

The cited driver scratch/detbwd_diet.py stays FROZEN (evidence
record, DIET-BRIDGE + LOCKSTEP-A2); this driver imports it and sets
module knobs only. All-integer, deterministic; traj shas are the
receipts; single runs (determinism class proven at R2/A1/A2).

Usage: NWIN=8 STEPS=4000 .venv/bin/python scratch/detbwd_plateau.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detbwd_diet as D  # noqa: E402

D.NWIN = int(os.environ.get("NWIN", "8"))
D.STEPS = int(os.environ.get("STEPS", "4000"))

if __name__ == "__main__":
    print(f"[plateau] NWIN={D.NWIN} STEPS={D.STEPS} "
          f"SHIFT={D.M.SHIFT}", flush=True)
    D.main()

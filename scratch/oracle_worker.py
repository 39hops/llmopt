"""Standalone oracle worker for timeboxed p.check (MOE-GT-6 v3).

A plain subprocess line-server: reads base64(pickle((problem, expr)))
lines on stdin, writes "1"/"0" per check on stdout. Imports sympy and
llmopt.mathgen only — never mlx/Metal.

Why not fork or spawn-pool: v1 forked the driver (30B Metal-wired
resident) and macOS SIGKILLed the PARENT (Killed: 9; three GT-6 runs
lost). v2's spawn ProcessPoolExecutor re-executes the driver script as
__main__ in the child AND turned a broken pool into a silent
(False, False) — a dead oracle scoring every answer wrong looks
exactly like a hard gate. v3 is a subprocess with an explicit
protocol: the parent owns the deadline, a crash is an EOF the parent
must handle LOUDLY, and nothing about the driver leaks in.

Usage (parent): scratch/moe_gt1_arm2.check_isolated.
Manual: echo <b64> | .venv/bin/python scratch/oracle_worker.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Phase 3 module 2 (2026-08-12): the worker body lives in
# llmopt.lab.oracle_worker (canonical since this commit); this file
# stays the by-path entry point scratch/moe_gt1_arm2.check_isolated
# spawns. Behavior pinned by tests/test_lab_oracle.py's typed
# failure-path battery (TIMEOUT via SLEEP, crash via real kill,
# MemoryError-as-"0" via BOMB).
from llmopt.lab.oracle_worker import main  # noqa: E402,F401


if __name__ == "__main__":
    main()

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

import base64
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    # HARD MEMORY CAP (v3.1): a pathological simplify can balloon GBs
    # in seconds — the 20s timeout window was enough for the worker's
    # spike to drive system memory pressure and get the 17GB DRIVER
    # jetsam-killed (four runs lost; kills always followed the first
    # 1-2 timeouts). Cap the worker so a blowup dies as its own
    # MemoryError -> caught -> "0", never as system pressure.
    import resource
    CAP = 2 << 30
    for lim in ("RLIMIT_DATA", "RLIMIT_AS"):
        try:
            resource.setrlimit(getattr(resource, lim), (CAP, CAP))
        except (ValueError, OSError):
            pass
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SLEEP":  # test affordance: exercises the parent's
            import time      # timeout+respawn path deterministically
            time.sleep(600)
        if line == "BOMB":   # test affordance: exercises the memory
            _x = []          # cap (must die as MemoryError -> "0",
            try:             # never as system pressure)
                while True:
                    _x.append(bytearray(50 << 20))
            except MemoryError:
                _x = None
                sys.stdout.write("0\n")
                sys.stdout.flush()
                continue
        try:
            # pickle is safe here: the only writer is our own parent
            # driver over a private pipe (same trusted codebase);
            # nothing untrusted can reach this stdin
            problem, expr = pickle.loads(base64.b64decode(line))
            ok = bool(problem.check(expr))
        except Exception:
            ok = False
        sys.stdout.write("1\n" if ok else "0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

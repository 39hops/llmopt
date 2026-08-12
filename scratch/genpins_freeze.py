"""One-shot freeze helper for tests/test_lab_verify_gen_battery.py:
prints sp.sstr of _gen_isolated problems on the pin grid, to be
pasted into GEN_PINS as frozen literals (string-seed house law
makes them stable across processes/machines)."""
import sys
from pathlib import Path


import sympy as sp  # noqa: E402

from llmopt.lab.gen import _gen_isolated  # noqa: E402

for level, seed in [(2, 7_000_000), (2, 7_000_050), (3, 7_000_001),
                    (3, 7_000_051), (4, 7_000_002), (4, 7_000_052)]:
    p = _gen_isolated(level, seed)
    expr = None if p is None else sp.sstr(p._expr)
    print(f"    ({level}, {seed}): {expr!r},")

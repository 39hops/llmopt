"""One-shot: print the sstr of the first gate problem per level for
the standard 120 gate's seed grid (GATE_BAND + 1000*lv + 0). Output
becomes the GATE_PINS literal in tests/test_gate_battery.py."""
import sympy as sp

from llmopt.lab.gate import GATE_BAND
from llmopt.lab.gen import _gen_isolated

for lv in (3, 4, 5, 6, 7):
    p = _gen_isolated(lv, GATE_BAND + 1000 * lv + 0)
    print(f"    {lv}: {sp.sstr(p._expr)!r},")

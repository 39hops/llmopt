"""External-slot callbacks for axiom's hybrid engine config.

Axiom's C++ successors()/verify_edge take a named-callback
registry; absent slots cost coverage, never soundness. This module
is llmopt's half of the contract: the sympy-powered slots, each
fork-walled (the nine-pathologies doctrine — no sympy call is
safely boxed except by fork).

Slots:
- heurisch(node_sstr) -> list[str]      (the L6 unlock: +21 by
  both engines' measurement; proposals only — axiom's verify_edge
  still gates every result)
- equivalence(lhs_sstr, rhs_sstr) -> "EQUIVALENT" |
  "NOT_EQUIVALENT" | "UNDECIDED"        (the UNDECIDED fallback
  from the qualification amendment)

Both return conservatively on wall/error: heurisch -> [],
equivalence -> "UNDECIDED".
"""
from __future__ import annotations

import multiprocessing as mp

_WALL_S = 60  # matches the engine's i_heurisch RULE_WALL


def _heurisch_worker(node_s: str, q) -> None:
    import sympy as sp

    from llmopt.search.rules import i_heurisch
    node = sp.sympify(node_s)
    if not isinstance(node, sp.Integral):
        q.put([])
        return
    q.put([sp.sstr(r) for r in i_heurisch(node)])


def heurisch(node_sstr: str) -> list[str]:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=_heurisch_worker, args=(node_sstr, q))
    pr.start()
    pr.join(_WALL_S)
    if pr.is_alive():
        pr.kill()
        pr.join()
        return []
    try:
        return q.get(timeout=5)
    except Exception:
        return []


def _equiv_worker(lhs: str, rhs: str, q) -> None:
    import sympy as sp
    d = sp.simplify(sp.sympify(lhs) - sp.sympify(rhs))
    if d == 0:
        q.put("EQUIVALENT")
        return
    # numeric witnesses before declaring inequality (axiom's own
    # two-witness standard)
    x = sp.Symbol("x")
    hits = 0
    for pt in (0.31, 0.73, 1.37):
        try:
            v = complex(d.subs(x, pt).evalf(chop=True))
            if abs(v) > 1e-7:
                hits += 1
        except Exception:
            q.put("UNDECIDED")
            return
    q.put("NOT_EQUIVALENT" if hits >= 2 else "UNDECIDED")


def equivalence(lhs_sstr: str, rhs_sstr: str) -> str:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=_equiv_worker,
                     args=(lhs_sstr, rhs_sstr, q))
    pr.start()
    pr.join(_WALL_S)
    if pr.is_alive():
        pr.kill()
        pr.join()
        return "UNDECIDED"
    try:
        return q.get(timeout=5)
    except Exception:
        return "UNDECIDED"


SLOTS = {"heurisch": heurisch, "equivalence": equivalence}

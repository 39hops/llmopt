"""lab.verify — the fast wave-verifier, ADOPTED VERBATIM from
scripts/bench_verify_fast.py (2026-08-06; that file stays frozen — it
backs the parity bench and every verdict that cites it). Function
bodies below are character-identical to the source; guarded by
tests/test_lab_adoption.py (source-identity + behavior parity). Fix a
bug here and there in the SAME commit, or the guard fails.

Design provenance (spec 2026-07-14-grpo-v2): (1) verdict cache by
(prev, cand); (2) ONE fork per wave, verdicts streamed (magic-bucket
blast radius); (3) never-integrate + numeric-first rejection — d/dx of
an unevaluated Integral is its integrand, numeric-screen at 3 generic
points is reject-only, sp.simplify paid only for survivors; acceptance
stays exact. Timeout is a FAILURE (conservative reject), never a skip.
"""
from __future__ import annotations

import multiprocessing as mp
import time

_WAVE_CACHE: dict[tuple[str, str], tuple[bool, bool]] = {}


def _wave_worker(prev_s: str, cands: list[str], q) -> None:
    """One fork verifies a whole wave; verdicts streamed per candidate
    so an outer kill loses only the wedged one (magic-bucket rule)."""
    import sympy as sp
    x = sp.Symbol("x")
    env = {"Integral": sp.Integral, "x": x, "sqrt": sp.sqrt,
           "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
           "exp": sp.exp, "log": sp.log, "atan": sp.atan,
           "asin": sp.asin, "pi": sp.pi, "E": sp.E}
    try:
        prev = sp.sympify(prev_s, locals=env)
    except Exception:
        for c in cands:
            q.put((c, False, False))
        q.put(None)
        return
    for cand_s in cands:
        try:
            cand = sp.sympify(cand_s, locals=env)
            # never integrate: d/dx of the difference; unevaluated
            # Integrals differentiate to their integrands
            d = sp.diff(prev - cand, x).doit(integrals=False)
            ok = None
            if d.has(sp.Integral, sp.Subs):
                ok = False  # unresolved carriers: conservative reject
            else:
                d2 = d
                # numeric-first: reject-only screen at 3 generic pts
                decided_zero = True
                for k in range(3):
                    try:
                        val = complex(d2.evalf(
                            20, subs={x: sp.Float("0.7183")
                                      + sp.Rational(17 * (k + 1), 100)}))
                    except Exception:
                        decided_zero = None
                        break
                    if abs(val) > 1e-8:
                        ok = False  # sound: valid steps vanish here
                        break
                    if abs(val) > 1e-16:
                        decided_zero = None  # suspicious: escalate
                        break
                if ok is None:
                    if decided_zero:
                        # numeric-zero at 3 points: CONFIRM symbolically
                        ok = bool(sp.simplify(d) == 0)
                    else:
                        ok = bool(sp.simplify(d) == 0)
                if ok and not (prev - cand).has(x):
                    ok = bool(sp.simplify(prev - cand) == 0)
            solved = bool(ok) and not cand.atoms(sp.Integral)
            q.put((cand_s, bool(ok), solved))
        except Exception:
            q.put((cand_s, False, False))
    q.put(None)


def verify_wave(prev_s: str, cands: list[str],
                wall: int = 20) -> dict[str, tuple[bool, bool]]:
    """Levers 1+2: cache, then one streamed fork for the misses."""
    out: dict[str, tuple[bool, bool]] = {}
    todo = []
    for c in dict.fromkeys(cands):  # dedup, order-stable
        hit = _WAVE_CACHE.get((prev_s, c))
        if hit is not None:
            out[c] = hit
        else:
            todo.append(c)
    if not todo:
        return out
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=_wave_worker, args=(prev_s, todo, q))
    pr.start()
    deadline = time.time() + wall
    while True:
        try:
            row = q.get(timeout=max(deadline - time.time(), 0.1))
        except Exception:
            break  # wall: unreturned candidates default to reject
        if row is None:
            break
        c, ok, solved = row
        out[c] = (ok, solved)
        _WAVE_CACHE[(prev_s, c)] = (ok, solved)
    pr.kill()
    pr.join()
    for c in todo:
        out.setdefault(c, (False, False))
    return out

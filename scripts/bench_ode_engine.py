"""ODE engine rung 1 (the ENGINE-shaped physics rung; generator
llmopt/mathgen/odes.py existed since the mathgen expansion but
nothing ever consumed it).

Structure rules reduce each ODE family to INTEGRALS, then the
existing integral engine (engine.solve, L8-grade) is the
subcontractor — the composition pattern of i_heurisch, one level up:
  separable y' = f*y     -> y = C1 * exp( INT f dx )
  linear1  y' + p y = q  -> y = e^{-P} ( INT e^{P} q dx + C1 ),
                            P = INT p dx
  cc2 (const coeff)      -> characteristic roots (algebra, no
                            integrals): exp/cos/sin basis
Oracle: sympy.checkodesol on the CANDIDATE (fork-isolated; hang =
wrong). Race vs sympy.dsolve at equal wall discipline. Bar
(pre-registered): engine matches dsolve's solve count on the
generated families; wall reported honestly either way.
"""
from __future__ import annotations

import multiprocessing as mp
import time

N_PER = 25
WALL = 60


def _solve_int(expr):
    """Subcontract an integral to the house engine; None if unsolved."""
    import sympy as sp

    from llmopt.search.engine import solve
    res = solve(sp.Integral(expr, sp.Symbol("x")), budget=200)
    if not res.solved:
        return None
    return res.state.expr


def _engine_worker(kind: str, level: int, seed: int, q: "mp.Queue") -> None:
    import sympy as sp

    from llmopt.mathgen.odes import (make_linear_first_order,
                                     make_second_order_cc,
                                     make_separable_growth)
    x = sp.Symbol("x")
    y = sp.Function("y")
    C1, C2 = sp.Symbol("C1"), sp.Symbol("C2")
    mk = {"separable": make_separable_growth,
          "linear1": make_linear_first_order,
          "cc2": make_second_order_cc}[kind]
    p = mk(level, seed)
    eq = p._expr[0] if isinstance(p._expr, tuple) else p._expr
    t0 = time.monotonic()
    cand = None
    lhs = (eq.lhs - eq.rhs).expand()
    dy = sp.Derivative(y(x), x)
    d2y = sp.Derivative(y(x), x, 2)
    if kind == "separable":
        # lhs = y' - f(x)*y  -> f = -coeff of y
        f = -sp.simplify(lhs.coeff(y(x)))
        F = _solve_int(f)
        if F is not None:
            cand = C1 * sp.exp(F)
    elif kind == "linear1":
        # lhs = y' + p*y - q
        pcoef = sp.simplify(lhs.coeff(y(x)))
        qrhs = -(lhs - dy - pcoef * y(x)).simplify()
        P = _solve_int(pcoef)
        if P is not None:
            inner = _solve_int(sp.exp(P) * qrhs)
            if inner is not None:
                cand = sp.exp(-P) * (inner + C1)
    else:  # cc2: characteristic polynomial, pure algebra
        a = sp.simplify(lhs.coeff(dy))
        b = sp.simplify(lhs.coeff(y(x)))
        r = sp.Symbol("r")
        roots = sp.roots(r**2 + a * r + b, r)
        rs = list(roots.items())
        if len(rs) == 1 and rs[0][1] == 2:
            r1 = rs[0][0]
            cand = (C1 + C2 * x) * sp.exp(r1 * x)
        elif len(rs) == 2:
            (r1, _), (r2, _) = rs
            if r1.is_real and r2.is_real:
                cand = C1 * sp.exp(r1 * x) + C2 * sp.exp(r2 * x)
            else:
                al, w = sp.re(r1), sp.im(r1)
                cand = sp.exp(al * x) * (C1 * sp.cos(sp.Abs(w) * x)
                                         + C2 * sp.sin(sp.Abs(w) * x))
    wall = time.monotonic() - t0
    ok = False
    if cand is not None:
        try:
            chk = sp.checkodesol(eq, sp.Eq(y(x), cand))
            ok = bool(chk[0])
        except Exception:
            ok = False
    q.put({"solved": ok, "wall": round(wall, 3)})


def _dsolve_worker(kind: str, level: int, seed: int, q: "mp.Ueue") -> None:  # type: ignore[name-defined]
    import sympy as sp

    from llmopt.mathgen.odes import (make_linear_first_order,
                                     make_second_order_cc,
                                     make_separable_growth)
    mk = {"separable": make_separable_growth,
          "linear1": make_linear_first_order,
          "cc2": make_second_order_cc}[kind]
    p = mk(level, seed)
    eq = p._expr[0] if isinstance(p._expr, tuple) else p._expr
    t0 = time.monotonic()
    try:
        sol = sp.dsolve(eq)
        ok = sol is not None
    except Exception:
        ok = False
    q.put({"solved": bool(ok), "wall": round(time.monotonic() - t0, 3)})


def run_arm(worker, kind, level, seed) -> dict:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=worker, args=(kind, level, seed, q))
    pr.start()
    pr.join(WALL)
    if pr.is_alive():
        pr.kill()
        pr.join()
        return {"solved": False, "wall": float(WALL)}
    try:
        return q.get(timeout=10)
    except Exception:
        return {"solved": False, "wall": float(WALL)}


def main() -> None:
    tot = {"engine": [0, 0.0], "dsolve": [0, 0.0]}
    for kind in ("separable", "linear1", "cc2"):
        k = {"engine": [0, 0.0], "dsolve": [0, 0.0]}
        for i in range(N_PER):
            for arm, worker in (("engine", _engine_worker),
                                ("dsolve", _dsolve_worker)):
                r = run_arm(worker, kind, 2, 11_000_000 + i)
                k[arm][0] += r["solved"]
                k[arm][1] += r["wall"]
                tot[arm][0] += r["solved"]
                tot[arm][1] += r["wall"]
        print(f"{kind}: engine {k['engine'][0]}/{N_PER} "
              f"({k['engine'][1]:.1f}s) | dsolve {k['dsolve'][0]}/{N_PER}"
              f" ({k['dsolve'][1]:.1f}s)", flush=True)
    n = 3 * N_PER
    print(f"TOTAL: engine {tot['engine'][0]}/{n} in {tot['engine'][1]:.0f}s"
          f" | dsolve {tot['dsolve'][0]}/{n} in {tot['dsolve'][1]:.0f}s")
    print("bar: engine matches dsolve's solves; wall honest either way")


if __name__ == "__main__":
    main()

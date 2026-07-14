"""Fast wave-verifier: the three lossless levers, parity-benched.

Levers (spec 2026-07-14-grpo-v2): (1) verdict cache by (prev, cand);
(2) ONE fork per wave, verdicts streamed (magic-bucket blast radius);
(3) never-integrate + numeric-first rejection — the verify_edge
lesson ported: the OLD _verify_step calls prev.doit(deep=True),
i.e. asks sympy to SOLVE the integral per verification; the fast
path differentiates the difference instead (d/dx of an unevaluated
Integral is its integrand), numeric-screens at 3 generic points
(reject-only: a valid step cannot be numerically nonzero), and pays
sp.simplify only for numeric survivors — acceptance stays exact.

Parity bar: on a battery of true pairs (corpus rows), perturbed
pairs (coefficient/sign tweaks), and garbage — ZERO accept flips
vs the old oracle, or no ship. Timing reported per class.

    .venv/bin/python scripts/bench_verify_fast.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _battery():
    rows = [json.loads(l) for l in
            Path("data/step_chains.jsonl").read_text().splitlines()
            if '"cur"' in l]
    rng = random.Random(0)
    rng.shuffle(rows)
    rows = [r for r in rows if "Integral" in r["cur"]][:120]
    battery = []
    for r in rows:
        battery.append((r["cur"], r["nxt"], "true"))
        # perturbations: coefficient tweak and sign flip
        pert = r["nxt"].replace("2", "3", 1) if "2" in r["nxt"] \
            else "2*" + r["nxt"]
        battery.append((r["cur"], pert, "perturbed"))
        if "+" in r["nxt"]:
            battery.append((r["cur"], r["nxt"].replace("+", "-", 1),
                            "signflip"))
    battery += [(rows[i]["cur"], "x***bogus((", "garbage")
                for i in range(0, 40)]
    return battery


def main() -> None:
    from bench_step_tokens import verify_step  # the old oracle
    battery = _battery()
    print(f"# parity battery: {len(battery)} pairs")
    flips_accept = flips_reject = 0
    t_old = t_new = 0.0
    from collections import defaultdict
    times = defaultdict(list)
    # group by prev to exercise wave batching
    by_prev: dict[str, list[tuple[str, str]]] = {}
    for prev, cand, cls in battery:
        by_prev.setdefault(prev, []).append((cand, cls))
    for prev, items in by_prev.items():
        cands = [c for c, _ in items]
        t0 = time.time()
        new = verify_wave(prev, cands)
        t_new += time.time() - t0
        for cand, cls in items:
            t0 = time.time()
            old_ok, _old_solved = verify_step(prev, cand)
            dt = time.time() - t0
            t_old += dt
            times[cls].append(dt)
            new_ok = new[cand][0]
            if new_ok and not old_ok:
                flips_accept += 1
                print(f"  ACCEPT FLIP: {prev[:50]} | {cand[:50]}")
            elif old_ok and not new_ok:
                flips_reject += 1
                print(f"  reject flip: {prev[:50]} | {cand[:50]}")
    print(f"\nold wall {t_old:.1f}s  new wall {t_new:.1f}s "
          f"-> {t_old / max(t_new, 1e-9):.1f}x")
    print(f"accept flips (unsound if >0): {flips_accept}")
    print(f"reject flips (conservative-ok, report): {flips_reject}")
    n = len(battery)
    print(f"bar: 0 accept flips AND speedup > 2x -> "
          f"{'SHIP' if flips_accept == 0 and t_old > 2 * t_new else 'NO SHIP'}")


if __name__ == "__main__":
    main()

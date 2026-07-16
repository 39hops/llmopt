"""Curriculum v2: farm the algebra/simplification shard (riff ledger
2026-07-15, staged curriculum pretraining — the L4-starvation fix).

The 19M's phase-1 diet was calculus-only; L4 (nested product/chain
integrands) is diet-thin and GRPO cannot self-feed a band that
produces no valid samples. This shard supplies the substrate:
verified algebraic rewrites in the same Current:/Step: format.

Families (levels 1-3 by depth):
  expand   — (ax+b)(cx+d)... -> expanded polynomial
  factor   — expanded polynomial -> factored form
  collect  — scrambled like terms -> collected polynomial
  cancel   — rational with a shared factor -> cancelled
  prodpoly — THE L4 TARGET: expand the polynomial factor inside a
             product with an opaque sin/cos/exp composition, e.g.
             (3*x**2+2)**2*sin(x**3+2*x+2) -> (9*x**4+...)*sin(...)

Safety: nxt is constructed BY expand/factor/cancel, so equality
holds by construction; the re-check is expand(cur - nxt) == 0 —
no simplify() anywhere (sympy timebox law). Rows stream to the
shard file incrementally (killed-worker selection-effect law).
Seeds: stable STRING seeds, band "algebra-v2". Charset is verified
against the MathTokenizer atom set; uncovered rows are dropped and
counted.

    .venv/bin/python scripts/farm_algebra.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer

x = sp.Symbol("x")
OUT = Path("data/micromodel_algebra_shard0.jsonl")
PER_CELL = 2000  # rows per (family, level) target


def _poly(rng, deg, cmax=9):
    """Random polynomial with small nonzero-lead integer coeffs."""
    cs = [rng.randint(-cmax, cmax) for _ in range(deg + 1)]
    cs[0] = rng.choice([c for c in range(-cmax, cmax + 1) if c])
    return sum(c * x ** (deg - i) for i, c in enumerate(cs))


def _opaque(rng, level):
    """Opaque composition factor (never expanded): f(inner poly)."""
    f = rng.choice([sp.sin, sp.cos, sp.exp])
    inner = _poly(rng, rng.randint(1, min(level + 1, 3)), 5)
    return f(inner)


def gen(family, level, i):
    rng = random.Random(f"algebra-v2-{family}-{level}-{i}")
    if family == "expand":
        n = 2 if level == 1 else rng.randint(2, level + 1)
        cur_e = sp.Mul(*[_poly(rng, rng.randint(1, level))
                         for _ in range(n)], evaluate=False)
        nxt_e = sp.expand(cur_e)
        think = "expand the product"
    elif family == "factor":
        base = sp.Mul(*[_poly(rng, 1) for _ in range(level + 1)])
        k = rng.randint(1, 6)
        nxt_e = sp.factor(k * base)
        cur_e = sp.expand(k * base)
        think = "factor the polynomial"
    elif family == "collect":
        terms = []
        for _ in range(3 + 2 * level):
            terms.append(rng.randint(-9, 9) * x ** rng.randint(0, level + 1))
        cur_e = sp.Add(*terms, evaluate=False)
        nxt_e = sp.expand(sp.Add(*terms))
        think = "collect like terms"
    elif family == "cancel":
        shared = _poly(rng, 1)
        num, den = _poly(rng, level), _poly(rng, 1)
        # expand top and bottom so the shared factor is hidden —
        # sympy would auto-cancel the factored form at construction
        cur_e = sp.expand(shared * num) / sp.expand(shared * den)
        nxt_e = sp.cancel(cur_e)
        think = "cancel the shared factor"
    elif family == "prodpoly":
        p = _poly(rng, rng.randint(1, 2), 5)
        k = rng.randint(2, 2 + (level > 1))
        op = _opaque(rng, level)
        cur_e = sp.Mul(p ** k, op, evaluate=False)
        nxt_e = sp.expand(p ** k) * op
        think = "expand the polynomial factor, keep the composition"
    else:
        raise ValueError(family)
    return cur_e, nxt_e, think


def main() -> None:
    tok = MathTokenizer()
    fams = ("expand", "factor", "collect", "cancel", "prodpoly")
    seen: set[tuple[str, str]] = set()
    stats = {"kept": 0, "dup": 0, "identity": 0, "charset": 0,
             "unverified": 0}
    with OUT.open("w") as f:
        for fam in fams:
            for lv in (1, 2, 3):
                kept = i = 0
                while kept < PER_CELL and i < PER_CELL * 4:
                    cur_e, nxt_e, think = gen(fam, lv, i)
                    i += 1
                    cur, nxt = sp.sstr(cur_e), sp.sstr(nxt_e)
                    if cur.replace(" ", "") == nxt.replace(" ", ""):
                        stats["identity"] += 1
                        continue
                    if (cur, nxt) in seen:
                        stats["dup"] += 1
                        continue
                    # verification: no simplify — expand-based check;
                    # cross-multiply so rationals verify too
                    n1, d1 = sp.fraction(sp.together(cur_e))
                    n2, d2 = sp.fraction(sp.together(nxt_e))
                    if sp.expand(n1 * d2 - n2 * d1) != 0:
                        stats["unverified"] += 1
                        continue
                    text = f"Current: {cur}\nHints: none\nStep: {nxt}\n"
                    if tok.decode(tok.encode(text)) != text:
                        stats["charset"] += 1
                        continue
                    seen.add((cur, nxt))
                    f.write(json.dumps(
                        {"cur": cur, "nxt": nxt, "level": lv,
                         "source": f"algebra-{fam}", "hints": [],
                         "think": think}) + "\n")
                    kept += 1
                    stats["kept"] += 1
                print(f"{fam} L{lv}: {kept} rows", flush=True)
    print(f"done: {stats} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

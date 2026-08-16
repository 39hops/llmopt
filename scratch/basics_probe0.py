"""BASICS-PROBE-0 — does the 19M already do arithmetic without ever
being shown an arithmetic row? (desk probe, frozen checkpoint)

THRESHOLD, named before the first number (see the RESULTS entry):
  KILL      arithmetic pass@1 >= 50% at stock_s3 -> the model
            already does this; the basics rung dies as a
            capability claim.
  PROMOTE   arithmetic pass@1 <= 20% WHILE the resident
            algebra-expand family reads >= 50% -> a real gap on a
            family the diet does teach; the rung lives.
  BETWEEN   reshape.

Three arms separate CAPABILITY from FORMAT, because the diet holds
only 12 bare-numeric rows (BASICS-CENSUS-0) and a bare numeric
prompt is nearly out-of-format:
  (a) expand   RESIDENT family (6,000 rows in the diet), held-out
               seeds — the instrument's own sanity arm.
  (b) arith    ABSENT family: bare numeric eval with * and /, the
               coefficient arithmetic calculus steps actually
               perform (-9/2, 24/6, 7*8).
  (c) numsum   NEAR-FORMAT: numeric-only additive chains, the exact
               shape of the 12 incidental rows.
Scoring is oracle-verified (sympy exact equality on parsed
expressions), never string match; no simplify() anywhere.

Usage: .venv/bin/python -u scratch/basics_probe0.py [ckpt]
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "3")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import sympy as sp  # noqa: E402
import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from llmopt.common.device import pick_device  # noqa: E402
from llmopt.lab.gate import sample_wave_lp  # noqa: E402
from llmopt.lab.hash import git_sha  # noqa: E402
from llmopt.lab.jsonl import append_jsonl  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/gallery19m_stock_s3.pt"
N = int(os.environ.get("PROBE_N", "120"))
B = int(os.environ.get("PROBE_B", "8"))
OUT = "logs/basics/probe0.jsonl"
x = sp.Symbol("x")


def _poly(rng, deg, cmax=9):
    cs = [rng.randint(-cmax, cmax) for _ in range(deg + 1)]
    cs[0] = rng.choice([c for c in range(-cmax, cmax + 1) if c])
    return sum(c * x ** (deg - i) for i, c in enumerate(cs))


def gen_expand(i):
    """Resident family, farm_algebra's level-1 expand shape, held-out
    seed band (probe- prefix, never algebra-v2-)."""
    rng = random.Random(f"basics-probe0-expand-{i}")
    cur = sp.Mul(_poly(rng, 1), _poly(rng, 1), evaluate=False)
    return sp.sstr(cur), sp.expand(cur)


def gen_arith(i):
    """Absent family: the coefficient arithmetic calculus performs."""
    rng = random.Random(f"basics-probe0-arith-{i}")
    kind = rng.choice(["mul", "div", "divred", "pow"])
    if kind == "mul":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        cur, val = f"{a}*{b}", sp.Integer(a * b)
    elif kind == "div":
        b = rng.randint(2, 9)
        q = rng.randint(2, 9) * rng.choice([1, -1])
        cur, val = f"{b * q}/{b}", sp.Integer(q)
    elif kind == "divred":
        a = rng.randint(2, 9) * rng.choice([1, -1])
        b = rng.choice([2, 3, 4, 5])
        cur, val = f"{a}/{b}", sp.Rational(a, b)
    else:
        a = rng.randint(2, 7)
        cur, val = f"{a}**2", sp.Integer(a * a)
    return cur, val


def gen_numsum(i):
    """Near-format: the shape of the 12 incidental diet rows."""
    rng = random.Random(f"basics-probe0-numsum-{i}")
    terms = [rng.randint(-9, 9) for _ in range(rng.randint(4, 6))]
    s = " ".join(("- " + str(-t)) if t < 0 else ("+ " + str(t))
                 for t in terms).lstrip("+ ").strip()
    if s.startswith("- "):
        s = "-" + s[2:]
    return s, sp.Integer(sum(terms))


ARMS = {"expand": gen_expand, "arith": gen_arith, "numsum": gen_numsum}


def correct(pred, truth):
    """Oracle equality: parse and compare exactly. No simplify."""
    if not pred:
        return False
    try:
        p = sp.sympify(pred, evaluate=True)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return False
    try:
        return bool(sp.expand(p - truth) == 0)
    except (TypeError, ValueError):
        return False


def main():
    os.makedirs("logs/basics", exist_ok=True)
    t0 = time.time()
    tok = TM.MathTokenizer()
    dev = pick_device()
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6,
                           ffn=1536).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    res = {}
    # expand-arm structural diagnostic: when the model misses, does it
    # keep the polynomial FORM and miss only an arithmetic coefficient?
    struct = {"parsed": 0, "deg2_ok": 0, "const_ok": 0,
              "ends_ok_mid_wrong": 0, "misses": 0}
    with torch.no_grad():
        for arm, gen in ARMS.items():
            p1 = pk = n = 0
            examples = []
            for i in range(N):
                cur, truth = gen(i)
                try:
                    prompt = tok.encode(
                        f"Current: {cur}\nHints: none\nStep: ")
                except ValueError:
                    continue
                texts, _, _ = sample_wave_lp(
                    model, tok, prompt,
                    [7_700_000 + i * 31 + b for b in range(B)], dev)
                n += 1
                ok1 = correct(texts[0], truth)
                p1 += ok1
                pk += any(correct(t, truth) for t in texts)
                if arm == "expand" and not ok1:
                    struct["misses"] += 1
                    try:
                        p = sp.Poly(sp.sympify(texts[0]), x)
                        t = sp.Poly(truth, x)
                        struct["parsed"] += 1
                        d2 = p.coeff_monomial(x**2) == t.coeff_monomial(x**2)
                        c0 = p.coeff_monomial(1) == t.coeff_monomial(1)
                        mid = p.coeff_monomial(x) == t.coeff_monomial(x)
                        struct["deg2_ok"] += bool(d2)
                        struct["const_ok"] += bool(c0)
                        struct["ends_ok_mid_wrong"] += bool(d2 and c0
                                                            and not mid)
                    except Exception:
                        pass
                if len(examples) < 5:
                    examples.append({"cur": cur, "truth": sp.sstr(truth),
                                     "pred1": texts[0]})
            res[arm] = {"n": n, "pass1": p1, "passk": pk,
                        "pass1_pct": round(100 * p1 / max(n, 1), 2),
                        "passk_pct": round(100 * pk / max(n, 1), 2),
                        "examples": examples}
            print(f"[{arm}] n={n} pass@1 {res[arm]['pass1_pct']}% "
                  f"pass@{B} {res[arm]['passk_pct']}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for e in examples[:3]:
                print(f"    {e['cur']!r} -> pred {e['pred1']!r} "
                      f"(truth {e['truth']!r})", flush=True)
    print(f"[struct] expand misses {struct['misses']}, parsed "
          f"{struct['parsed']}, deg2 right {struct['deg2_ok']}, const "
          f"right {struct['const_ok']}, ends-right-middle-wrong "
          f"{struct['ends_ok_mid_wrong']}", flush=True)
    row = {"probe": "basics-probe0", "ckpt": CKPT, "device": dev,
           "n_per_arm": N, "wave_b": B,
           "code_commit": git_sha(short=True),
           "arms": res, "expand_struct": struct,
           "wall_s": round(time.time() - t0, 1)}
    append_jsonl(OUT, row)
    print(f"[done] {OUT}  wall {row['wall_s']}s", flush=True)
    print(json.dumps({a: r["pass1_pct"] for a, r in res.items()}))


if __name__ == "__main__":
    main()

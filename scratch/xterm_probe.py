"""XTERM-DIET-1 probe — sibling of the frozen basics_probe0.py
(pre-reg RESULTS 2026-08-16, L30565). Same three arms, same seeds,
same sampler, same oracle, PLUS the canonical-form counter the
pre-reg pins: sympify silently evaluates (3*4 + 2*5), so a
prediction counts as FULLY-EVALUATED only if its text is already
sympy-canonical (norm(pred) == norm(sstr(sympify(pred)))).
Intermediate-form corrects are counted separately, never inside a
bar. All BAR scoring is on the fully-evaluated counts.

Usage: .venv/bin/python -u scratch/xterm_probe.py [ckpt]
"""
import json
import os
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
from basics_probe0 import ARMS, correct  # noqa: E402  (frozen, import-only)
from llmopt.common.device import pick_device  # noqa: E402
from llmopt.lab.gate import sample_wave_lp  # noqa: E402
from llmopt.lab.hash import git_sha  # noqa: E402
from llmopt.lab.jsonl import append_jsonl  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/gallery19m_xtermcontrol_s3.pt"
N = int(os.environ.get("PROBE_N", "120"))
B = int(os.environ.get("PROBE_B", "8"))
OUT = "logs/xtermdiet1/probe.jsonl"
x = sp.Symbol("x")


def _n(s):
    return str(s).replace(" ", "")


def fully_evaluated(pred):
    """True iff pred's TEXT is already sympy-canonical — an
    intermediate like (3*4 + 2*5)*x parses correct but fails this."""
    try:
        p = sp.sympify(pred, evaluate=True)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return False
    return _n(pred) == _n(sp.sstr(p))


def main():
    os.makedirs("logs/xtermdiet1", exist_ok=True)
    t0 = time.time()
    tok = TM.MathTokenizer()
    dev = pick_device()
    model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6,
                           ffn=1536).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    res = {}
    # expand-arm structural diagnostic on FULL misses: does the model
    # keep the polynomial form and miss only the cross coefficient?
    struct = {"parsed": 0, "deg2_ok": 0, "const_ok": 0,
              "ends_ok_mid_wrong": 0, "misses": 0,
              "intermediate_form": 0}
    with torch.no_grad():
        for arm, gen in ARMS.items():
            p1 = pk = p1_any = pk_any = interk = n = 0
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
                oks = [correct(t, truth) for t in texts]
                fulls = [ok and fully_evaluated(t)
                         for ok, t in zip(oks, texts)]
                ok1, okk = fulls[0], any(fulls)
                p1 += ok1
                pk += okk
                p1_any += oks[0]
                pk_any += any(oks)
                interk += any(o and not f for o, f in zip(oks, fulls))
                if arm == "expand" and not ok1:
                    struct["misses"] += 1
                    if oks[0]:
                        struct["intermediate_form"] += 1
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
                        "pass1_any": p1_any, "passk_any": pk_any,
                        "passk_intermediate_only": interk,
                        "pass1_pct": round(100 * p1 / max(n, 1), 2),
                        "passk_pct": round(100 * pk / max(n, 1), 2),
                        "examples": examples}
            print(f"[{arm}] n={n} pass@1 {res[arm]['pass1_pct']}% "
                  f"pass@{B} {res[arm]['passk_pct']}% "
                  f"(any-form {p1_any}/{pk_any}, inter-only {interk}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for e in examples[:3]:
                print(f"    {e['cur']!r} -> pred {e['pred1']!r} "
                      f"(truth {e['truth']!r})", flush=True)
    print(f"[struct] expand misses {struct['misses']}, parsed "
          f"{struct['parsed']}, deg2 right {struct['deg2_ok']}, const "
          f"right {struct['const_ok']}, ends-right-middle-wrong "
          f"{struct['ends_ok_mid_wrong']}, intermediate-form "
          f"{struct['intermediate_form']}", flush=True)
    row = {"probe": "xterm-probe", "ckpt": CKPT, "device": dev,
           "n_per_arm": N, "wave_b": B,
           "code_commit": git_sha(short=True),
           "arms": res, "expand_struct": struct,
           "wall_s": round(time.time() - t0, 1)}
    append_jsonl(OUT, row)
    print(f"[done] {OUT}  wall {row['wall_s']}s", flush=True)
    print(json.dumps({a: r["pass1_pct"] for a, r in res.items()}))


if __name__ == "__main__":
    main()

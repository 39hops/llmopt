"""Validity autopsy: WHERE do the ~38% invalid steps go wrong?

The NNUE understood its game because every position carried a dense
labeled signal; the step model gets one sparse bit per candidate. This
keeps every REJECTED candidate from a gate-protocol pass and classifies
the wrongness, per level:

  empty        — no text / immediate EOS
  unparseable  — sympy can't parse the candidate
  identity     — equals the current state (stripped)
  repeat       — equals an earlier state in this chain
  scaled       — parses, wrong value, but equals c * (a valid sibling
                 from the same wave) for constant c: c<0 sign error,
                 else coefficient error (the Arena's one-sign class)
  structural   — parses, wrong value, no valid-sibling scaling match
                 (the model ran the wrong pattern)

If one class dominates, run-4 reward shaping targets it; if it's
diffuse structural wrongness, the lever is diet/capacity, not shaping.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model
from step_grpo_micro import (B, GATE_BAND, GATE_LEVELS, GATE_N,
                             sample_wave_lp)


def classify(cand: str, cur: str, visited: set, valids: list,
             sp) -> str:
    if not cand or not cand.strip():
        return "empty"
    c = cand.replace(" ", "")
    if c == cur.replace(" ", ""):
        return "identity"
    if c in visited:
        return "repeat"
    try:
        ce = sp.sympify(cand)
    except Exception:
        return "unparseable"
    for v in valids:
        try:
            ratio = sp.simplify(ce / sp.sympify(v))
            if ratio.is_constant() and ratio.is_number:
                return "scaled_neg" if ratio.is_negative else \
                    "scaled_pos"
        except Exception:
            continue
    return "structural"


def main(ckpt: str, d: int, layers: int, ffn: int, heads: int,
         n: int, out: str) -> None:
    import sympy as sp
    import torch

    from bench_step_tokens import _gen_isolated
    from bench_verify_fast import verify_wave

    tok = MathTokenizer()
    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(dev).eval()

    per_lv: dict[int, Counter] = defaultdict(Counter)
    rows = []
    with torch.no_grad():
        for lv in GATE_LEVELS:
            for i in range(n):
                p = _gen_isolated(lv, GATE_BAND + 1000 * lv + i)
                if p is None:
                    continue
                cur = f"Integral({sp.sstr(p._expr)}, x)"
                visited = {cur.replace(" ", "")}
                for ply in range(12):
                    prompt = tok.encode(
                        f"Current: {cur}\nHints: none\nStep: ")
                    texts, _, _ = sample_wave_lp(
                        model, tok, prompt,
                        [GATE_BAND + i * 31 + ply * 7 + b
                         for b in range(B)], dev)
                    distinct = [t for t in dict.fromkeys(texts)
                                if t and t.replace(" ", "")
                                not in visited]
                    wv = verify_wave(cur, distinct) if distinct else {}
                    valids = [t for t in distinct
                              if wv.get(t, (False, False))[0]]
                    nxt = None
                    for t in texts:
                        ok, so = wv.get(t, (False, False))
                        if ok and t.replace(" ", "") not in visited:
                            per_lv[lv]["valid"] += 1
                            if nxt is None:
                                nxt = "SOLVED" if so else t
                        else:
                            k = classify(t, cur, visited, valids, sp)
                            per_lv[lv][k] += 1
                            if k == "structural" and len(rows) < 400:
                                rows.append({"level": lv, "cur": cur,
                                             "cand": t})
                    if nxt == "SOLVED" or nxt is None:
                        break
                    cur = nxt
                    visited.add(cur.replace(" ", ""))
            tot = sum(per_lv[lv].values())
            print(f"L{lv} ({tot} candidates): "
                  f"{ {k: f'{100*v/tot:.1f}%' for k, v in per_lv[lv].most_common()} }",
                  flush=True)
    allc = Counter()
    for c in per_lv.values():
        allc.update(c)
    tot = sum(allc.values())
    print(f"ALL ({tot}): "
          f"{ {k: f'{100*v/tot:.1f}%' for k, v in allc.most_common()} }",
          flush=True)
    if out:
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} structural samples to {out}",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n", type=int, default=GATE_N)
    ap.add_argument("--out", default="data/autopsy_structural.jsonl")
    a = ap.parse_args()
    main(a.ckpt, a.d, a.layers, a.ffn, a.heads, a.n, a.out)

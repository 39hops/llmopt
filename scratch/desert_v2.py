"""Desert test v2 — cross-grammar composition probe (union eq coefficient iv).

Feed Liouville-dead integrals to everything-crystals holding BOTH the
integral and series grammars. Classify sampled steps per state:
  HONEST-STALL  zero oracle-valid proposals
  VALID-REWRITE valid integral-grammar rewrite (state stays dead)
  SERIES-REACH  valid proposal that leaves integral grammar for a
                truncated-series representation (the jailbreak move)
  BLUFF         invalid proposals only
Pre-reg (RIFF-LEDGER 2026-07-24): house predicts HONEST-STALL.
Usage: desert_v2.py <ckpt> [n_samples]
"""
import sys, json
sys.path.insert(0, ".")
import torch
import sympy as sp
import multiprocessing as mp
mp.set_start_method("fork", force=True)
from llmopt.train.mathnative import MathTokenizer, build_model

x = sp.Symbol("x")
DEAD = [  # Liouville-certified non-elementary integrands
    "Integral(exp(-x**2), x)",
    "Integral(exp(x**2), x)",
    "Integral(sin(x)/x, x)",
    "Integral(cos(x)/x, x)",
    "Integral(exp(x)/x, x)",
    "Integral(1/log(x), x)",
    "Integral(sin(x**2), x)",
    "Integral(exp(exp(x)), x)",
]

ckpt = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
tok = MathTokenizer(extra=["t"])
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab)).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
eos = tok.vocab.index("<eos>")


def sample_step(cur, temp=0.7):
    ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
    xt = torch.tensor([ids], device=dev)
    out = []
    with torch.no_grad():
        logits, past = model(xt, use_cache=True)
        for _ in range(160):
            p = torch.softmax(logits[0, -1] / temp, -1)
            nxt = torch.multinomial(p, 1).item()
            if nxt == eos:
                break
            out.append(nxt)
            logits, past = model(torch.tensor([[nxt]], device=dev),
                                 use_cache=True, past=past)
    return tok.decode(out).strip()


def verify(cur, pred, q):
    try:
        d = sp.simplify(sp.diff(sp.sympify(cur), x) - sp.diff(sp.sympify(pred), x))
        q.put(bool(d == 0))
    except Exception:
        q.put(False)


results = []
for cur in DEAD:
    votes = {"valid": 0, "invalid": 0, "series_marks": 0, "identity": 0}
    valid_preds = []
    for k in range(K):
        pred = sample_step(cur)
        if pred.replace(" ", "") == cur.replace(" ", ""):
            votes["identity"] += 1
            continue
        if "t" in pred and "Integral" not in pred:
            votes["series_marks"] += 1  # candidate representation switch
        q = mp.Queue()
        p = mp.Process(target=verify, args=(cur, pred, q))
        p.start(); p.join(30)
        if p.is_alive():
            p.kill(); p.join(); ok = False
        else:
            ok = q.get() if not q.empty() else False
        if ok:
            votes["valid"] += 1
            valid_preds.append(pred)
        else:
            votes["invalid"] += 1
    if votes["valid"] == 0:
        tag = "HONEST-STALL" if votes["invalid"] < K else "BLUFF"
        if votes["identity"] == K:
            tag = "HONEST-STALL"  # pure identity echo = stall
    else:
        tag = "VALID-REWRITE"
    results.append((cur, tag, dict(votes)))
    print(f"{tag:14s} {cur}  {votes}")
    for vp in valid_preds[:2]:
        print(f"    valid: {vp[:100]}")

from collections import Counter
print("SUMMARY:", Counter(t for _, t, _ in results), f"K={K} ckpt={ckpt}")

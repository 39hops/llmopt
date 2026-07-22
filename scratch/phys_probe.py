"""Physics rung 1 probe: greedy emission on held-out phys steps
(seeds 17-19), sympy-equivalence in t, fork-isolated. No math gate —
the physics expert is vocab-41, a separate model class by design.
Usage: phys_probe.py <ckpt>"""
import sys, json
import multiprocessing as mp
mp.set_start_method("fork", force=True)
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
from llmopt.train.mathnative import MathTokenizer, build_model

def _equiv(q, pred, gold):
    import sympy as sp
    try:
        t = sp.Symbol("t")
        d = sp.cancel(sp.sympify(pred, {"t": t})
                      - sp.sympify(gold, {"t": t}))
        q.put(d == 0)
    except Exception:
        q.put(False)

def equiv(pred, gold, deadline=10):
    q = mp.Queue()
    p = mp.Process(target=_equiv, args=(q, pred, gold))
    p.start(); p.join(deadline)
    if p.is_alive():
        p.kill(); p.join()
        return False
    return q.get() if not q.empty() else False

ckpt = sys.argv[1]
tok = MathTokenizer(extra=["t"])
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab)).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

rows = [json.loads(l) for l in open("data/phys_probe.jsonl")]
hit = 0
by_fam = {}
with torch.no_grad():
    for r in rows:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\nStep: ")
        xt = torch.tensor([ids], device=dev)
        logits, past = model(xt, use_cache=True)
        out = []
        nxt_id = int(logits[0, -1].argmax())
        for _ in range(160):
            if nxt_id == tok.eos_id or tok.vocab[nxt_id] == "\n":
                break
            out.append(nxt_id)
            col = torch.tensor([[nxt_id]], device=dev)
            logits, past = model(col, past=past)
            nxt_id = int(logits[0, -1].argmax())
        pred = tok.decode(out).strip()
        ok = equiv(pred, r["nxt"])
        hit += ok
        f = by_fam.setdefault(r["family"], [0, 0])
        f[0] += ok; f[1] += 1
print(f"PHYS probe: {hit}/{len(rows)} ({100*hit/len(rows):.1f}%)")
for k, (a, b) in sorted(by_fam.items()):
    print(f"  {k}: {a}/{b}")

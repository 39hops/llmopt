"""Series rung 1 probe: greedy next-partial-sum emission on the 142
held-out steps (seeds 17-19), scored by sympy polynomial equivalence
in fork-isolated workers (the solve_isolated doctrine). Also runs the
standard 120 gate for the paired regression read vs seedvar-1 (65).
Usage: series_probe.py <ckpt>"""
import os
import sys, json
import multiprocessing as mp
mp.set_start_method("fork", force=True)
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model

def _equiv(q, pred, gold):
    import sympy as sp
    try:
        x = sp.Symbol("x")
        d = sp.expand(sp.sympify(pred, {"x": x})
                      - sp.sympify(gold, {"x": x}))
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
tok = (MathTokenizer(extra=os.environ["VOCAB_EXTRA"].split(","))
       if os.environ.get("VOCAB_EXTRA") else MathTokenizer())
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab)).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

rows = [json.loads(l) for l in open("data/series_probe.jsonl")]
preds = []
hit = 0
by_fam = {}
with torch.no_grad():
    for r in rows:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\nStep: ")
        x = torch.tensor([ids], device=dev)
        logits, past = model(x, use_cache=True)
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
        preds.append({**r, "pred": pred})
        ok = equiv(pred, r["nxt"])
        hit += ok
        f = by_fam.setdefault(r["family"], [0, 0])
        f[0] += ok; f[1] += 1
json.dump(preds, open("series_preds.json", "w"))
print(f"SERIES probe: {hit}/{len(rows)} "
      f"({100*hit/len(rows):.1f}%) exact-coefficient steps")
for k, (a, b) in sorted(by_fam.items()):
    print(f"  {k}: {a}/{b}")
solves, valid = G.gate_eval(model, tok, dev)
print(f"SERIES-19M gate: {solves} = {sum(solves.values())}/120 "
      f"@ {valid:.2f}%", flush=True)

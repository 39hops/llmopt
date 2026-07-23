"""vm-asm rung 1 probe: greedy emission on 401 held-out steps.
Score: pred parses AND is symbolically equivalent to cur AND
differs from cur (a valid productive rewrite — any equivalent
answer accepted, string match never used). Exact-gold reported
separately. Usage: vmasm_probe.py <ckpt>"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scratch")
import torch
from vmasm import parse, run
from llmopt.train.mathnative import MathTokenizer, build_model

EXTRA = list("movadsublhngre;")  # unique chars for the ISA
tok = MathTokenizer(extra=sorted(set(EXTRA)))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab)).to(dev)
model.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
model.eval()
rows = [json.loads(l) for l in open("data/vmasm_probe.jsonl")]
valid = exact = 0
by_rule = {}
with torch.no_grad():
    for r in rows:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\nStep: ")
        xt = torch.tensor([ids], device=dev)
        logits, past = model(xt, use_cache=True)
        out = []
        nxt_id = int(logits[0, -1].argmax())
        for _ in range(200):
            if nxt_id == tok.eos_id or tok.vocab[nxt_id] == "\n":
                break
            out.append(nxt_id)
            col = torch.tensor([[nxt_id]], device=dev)
            logits, past = model(col, past=past)
            nxt_id = int(logits[0, -1].argmax())
        pred = tok.decode(out).strip()
        p = parse(pred)
        ok = (p is not None and pred != r["cur"]
              and run(p) == run(parse(r["cur"])))
        valid += ok
        exact += pred == r["nxt"]
        f = by_rule.setdefault(r["rule"], [0, 0])
        f[0] += ok; f[1] += 1
print(f"VMASM probe: valid-rewrite {valid}/{len(rows)} "
      f"({100*valid/len(rows):.1f}%) | exact-gold {exact}")
for k, (a, b) in sorted(by_rule.items()):
    print(f"  {k}: {a}/{b}")

"""CE-400: fixed-sample CE proxy (the standing instrument from the
CE-gate study). Usage: ce400.py <ckpt> <label>"""
from llmopt.common.device import pick_device
import random, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch, torch.nn.functional as F
from train_mathnative import load_rows
from llmopt.train.mathnative import MathTokenizer, build_model

ckpt, label = sys.argv[1], sys.argv[2]
tok = MathTokenizer()
rows = load_rows(gen4=True)
rows = [r for r in rows if r["cur"].replace(" ","") != r["nxt"].replace(" ","")]
random.Random(7).shuffle(rows)
dev = pick_device()
model = build_model(len(tok.vocab), d=256, layers=8, heads=4, ffn=1024).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
tot = n = 0
with torch.no_grad():
    for r in rows[:400]:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try: ids = torch.tensor([tok.encode(t) + [tok.eos_id]], device=dev)
        except ValueError: continue
        logits = model(ids[:, :-1])
        tot += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[0, 1:])); n += 1
print(f"{label} CE-400: {tot/n:.4f} (n={n})", flush=True)

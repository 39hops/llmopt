"""Atlas-2 Lyapunov leg: function-space divergence between twin
births. Observable (weight distance forbidden by doctrine):
teacher-forced argmax disagreement on 200 fixed gen-4 rows —
fraction of non-pad positions where the two models' greedy next-
token predictions differ. Usage:
  lyap_compare.py ckptA ckptB TAG   (d64/ffn256/heads4 assumed)
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

A, B, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
tok = MathTokenizer()

rows = load_rows(gen4=True)
enc = []
for r in rows:
    try:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
    except ValueError:
        continue
    if len(ids) <= 256:
        enc.append(ids)
    if len(enc) == 200:
        break

preds = []
for path in (A, B):
    m = build_model(len(tok.vocab), d=64, layers=8, heads=4,
                    ffn=256).to(dev)
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=True))
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, 200, 50):
            batch = enc[i:i + 50]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in batch], device=dev)
            am = m(x).argmax(-1).cpu()
            for j, q in enumerate(batch):
                out.append(am[j, :len(q) - 1])
    preds.append(out)

diff = tot = 0
for a, b in zip(*preds):
    diff += int((a != b).sum())
    tot += a.numel()
print(f"LYAP {TAG}: disagree {diff}/{tot} = {diff/tot:.4f}",
      flush=True)

"""Night-28b soup instrument: plain parameter mean of N
checkpoints (same shape), then gate. Usage:
  soup_gate.py TAG d layers ffn heads ckpt1 ckpt2 [ckpt3 ...]
VOCAB_EXTRA rides (atom order must match the births).
"""
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

TAG = sys.argv[1]
d, layers, ffn, heads = (int(a) for a in sys.argv[2:6])
paths = sys.argv[6:]
_extra = os.environ.get("VOCAB_EXTRA", "")
tok = MathTokenizer(extra=_extra.split(",") if _extra else None)

acc = None
for p in paths:
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if acc is None:
        acc = {k: v.float().clone() for k, v in sd.items()}
    else:
        for k in acc:
            acc[k] += sd[k].float()
soup = {k: v / len(paths) for k, v in acc.items()}

dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
m = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                ffn=ffn).to(dev)
m.load_state_dict({k: v.to(dev) for k, v in soup.items()})
m.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(m, tok, dev)
print(f"SOUP {TAG} ({len(paths)} ckpts): {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)

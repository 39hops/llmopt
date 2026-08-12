"""Build row embeddings for the delta-chained format (spec
2026-07-26-format-ladder): mean-pooled final-norm hidden states of
the pairs-trained control crystal (wfloor_d256) over each pair
text. Output: checkpoints/fmt_row_emb.pt (N, d) unit vectors,
row-aligned with the filtered gen-4 row list."""
from llmopt.common.device import pick_device
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

tok = MathTokenizer()
rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
dev = pick_device()
model = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
model.load_state_dict(
    torch.load("checkpoints/mathnative_wfloor_d256.pt",
               map_location="cpu"))
model.eval()

hidden = {}
model.norm.register_forward_hook(
    lambda m, i, o: hidden.__setitem__("h", o))

enc = []
bad = 0
for r in rows:
    t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
    try:
        enc.append(tok.encode(t)[:512])
    except ValueError:  # out-of-language rows (the 388): zero
        enc.append(None)  # vector below — never similar, never
        bad += 1          # walked; their groups drop at encode
print(f"{bad} out-of-language rows -> zero embeddings", flush=True)
out = torch.empty(len(rows), 256)
B = 256
good = [i for i, s in enumerate(enc) if s is not None]
out.zero_()
with torch.no_grad():
    for a in range(0, len(good), B):
        idx = good[a:a + B]
        batch = [enc[i] for i in idx]
        L = max(len(s) for s in batch)
        ids = torch.tensor(
            [s + [tok.pad_id] * (L - len(s)) for s in batch], device=dev)
        mask = torch.tensor(
            [[1] * len(s) + [0] * (L - len(s)) for s in batch],
            device=dev)
        model(ids, mask)
        h = hidden["h"]
        m = mask.unsqueeze(-1).float()
        emb = ((h * m).sum(1) / m.sum(1)).cpu()
        out[torch.tensor(idx)] = emb
        if (a // B) % 50 == 0:
            print(f"  {a}/{len(good)}", flush=True)
out[torch.tensor(good)] = torch.nn.functional.normalize(
    out[torch.tensor(good)], dim=1)
torch.save(out, "checkpoints/fmt_row_emb.pt")
print(f"saved checkpoints/fmt_row_emb.pt {tuple(out.shape)}", flush=True)

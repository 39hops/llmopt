"""C8-retrofit at 45M (pre-reg 2026-07-28 ~5PM): project
union_45m gates onto the C8 commutant (params/8), one warm
epoch on the union diet, ramped permutation penalty. Prints
projected-init math gate, then trains and saves; math+ZX final
gates run via gate scripts after. cuda/bf16 autocast.
"""
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS, BS, NB = 512, 12, 2048, 8, 16, 8
CKPT = "checkpoints/union_45m.pt"
OUT = "checkpoints/union_45m_c8.pt"
DIET = "data/union_math_zx.jsonl"


def shift_perm(n, sh):
    """index map: row r <- r shifted by sh within its 8-block."""
    return torch.tensor([NB * (r // NB) + (r % NB - sh) % NB
                         for r in range(n)])


def project(W):
    """C8 group average via double permutations (cheap, exact)."""
    acc = torch.zeros_like(W)
    for sh in range(NB):
        po = shift_perm(W.shape[0], sh)
        pi = shift_perm(W.shape[1], sh)
        acc += W[po][:, pi]
    return acc / NB


def anti_mass(W):
    P = project(W)
    return float(1.0 - (P.norm() ** 2 / W.norm() ** 2))


torch.manual_seed(1)
torch.backends.cuda.matmul.allow_tf32 = True
tok = MathTokenizer()
dev = "cuda"
base = torch.load(CKPT, map_location="cpu", weights_only=True)
for li in range(LAYERS):
    k = f"blocks.{li}.gate.weight"
    base[k] = project(base[k].float()).to(base[k].dtype)

model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                    heads=HEADS, ffn=FFN).to(dev)
model.load_state_dict(base)
torch.save({k: v.cpu() for k, v in model.state_dict().items()},
           "checkpoints/union_45m_c8_projinit.pt")
po1 = {}  # shift-1 index maps per shape (generator penalty)
pi1 = shift_perm(D, 1).to(dev)
po1 = shift_perm(FFN, 1).to(dev)

rows = [json.loads(ln) for ln in open(DIET)]
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
enc = []
for r in rows:
    try:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
    except ValueError:
        continue
    if len(ids) <= 512:
        enc.append(ids)
enc.sort(key=len)
print(f"{len(enc)} rows", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                        weight_decay=0.01)
order = list(range(0, len(enc) - BS + 1, BS))
random.Random(0).shuffle(order)
steps_total = len(order)
for bi, off in enumerate(order):
    batch = enc[off:off + BS]
    L = max(len(q) for q in batch)
    x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                      for q in batch], device=dev)
    with torch.autocast("cuda", torch.bfloat16):
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            y.reshape(-1), ignore_index=tok.pad_id)
    lam = 0.1 + 0.9 * bi / steps_total
    pen = 0.0
    for li in range(LAYERS):
        W = dict(model.named_parameters())[
            f"blocks.{li}.gate.weight"]
        C = W[:, pi1] - W[po1, :]
        pen = pen + (C.norm() ** 2) / (W.norm() ** 2)
    loss = loss + lam * pen / LAYERS
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if bi % 500 == 0:
        print(f"{bi}/{steps_total} loss {float(loss):.4f}",
              flush=True)

sd = {k: v.detach().float().cpu()
      for k, v in model.state_dict().items()}
torch.save(sd, OUT)
am = sum(anti_mass(sd[f"blocks.{li}.gate.weight"])
         for li in range(LAYERS)) / LAYERS
print(f"final anti-mass {am:.4f}", flush=True)

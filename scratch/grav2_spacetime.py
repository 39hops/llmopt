"""GRAV-2 (pre-reg 2026-07-30): engineered spacetime — birth a
d64h8 crystal with a contractivity penalty and price the toll.

ARM=ctl      plain dense birth (lambda 0)
ARM=contract per-step random-block expansion tax, lambda 0.1

After training: gate, epsilon-kick falloff profile (displacement
by depth at gentle amplitude), penalty-bind curve printed during
training. Mac. Usage: ARM=contract python scratch/grav2_spacetime.py
"""
import os
import random
import sys

os.environ.setdefault("ARM", "dense")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

ARM = os.environ["ARM"]
LAM = 0.1 if ARM == "contract" else 0.0
D, LAYERS, HEADS, FFN = 64, 8, 8, 256
BS, EPOCHS, LR = 8, 3, 1.5e-3
SEED = int(os.environ.get("SEED", "1"))
OUT = f"checkpoints/grav2_{ARM}_s{SEED}.pt"


def falloff(model, enc, tok, dev, eps=0.05):
    """Gentle-kick displacement profile: perturb block-k input by
    eps*rms once per k, report displacement at each later depth."""
    xb = torch.tensor(
        [enc[i] + [tok.pad_id] * (max(len(q) for q in enc[:8])
                                  - len(enc[i])) for i in range(8)],
        device=dev)
    prof = []
    for k in (1, 4):
        acts = {}

        def kick(mod, i, o, k=k):
            return None
        base, kicked = [], []
        hs = [model.blocks[j].register_forward_hook(
            lambda m, i, o, s=base: s.append(o[0].detach()))
            for j in range(LAYERS)]
        with torch.no_grad():
            model(xb)
        for h in hs:
            h.remove()
        g = torch.Generator(device="cpu").manual_seed(99)
        noise = None

        def pre(mod, args, k=k):
            x, mask, past = args
            n = torch.randn(x.shape, generator=g).to(x.device)
            n = n / n.norm(dim=-1, keepdim=True) \
                * eps * x.norm(dim=-1, keepdim=True)
            return (x + n, mask, past)
        ph = model.blocks[k].register_forward_pre_hook(pre)
        hs = [model.blocks[j].register_forward_hook(
            lambda m, i, o, s=kicked: s.append(o[0].detach()))
            for j in range(LAYERS)]
        with torch.no_grad():
            model(xb)
        for h in hs:
            h.remove()
        ph.remove()
        d = [float((a - b).norm(dim=-1).mean())
             for a, b in zip(kicked, base)][k:]
        prof.append((k, [round(v, 4) for v in d]))
    return prof


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    rows = load_rows(gen4=True)
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
    print(f"[grav2] ARM={ARM} lam={LAM} seed={SEED} dev={dev} "
          f"{len(enc)} seqs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    step, pen_run = 0, 0.0
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(order)
        for off in order:
            batch = enc[off:off + BS]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in batch], device=dev)
            logits = model(x)[:, :-1]
            y = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                ignore_index=tok.pad_id)
            if LAM > 0:
                # expansion tax on one random block
                li = random.randrange(LAYERS)
                blk = model.blocks[li]
                with torch.no_grad():
                    h_in = model.emb(x)
                    for b in model.blocks[:li]:
                        h_in, _ = b(h_in, None)
                delta = torch.randn_like(h_in)
                delta = delta / delta.norm(dim=-1, keepdim=True) \
                    * 0.01 * h_in.norm(dim=-1, keepdim=True)
                o1, _ = blk(h_in, None)
                o2, _ = blk(h_in + delta, None)
                ratio = ((o2 - o1).norm(dim=-1)
                         / delta.norm(dim=-1).clamp(min=1e-8))
                pen = F.relu(ratio - 1.0).mean()
                pen_run += float(pen)
                loss = loss + LAM * pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 1000 == 0:
                print(f"  ep{ep} step {step} loss {float(loss):.3f}"
                      f" pen {pen_run / 1000:.3f}", flush=True)
                pen_run = 0.0
    torch.save({"sd": model.state_dict()}, OUT)
    model.eval()
    solves, valid = G.gate_eval(model, tok, dev)
    print(f"[gate] ARM={ARM} solves {solves}/120 valid {valid}",
          flush=True)
    for k, prof in falloff(model, enc[-64:], tok, dev):
        print(f"[falloff] kick L{k}: {prof}", flush=True)


if __name__ == "__main__":
    main()

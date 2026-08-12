"""GRAV-1b (pre-reg 2026-07-30): the field in router coordinates.
Bin tokens by router probability p_e on the ablated expert; report
ablation dNLL per distance bin. Mac, umoe_lb_s{1,2}.
Usage: SEED=1 python scratch/grav1b_distance.py
"""
from llmopt.common.device import pick_device
import os
import sys

os.environ.setdefault("ARM", "lb")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import umoe_conserve as U  # noqa: E402
import grav_probe as GP  # noqa: E402
from train_mathnative import load_rows  # noqa: E402

SEED = int(os.environ.get("SEED", "1"))
CKPT = f"checkpoints/umoe_lb_s{SEED}.pt"
BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5),
        (0.5, 0.75), (0.75, 1.01)]


def main():
    dev = pick_device()
    U.ARM = "lb"
    tok, model = U.build()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["sd"]
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    rows = load_rows(gen4=True)
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    enc = []
    for r in rows[-4000:]:
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 256:
            enc.append(ids)
    enc = enc[:400]
    print(f"[grav1b] seed {SEED} dev {dev} battery {len(enc)}")

    # record router probs at probe layers during baseline pass
    probs = {}   # li -> [N, T, NE]
    for li in (1, 4, 6):
        probs[li] = []
        def mk(li):
            def hk(mod, i, o, li=li):
                p = F.softmax(mod.router(i[0]), -1)
                probs[li].append(p.detach().cpu())
            return hk
        model.blocks[li].moe.register_forward_hook(mk(li))
    base_nll = []
    with torch.no_grad():
        for x in GP.batches(enc, tok, dev):
            lg = model(x)[:, :-1]
            y = x[:, 1:]
            nll = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), y.reshape(-1),
                ignore_index=tok.pad_id, reduction="none").view(y.shape)
            base_nll.append(nll.cpu())
    base_nll = torch.cat(base_nll)
    mask = base_nll != 0
    P = {li: torch.cat(v) for li, v in probs.items()}

    print("bin: p_e range -> dNLL (n)  [monotone-in-p = lawful]")
    for li in (1, 4, 6):
        blk = model.blocks[li].moe
        orig = blk._one
        for e in range(U.NE):
            def gone(ex, h, e=e, orig=orig):
                y = orig(ex, h)
                return torch.zeros_like(y) if ex is blk.exp[e] else y
            blk._one = gone
            nll_a = []
            with torch.no_grad():
                for x in GP.batches(enc, tok, dev):
                    lg = model(x)[:, :-1]
                    y = x[:, 1:]
                    nll = F.cross_entropy(
                        lg.reshape(-1, lg.shape[-1]), y.reshape(-1),
                        ignore_index=tok.pad_id,
                        reduction="none").view(y.shape)
                    nll_a.append(nll.cpu())
            blk._one = orig
            d = torch.cat(nll_a) - base_nll
            pe = P[li][:, 1:, e]          # align with next-token nll
            line = []
            for lo, hi in BINS:
                m = (pe >= lo) & (pe < hi) & mask
                if m.sum() < 50:
                    line.append("     --    ")
                    continue
                line.append(f"{float(d[m].mean()):+.3f}({int(m.sum())})")
            print(f"  L{li}e{e}: " + " ".join(line))


if __name__ == "__main__":
    main()

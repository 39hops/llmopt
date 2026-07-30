"""BASIN-1: routing basin radius v usage. CPU.
Usage: SEED=1 python scratch/basin_probe.py"""
import os
import sys

os.environ.setdefault("ARM", "lb")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import umoe_conserve as U  # noqa: E402
import grav_probe as GP  # noqa: E402
from train_mathnative import load_rows  # noqa: E402

SEED = int(os.environ.get("SEED", "1"))
EPS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]


def main():
    dev = "cpu"
    U.ARM = "lb"
    tok, model = U.build()
    sd = torch.load(f"checkpoints/umoe_lb_s{SEED}.pt",
                    map_location="cpu", weights_only=True)["sd"]
    model.load_state_dict(sd)
    model = model.eval()
    rows = load_rows(gen4=True)
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    enc = []
    for r in rows[-2000:]:
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 256:
            enc.append(ids)
    enc = enc[:128]
    # capture pre-router hiddens (input to moe) per layer
    hid = {li: [] for li in range(U.LAYERS)}
    for li, blk in enumerate(model.blocks):
        def pre(mod, args, li=li):
            hid[li].append(args[0].detach())
        blk.moe.register_forward_pre_hook(pre)
    with torch.no_grad():
        for x in GP.batches(enc, tok, dev):
            model(x)
    g = torch.Generator().manual_seed(1234)
    print(f"[basin] seed {SEED}: eps -> retention (all layers)")
    per_exp = {}   # (li, e) -> (usage, radius_proxy)
    rets = []
    for li in range(U.LAYERS):
        H = torch.cat([h.reshape(-1, U.D) for h in hid[li]])
        r = model.blocks[li].moe.router
        with torch.no_grad():
            base = r(H).argmax(-1)
        usage = [float((base == e).float().mean())
                 for e in range(U.NE)]
        ret_e = np.zeros((U.NE, len(EPS)))
        for k, eps in enumerate(EPS):
            n = torch.randn(H.shape, generator=g)
            n = n / n.norm(dim=-1, keepdim=True) \
                * eps * H.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                pert = r(H + n).argmax(-1)
            for e in range(U.NE):
                m = base == e
                if m.any():
                    ret_e[e, k] = float((pert[m] == e).float().mean())
        rets.append(ret_e.mean(0))
        for e in range(U.NE):
            # radius proxy = area under retention curve
            per_exp[(li, e)] = (usage[e], float(ret_e[e].mean()))
    print("  mean retention:", [round(float(v), 3)
                                for v in np.mean(rets, axis=0)])
    us = np.array([v[0] for v in per_exp.values()])
    ra = np.array([v[1] for v in per_exp.values()])
    rc = np.corrcoef(np.argsort(np.argsort(us)),
                     np.argsort(np.argsort(ra)))[0, 1]
    print(f"[basin verdict-inputs] seed {SEED}: rank corr(usage, "
          f"basin) {rc:.3f} over {len(us)} (layer,expert) cells")


if __name__ == "__main__":
    main()

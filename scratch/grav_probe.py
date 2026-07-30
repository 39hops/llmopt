"""GRAV-1 (pre-reg 2026-07-30): expert gravity in the micro-MoE.

Per expert e in umoe_lb_s{1,2}: MASS (mean residual-write norm x
usage), FIELD FALLOFF (displacement at depth l+k after ablating e
at layer l), SCREENING (on-target v off-target delta-NLL under
ablation). Mac, held-out gen-4 rows.
Usage: SEED=1 python scratch/grav_probe.py
"""
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
from train_mathnative import load_rows  # noqa: E402

SEED = int(os.environ.get("SEED", "1"))
CKPT = f"checkpoints/umoe_lb_s{SEED}.pt"
NE, LAYERS = U.NE, U.LAYERS
N_EVAL = 400


def batches(enc, tok, dev, bs=8):
    L = max(len(q) for q in enc)     # GLOBAL pad: aligned concat
    for off in range(0, len(enc) - bs + 1, bs):
        b = enc[off:off + bs]
        yield torch.tensor([q + [tok.pad_id] * (L - len(q))
                            for q in b], device=dev)


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    U.ARM = "lb"
    tok, model = U.build()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["sd"]
    model.load_state_dict(sd)
    model = model.to(dev).eval()

    rows = load_rows(gen4=True)
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    enc = []
    for r in rows[-4000:]:          # tail rows = held-out-ish battery
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 256:
            enc.append(ids)
    enc = enc[:N_EVAL]
    print(f"[grav] seed {SEED} dev {dev} battery {len(enc)} rows")

    # ---- pass 0: baseline NLL per token + routing map + writes ----
    writes = [[0.0, 0] for _ in range(LAYERS)]  # sum norm, n (per L)
    wr_e = [[[0.0, 0] for _ in range(NE)] for _ in range(LAYERS)]
    hooks = []
    for li, blk in enumerate(model.blocks):
        def hk(mod, i, o, li=li):
            n = o.norm(dim=-1)
            idx = mod.last_idx
            for e in range(NE):
                m = idx == e
                if m.any():
                    wr_e[li][e][0] += float(n[m].sum())
                    wr_e[li][e][1] += int(m.sum())
        hooks.append(blk.moe.register_forward_hook(hk))

    base_nll, routes = [], []
    with torch.no_grad():
        for x in batches(enc, tok, dev):
            lg = model(x)[:, :-1]
            y = x[:, 1:]
            nll = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), y.reshape(-1),
                ignore_index=tok.pad_id, reduction="none"
            ).view(y.shape)
            base_nll.append(nll.cpu())
            routes.append(torch.stack(
                [blk.moe.last_idx.cpu() for blk in model.blocks]))
    for h in hooks:
        h.remove()
    base_nll = torch.cat(base_nll)                      # [N, T-1]
    routes = torch.cat(routes, dim=1)                   # [L, N, T]
    mask = base_nll != 0

    # mass = mean write norm x usage share, per (layer, expert)
    print("\n[mass] layer x expert: mean-write-norm x usage")
    mass = np.zeros((LAYERS, NE))
    for li in range(LAYERS):
        tot = sum(n for _, n in wr_e[li])
        for e in range(NE):
            s, n = wr_e[li][e]
            usage = n / max(tot, 1)
            mass[li][e] = (s / max(n, 1)) * usage
        print(f"  L{li}: " + " ".join(
            f"e{e}:{mass[li][e]:.2f}" for e in range(NE)))

    # ---- ablation passes: zero expert e at layer l ----
    print("\n[field+screen] ablate (l,e) -> on/off-target dNLL, "
          "displacement by depth")
    on_off, damages = [], []
    probe_layers = [1, 4, 6]
    for li in probe_layers:
        blk = model.blocks[li].moe
        orig = blk._one
        for e in range(NE):
            def gone(ex, h, e=e, orig=orig):
                y = orig(ex, h)
                if ex is blk.exp[e]:
                    return torch.zeros_like(y)
                return y
            blk._one = gone
            # displacement probe on one batch
            xb = next(batches(enc, tok, dev))
            acts_a, acts_b = [], []

            def rec(mod, i, o, store=None):
                store.append(o[0] if isinstance(o, tuple) else o)
            hs = [model.blocks[k].register_forward_hook(
                lambda m, i, o, s=acts_a: s.append(o[0].detach()))
                for k in range(LAYERS)]
            with torch.no_grad():
                model(xb)
            for h in hs:
                h.remove()
            blk._one = orig
            hs = [model.blocks[k].register_forward_hook(
                lambda m, i, o, s=acts_b: s.append(o[0].detach()))
                for k in range(LAYERS)]
            with torch.no_grad():
                model(xb)
            for h in hs:
                h.remove()
            blk._one = gone
            disp = [float((a - b).norm(dim=-1).mean())
                    for a, b in zip(acts_a, acts_b)]
            # full-battery NLL split by routed-through-e or not
            nll_a = []
            with torch.no_grad():
                for x in batches(enc, tok, dev):
                    lg = model(x)[:, :-1]
                    y = x[:, 1:]
                    nll = F.cross_entropy(
                        lg.reshape(-1, lg.shape[-1]), y.reshape(-1),
                        ignore_index=tok.pad_id, reduction="none"
                    ).view(y.shape)
                    nll_a.append(nll.cpu())
            blk._one = orig
            nll_a = torch.cat(nll_a)
            d = (nll_a - base_nll)
            hit = routes[li][:, 1:] == e                # routed thru
            on = float(d[hit & mask].mean()) if (hit & mask).any() \
                else float("nan")
            off = float(d[(~hit) & mask].mean())
            on_off.append((li, e, on, off))
            damages.append((mass[li][e], on))
            k_disp = [round(v, 3) for v in disp[li:]]
            print(f"  L{li}e{e}: on {on:+.3f} off {off:+.3f} "
                  f"ratio {abs(on) / max(abs(off), 1e-6):5.1f}x | "
                  f"falloff {k_disp}")

    m = np.array([a for a, _ in damages])
    dmg = np.array([b for _, b in damages])
    ok = ~np.isnan(dmg)
    rc = np.corrcoef(np.argsort(np.argsort(m[ok])),
                     np.argsort(np.argsort(dmg[ok])))[0, 1]
    ratios = [abs(on) / max(abs(off), 1e-6)
              for _, _, on, off in on_off if not np.isnan(on)]
    print(f"\n[grav verdict-inputs] seed {SEED}: "
          f"mass-damage rank corr {rc:.3f} | "
          f"median on/off ratio {np.median(ratios):.1f}x | "
          f"cells {len(ratios)}")


if __name__ == "__main__":
    main()

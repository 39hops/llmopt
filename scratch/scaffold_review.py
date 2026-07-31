"""Overnight scaffold review (2026-07-30 night): on the seed-2
checkpoints — MERGE-1 on gravmoe_s2, channel-ablation on
channel_s2. CPU. Runs as part of the overnight chain."""
import os
import sys

os.environ.setdefault("ARM", "gravmoe")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import umoe_conserve as U  # noqa: E402
import step_grpo_micro as G  # noqa: E402


def gate(m, tok):
    m.eval()
    return G.gate_eval(m, tok, "cpu")


def main():
    # merge gravmoe_s2
    U.ARM = "gravmoe"
    os.environ["ARM"] = "gravmoe"
    tok, m = U.build()
    sd = torch.load("checkpoints/umoe_gravmoe_s2.pt",
                    map_location="cpu", weights_only=True)["sd"]
    m.load_state_dict(sd)
    corrs = []
    with torch.no_grad():
        for blk in m.blocks:
            vs = [torch.cat([e[k].weight.flatten()
                             for k in ("g", "u", "d")]).detach()
                  for e in blk.moe.exp]
            for i in range(4):
                for j in range(i + 1, 4):
                    a = vs[i] - vs[i].mean()
                    b = vs[j] - vs[j].mean()
                    corrs.append(float((a @ b)
                                       / (a.norm() * b.norm())))
        print(f"[review gravmoe_s2] corr "
              f"{sum(corrs) / len(corrs):.4f}", flush=True)
        for blk in m.blocks:
            for k in ("g", "u", "d"):
                mw = torch.stack([e[k].weight
                                  for e in blk.moe.exp]).mean(0)
                for e in blk.moe.exp:
                    e[k].weight.copy_(mw)
    s, v = gate(m, tok)
    print(f"[review merge_s2] gate {s}/120 valid {v}", flush=True)
    # channel_s2 ablation
    U.ARM = "channel"
    os.environ["ARM"] = "channel"
    tok, m = U.build()
    sd = torch.load("checkpoints/umoe_channel_s2.pt",
                    map_location="cpu", weights_only=True)["sd"]
    m.load_state_dict(sd)
    amax = max(float(a.abs().max())
               for blk in m.blocks for a in [blk.moe.a])
    with torch.no_grad():
        for blk in m.blocks:
            for pnm in (blk.moe.sg, blk.moe.sg2, blk.moe.su,
                        blk.moe.su2, blk.moe.sd, blk.moe.sd2):
                pnm.zero_()
    s, v = gate(m, tok)
    print(f"[review channel_s2] max|a_i| {amax:.3f}; S-zeroed "
          f"gate {s}/120 valid {v}", flush=True)


if __name__ == "__main__":
    main()

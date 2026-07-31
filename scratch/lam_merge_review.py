"""Lambda-merge rider (pre-reg in the review-adoption amendment,
2026-07-31): merge reviews on the three lambda-arm checkpoints —
is merge-free lambda-independent, and does LOW lambda (weak
collapse) merge badly? Runs on the device holding the ckpts.
Usage: python scratch/lam_merge_review.py
"""
import os
import sys

os.environ.setdefault("ARM", "gravmoe")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import umoe_conserve as U  # noqa: E402
import step_grpo_micro as G  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for lam in ("0.1", "0.25", "1.0"):
        ck = f"checkpoints/umoe_gravmoe_g{lam}_cuda_s1.pt"
        if not os.path.exists(ck):
            print(f"[lam-merge g{lam}] MISSING {ck}", flush=True)
            continue
        tok, m = U.build()
        sd = torch.load(ck, map_location="cpu",
                        weights_only=True)["sd"]
        m.load_state_dict(sd)
        m = m.to(dev).eval()
        s, v = G.gate_eval(m, tok, dev)
        print(f"[lam-merge g{lam}] UNMERGED {s}/120 valid {v}",
              flush=True)
        with torch.no_grad():
            for blk in m.blocks:
                for k in ("g", "u", "d"):
                    mw = torch.stack([e[k].weight
                                      for e in blk.moe.exp]).mean(0)
                    for e in blk.moe.exp:
                        e[k].weight.copy_(mw)
        s, v = G.gate_eval(m, tok, dev)
        print(f"[lam-merge g{lam}] MERGED {s}/120 valid {v}",
              flush=True)


if __name__ == "__main__":
    main()

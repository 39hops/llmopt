"""CAL-DK-1 (pre-reg 2026-07-30): does the crystal know when it
doesn't know? Teacher-forced token-level reliability + per-level
(3..7) confidence-v-accuracy on the d64h8 EMA crystal, Mac.
Usage: python scratch/cal_dk_probe.py
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

import os
CKPT = os.environ.get("CKPT", "checkpoints/sym_birth_dense_mps_h8_ema.pt")
D, LAYERS, HEADS, FFN = 64, 8, 8, 256
N_PER_LEVEL = 300


def main():
    dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd.get("sd", sd))
    model.load_state_dict(sd)
    model = model.to(dev).eval()

    rows = load_rows(gen4=True)
    rows = [r for r in rows if r.get("level") in (3, 4, 5, 6, 7)
            and r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    by_level = {lv: [] for lv in (3, 4, 5, 6, 7)}
    for r in reversed(rows):                     # tail = held-out-ish
        lv = r["level"]
        if len(by_level[lv]) < N_PER_LEVEL:
            by_level[lv].append(r)

    conf_all, corr_all = [], []
    print(f"[cal-dk] {CKPT} dev {dev}")
    print("level |    n_tok |  conf  |  acc   | conf-acc (DK gap)")
    for lv in (3, 4, 5, 6, 7):
        confs, corrs = [], []
        for r in by_level[lv]:
            try:
                ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                                 f"Step: {r['nxt']}\n") + [tok.eos_id]
            except ValueError:
                continue
            if len(ids) > 256:
                continue
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                lg = model(x)[0, :-1]
            p = F.softmax(lg.float(), -1)
            y = x[0, 1:]
            # score only the Step: span (the answer region)
            txt = tok.decode(ids)
            step_at = txt.index("Step:")
            start = len(tok.encode(txt[:step_at], strict=False))
            c = p.max(-1).values[start:]
            a = (p.argmax(-1) == y)[start:]
            confs.append(c.cpu())
            corrs.append(a.cpu())
        conf = torch.cat(confs)
        corr = torch.cat(corrs).float()
        conf_all.append(conf)
        corr_all.append(corr)
        print(f"  L{lv}  | {len(conf):8d} | {float(conf.mean()):.4f} "
              f"| {float(corr.mean()):.4f} | "
              f"{float(conf.mean() - corr.mean()):+.4f}")

    conf = torch.cat(conf_all)
    corr = torch.cat(corr_all)
    # reliability curve + ECE (10 equal-width bins on conf)
    print("\nreliability (bin -> conf, acc, n):")
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf >= lo) & (conf < hi if b < 9 else conf <= hi)
        if m.sum() < 20:
            continue
        c, a, n = float(conf[m].mean()), float(corr[m].mean()), int(m.sum())
        ece += n / len(conf) * abs(c - a)
        print(f"  [{lo:.1f},{hi:.1f}) conf {c:.3f} acc {a:.3f} n {n}")
    # AUROC of conf as an error detector
    s = conf.numpy()
    e = corr.numpy()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(s))
    pos = e == 1
    auroc = ((ranks[pos].mean() - (pos.sum() - 1) / 2)
             / max((~pos).sum(), 1))
    print(f"\n[cal-dk verdict-inputs] ECE {ece:.4f} | "
          f"AUROC(conf detects correct) {auroc:.3f} | "
          f"overall conf {float(conf.mean()):.4f} "
          f"acc {float(corr.mean()):.4f}")


if __name__ == "__main__":
    main()

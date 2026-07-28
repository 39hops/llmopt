"""Calibration probe (spec 2026-07-28 rung 1): flips-per-token under
a Q-lattice snap. Teacher-forced greedy argmax on a fixed 400-row
probe set; count positions where the snapped twin's argmax differs
from the unsnapped model's. Control arm: Q=0 (no snap) must read
exactly 0 flips. Also reports the logit-margin distribution at flip
sites (the snap-anatomy read: flips should sit at tiny margins).
Usage: calib_probe.py <ckpt> <d> <layers> <ffn> <heads> [Q=16]
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

PROBE_SEED = 99_000_001


def rat_snap(sd, Q):
    # EXACTLY rational_snap.py's operator (direct w -> best p/q,
    # q <= Q, NO absmean scale) — the instrument that measured the
    # 49->26 ground-truth crack. rat_deploy's scaled snap is a
    # DIFFERENT (finer) instrument; snap operators are instruments
    # and fences travel with them.
    out = {}
    for k, w in sd.items():
        if w.ndim != 2 or not w.is_floating_point():
            out[k] = w
            continue
        wf = w.float()
        best = torch.round(wf)  # q = 1
        err = (wf - best).abs()
        for q in range(2, Q + 1):
            cand = torch.round(wf * q) / q
            e = (wf - cand).abs()
            m = e < err
            best = torch.where(m, cand, best)
            err = torch.where(m, e, err)
        out[k] = best.to(w.dtype)
    return out


@torch.no_grad()
def flips_per_token(ckpt, d, layers, ffn, heads, Q=16, dev=None):
    dev = dev or ("mps" if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
    tok = MathTokenizer()
    rows = load_rows(gen4=True)
    random.Random(PROBE_SEED).shuffle(rows)
    probe = rows[:400]
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)

    def run(state):
        m = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
        m.load_state_dict(state)
        m.eval()
        args, tops = [], []
        for r in probe:
            t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
            try:
                ids = torch.tensor([tok.encode(t) + [tok.eos_id]],
                                   device=dev)
            except ValueError:
                continue
            lg = m(ids[:, :-1])[0]
            top2 = lg.topk(2, dim=-1)
            args.append(top2.indices[:, 0].cpu())
            tops.append((top2.values[:, 0] - top2.values[:, 1]).cpu())
        del m
        return torch.cat(args), torch.cat(tops)

    a0, m0 = run(sd)
    a1, _ = run(rat_snap(sd, Q) if Q else sd)
    flips = a0 != a1
    n = len(a0)
    return {
        "flips_per_token": flips.sum().item() / n,
        "n_tokens": n,
        "margin_median": m0.median().item(),
        "margin_at_flips": (m0[flips].median().item()
                            if flips.any() else float("nan")),
    }


if __name__ == "__main__":
    ckpt = sys.argv[1]
    d, layers, ffn, heads = map(int, sys.argv[2:6])
    Q = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    r = flips_per_token(ckpt, d, layers, ffn, heads, Q)
    print(f"{ckpt} Q={Q}: flips_per_token={r['flips_per_token']:.5f} "
          f"n_tokens={r['n_tokens']} "
          f"margin_median={r['margin_median']:.3f} "
          f"margin_at_flips={r['margin_at_flips']:.2e}", flush=True)

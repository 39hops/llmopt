"""FOURIER-1: does the crystal implement the roots-of-unity filter?
Per-neuron Fourier v indicator regression of answer-position
activations over n mod k. CPU. Usage: python scratch/fourier_probe.py
"""
import sys

import numpy as np  # noqa: E402
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = "checkpoints/sym_birth_dense_mps_h8_ema.pt"
D, LAYERS, HEADS, FFN = 64, 8, 8, 256
N = 400


def main():
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("model", sd.get("sd", sd)))
    model = model.eval()
    rng = np.random.default_rng(11)
    for k in (10,):     # digits ARE residues: units-digit structure
        acts = []   # [N, LAYERS, D] residual after each block
        ns = list(range(10, 10 + N))
        for n in ns:
            txt = f"Current: {n}+7\nHints: none\nStep: "
            ids = torch.tensor([tok.encode(txt, strict=False)])
            hs = []
            hooks = [model.blocks[j].register_forward_hook(
                lambda m, i, o, s=hs: s.append(
                    o[0][0, -1].detach().numpy()))
                for j in range(LAYERS)]
            with torch.no_grad():
                model(ids)
            for h in hooks:
                h.remove()
            acts.append(np.stack(hs))
        A = np.stack(acts)              # [N, LAYERS, D]
        res = np.array(ns) % k
        # bases
        four = []
        for m in range(1, (k - 1) // 2 + 1):
            four.append(np.cos(2 * np.pi * m * np.array(ns) / k))
            four.append(np.sin(2 * np.pi * m * np.array(ns) / k))
        four = np.stack(four, 1)        # [N, k-1]
        ind = np.eye(k)[res]            # [N, k] indicator
        ind = ind - ind.mean(0)

        def r2(X, y):
            X = np.column_stack([np.ones(len(X)), X])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ b
            ssr = ((y - pred) ** 2).sum()
            sst = ((y - y.mean()) ** 2).sum() + 1e-12
            return 1 - ssr / sst

        for name in ("true", "shuf"):
            n_per, conc = 0, []
            for li in range(LAYERS):
                for d in range(D):
                    y = A[:, li, d]
                    y = (y - y.mean()) / (y.std() + 1e-9)
                    if name == "true":
                        rf = r2(four, y)
                        ri = r2(ind, y)
                    else:
                        sh = rng.permutation(len(y))
                        rf = r2(four, y[sh])
                        ri = r2(ind, y[sh])
                    if max(rf, ri) > 0.5:
                        n_per += 1
                        # concentration: best single-freq share
                        best = 0
                        for m in range(0, four.shape[1], 2):
                            r1f = r2(four[:, m:m + 2], y if name == "true"
                                     else y[sh])
                            best = max(best, r1f)
                        conc.append(best / max(rf, 1e-9))
            print(f"[fourier k={k} {name}] periodic neurons "
                  f"{n_per}/{LAYERS * D} | median top-freq share "
                  f"{np.median(conc) if conc else float('nan'):.3f}")


if __name__ == "__main__":
    main()

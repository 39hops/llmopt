"""FOURIER-2: birth a Mod-diet crystal (nt pilot 500, callspan
plain-arm recipe) and run the roots-of-unity probe properly.
Usage: python scratch/fourier2_modbirth.py (pilot at data/)
"""
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

ATOMS = ["gcd", "Mod", "**", "call:", "->",  # sidecar order (fence)
         "Hints: ", ";"]
D, LAYERS, FFN, HEADS, BS, EPOCHS, LR = 64, 8, 256, 4, 8, 20, 1.5e-3
PILOT = "data/nt_callspan_pilot500.jsonl"
OUT = "checkpoints/fourier2_modbirth.pt"


def main():
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    rows = [json.loads(ln) for ln in open(PILOT)]
    random.Random(7).shuffle(rows)
    torch.manual_seed(1)
    tok = MathTokenizer(extra=ATOMS)
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    enc = []
    for r in rows:
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 256:
            enc.append(ids)
    enc.sort(key=len)
    print(f"[f2] {len(enc)} rows dev {dev} vocab {len(tok.vocab)}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(order)
        for off in order:
            b = enc[off:off + BS]
            L = max(len(q) for q in b)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in b], device=dev)
            lg = model(x)[:, :-1]
            loss = F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                ignore_index=tok.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save({"sd": model.state_dict()}, OUT)
    print(f"[f2] birth done, final loss {float(loss):.3f}", flush=True)

    # roots-of-unity probe, proper substrate
    model.eval()
    rng = np.random.default_rng(11)
    N = 400
    for k in (5, 7):
        acts, ns = [], list(range(10, 10 + N))
        for n in ns:
            txt = f"Current: Mod({n}, {k})\nHints: none\nStep: "
            ids = torch.tensor([tok.encode(txt)]).to(dev)
            hs = []
            hooks = [model.blocks[j].register_forward_hook(
                lambda m, i, o, s=hs: s.append(
                    o[0][0, -1].detach().float().cpu().numpy()))
                for j in range(LAYERS)]
            with torch.no_grad():
                model(ids)
            for h in hooks:
                h.remove()
            acts.append(np.stack(hs))
        A = np.stack(acts)
        four = []
        for m in range(1, (k - 1) // 2 + 1):
            four.append(np.cos(2 * np.pi * m * np.array(ns) / k))
            four.append(np.sin(2 * np.pi * m * np.array(ns) / k))
        four = np.stack(four, 1)

        def r2(X, y):
            X = np.column_stack([np.ones(len(X)), X])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            ssr = ((y - X @ b) ** 2).sum()
            return 1 - ssr / (((y - y.mean()) ** 2).sum() + 1e-12)

        for name in ("true", "shuf"):
            n_per, conc = 0, []
            for li in range(LAYERS):
                for d in range(D):
                    y = A[:, li, d]
                    y = (y - y.mean()) / (y.std() + 1e-9)
                    if name == "shuf":
                        y = y[rng.permutation(len(y))]
                    rf = r2(four, y)
                    if rf > 0.5:
                        n_per += 1
                        best = max(r2(four[:, m:m + 2], y)
                                   for m in range(0, four.shape[1], 2))
                        conc.append(best / max(rf, 1e-9))
            print(f"[fourier2 k={k} {name}] periodic "
                  f"{n_per}/{LAYERS * D} | median top-freq share "
                  f"{np.median(conc) if conc else float('nan'):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()

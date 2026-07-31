"""B6 (revival-sweep Tier B, 2026-07-31): G9 zeta-8 ON THE MOD
DIET — the declared rotation reopening, fired on the one substrate
where the target computation is provably rotational (clock-
placement law). Completes the causal square: diet-forced clocks
exist (FOURIER-2b); does architecture-PROVIDED rotation get
adopted where the diet wants it?

ARM=G9   complex-FFN, phases snapped exactly to Z[zeta_8] (STE)
ARM=none complex-FFN, no snap (separates complex structure from
         the exact-phase alphabet)
Paired against the booked REAL crystal (fourier2b_widemod: acc
0.526; k=4/5/10 solved, k=8 partial 0.63; clocks 276/512 at k=5).
Same diet/eval/seed/epochs/device. Watch k=8 especially — zeta_8's
own modulus. Usage: ALPHA=G9 python scratch/fourier_g9.py
"""
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import complex_model as CM  # noqa: E402
from fourier2b_widemod import gen_rows, fmt, ATOMS  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402
from llmopt.runlog import get_logger, timed  # noqa: E402

ALPHA = os.environ.get("ALPHA", "G9")
D, LAYERS, FFN, HEADS, BS, LR = 64, 8, 256, 4, 8, 1.5e-3
EPOCHS = int(os.environ.get("EPOCHS", "30"))
OUT = f"checkpoints/fourier_g9_{ALPHA}.pt"
log = get_logger("llmopt.fourier_g9")


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    CM.set_alpha(ALPHA)
    tr, ev = gen_rows()                 # SAME split as fourier2b
    torch.manual_seed(1)
    tok = MathTokenizer(extra=ATOMS)
    model = CM.build_complex_model(len(tok.vocab), d=D,
                                   layers=LAYERS, heads=HEADS,
                                   ffn=FFN).to(dev)
    n = sum(p.numel() for p in model.parameters())
    enc = [tok.encode(fmt(nn_, k)) + [tok.eos_id] for nn_, k in tr]
    enc.sort(key=len)
    log.info("ALPHA=%s train %d eval %d dev %s params %.2fM",
             ALPHA, len(enc), len(ev), dev, n / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    with timed("birth", log):
        for ep in range(EPOCHS):
            random.Random(ep).shuffle(order)
            for off in order:
                b = enc[off:off + BS]
                L = max(len(q) for q in b)
                x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                                  for q in b], device=dev)
                lg = model(x)[:, :-1]
                loss = F.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]),
                    x[:, 1:].reshape(-1), ignore_index=tok.pad_id)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if ep % 5 == 0 or ep == EPOCHS - 1:
                log.info("ep%d loss %.4f", ep, float(loss.detach()))
    torch.save({"sd": model.state_dict(), "alpha": ALPHA}, OUT)

    # held-out gate, per k (same protocol as fourier2b)
    model.eval()
    hit, tot = {}, {}
    with timed("held-out gate", log):
        for nn_, k in ev:
            prompt = f"Current: Mod({nn_}, {k})\nHints: none\nStep: "
            ids = tok.encode(prompt)
            x = torch.tensor([ids], device=dev)
            outp = []
            with torch.no_grad():
                for _ in range(8):
                    nx = int(model(x)[0, -1].argmax())
                    if nx == tok.eos_id:
                        break
                    outp.append(nx)
                    x = torch.cat(
                        [x, torch.tensor([[nx]], device=dev)], 1)
            tot[k] = tot.get(k, 0) + 1
            if tok.decode(outp).strip() == str(nn_ % k):
                hit[k] = hit.get(k, 0) + 1
    accs = {k: round(hit.get(k, 0) / tot[k], 2) for k in sorted(tot)}
    overall = sum(hit.values()) / sum(tot.values())
    log.info("[g9 gate ALPHA=%s] per-k %s | overall %.3f "
             "(real ctrl 0.526)", ALPHA, accs, overall)

    # clock probe at k in {5, 8, 7} (8 = zeta_8's own modulus)
    rng = np.random.default_rng(11)
    for k in (5, 8, 7):
        ns = [nn_ for nn_, kk in ev if kk == k]
        acts = []
        for nn_ in ns:
            txt = f"Current: Mod({nn_}, {k})\nHints: none\nStep: "
            ids = torch.tensor([tok.encode(txt)]).to(dev)
            hs = []
            hooks = [model.blocks[j].register_forward_hook(
                lambda m, i, o, s=hs: s.append(
                    (o[0] if isinstance(o, tuple) else o)
                    [0, -1].detach().float().cpu().numpy()))
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
            n_per = 0
            for li in range(LAYERS):
                for dd in range(D):
                    y = A[:, li, dd]
                    y = (y - y.mean()) / (y.std() + 1e-9)
                    if name == "shuf":
                        y = y[rng.permutation(len(y))]
                    if r2(four, y) > 0.5:
                        n_per += 1
            log.info("[g9 clock ALPHA=%s k=%d %s] periodic %d/%d",
                     ALPHA, k, name, n_per, LAYERS * D)


if __name__ == "__main__":
    main()

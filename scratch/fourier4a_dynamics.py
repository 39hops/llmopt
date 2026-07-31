"""FOURIER-4a (pre-reg 2026-07-31): clock-FORMATION dynamics.

Rerun the FOURIER-2b birth (same diet/split/seed/recipe) and probe
every 2 epochs: held-out per-k accuracy (quick greedy subsets) +
periodic-neuron counts at k in {5, 8, 7}. Question: does the clock
form BEFORE, WITH, or AFTER per-modulus competence?
  (a) clock leads acc  -> training-time progress instrument;
  (b) clock lags acc   -> clocks are consolidation, not mechanism;
  (c) co-arrival       -> single transition (grokking-shaped).
k=7 must stay 0 throughout (control). Either branch books.
Mac. Usage: python scratch/fourier4a_dynamics.py
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fourier2b_widemod import gen_rows, fmt, ATOMS  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from llmopt.runlog import get_logger, timed  # noqa: E402

D, LAYERS, FFN, HEADS, BS, LR = 64, 8, 256, 4, 8, 1.5e-3
EPOCHS, PROBE_EVERY = 30, 2
PROBE_KS = (5, 8, 7)
ACC_KS = (4, 5, 8, 7)
N_ACC = 40                     # quick per-k greedy subset
OUT = "checkpoints/fourier4a_dynamics.pt"
log = get_logger("llmopt.fourier4a")


def acc_at(model, tok, dev, ev, k, n=N_ACC):
    hit = tot = 0
    with torch.no_grad():
        for nn_, kk in ev:
            if kk != k or tot >= n:
                continue
            ids = tok.encode(f"Current: Mod({nn_}, {k})\n"
                             f"Hints: none\nStep: ")
            x = torch.tensor([ids], device=dev)
            out = []
            for _ in range(8):
                nx = int(model(x)[0, -1].argmax())
                if nx == tok.eos_id:
                    break
                out.append(nx)
                x = torch.cat(
                    [x, torch.tensor([[nx]], device=dev)], 1)
            tot += 1
            if tok.decode(out).strip() == str(nn_ % k):
                hit += 1
    return hit / max(tot, 1)


def clock_at(model, tok, dev, ev, k, rng):
    ns = [nn_ for nn_, kk in ev if kk == k]
    acts = []
    with torch.no_grad():
        for nn_ in ns:
            ids = torch.tensor([tok.encode(
                f"Current: Mod({nn_}, {k})\nHints: none\nStep: ")]
            ).to(dev)
            hs = []
            hooks = [model.blocks[j].register_forward_hook(
                lambda m, i, o, s=hs: s.append(
                    o[0][0, -1].detach().float().cpu().numpy()))
                for j in range(LAYERS)]
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

    # threshold-free (Artin 2026-07-31: "can't R^2 change
    # dynamically?"): report the count at two thresholds AND the
    # total periodic variance sum(R^2) — if count@0.5 falls while
    # sum(R^2) holds, "pruning" is really BLURRING (mass
    # redistributes below the cutoff); if both fall, real pruning.
    r2s = []
    for li in range(LAYERS):
        for dd in range(D):
            y = A[:, li, dd]
            y = (y - y.mean()) / (y.std() + 1e-9)
            r2s.append(max(r2(four, y), 0.0))
    r2s = np.array(r2s)
    return {"n50": int((r2s > 0.5).sum()),
            "n25": int((r2s > 0.25).sum()),
            "sumR2": round(float(r2s.sum()), 1)}


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr, ev = gen_rows()               # SAME split as 2b
    torch.manual_seed(1)
    tok = MathTokenizer(extra=ATOMS)
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    enc = [tok.encode(fmt(n, k)) + [tok.eos_id] for n, k in tr]
    enc.sort(key=len)
    log.info("train %d eval %d dev %s probe every %d ep",
             len(enc), len(ev), dev, PROBE_EVERY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    rng = np.random.default_rng(11)

    def probe(tag):
        model.eval()
        accs = {k: round(acc_at(model, tok, dev, ev, k), 2)
                for k in ACC_KS}
        clocks = {k: clock_at(model, tok, dev, ev, k, rng)
                  for k in PROBE_KS}
        log.info("[f4a %s] acc %s | clocks %s", tag, accs, clocks)
        model.train()

    probe("ep-1(init)")
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
        if ep % PROBE_EVERY == 1 or ep == EPOCHS - 1:
            with timed(f"probe ep{ep}", log):
                probe(f"ep{ep}")
    torch.save({"sd": model.state_dict()}, OUT)


if __name__ == "__main__":
    main()

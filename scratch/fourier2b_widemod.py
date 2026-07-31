"""FOURIER-2b (pre-reg 2026-07-31): wide-Mod birth + roots-of-unity
probe, with the memorization check FOURIER-2 lacked.

Generator: Mod(n, k) rows, n uniform in [10, 99999], k in 3..11,
string-seeded; eval n-set EXCLUDED from train (exclude= guard, not
seed offsets). Memorization check = greedy exact-match on held-out
n's BEFORE the probe: if the task didn't generalize, the probe is
VOID (that's what killed FOURIER-2's 500-row pilot). Probe: per-
neuron roots-of-unity R^2 at k=5,7 on held-out n's v shuffle.
Mac. Usage: python scratch/fourier2b_widemod.py
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from llmopt.runlog import get_logger, timed  # noqa: E402

ATOMS = ["gcd", "Mod", "**", "call:", "->",  # sidecar order (fence)
         "Hints: ", ";"]
D, LAYERS, FFN, HEADS, BS, LR = 64, 8, 256, 4, 8, 1.5e-3
EPOCHS = int(__import__("os").environ.get("EPOCHS", "6"))
N_TRAIN, N_EVAL = 20000, 500
KS = list(range(3, 12))
OUT = "checkpoints/fourier2b_widemod.pt"
log = get_logger("llmopt.fourier2b")


def gen_rows():
    """Eval n's drawn first, excluded from train (prompt-set guard)."""
    r_ev = random.Random("fourier2b-eval-1")
    ev = []
    seen = set()
    while len(ev) < N_EVAL:
        n, k = r_ev.randint(10, 99999), r_ev.choice(KS)
        if (n, k) not in seen:
            seen.add((n, k))
            ev.append((n, k))
    excl = {n for n, _ in ev}          # exclude by n, the wide axis
    r_tr = random.Random("fourier2b-train-1")
    tr = []
    while len(tr) < N_TRAIN:
        n, k = r_tr.randint(10, 99999), r_tr.choice(KS)
        if n not in excl:
            tr.append((n, k))
    assert not excl & {n for n, _ in tr}, "train/eval n-overlap"
    return tr, ev


def fmt(n, k):
    return f"Current: Mod({n}, {k})\nHints: none\nStep: {n % k}\n"


def main():
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    tr, ev = gen_rows()
    torch.manual_seed(1)
    tok = MathTokenizer(extra=ATOMS)
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    enc = [tok.encode(fmt(n, k)) + [tok.eos_id] for n, k in tr]
    enc.sort(key=len)
    log.info("train %d eval %d dev %s vocab %d",
             len(enc), len(ev), dev, len(tok.vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    with timed("birth", log):
        for ep in range(EPOCHS):
            random.Random(ep).shuffle(order)
            for step, off in enumerate(order):
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
            log.info("ep%d loss %.4f", ep, float(loss.detach()))
    torch.save({"sd": model.state_dict()}, OUT)

    # ---- MEMORIZATION CHECK: greedy exact-match on held-out n ----
    model.eval()
    hit = 0
    with timed("held-out gate", log):
        for n, k in ev:
            prompt = f"Current: Mod({n}, {k})\nHints: none\nStep: "
            ids = tok.encode(prompt)
            x = torch.tensor([ids], device=dev)
            outp = []
            with torch.no_grad():
                for _ in range(8):
                    nxt = int(model(x)[0, -1].argmax())
                    if nxt == tok.eos_id:
                        break
                    outp.append(nxt)
                    x = torch.cat(
                        [x, torch.tensor([[nxt]], device=dev)], 1)
            if tok.decode(outp).strip().rstrip("\n") == str(n % k):
                hit += 1
    acc = hit / len(ev)
    log.info("[f2b gate] held-out Mod acc %.3f (%d/%d)",
             acc, hit, len(ev))
    if acc < 0.5:
        log.info("[f2b] VOID — task did not generalize; "
                 "probe skipped (FOURIER-2 lesson)")
        return

    # ---- roots-of-unity probe on held-out n's ----
    rng = np.random.default_rng(11)
    for k in (5, 7):
        ns = [n for n, kk in ev if kk == k] or \
            [n for n, _ in ev][:150]
        acts = []
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
            log.info("[f2b k=%d %s] periodic %d/%d | median "
                     "top-freq share %.3f", k, name, n_per,
                     LAYERS * D,
                     float(np.median(conc)) if conc else float("nan"))


if __name__ == "__main__":
    main()

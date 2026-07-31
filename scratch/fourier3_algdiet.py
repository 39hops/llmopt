"""FOURIER-3 (pre-reg 2026-07-31): the causal arrow — put the
ALGORITHM in the diet and watch where clocks appear.

Diet = FOURIER-2b wide generator, but for k in {3, 9} the rows
teach digit-sum decomposition: n >= 10 rewrites to
Mod(digitsum(n), k); n < 10 answers n % k. Other moduli stay
single-step (4/5/10 shortcuts; 7/11 as UNTAUGHT hard controls).
Eval: greedy multi-hop rollout (follow decomposition up to 4
steps) on held-out n's, per-k accuracy. Probe: roots-of-unity
per-neuron R^2 at k in {5, 9, 7} at the prompt-end position.

The design separates two futures, both informative:
(a) k=9 competence arrives WITHOUT a k=9 clock — the algorithm
    SUBSTITUTES for the rotational representation (single-pass
    activations never need n mod 9; the chain computes it);
(b) k=9 competence arrives WITH a clock — practice on reduced
    forms builds the representation anyway.
Either way k=7/11 (untaught) must stay dead on both axes.
Mac. Usage: EPOCHS=30 python scratch/fourier3_algdiet.py
"""
import os
import random
import re
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
EPOCHS = int(os.environ.get("EPOCHS", "30"))
N_TRAIN, N_EVAL = 24000, 500
KS = list(range(3, 12))
ALG_KS = {3, 9}                      # taught the digit-sum rewrite
OUT = "checkpoints/fourier3_algdiet.pt"
log = get_logger("llmopt.fourier3")


def dsum(n):
    return sum(int(c) for c in str(n))


def nxt_of(n, k):
    """One teaching step: decompose for ALG_KS, else answer."""
    if k in ALG_KS and n >= 10:
        return f"Mod({dsum(n)}, {k})"
    return str(n % k)


def gen_rows():
    r_ev = random.Random("fourier3-eval-1")
    ev, seen = [], set()
    while len(ev) < N_EVAL:
        n, k = r_ev.randint(10, 99999), r_ev.choice(KS)
        if (n, k) not in seen:
            seen.add((n, k))
            ev.append((n, k))
    excl = {n for n, _ in ev}
    r_tr = random.Random("fourier3-train-2")   # v2: length-uniform
    tr = []
    while len(tr) < N_TRAIN:
        # v2 fix (diet-share lesson): uniform n starves the chain's
        # base case (~90% of uniform [2,99999] is 5-digit; reduced
        # forms n<100 got ~0.1% share and the rollout looped there).
        # Draw digit-LENGTH uniformly so every recursion depth gets
        # equal exposure.
        length = r_tr.randint(1, 5)
        n = r_tr.randint(max(2, 10 ** (length - 1)),
                         10 ** length - 1)
        k = r_tr.choice(KS)
        if n not in excl:
            tr.append((n, k))    # includes n<10 terminal practice
    assert not excl & {n for n, _ in tr}, "train/eval n-overlap"
    return tr, ev


def fmt(n, k):
    return (f"Current: Mod({n}, {k})\nHints: none\n"
            f"Step: {nxt_of(n, k)}\n")


def rollout(model, tok, dev, n, k, hops=4):
    """Greedy; follow Mod(m, k) rewrites until a bare number."""
    for _ in range(hops):
        ids = tok.encode(f"Current: Mod({n}, {k})\nHints: none\n"
                         f"Step: ")
        x = torch.tensor([ids], device=dev)
        out = []
        with torch.no_grad():
            for _ in range(16):
                nx = int(model(x)[0, -1].argmax())
                if nx == tok.eos_id:
                    break
                out.append(nx)
                x = torch.cat(
                    [x, torch.tensor([[nx]], device=dev)], 1)
        s = tok.decode(out).strip()
        m = re.fullmatch(r"Mod\((\d+), (\d+)\)", s)
        if m and int(m.group(2)) == k:
            n = int(m.group(1))       # follow the rewrite
            continue
        return s
    return s


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
    log.info("train %d eval %d dev %s alg_ks %s",
             len(enc), len(ev), dev, sorted(ALG_KS))
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
    torch.save({"sd": model.state_dict()}, OUT)

    # ---- multi-hop held-out gate, per k ----
    model.eval()
    hit, tot = {}, {}
    with timed("held-out rollout gate", log):
        for n, k in ev:
            tot[k] = tot.get(k, 0) + 1
            if rollout(model, tok, dev, n, k) == str(n % k):
                hit[k] = hit.get(k, 0) + 1
    accs = {k: hit.get(k, 0) / tot[k] for k in sorted(tot)}
    log.info("[f3 gate] per-k acc: %s",
             {k: round(a, 2) for k, a in accs.items()})
    overall = sum(hit.values()) / sum(tot.values())
    log.info("[f3 gate] overall %.3f", overall)

    # ---- clock probe at prompt-end, k in {5, 9, 7} ----
    rng = np.random.default_rng(11)
    for k in (5, 9, 7):
        ns = [n for n, kk in ev if kk == k]
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
            n_per = 0
            for li in range(LAYERS):
                for d in range(D):
                    y = A[:, li, d]
                    y = (y - y.mean()) / (y.std() + 1e-9)
                    if name == "shuf":
                        y = y[rng.permutation(len(y))]
                    if r2(four, y) > 0.5:
                        n_per += 1
            log.info("[f3 clock k=%d %s] periodic %d/%d",
                     k, name, n_per, LAYERS * D)


if __name__ == "__main__":
    main()

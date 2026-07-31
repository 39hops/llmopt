"""UMOE-1 (pre-reg 2026-07-30): micro-MoE conservation 3-arm.
First house MoE births. d64 h8 L8, FFN -> 4 experts (SwiGLU
ffn_e=128) + top-1 switch router per block; gen-4 diet, 3 epochs,
seed 1, all arms one device (3080).

ARM=lb    switch load-balance aux 0.01 (standard)
ARM=free  aux 0 (correlation permitted)
ARM=tied  expert_i = base + 0.1-init delta_i, aux 0.01
ARM=dense plain d64h8 control (gate reference, same seed/device)
ARM=treegrav rung-2 combo: tree parameterization + Hebbian
          relaxation restricted to tree edges (siblings only)
ARM=chantree rung-2 combo: per-sibling-pair low-rank channels
          (experts talk only through their pair's channel)

Measures after training: gate (G.gate_eval), mean pairwise
expert-weight corr per block (N3), adjacent-layer co-routing MI v
token-shuffle (B4, 4x4 joint), meter M per expert group.
Usage: ARM=lb python scratch/umoe_conserve.py
"""
import math
import os
import random
import sys
import types

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from llmopt.quantize.meter import meter_group  # noqa: E402

ARM = os.environ["ARM"]
D, LAYERS, HEADS, FFN = 64, 8, 8, 256
NE, FFN_E = 4, int(os.environ.get("FFN_E", "128"))
BS, EPOCHS, LR = 8, 3, 1.5e-3
AUX = {"lb": 0.01, "free": 0.0, "tied": 0.01, "soft": 0.0,
       "channel": 0.01, "gravmoe": 0.01, "tree": 0.01,
       "treegrav": 0.01, "chantree": 0.01}.get(ARM, 0.0)
CH_R = 16          # channel arm: shared base rank
GRAV_LAM = float(os.environ.get("GRAV_LAM", "0.5"))  # relaxation
GRAV_EVERY = 100   # gravmoe: apply every N steps
SEED = int(os.environ.get("SEED", "1"))
CONTRACT = float(os.environ.get("CONTRACT", "0"))
TAG = "_ct" if CONTRACT > 0 else ""
FTAG = f"_f{FFN_E}" if FFN_E != 128 else ""
GTAG = f"_g{GRAV_LAM}" if GRAV_LAM != 0.5 else ""
OTAG = os.environ.get("OTAG", "")  # device/run namespace
OUT = f"checkpoints/umoe_{ARM}{TAG}{FTAG}{GTAG}{OTAG}_s{SEED}.pt"


class MoEFFN(nn.Module):
    """4-expert SwiGLU with top-1 switch routing. Stores the last
    batch's aux loss + per-token expert choice for the probes."""

    def __init__(self, tied: bool):
        super().__init__()
        self.tied = tied
        self.channel = ARM == "channel"
        self.chantree = ARM == "chantree"
        self.tree = ARM in ("tree", "treegrav")
        self.router = nn.Linear(D, NE, bias=False)
        mk = (lambda i, o: nn.Linear(i, o, bias=False))
        if self.channel:  # thin shared base: low-rank triple + a_i
            self.sg = nn.Parameter(torch.randn(FFN_E, CH_R) * 0.05)
            self.sg2 = nn.Parameter(torch.randn(CH_R, D) * 0.05)
            self.su = nn.Parameter(torch.randn(FFN_E, CH_R) * 0.05)
            self.su2 = nn.Parameter(torch.randn(CH_R, D) * 0.05)
            self.sd = nn.Parameter(torch.randn(D, CH_R) * 0.05)
            self.sd2 = nn.Parameter(torch.randn(CH_R, FFN_E) * 0.05)
            self.a = nn.Parameter(torch.zeros(NE))  # talk iff needed
        if self.chantree:  # per-sibling-pair channels (0,1 | 2,3)
            self.psg = nn.Parameter(torch.randn(2, FFN_E, CH_R) * 0.05)
            self.psg2 = nn.Parameter(torch.randn(2, CH_R, D) * 0.05)
            self.psu = nn.Parameter(torch.randn(2, FFN_E, CH_R) * 0.05)
            self.psu2 = nn.Parameter(torch.randn(2, CH_R, D) * 0.05)
            self.psd = nn.Parameter(torch.randn(2, D, CH_R) * 0.05)
            self.psd2 = nn.Parameter(torch.randn(2, CH_R, FFN_E) * 0.05)
            self.a = nn.Parameter(torch.zeros(NE))
        self.ema = None  # gravmoe router-overlap EMA [NE, NE]
        if self.tree:  # root + sibling-pair mids (0,1 | 2,3)
            mk3 = lambda: nn.ModuleDict(
                {"g": mk(D, FFN_E), "u": mk(D, FFN_E),
                 "d": mk(FFN_E, D)})
            self.base = mk3()                 # root, full init
            self.mid = nn.ModuleList([mk3(), mk3()])
            with torch.no_grad():
                for md in self.mid:
                    for m in md.values():
                        m.weight.mul_(0.1)
        if tied:
            self.base = nn.ModuleDict(
                {"g": mk(D, FFN_E), "u": mk(D, FFN_E),
                 "d": mk(FFN_E, D)})
        self.exp = nn.ModuleList()
        for _ in range(NE):
            e = nn.ModuleDict({"g": mk(D, FFN_E), "u": mk(D, FFN_E),
                               "d": mk(FFN_E, D)})
            if tied:  # deltas: 0.1-scale init
                with torch.no_grad():
                    for m in e.values():
                        m.weight.mul_(0.1)
            self.exp.append(e)
        self.aux = torch.tensor(0.0)
        self.last_idx = None

    def _one(self, e, h):
        if self.tree:
            i = list(self.exp).index(e)
            md = self.mid[i // 2]
            b = self.base
            g = F.linear(h, b["g"].weight + md["g"].weight
                         + e["g"].weight)
            u = F.linear(h, b["u"].weight + md["u"].weight
                         + e["u"].weight)
            return F.linear(F.silu(g) * u,
                            b["d"].weight + md["d"].weight
                            + e["d"].weight)
        if self.chantree:
            i = list(self.exp).index(e)
            pi = i // 2                      # sibling-pair channel only
            a = self.a[i]
            g = F.linear(h, e["g"].weight
                         + a * (self.psg[pi] @ self.psg2[pi]))
            u = F.linear(h, e["u"].weight
                         + a * (self.psu[pi] @ self.psu2[pi]))
            return F.linear(F.silu(g) * u,
                            e["d"].weight
                            + a * (self.psd[pi] @ self.psd2[pi]))
        if self.channel:
            i = list(self.exp).index(e)
            a = self.a[i]
            g = F.linear(h, e["g"].weight + a * (self.sg @ self.sg2))
            u = F.linear(h, e["u"].weight + a * (self.su @ self.su2))
            return F.linear(F.silu(g) * u,
                            e["d"].weight + a * (self.sd @ self.sd2))
        if self.tied:  # expert_i = base + delta_i, weight-level
            b = self.base
            g = F.linear(h, b["g"].weight + e["g"].weight)
            u = F.linear(h, b["u"].weight + e["u"].weight)
            return F.linear(F.silu(g) * u,
                            b["d"].weight + e["d"].weight)
        return e["d"](F.silu(e["g"](h)) * e["u"](h))

    def forward(self, h):
        p = F.softmax(self.router(h), -1)          # [B,T,NE]
        top_p, top_i = p.max(-1)                   # [B,T]
        if os.environ.get("ARM") == "soft":        # full mixture
            y = sum(p[..., i:i + 1] * self._one(self.exp[i], h)
                    for i in range(NE))
            self.aux = torch.tensor(0.0)
            self.last_idx = top_i.detach()
            return y
        y = torch.zeros_like(h)
        for i in range(NE):
            m = top_i == i
            if m.any():
                y[m] = self._one(self.exp[i], h[m])
        y = y * top_p.unsqueeze(-1)
        if ARM in ("gravmoe", "treegrav"):  # router-overlap EMA
            ov = torch.einsum("bti,btj->ij", p, p) / p.shape[0] \
                / p.shape[1]
            ov = ov.detach().cpu()
            self.ema = ov if self.ema is None else \
                0.99 * self.ema + 0.01 * ov
        # switch aux: NE * sum_i f_i * mean-prob_i
        f = F.one_hot(top_i, NE).float().mean((0, 1))
        self.aux = NE * (f * p.mean((0, 1))).sum()
        self.last_idx = top_i.detach()
        return y


def rope(q, k, pos0=0):
    B, H, T, Dh = q.shape
    half = Dh // 2
    freq = torch.exp(-math.log(10000.0)
                     * torch.arange(half, device=q.device) / half)
    t = torch.arange(pos0, pos0 + T, device=q.device)
    ang = t[:, None] * freq[None, :]
    cos, sin = ang.cos(), ang.sin()

    def rot(v):
        v1, v2 = v[..., :half], v[..., half:]
        return torch.cat([v1 * cos - v2 * sin,
                          v1 * sin + v2 * cos], -1)
    return rot(q), rot(k)


def moe_forward(self, x, mask, past=None):
    """Block.forward twin with the FFN swapped for self.moe."""
    B, T, _ = x.shape
    h = self.n1(x)
    q, k, v = self.qkv(h).chunk(3, -1)
    q = q.view(B, T, HEADS, -1).transpose(1, 2)
    k = k.view(B, T, HEADS, -1).transpose(1, 2)
    v = v.view(B, T, HEADS, -1).transpose(1, 2)
    pos0 = past[0].shape[2] if past is not None else 0
    q, k = rope(q, k, pos0)
    if past is not None:
        k = torch.cat([past[0], k], 2)
        v = torch.cat([past[1], v], 2)
    new_past = (k, v)
    a = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask,
        is_causal=(mask is None and past is None))
    a = a.transpose(1, 2).reshape(B, T, D)
    x = x + self.o(a)
    x = x + self.moe(self.n2(x))
    return x, new_past


def build():
    torch.manual_seed(SEED)
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN)
    if ARM != "dense":
        for blk in model.blocks:
            del blk.gate, blk.up, blk.down
            blk.moe = MoEFFN(tied=(ARM == "tied"))
            blk.forward = types.MethodType(moe_forward, blk)
    return tok, model


def probes(model, enc, dev):
    """corr / MI / meter on the trained model."""
    if ARM == "dense":
        return
    # N3: mean pairwise expert corr per block (flattened g|u|d)
    corrs = []
    for blk in model.blocks:
        vs = []
        for e in blk.moe.exp:
            vs.append(torch.cat([e[k].weight.flatten()
                                 for k in ("g", "u", "d")]).cpu())
        c = []
        for i in range(NE):
            for j in range(i + 1, NE):
                a, b = vs[i] - vs[i].mean(), vs[j] - vs[j].mean()
                c.append(float((a @ b) / (a.norm() * b.norm())))
        corrs.append(sum(c) / len(c))
    print(f"[probe corr] per-block mean pairwise expert corr: "
          f"{[round(c, 3) for c in corrs]} | "
          f"mean {sum(corrs) / len(corrs):.4f}")
    if ARM in ("tree", "treegrav"):
        m0 = model.blocks[0].moe
        nrm = lambda md: float(torch.cat(
            [md[k].weight.flatten() for k in ("g", "u", "d")]).norm())
        print(f"[probe tree] block0 norms root {nrm(m0.base):.2f} "
              f"mids {nrm(m0.mid[0]):.2f}/{nrm(m0.mid[1]):.2f} "
              f"leaves {[round(nrm(e), 2) for e in m0.exp]}")
        wi, ac = [], []
        for blk in model.blocks:
            vs = [torch.cat([e[k].weight.flatten()
                             for k in ("g", "u", "d")]).cpu()
                  for e in blk.moe.exp]
            def cc(a, b):
                a, b = a - a.mean(), b - b.mean()
                return float((a @ b) / (a.norm() * b.norm()))
            wi += [cc(vs[0], vs[1]), cc(vs[2], vs[3])]
            ac += [cc(vs[0], vs[2]), cc(vs[0], vs[3]),
                   cc(vs[1], vs[2]), cc(vs[1], vs[3])]
        print(f"[probe tree] leaf-delta corr within-pair "
              f"{sum(wi) / len(wi):.4f} v across "
              f"{sum(ac) / len(ac):.4f}")
    if ARM in ("channel", "chantree"):
        avals = [[round(float(a), 3) for a in blk.moe.a]
                 for blk in model.blocks]
        print(f"[probe channel] a_i per block: {avals}")
    if ARM in ("gravmoe", "treegrav"):
        E = model.blocks[0].moe.ema
        print(f"[probe grav] block0 overlap EMA:\n{E}")
    if ARM == "tied":
        bn = torch.cat([model.blocks[0].moe.base[k].weight.flatten()
                        for k in ("g", "u", "d")]).norm()
        dn = torch.cat([model.blocks[0].moe.exp[0][k].weight.flatten()
                        for k in ("g", "u", "d")]).norm()
        print(f"[probe tied] block0 base norm {float(bn):.2f} "
              f"v delta norm {float(dn):.2f}")
    # B4: adjacent-layer co-routing MI v token shuffle
    model.eval()
    idxs = [[] for _ in range(LAYERS)]
    with torch.no_grad():
        for off in range(0, min(len(enc), 512), BS):
            batch = enc[off:off + BS]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [0] * (L - len(q)) for q in batch],
                             device=dev)
            model(x)
            for li, blk in enumerate(model.blocks):
                idxs[li].append(blk.moe.last_idx.flatten().cpu())
    idxs = [torch.cat(v) for v in idxs]

    def mi(a, b):
        j = torch.zeros(NE, NE)
        for i in range(NE):
            for k in range(NE):
                j[i, k] = ((a == i) & (b == k)).float().sum()
        j /= j.sum()
        pa, pb = j.sum(1, keepdim=True), j.sum(0, keepdim=True)
        nz = j > 0
        return float((j[nz] * (j[nz] / (pa @ pb)[nz]).log()).sum())

    g = torch.Generator().manual_seed(7)
    mis, mis_sh = [], []
    for li in range(LAYERS - 1):
        a, b = idxs[li], idxs[li + 1]
        mis.append(mi(a, b))
        mis_sh.append(mi(a, b[torch.randperm(len(b), generator=g)]))
    usage = [float((idxs[0] == i).float().mean()) for i in range(NE)]
    print(f"[probe usage] L0 expert shares "
          f"{[round(u, 3) for u in usage]}")
    print(f"[probe MI] adjacent-layer MI "
          f"{[round(m, 4) for m in mis]} | shuffle "
          f"{[round(m, 4) for m in mis_sh]} | ratio "
          f"{sum(mis) / max(sum(mis_sh), 1e-9):.1f}x")
    # meter (exploratory): all expert tensors, param-weighted
    ts = []
    for blk in model.blocks:
        for e in blk.moe.exp:
            for k in ("g", "u", "d"):
                w = e[k].weight.detach()
                if ARM == "tied":  # meter the effective expert
                    w = w + blk.moe.base[k].weight.detach()
                ts.append(w.cpu())
    m, kurt, n = meter_group(ts)
    print(f"[probe meter] experts M={m:.2f} kurt={kurt:.2f} "
          f"({n / 1e6:.2f}M params)")


def main():
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    tok, model = build()
    model = model.to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[umoe] ARM={ARM} seed={SEED} dev={dev} "
          f"params {nparam / 1e6:.2f}M aux={AUX}", flush=True)
    rows = load_rows(gen4=True)
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    enc = []
    for r in rows:
        try:
            ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: {r['nxt']}\n") + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    order = list(range(0, len(enc) - BS + 1, BS))
    step = 0
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(order)
        for off in order:
            batch = enc[off:off + BS]
            L = max(len(q) for q in batch)
            x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                              for q in batch], device=dev)
            logits = model(x)[:, :-1]
            y = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                ignore_index=tok.pad_id)
            if ARM != "dense" and AUX > 0:
                loss = loss + AUX * sum(
                    blk.moe.aux for blk in model.blocks) / LAYERS
            if CONTRACT > 0:  # GRAV-2 expansion tax, MoE edition
                li_c = random.randrange(LAYERS)
                blk_c = model.blocks[li_c]
                with torch.no_grad():
                    h_in = model.emb(x)
                    for b_ in model.blocks[:li_c]:
                        h_in, _ = b_(h_in, None)
                delta = torch.randn_like(h_in)
                delta = delta / delta.norm(dim=-1, keepdim=True) \
                    * 0.01 * h_in.norm(dim=-1, keepdim=True)
                o1, _ = blk_c(h_in, None)
                o2, _ = blk_c(h_in + delta, None)
                ratio = ((o2 - o1).norm(dim=-1)
                         / delta.norm(dim=-1).clamp(min=1e-8))
                loss = loss + CONTRACT * F.relu(ratio - 1.0).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if (ARM in ("gravmoe", "treegrav")
                    and step % GRAV_EVERY == 0):
                with torch.no_grad():  # relax toward co-used peers
                    for blk in model.blocks:
                        E = blk.moe.ema
                        if E is None:
                            continue
                        for i in range(NE):
                            for j in range(NE):
                                if i == j:
                                    continue
                                if (ARM == "treegrav"
                                        and j // 2 != i // 2):
                                    continue  # gravity on tree edges
                                c = GRAV_LAM * float(E[i, j])
                                for k in ("g", "u", "d"):
                                    wi = blk.moe.exp[i][k].weight
                                    wj = blk.moe.exp[j][k].weight
                                    wi.add_(c * (wj - wi))
            if step % 500 == 0:
                print(f"  ep{ep} step {step} loss {float(loss):.3f}",
                      flush=True)
    torch.save({"sd": model.state_dict(), "arm": ARM, "seed": SEED},
               OUT)
    solves, valid = G.gate_eval(model, tok, dev)
    print(f"[gate] ARM={ARM} solves {solves}/120 valid {valid}",
          flush=True)
    probes(model, enc, dev)


if __name__ == "__main__":
    main()

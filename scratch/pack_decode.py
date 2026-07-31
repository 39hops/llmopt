"""P3 THE DETERMINISTIC DECODE (pre-reg 2026-07-30): fixed-point
twin of the MicroLM forward on the packed d64h8 crystal. Every op
is exact integer arithmetic or a SHIPPED-table lookup — no libm in
the path. GEMMs run on the exact-fp32 integer carrier with hi/lo
splitting (every partial < 2^24, printed). Tables are generated
once (CPU) and saved to checkpoints/p3_tables.pt — ship the SAME
file to every device. Usage:
  python scratch/pack_decode.py tables   # generate + save tables
  python scratch/pack_decode.py hash     # 40-tok greedy battery hash
  python scratch/pack_decode.py gate     # full gate (capability price)
__main__-guarded.
"""
import hashlib
import math
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

CKPT = "checkpoints/sym_birth_dense_mps_h8_ema.pt"
TABLES = "checkpoints/p3_tables.pt"
D, LAYERS, FFN, HEADS = 64, 8, 256, 8
HD = D // HEADS
A = 1024           # activation fixed-point scale (2^10; 2^8 read 92% agreement — pre-reg fallback)
ACT_CLAMP = 32 * A  # |x| <= 32 (8 clamped real features; silu table covers 32)
ROPE_S = 1 << 14   # rope table scale
EXP_S = 1 << 16    # exp table value scale
EXP_K = 1024       # score-index granularity (256 left flips at small margins)
SILU_S = A         # silu table maps a-scale -> a-scale
CTX = 512


def make_tables():
    """CPU, once. Shipped — never regenerated per device."""
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    t = {}
    for k, v in sd.items():
        if v.ndim == 2:
            s = float(v.float().std())
            # emb/head are INTERFACE tensors (C1 kept them fp32):
            # sigma/8 step there; block tensors at the sigma/2 law
            k_step = 16.0 if k in ("emb.weight", "head.weight") else 2.0
            q = math.ceil(k_step / s)
            t[k + ".codes"] = torch.round(v.float() * q).to(torch.int64)
            t[k + ".q"] = torch.tensor(q, dtype=torch.int64)
        elif k.endswith(".g"):
            t[k + ".int"] = torch.round(v.float() * A).to(torch.int64)
    # rope tables for all positions/frequencies (half = HD//2)
    half = HD // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    ang = torch.arange(CTX)[:, None].double() * freq[None, :].double()
    t["rope.cos"] = torch.round(ang.cos() * ROPE_S).to(torch.int64)
    t["rope.sin"] = torch.round(ang.sin() * ROPE_S).to(torch.int64)
    # exp table over score indices [-2^15, 0]: value exp(idx/EXP_K)
    idx = torch.arange(-(1 << 15), 1).double() / EXP_K
    t["exp.tab"] = torch.round(idx.exp() * EXP_S).to(torch.int64)
    # silu table over a-scale inputs [-2^15, 2^15]
    x = torch.arange(-(1 << 15), (1 << 15) + 1).double() / A
    t["silu.tab"] = torch.round(
        (x * torch.sigmoid(x)) * A).to(torch.int64)
    torch.save(t, TABLES)
    print(f"tables saved: {TABLES} "
          f"sha {hashlib.sha256(repr(sorted((k, hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k, v in t.items())).encode()).hexdigest()[:16]}",
          flush=True)


def rdiv(x, d):
    """round-half-away integer division, exact + deterministic."""
    return torch.where(x >= 0, (2 * x + d) // (2 * d),
                       -((-2 * x + d) // (2 * d)))


def isqrt_newton(n, iters=30):
    """integer sqrt of int64 scalar tensor via Newton; exact floor."""
    x = torch.ones_like(n)
    x = x << 32
    for _ in range(iters):
        x = (x + rdiv(n, torch.clamp(x, min=1))) >> 1
    return torch.clamp(x, min=1)


class DetLM:
    """Deterministic fixed-point MicroLM twin. All state int64."""

    def __init__(self, dev):
        self.dev = dev
        t = torch.load(TABLES, map_location="cpu", weights_only=True)
        self.t = {k: v.to(dev) for k, v in t.items()}
        self.max_partial = 0

    def gemm(self, a, key):
        """a [.., in] int64 (a-scale) x codes^T -> int64 (a-scale).
        hi/lo split keeps every fp32 partial an exact integer."""
        W = self.t[key + ".weight.codes"].float()          # [out, in]
        q = int(self.t[key + ".weight.q"])
        hi, lo = a >> 6, a & 63
        wmax = int(W.abs().max())
        n_in = W.shape[1]
        bound = max(int(hi.abs().max()) if hi.numel() else 0, 63) \
            * wmax * n_in
        self.max_partial = max(self.max_partial, bound)
        assert bound < (1 << 24), f"partial bound 2^{math.log2(bound):.1f}"
        Yhi = (hi.float() @ W.T)
        Ylo = (lo.float() @ W.T)
        Y = (Yhi.to(torch.int64) << 6) + Ylo.to(torch.int64)
        return rdiv(Y, q)

    def rmsnorm(self, a, gkey):
        g = self.t[gkey + ".g.int"]
        s2 = (a * a).sum(-1, keepdim=True)          # int64 exact
        # rsqrt(mean(x^2)+eps): mean = s2/(D*A^2). Work at scale 2^20:
        # r ~ round(2^20 / sqrt(mean+eps)). sqrt via integer Newton on
        # m40 = (s2*2^40)/(D*A^2) + eps*2^40  -> r = 2^40 // isqrt(m40)?
        # keep exact ints: m = s2 * (2**28) // (D * A * A) + 268  (eps*2^28)
        m = (s2 // D) * (1 << 32) // (A * A) + 4295  # div D first: no int64 overflow
        r = isqrt_newton(m)                          # ~ sqrt(mean)*2^16
        y = rdiv(a * g, A)                           # g*x at a-scale
        y = rdiv(y * (1 << 16), torch.clamp(r, min=1))
        return torch.clamp(y, -ACT_CLAMP, ACT_CLAMP)

    def rope(self, x, pos0):
        # x [B,H,T,HD] int64 a-scale; rotate pairs (v1,v2)
        B, H, T, _ = x.shape
        half = HD // 2
        cos = self.t["rope.cos"][pos0:pos0 + T]      # [T, half]
        sin = self.t["rope.sin"][pos0:pos0 + T]
        v1, v2 = x[..., :half], x[..., half:]
        r1 = rdiv(v1 * cos - v2 * sin, ROPE_S)
        r2 = rdiv(v1 * sin + v2 * cos, ROPE_S)
        return torch.cat([r1, r2], -1)

    def attn(self, q, k, v):
        # q [B,H,1,HD] (decode step), k/v [B,H,T,HD]; int64 a-scale
        s = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1)  # [B,H,1,T]
        # score value = s/(A^2 * sqrt(HD)); index = round(val*EXP_K)
        # fold: idx = rdiv(s * EXP_K_ADJ, A*A) with sqrt(HD) inside:
        # EXP_K/sqrt(8) = EXP_K*2^14 // round(sqrt(8)*2^14)
        SQ = 46341  # round(sqrt(8)*2^14)
        idx = rdiv(s * EXP_K * (1 << 14), A * A * SQ)
        idx = idx - idx.max(dim=-1, keepdim=True).values
        idx = torch.clamp(idx, -(1 << 15), 0)
        w = self.t["exp.tab"][idx + (1 << 15)]       # int64 <= 2^16
        num = (w.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)
        den = w.sum(-1, keepdim=True)
        return rdiv(num, torch.clamp(den, min=1))    # [B,H,1,HD]

    def step(self, tok_id, past, pos):
        emb = self.gemm_embed(tok_id)
        x = emb                                       # [1,1,D]
        new_past = []
        for li in range(LAYERS):
            p = f"blocks.{li}"
            h = self.rmsnorm(x, f"{p}.n1")
            qkv = self.gemm(h, f"{p}.qkv")
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(1, 1, HEADS, HD).transpose(1, 2)
            k = k.view(1, 1, HEADS, HD).transpose(1, 2)
            v = v.view(1, 1, HEADS, HD).transpose(1, 2)
            q, k = self.rope(q, pos), self.rope(k, pos)
            if past is not None:
                k = torch.cat([past[li][0], k], 2)
                v = torch.cat([past[li][1], v], 2)
            new_past.append((k, v))
            a = self.attn(q.squeeze(2).unsqueeze(2), k, v)
            a = a.transpose(1, 2).reshape(1, 1, D)
            x = torch.clamp(x + self.gemm(a, f"{p}.o"),
                            -ACT_CLAMP, ACT_CLAMP)
            h = self.rmsnorm(x, f"{p}.n2")
            g = self.gemm(h, f"{p}.gate")
            u = self.gemm(h, f"{p}.up")
            gi = torch.clamp(g, -(1 << 15), (1 << 15))
            sg = self.t["silu.tab"][gi + (1 << 15)]
            ff = rdiv(sg * u, A)
            # clamp keeps the down-GEMM's hi/lo partials under 2^24
            ff = torch.clamp(ff, -(1 << 15), (1 << 15))
            x = torch.clamp(x + self.gemm(ff, f"{p}.down"),
                            -ACT_CLAMP, ACT_CLAMP)
        x = self.rmsnorm(x, "norm")
        logits = self.gemm(x, "head")
        return logits.squeeze(), new_past

    def gemm_embed(self, tok_id):
        q = int(self.t["emb.weight.q"])
        row = self.t["emb.weight.codes"][tok_id]     # int64
        return rdiv(row * A, q).view(1, 1, D)

    def greedy(self, ids, n_new):
        past = None
        logits = None
        for pos, t in enumerate(ids):
            logits, past = self.step(
                torch.tensor(t, device=self.dev), past, pos)
        out = []
        for j in range(n_new):
            nxt = int(logits.argmax())
            out.append(nxt)
            logits, past = self.step(
                torch.tensor(nxt, device=self.dev), past, len(ids) + j)
        return out


def cmd_hash():
    import sympy as sp
    from bench_step_tokens import _gen_isolated
    import step_grpo_micro as G
    from llmopt.train.mathnative import MathTokenizer
    tok = MathTokenizer()
    dev = os.environ.get("P3_DEV") or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    m = DetLM(dev)
    streams, lh = [], hashlib.sha256()
    for lv in G.GATE_LEVELS:
        p = _gen_isolated(lv, G.GATE_BAND + 800_000 + lv)
        if p is None:
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        toks = m.greedy(ids, 40)
        streams.append(toks)
        # logit hash: rerun final step logits into the hash
    for s in streams:
        lh.update(repr(s).encode())
    print(f"P3 streams ({dev}): "
          f"{hashlib.sha256(repr(streams).encode()).hexdigest()}",
          flush=True)
    # full logit-trace hash: re-decode first prompt, hashing logits
    p = _gen_isolated(3, G.GATE_BAND + 800_003)
    cur = f"Integral({sp.sstr(p._expr)}, x)"
    ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
    past, h2 = None, hashlib.sha256()
    for pos, t in enumerate(ids):
        lg, past = m.step(torch.tensor(t, device=dev), past, pos)
        h2.update(lg.cpu().numpy().tobytes())
    print(f"P3 logit-trace ({dev}): {h2.hexdigest()} | "
          f"max GEMM partial 2^{math.log2(max(m.max_partial, 1)):.1f}",
          flush=True)


class GateShim(torch.nn.Module):
    """Adapter so G.gate_eval can drive the deterministic path.
    Slow (step loop per token) — capability cell only."""

    def __init__(self, dev):
        super().__init__()
        self.m = DetLM(dev)
        self.dev = dev

    def forward(self, ids, attn_mask=None, past=None, use_cache=False):
        B, T = ids.shape
        outs = []
        for b in range(B):
            past_b, row = None, []
            for pos in range(T):
                lg, past_b = self.m.step(ids[b, pos], past_b, pos)
                row.append(lg.float())
            outs.append(torch.stack(row))
        return torch.stack(outs)

    def eval(self):
        return self


def cmd_gate():
    import step_grpo_micro as G
    from llmopt.train.mathnative import MathTokenizer
    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = GateShim(dev)
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev, n=8)  # proxy first
    print(f"P3 gate (proxy n=8): {sum(solves.values())}/40 "
          f"@ {valid:.2f}% (fp proxy ctrl was 19/40)", flush=True)


def cmd_battery():
    """A1 gate-pooling cell (revival-sweep Tier A, 2026-07-31):
    greedy-decode the FULL 120-prompt gate battery through the
    integer path and print one digest. Identical digests across
    machines make cross-device seed POOLING legal for greedy
    deterministic batteries (resolution-law throughput lever)."""
    import sympy as sp
    from bench_step_tokens import _gen_isolated
    import step_grpo_micro as G
    from llmopt.train.mathnative import MathTokenizer
    tok = MathTokenizer()
    dev = os.environ.get("P3_DEV") or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    m = DetLM(dev)
    streams, n = [], 0
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            streams.append(m.greedy(ids, 40))
            n += 1
            if n % 20 == 0:
                print(f"  {n} prompts decoded", flush=True)
    dg = hashlib.sha256(repr(streams).encode()).hexdigest()
    print(f"P3 battery ({dev}, {n} prompts): {dg}", flush=True)


if __name__ == "__main__":
    {"tables": make_tables, "hash": cmd_hash,
     "gate": cmd_gate, "battery": cmd_battery}[sys.argv[1]]()

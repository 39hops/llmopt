"""FX-V3 house reproduction (pre-reg 2026-07-31 night): the house
integer reference for the MERGED crystal — P3 DetLM + axiom's
switch_top1 gate spec (relay 2026-07-31-3), decoding THEIR shipped
tables (never regenerated). PASS = both published stream digests.
Usage: AXIOM=~/code/axiom python scratch/fx3_house.py
"""
import hashlib
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import pack_decode as P  # noqa: E402
from pack_decode import ACT_CLAMP, D, LAYERS, A, rdiv  # noqa: E402

AXIOM = os.path.expanduser(os.environ.get("AXIOM", "~/code/axiom"))
Q = 512
TSE = 8 * Q                      # rexp domain [-TSE, 0]
PINS = {
    "s1": "e377201c79bc2034ad74bc039f5c2bddbd5c3d2c16f2d8aa"
          "0b916a6fea4917f7",
    "s2": "f5013f2b34c00f8a0a47b26630c284f0bf7d5d8ed0bea3df"
          "923ceae212fbd82b",
}


class Fx3LM(P.DetLM):
    """DetLM with the router gate; tables from axiom's shipped file."""

    def __init__(self, dev, tables_path):
        self.dev = dev
        t = torch.load(tables_path, map_location="cpu",
                       weights_only=True)
        self.t = {k: v.to(dev) for k, v in t.items()}
        self.max_partial = 0

    def step(self, tok_id, past, pos):
        emb = self.gemm_embed(tok_id)
        x = emb
        new_past = []
        for li in range(LAYERS):
            p = f"blocks.{li}"
            h = self.rmsnorm(x, f"{p}.n1")
            qkv = self.gemm(h, f"{p}.qkv")
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(1, 1, P.HEADS, P.HD).transpose(1, 2)
            k = k.view(1, 1, P.HEADS, P.HD).transpose(1, 2)
            v = v.view(1, 1, P.HEADS, P.HD).transpose(1, 2)
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
            # switch_top1 gate (axiom spec, relay 2026-07-31-3):
            rl = self.gemm(h, f"{p}.router")           # a-scale [E]
            s = rdiv(rl * Q, A)                        # Q units
            d = s - s.max(-1, keepdim=True).values
            e = torch.where(d < -TSE, torch.zeros_like(d),
                            self.t["rexp.tab"][
                                torch.clamp(d, min=-TSE) + TSE])
            z = e.sum(-1, keepdim=True)                # >= Q always
            top_p = rdiv(self.t["rexp.tab"][TSE] * Q, z)
            g = self.gemm(h, f"{p}.gate")
            u = self.gemm(h, f"{p}.up")
            gi = torch.clamp(g, -(1 << 15), (1 << 15))
            sg = self.t["silu.tab"][gi + (1 << 15)]
            ff = rdiv(sg * u, A)
            ff = torch.clamp(ff, -(1 << 15), (1 << 15))
            down = self.gemm(ff, f"{p}.down")
            x = torch.clamp(x + rdiv(down * top_p, Q),
                            -ACT_CLAMP, ACT_CLAMP)
        x = self.rmsnorm(x, "norm")
        logits = self.gemm(x, "head")
        return logits.squeeze(), new_past


def battery(m, tok):
    import sympy as sp
    from bench_step_tokens import _gen_isolated
    import step_grpo_micro as G
    streams = []
    for lv in G.GATE_LEVELS:
        p = _gen_isolated(lv, G.GATE_BAND + 800_000 + lv)
        if p is None:
            continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        streams.append(m.greedy(ids, 40))
    return hashlib.sha256(repr(streams).encode()).hexdigest()


def main():
    from llmopt.train.mathnative import MathTokenizer
    tok = MathTokenizer()
    dev = "cpu"     # exact integer path; cpu is canonical
    for seed in ("s1", "s2"):
        path = f"{AXIOM}/tools/fx_v3/fx3_tables_{seed}.pt"
        m = Fx3LM(dev, path)
        dg = battery(m, tok)
        ok = "PASS" if dg == PINS[seed] else "MISMATCH"
        print(f"[fx3-house] {seed} digest {dg} -> {ok}", flush=True)


if __name__ == "__main__":
    main()

"""Tier-retry controller (pre-reg 2026-07-29: attention-core Leg
0). d56 matryoshka pair: attempt each gate row on the CHEAP tier
(commutant projection, 1/8 gate params); on failure retry the
same row on the DENSE tier. Oracle-fail = the free difficulty
signal. Reports retry solves, the overlap census, and effective
gate-params per row. Desk only, MPS.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import sympy as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn.utils.parametrize as P  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS, NB = 56, 8, 224, 4, 8
TIER = {"on": False}


def shift_perm(n, sh, dev):
    return torch.tensor([NB * (r // NB) + (r % NB - sh) % NB
                         for r in range(n)], device=dev)


class TierP(torch.nn.Module):
    """Inlined from matryoshka_r1 (module-level script — importing
    it would re-run the training; the rot_commutant scar)."""

    def __init__(self, n_out, n_in, dev):
        super().__init__()
        self.po = [shift_perm(n_out, s, dev) for s in range(NB)]
        self.pi = [shift_perm(n_in, s, dev) for s in range(NB)]

    def project(self, W):
        acc = torch.zeros_like(W)
        for po, pi in zip(self.po, self.pi):
            acc = acc + W[po][:, pi]
        return acc / NB

    def forward(self, W):
        if not TIER["on"]:
            return W
        return self.project(W)  # eval-only: no STE needed

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load("checkpoints/matryoshka_d56.pt",
                                 map_location="cpu", weights_only=True))
for blk in model.blocks:
    P.register_parametrization(blk.gate, "weight", TierP(FFN, D, dev))
model.eval()


def try_row(lv, i, tier_on):
    """One gate row under the given tier; same seeds as gate_eval."""
    TIER["on"] = tier_on
    p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
    if p is None:
        return None
    cur = f"Integral({sp.sstr(p._expr)}, x)"
    visited = {cur.replace(" ", "")}
    for ply in range(12):
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        texts, _, _ = G.sample_wave_lp(
            model, tok, prompt,
            [G.GATE_BAND + i * 31 + ply * 7 + b for b in range(G.B)],
            dev)
        distinct = [t_ for t_ in dict.fromkeys(texts)
                    if t_ and t_.replace(" ", "") not in visited]
        wv = verify_wave(cur, distinct) if distinct else {}
        nxt = None
        for t_ in texts:
            ok, so = wv.get(t_, (False, False))
            if ok and t_.replace(" ", "") not in visited:
                nxt = "SOLVED" if so else t_
                break
        if nxt == "SOLVED":
            return True
        if nxt is None:
            return False
        cur = nxt
        visited.add(cur.replace(" ", ""))
    return False


census = {"both": 0, "cheap_only": 0, "dense_only": 0, "neither": 0}
retry_solves = {lv: 0 for lv in G.GATE_LEVELS}
cheap_fails = rows = 0
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            c = try_row(lv, i, True)
            if c is None:
                continue
            rows += 1
            d = try_row(lv, i, False)  # dense pass for census + retry
            if c:
                retry_solves[lv] += 1
            else:
                cheap_fails += 1
                if d:
                    retry_solves[lv] += 1
            key = ("both" if c and d else "cheap_only" if c
                   else "dense_only" if d else "neither")
            census[key] += 1
        print(f"L{lv}: retry {retry_solves[lv]}", flush=True)

gate_p = sum(m.numel() for blk in model.blocks
             for m in [blk.gate.parametrizations.weight.original])
eff = gate_p / 8 + (cheap_fails / rows) * gate_p
print(f"TIER-RETRY: {retry_solves} = {sum(retry_solves.values())}/120",
      flush=True)
print(f"census {census} | cheap-fail rate {cheap_fails/rows:.3f} | "
      f"effective gate params/row {eff/1e3:.1f}k v dense {gate_p/1e3:.1f}k"
      f" ({eff/gate_p:.2%})", flush=True)

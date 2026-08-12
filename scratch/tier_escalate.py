"""3-rung escalation policy (pre-reg 2026-07-29 night: cell 1).
matryoshka_d56_3tier.pt: per gate row eighth -> half -> dense,
escalate on oracle-fail. TierP inlined (module-level-script
scar). Desk, MPS.
"""
from llmopt.common.device import pick_device
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

D, LAYERS, FFN, HEADS = 56, 8, 224, 4
TIER = {"nb": 0}


def shift_perm(n, nb, sh, dev):
    return torch.tensor([nb * (r // nb) + (r % nb - sh) % nb
                         for r in range(n)], device=dev)


class TierP(torch.nn.Module):
    def __init__(self, n_out, n_in, dev):
        super().__init__()
        self.perms = {}
        for nb in (2, 8):
            self.perms[nb] = (
                [shift_perm(n_out, nb, s, dev) for s in range(nb)],
                [shift_perm(n_in, nb, s, dev) for s in range(nb)])

    def forward(self, W):
        nb = TIER["nb"]
        if not nb:
            return W
        po, pi = self.perms[nb]
        acc = torch.zeros_like(W)
        for a, b in zip(po, pi):
            acc = acc + W[a][:, b]
        return acc / nb


tok = MathTokenizer()
dev = pick_device()
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/matryoshka_d56_3tier.pt", map_location="cpu",
    weights_only=True))
for blk in model.blocks:
    P.register_parametrization(blk.gate, "weight", TierP(FFN, D, dev))
model.eval()


def try_row(lv, i, nb):
    TIER["nb"] = nb
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


census = {}
solves = {lv: 0 for lv in G.GATE_LEVELS}
esc_half = esc_dense = rows = 0
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            e = try_row(lv, i, 8)
            if e is None:
                continue
            rows += 1
            h = d = None
            if e:
                solves[lv] += 1
            else:
                esc_half += 1
                h = try_row(lv, i, 2)
                if h:
                    solves[lv] += 1
                else:
                    esc_dense += 1
                    d = try_row(lv, i, 0)
                    if d:
                        solves[lv] += 1
            key = (("E" if e else "e")
                   + ("-" if h is None else "H" if h else "h")
                   + ("-" if d is None else "D" if d else "d"))
            census[key] = census.get(key, 0) + 1
        print(f"L{lv}: policy {solves[lv]}", flush=True)

gp = sum(blk.gate.parametrizations.weight.original.numel()
         for blk in model.blocks)
eff = (gp / 8 + (esc_half / rows) * gp / 2
       + (esc_dense / rows) * gp)
print(f"ESCALATE: {solves} = {sum(solves.values())}/120", flush=True)
print(f"census {census} | esc half {esc_half}/{rows} dense "
      f"{esc_dense}/{rows} | eff gate params/row {eff/1e3:.1f}k "
      f"v dense {gp/1e3:.1f}k ({eff/gp:.2%})", flush=True)

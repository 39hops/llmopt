"""Margin-vs-hardness probe (PRE-REG DATA-CEIL rung 0, 2026-08-10).

Greedy ply-1 decode on the d256 control crystal over a FRESH
problem band (91M offset — disjoint from GATE_BAND 9.9M and
training exposure), levels 1-7, n=40/level. Records the top1-top2
logit gap at every emitted token. Registered observables: per-level
median of per-problem MIN margin; per-level fraction of tokens
with margin < 0.1 and < 0.02 (the fp16 near-tie zone).
P-MARGIN-HARDNESS: median min-margin decreases with level
(Spearman rho <= -0.8). Rows STREAM to JSONL (killed-worker
doctrine); generation failures book as their own class (rule 4).

CPU ONLY by fence — the Mac GPU belongs to the crown battery.
Usage: .venv/bin/python scratch/margin_by_level.py
"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402

CKPT = "checkpoints/mathnative_wfloor_d256.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BAND = 91_000_000  # fresh: gate band is 9.9M, clade probe 77M, ce study 88M
LEVELS = (1, 2, 3, 4, 5, 6, 7)
N_PER = 40
MAX_NEW = 120
OUT = "logs/data_ceil/margins_d256_L1-7.jsonl"
DEV = "cpu"  # fence: Mac GPU is the crown battery's


@torch.no_grad()
def greedy_margins(model, tok, prompt_ids):
    """-> list of top1-top2 gaps along the greedy trajectory."""
    ids = torch.tensor([prompt_ids], device=DEV)
    margins = []
    for _ in range(MAX_NEW):
        lg = model(ids)[0, -1]
        top2 = lg.topk(2)
        margins.append(float(top2.values[0] - top2.values[1]))
        nxt = int(top2.indices[0])
        if nxt == tok.eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
    return margins


def main():
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(DEV)
    model.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    import os
    os.makedirs("logs/data_ceil", exist_ok=True)
    out = open(OUT, "a")
    n_fail = {lv: 0 for lv in LEVELS}
    for lv in LEVELS:
        for i in range(N_PER):
            p = _gen_isolated(lv, BAND + 1000 * lv + i)
            if p is None:  # generator failure: its own class, counted
                n_fail[lv] += 1
                out.write(json.dumps(
                    {"level": lv, "i": i, "status": "genfail"}) + "\n")
                out.flush()
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            try:
                ids = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            except ValueError:
                n_fail[lv] += 1
                out.write(json.dumps(
                    {"level": lv, "i": i, "status": "tokfail"}) + "\n")
                out.flush()
                continue
            m = greedy_margins(model, tok, ids)
            out.write(json.dumps({
                "level": lv, "i": i, "status": "ok", "n_tok": len(m),
                "min_margin": min(m), "margins": [round(x, 4) for x in m],
            }) + "\n")
            out.flush()
        print(f"[L{lv}] done ({N_PER - n_fail[lv]} ok, "
              f"{n_fail[lv]} failed)", flush=True)

    # summary computed FROM THE ARTIFACT (fit the artifact rule)
    rows = [json.loads(x) for x in open(OUT)]
    ok = [r for r in rows if r.get("status") == "ok"]
    print("level  n   med_min_margin  frac<0.1  frac<0.02")
    for lv in LEVELS:
        rl = [r for r in ok if r["level"] == lv]
        if not rl:
            continue
        mins = sorted(r["min_margin"] for r in rl)
        med = mins[len(mins) // 2]
        toks = [m for r in rl for m in r["margins"]]
        f10 = sum(m < 0.1 for m in toks) / len(toks)
        f02 = sum(m < 0.02 for m in toks) / len(toks)
        print(f"L{lv}     {len(rl):3d}  {med:14.4f}  {f10:8.4f}  "
              f"{f02:9.4f}", flush=True)


if __name__ == "__main__":
    main()

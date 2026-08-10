"""Margin-vs-ply-depth probe (PRE-REG DATA-CEIL-0B, 2026-08-10).

Greedy chains up to 12 plies on the d256 control crystal, each
ply's emission verified (verify_wave); advance on valid, stop on
invalid/solve with the stop class recorded (rule 4). Levels 3-7,
n=30/level, fresh band 92M. Per emitted token: top1-top2 logit
gap. Rows STREAM. P-PLY-COMPRESSION: per-ply median min-margin
falls with ply (Spearman <= -0.8 over plies with n>=20 chains).
Survivorship counts book alongside (registered confound).

CPU only (crown battery owns the Mac GPU).
Usage: .venv/bin/python scratch/margin_by_ply.py
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

CKPT = "checkpoints/mathnative_wfloor_d256.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BAND = 92_000_000  # fresh: 9.9M gate / 77M / 88M / 91M taken
LEVELS = (3, 4, 5, 6, 7)
N_PER = 30
PLIES = 12
MAX_NEW = 120
OUT = "logs/data_ceil/margins_by_ply_d256.jsonl"
DEV = "cpu"


@torch.no_grad()
def greedy_step(model, tok, prompt_ids):
    """-> (text, margins) greedy decode of one step."""
    ids = torch.tensor([prompt_ids], device=DEV)
    margins, toks = [], []
    for _ in range(MAX_NEW):
        lg = model(ids)[0, -1]
        top2 = lg.topk(2)
        margins.append(float(top2.values[0] - top2.values[1]))
        nxt = int(top2.indices[0])
        if nxt == tok.eos_id:
            break
        toks.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
    return tok.decode(toks), margins


def main():
    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(DEV)
    model.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    os.makedirs("logs/data_ceil", exist_ok=True)
    out = open(OUT, "a")
    for lv in LEVELS:
        for i in range(N_PER):
            p = _gen_isolated(lv, BAND + 1000 * lv + i)
            if p is None:
                out.write(json.dumps(
                    {"level": lv, "i": i, "ply": None,
                     "status": "genfail"}) + "\n")
                out.flush()
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            visited = {cur.replace(" ", "")}
            for ply in range(PLIES):
                try:
                    ids = tok.encode(
                        f"Current: {cur}\nHints: none\nStep: ")
                except ValueError:
                    out.write(json.dumps(
                        {"level": lv, "i": i, "ply": ply,
                         "status": "tokfail"}) + "\n")
                    out.flush()
                    break
                text, margins = greedy_step(model, tok, ids)
                row = {"level": lv, "i": i, "ply": ply,
                       "n_tok": len(margins),
                       "min_margin": min(margins),
                       "margins": [round(m, 4) for m in margins]}
                t = text.strip()
                if not t or t.replace(" ", "") in visited:
                    row["status"] = "stall"  # identity/empty emission
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    break
                wv = verify_wave(cur, [t])
                ok_, solved = wv.get(t, (False, False))
                if not ok_:
                    row["status"] = "invalid"
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    break
                row["status"] = "solved" if solved else "valid"
                out.write(json.dumps(row) + "\n")
                out.flush()
                if solved:
                    break
                cur = t
                visited.add(cur.replace(" ", ""))
        print(f"[L{lv}] done", flush=True)

    # summary FROM THE ARTIFACT
    rows = [json.loads(x) for x in open(OUT)]
    ok = [r for r in rows if r.get("ply") is not None
          and "min_margin" in r]
    print("ply  n_chains  med_min_margin  frac<0.1  frac<0.02  stops")
    for ply in range(PLIES):
        rl = [r for r in ok if r["ply"] == ply]
        if not rl:
            continue
        mins = sorted(r["min_margin"] for r in rl)
        med = mins[len(mins) // 2]
        toks = [m for r in rl for m in r["margins"]]
        f10 = sum(m < 0.1 for m in toks) / len(toks)
        f02 = sum(m < 0.02 for m in toks) / len(toks)
        stops = {s: sum(1 for r in rl if r["status"] == s)
                 for s in ("valid", "solved", "invalid", "stall")}
        print(f"{ply:3d}  {len(rl):8d}  {med:14.4f}  {f10:8.4f}  "
              f"{f02:9.4f}  {stops}", flush=True)


if __name__ == "__main__":
    main()

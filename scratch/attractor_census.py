"""Free-running iteration census (PRE-REG ATTRACTOR-0, 2026-08-10).

The raw map: cur -> greedy(nxt), NO verify, NO visited-set, NO
solve-stop. 40 problems/level x L3-L7, fresh band 93M, 50 iters.
Terminal classes: fixed_point / cycle_k / solved_form / wandering
/ malformed. Margin join: min-margin per step, tagged transit v
attractor. Rows STREAM. CPU only.

Usage: .venv/bin/python scratch/attractor_census.py
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

CKPT = "checkpoints/mathnative_wfloor_d256.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BAND = 93_000_000  # fresh: 9.9M/77M/88M/91M/92M taken
LEVELS = (3, 4, 5, 6, 7)
N_PER = 40
ITERS = 50
MAX_NEW = 120
OUT = "logs/data_ceil/attractor_census_d256.jsonl"
DEV = "cpu"


@torch.no_grad()
def greedy_step(model, tok, prompt_ids):
    ids = torch.tensor([prompt_ids], device=DEV)
    margins, toks = [], []
    for _ in range(MAX_NEW):
        lg = model(ids)[0, -1]
        top2 = lg.topk(2)
        margins.append(float(top2.values[0] - top2.values[1]))
        n = int(top2.indices[0])
        if n == tok.eos_id:
            break
        toks.append(n)
        ids = torch.cat([ids, torch.tensor([[n]], device=DEV)], dim=1)
    return tok.decode(toks).strip(), min(margins) if margins else 0.0


def norm(s):
    return s.replace(" ", "")


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
                    {"level": lv, "i": i, "cls": "genfail"}) + "\n")
                out.flush()
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            seen = {norm(cur): 0}
            traj = []  # (min_margin, state_norm)
            cls, k, t_abs = "wandering", None, None
            state = cur
            for t in range(1, ITERS + 1):
                try:
                    ids = tok.encode(
                        f"Current: {state}\nHints: none\nStep: ")
                except ValueError:
                    cls, t_abs = "malformed", t
                    break
                nxt, mm = greedy_step(model, tok, ids)
                traj.append({"t": t, "mm": round(mm, 4)})
                if not nxt:
                    cls, t_abs = "malformed", t
                    break
                nn = norm(nxt)
                if nn == norm(state):
                    cls, k, t_abs = "fixed_point", 0, t
                    break
                if nn in seen:
                    cls, k, t_abs = "cycle", t - seen[nn], t
                    break
                if "Integral(" not in nxt:
                    cls, t_abs = "solved_form", t
                    break
                seen[nn] = t
                state = nxt
            out.write(json.dumps({
                "level": lv, "i": i, "cls": cls, "cycle_k": k,
                "t_absorb": t_abs, "n_steps": len(traj),
                "margins": [x["mm"] for x in traj],
            }) + "\n")
            out.flush()
        print(f"[L{lv}] done", flush=True)

    # summary FROM THE ARTIFACT
    rows = [json.loads(x) for x in open(OUT)]
    ok = [r for r in rows if r.get("cls") != "genfail"]
    from collections import Counter
    print("classes:", dict(Counter(r["cls"] for r in ok)))
    cyc = [r["cycle_k"] for r in ok if r["cls"] == "cycle"]
    print("cycle lengths:", dict(Counter(cyc)))
    absorbed = [r for r in ok if r["cls"] in ("fixed_point", "cycle")]
    if absorbed:
        ts = sorted(r["t_absorb"] for r in absorbed)
        print(f"time-to-absorption median: {ts[len(ts)//2]}")
        # margin join: attractor step (last) v transit (all before)
        att = [r["margins"][-1] for r in absorbed if r["margins"]]
        tra = [m for r in absorbed for m in r["margins"][:-1]]
        att.sort(); tra.sort()
        if att and tra:
            print(f"median margin ON attractor: {att[len(att)//2]:.3f}"
                  f" | transit: {tra[len(tra)//2]:.3f}")


if __name__ == "__main__":
    main()

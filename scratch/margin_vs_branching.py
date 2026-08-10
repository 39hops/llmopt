"""Margin-vs-branching probe (PRE-REG DATA-CEIL-0C rung C1).

Greedy ply-1 margins on the 4,000 censused diet states (n_succ
known from logs/data_ceil/determinability_gen4.jsonl), joined to
n_succ. P-AMBIGUITY: median min-margin decreases across n_succ
buckets (1 / 2 / 3 / 4-5 / 6+; qualify at n >= 100) with Spearman
<= -0.8 on bucket medians. Rows STREAM. CPU only.

Usage: .venv/bin/python scratch/margin_vs_branching.py
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = "checkpoints/mathnative_wfloor_d256.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
CENSUS = "logs/data_ceil/determinability_gen4.jsonl"
OUT = "logs/data_ceil/margin_vs_branching_d256.jsonl"
MAX_NEW = 120
DEV = "cpu"
# sharding: SHARD/NSHARD envs; each worker streams to OUT.shardN,
# reader concatenates. torch threads capped so 8 workers coexist.
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))


@torch.no_grad()
def greedy_min_margin(model, tok, prompt_ids):
    ids = torch.tensor([prompt_ids], device=DEV)
    mn = float("inf")
    for _ in range(MAX_NEW):
        lg = model(ids)[0, -1]
        top2 = lg.topk(2)
        mn = min(mn, float(top2.values[0] - top2.values[1]))
        nxt = int(top2.indices[0])
        if nxt == tok.eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
    return mn


def bucket(n):
    if n <= 3:
        return str(n)
    return "4-5" if n <= 5 else "6+"


def main():
    torch.set_num_threads(max(1, 10 // NSHARD))
    states = [json.loads(x) for x in open(CENSUS)]
    states = [s for s in states if s.get("status") == "ok"]
    states = states[SHARD::NSHARD]
    print(f"shard {SHARD}/{NSHARD}: {len(states)} states", flush=True)

    tok = MathTokenizer()
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(DEV)
    model.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    os.makedirs("logs/data_ceil", exist_ok=True)
    path = OUT if NSHARD == 1 else f"{OUT}.shard{SHARD}"
    out = open(path, "a")
    n_fail = 0
    for k, s in enumerate(states):
        try:
            ids = tok.encode(f"Current: {s['cur']}\nHints: none\nStep: ")
        except ValueError:
            n_fail += 1
            out.write(json.dumps(
                {"cur": s["cur"], "status": "tokfail"}) + "\n")
            out.flush()
            continue
        mn = greedy_min_margin(model, tok, ids)
        out.write(json.dumps(
            {"cur": s["cur"], "n_succ": s["n_succ"],
             "min_margin": round(mn, 4), "status": "ok"}) + "\n")
        out.flush()
        if (k + 1) % 500 == 0:
            print(f"[{k + 1}/{len(states)}] tokfail={n_fail}", flush=True)

    if NSHARD > 1:
        print("shard done; run summary after all shards land",
              flush=True)
        return
    # summary FROM THE ARTIFACT
    rows = [json.loads(x) for x in open(OUT)]
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\nscored {len(ok)}, tokfail {n_fail}")
    print("bucket  n      med_min_margin")
    for b in ("1", "2", "3", "4-5", "6+"):
        rl = sorted(r["min_margin"] for r in ok
                    if bucket(r["n_succ"]) == b)
        if not rl:
            continue
        print(f"{b:6s}  {len(rl):5d}  {rl[len(rl) // 2]:.4f}", flush=True)


if __name__ == "__main__":
    main()

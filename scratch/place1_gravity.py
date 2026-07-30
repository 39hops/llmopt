"""PLACE-1 (pre-reg 2026-07-30): inference-time gravity — co-routing
prefetch v popularity on real OLMoE traces. Mac, after UMOE-3.
Usage: python scratch/place1_gravity.py
"""
import collections
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
NE, TOPK, C = 64, 8, 16


def collect():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pack_c6 import PROMPTS
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(dev).eval()
    routers = {n: m for n, m in model.named_modules()
               if n.endswith("mlp.gate")}
    sel = collections.defaultdict(list)
    hooks = []
    for n, mod in routers.items():
        li = int(n.split("layers.")[1].split(".")[0])

        def hk(m_, i, o, li=li):
            top = o.detach().float().topk(TOPK, dim=-1).indices
            for row in top.reshape(-1, TOPK):
                sel[li].append(sorted(row.tolist()))
        hooks.append(mod.register_forward_hook(hk))
    texts = [open("README.md", encoding="utf-8").read(),
             open("docs/RESULTS.md", encoding="utf-8").read()[-40000:],
             open("docs/THEORY.md", encoding="utf-8").read()]
    with torch.no_grad():
        for p in PROMPTS:
            model(tok(p, return_tensors="pt").input_ids.to(dev))
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids[:, :2048]
            model(ids.to(dev))
    for h in hooks:
        h.remove()
    np.save("logs/place1_traces.npy",
            {li: np.array(v) for li, v in sel.items()},
            allow_pickle=True)
    return {li: np.array(v) for li, v in sel.items()}


def main():
    import os
    if os.path.exists("logs/place1_traces.npy"):
        sel = np.load("logs/place1_traces.npy",
                      allow_pickle=True).item()
    else:
        sel = collect()
    layers = sorted(sel)
    T = min(len(sel[li]) for li in layers)
    print(f"[place1] {len(layers)} layers, {T} tokens")
    half = T // 2

    # (a) next-layer prefetch: conditional v marginal v uniform
    rec = {"cond": [], "marg": [], "unif": []}
    for a, b in zip(layers[:-1], layers[1:]):
        A, B = sel[a][:T], sel[b][:T]
        # fit half: co-occurrence counts expert_at_a -> expert_at_b
        co = np.ones((NE, NE)) * 0.1
        freq = np.ones(NE) * 0.1
        for t in range(half):
            for i in A[t]:
                for j in B[t]:
                    co[i, j] += 1
            for j in B[t]:
                freq[j] += 1
        for k in (8, 16):
            for t in range(half, T):
                score = co[A[t]].sum(0)          # conditional on l-set
                pred_c = set(np.argsort(-score)[:k])
                pred_m = set(np.argsort(-freq)[:k])
                actual = set(B[t])
                if k == 8:
                    rec["cond"].append(len(pred_c & actual) / TOPK)
                    rec["marg"].append(len(pred_m & actual) / TOPK)
                    rec["unif"].append(k / NE)
    print(f"[prefetch recall@8] cond {np.mean(rec['cond']):.4f} "
          f"v marg {np.mean(rec['marg']):.4f} "
          f"v unif {np.mean(rec['unif']):.4f} "
          f"(delta {100 * (np.mean(rec['cond']) - np.mean(rec['marg'])):+.1f} pts)")

    # (b) cache sim per layer: LRU v FREQ-pin v LRU+prefetch
    miss = {"lru": 0, "freq": 0, "lru+pf": 0}
    tot = 0
    for idx, li in enumerate(layers):
        S = sel[li][:T]
        freq = np.zeros(NE)
        for t in range(half):
            freq[S[t]] += 1
        pin = set(np.argsort(-freq)[:C])
        # conditional table from previous layer for prefetch
        if idx > 0:
            P = sel[layers[idx - 1]][:T]
            co = np.ones((NE, NE)) * 0.1
            for t in range(half):
                for i in P[t]:
                    for j in S[t]:
                        co[i, j] += 1
        cache_l, order_l = set(), []
        cache_p, order_p = set(), []
        for t in range(half, T):
            need = set(S[t])
            tot += len(need)
            miss["freq"] += len(need - pin)
            # plain LRU
            for e in S[t]:
                if e not in cache_l:
                    miss["lru"] += 1
                    cache_l.add(e)
                    if len(cache_l) > C:
                        cache_l.discard(order_l.pop(0))
                if e in order_l:
                    order_l.remove(e)
                order_l.append(e)
            # LRU + prefetch from previous layer's actual set
            if idx > 0:
                score = co[P[t]].sum(0)
                for e in np.argsort(-score)[:TOPK]:
                    if e not in cache_p:
                        cache_p.add(int(e))      # prefetched (free slot
                        if len(cache_p) > C:     # accounting: counts
                            cache_p.discard(order_p.pop(0))
                    if int(e) in order_p:
                        order_p.remove(int(e))
                    order_p.append(int(e))
            for e in S[t]:
                if e not in cache_p:
                    miss["lru+pf"] += 1
                    cache_p.add(e)
                    if len(cache_p) > C:
                        cache_p.discard(order_p.pop(0))
                if e in order_p:
                    order_p.remove(e)
                order_p.append(e)
    print(f"[cache C={C}] miss rates: "
          + " ".join(f"{k} {v / tot:.4f}" for k, v in miss.items()))


if __name__ == "__main__":
    main()

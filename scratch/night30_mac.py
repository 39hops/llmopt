"""NIGHT-30b Mac chain (pre-reg 2026-07-30): B3 K2 depth curve ->
B4 entangled-experts MI (OLMoE) -> P6 entropy accounting of the
Qwen3 parts. Streaming discipline; K2 shards deleted after B3.
__main__-guarded.
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
from capacity_meter import meter  # noqa: E402


def b3():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    for sh in (5, 55, 37):
        try:
            p = hf_hub_download("moonshotai/Kimi-K2-Instruct",
                                f"model-{sh}-of-61.safetensors")
        except Exception as e:
            print(f"B3 shard {sh}: download failed {e}", flush=True)
            continue
        lay = collections.defaultdict(lambda: [0.0, 0.0, 0])
        with safe_open(p, framework="pt") as f:
            names = [n for n in f.keys() if ".experts." in n
                     and n.endswith("weight") and "scale_inv" not in n
                     and "shared" not in n]
            for n in names[:64]:
                w = f.get_tensor(n).float()
                try:
                    sc = f.get_tensor(n + "_scale_inv").float()
                    bo = (w.shape[0] + sc.shape[0] - 1) // sc.shape[0]
                    bi = (w.shape[1] + sc.shape[1] - 1) // sc.shape[1]
                    w = w * sc.repeat_interleave(bo, 0)[:w.shape[0]] \
                        .repeat_interleave(bi, 1)[:, :w.shape[1]]
                except Exception:
                    pass
                li = int(n.split("layers.")[1].split(".")[0])
                m, k = meter(w)
                c = w.numel()
                lay[li][0] += m * c
                lay[li][1] += k * c
                lay[li][2] += c
        for li in sorted(lay):
            wm, wk, c = lay[li]
            print(f"B3 K2 shard {sh} layer {li}: M {wm / c:.2f} "
                  f"kurt {wk / c:.2f} ({c / 1e6:.0f}M)", flush=True)
        os.remove(os.path.realpath(p))
    print("B3 done; K2 shards deleted", flush=True)


def b4():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "scratch")
    from pack_c6 import PROMPTS
    MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(dev).eval()
    routers = {n: m for n, m in model.named_modules()
               if n.endswith("mlp.gate")}
    sel = collections.defaultdict(list)  # layer -> [set(top8) per tok]
    hooks = []
    for n, mod in routers.items():
        li = int(n.split("layers.")[1].split(".")[0])

        def hk(m_, i, o, li=li):
            top = o.detach().float().topk(8, dim=-1).indices
            for row in top.reshape(-1, 8):
                sel[li].append(set(row.tolist()))
        hooks.append(mod.register_forward_hook(hk))
    text = open("README.md", encoding="utf-8").read()[:8000]
    with torch.no_grad():
        for p in PROMPTS:
            model(tok(p, return_tensors="pt").input_ids.to(dev))
        model(tok(text, return_tensors="pt").input_ids[:, :1024].to(dev))
    for h in hooks:
        h.remove()
    NE = 64
    best = (0.0, None, None)
    for li, toks in sel.items():
        T = len(toks)
        ind = np.zeros((T, NE), dtype=bool)
        for t, s in enumerate(toks):
            ind[t, list(s)] = True
        pi = ind.mean(0)
        top_mi, top_pair = 0.0, None
        C = (ind.astype(np.float32).T @ ind.astype(np.float32)) / T
        for a in range(NE):
            for b in range(a + 1, NE):
                pab, pa, pb = C[a, b], pi[a], pi[b]
                if pab <= 0 or pa * pb <= 0:
                    continue
                mi = pab * math.log2(pab / (pa * pb))
                if mi > top_mi:
                    top_mi, top_pair = mi, (a, b)
        print(f"B4 L{li}: top-pair MI {top_mi:.4f} bits {top_pair} "
              f"(shuffle-scale ~{1 / T:.5f})", flush=True)
        if top_mi > best[0]:
            best = (top_mi, li, top_pair)
    print(f"B4 BEST: L{best[1]} pair {best[2]} MI {best[0]:.4f}",
          flush=True)
    # merge read: mean the pair's expert weights into both slots
    mi_, li, (a, b) = best
    pref = f"model.layers.{li}.mlp.experts"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        wa = dict(model.named_parameters())[f"{pref}.{a}.{proj}.weight"]
        wb = dict(model.named_parameters())[f"{pref}.{b}.{proj}.weight"]
        mean = ((wa.detach().float() + wb.detach().float()) / 2).half()
        wa.data.copy_(mean)
        wb.data.copy_(mean)
    pids = [tok(p, return_tensors="pt").input_ids.to(dev)
            for p in PROMPTS]
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].to(dev)
    with torch.no_grad():
        loss = float(model(ids, labels=ids).loss)
    print(f"B4 merged L{li} ({a},{b}): ppl {math.exp(loss):.3f} "
          f"(fp control 75.739)", flush=True)
    del model


def p6():
    tot = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for p in sorted(glob.glob("checkpoints/blackhole_q3_parts/part-*.npz")):
        z = np.load(p)
        for n in z.files:
            if not n.endswith(".codes"):
                continue
            g = ("expert" if ".experts." in n else
                 "router" if ".mlp.gate." in n else "other")
            c = z[n].ravel()
            span = int(c.max()) - int(c.min()) + 1
            bits = max(1, math.ceil(math.log2(span)))
            _, cnt = np.unique(c, return_counts=True)
            pr = cnt / cnt.sum()
            ent = float(-(pr * np.log2(pr)).sum())
            tot[g][0] += bits * c.size
            tot[g][1] += ent * c.size
            tot[g][2] += c.size
    for g, (raw, ent, n) in sorted(tot.items()):
        print(f"P6 {g}: raw {raw / n:.2f} bits/wt | entropy "
              f"{ent / n:.2f} | penalty {(raw - ent) / n:.2f} "
              f"({n / 1e9:.1f}B params)", flush=True)


def main():
    b3()
    b4()
    p6()
    print("NIGHT-30b MAC CHAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()

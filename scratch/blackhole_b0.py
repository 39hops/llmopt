"""BLACK HOLE MoEs B0+B1+B2 (pre-reg 2026-07-29 close): capacity
atlas + dial-routed streaming pack + function-space spot check of
Qwen3-30B-A3B. One shard on disk at a time (download -> process ->
DELETE — the C7 OOM lesson, applied to disk). Zero calibration.
Atlas rows to logs/blackhole_atlas.jsonl; packed parts to
checkpoints/blackhole_q3_parts/ (codes npz per shard).
Env: START/END shard 1-indexed bounds. __main__-guarded.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
from capacity_meter import meter  # noqa: E402

REPO = "Qwen/Qwen3-30B-A3B"
NSH = 16
OUT = "checkpoints/blackhole_q3_parts"
ATLAS = "logs/blackhole_atlas.jsonl"


def group_of(name):
    if ".experts." in name:
        return "expert"
    if "self_attn" in name:
        return "attn"
    if ".mlp.gate." in name or name.endswith("mlp.gate.weight"):
        return "router"
    return "shared"


def pack_codes(w, sigma_law):
    """-> (codes int16, scales fp32 [rows], bits, err). sigma_law:
    per-row sigma grid (step sigma/2); else per-row max-anchored
    6-bit grid (outliers kept at full range — the P2a lesson)."""
    wf = w.float()
    if sigma_law:
        s = wf.std(dim=1, keepdim=True).clamp(min=1e-8)
        q = torch.ceil(2.0 / s)
        codes = torch.round(wf * q)
        scale = (1.0 / q).squeeze(1)
    else:
        qmax = 31.0  # 6-bit symmetric
        am = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        codes = torch.round(wf / (am / qmax)).clamp(-qmax, qmax)
        scale = (am / qmax).squeeze(1)
    span = int(codes.max() - codes.min()) + 1
    bits = max(1, math.ceil(math.log2(span)))
    wq = codes * scale.unsqueeze(1)
    x = torch.randn(64, wf.shape[1])
    err = float((x @ (wq - wf).T).norm() / (x @ wf.T).norm().clamp(min=1e-12))
    return codes.to(torch.int16), scale, bits, err


def main():
    from huggingface_hub import hf_hub_download
    os.makedirs(OUT, exist_ok=True)
    start = int(os.environ.get("START", "1"))
    end = int(os.environ.get("END", str(NSH)))
    t_all = time.time()
    from safetensors import safe_open
    for i in range(start, end + 1):
        fn = f"model-{i:05d}-of-{NSH:05d}.safetensors"
        t0 = time.time()
        path = hf_hub_download(REPO, fn)
        rows, part = [], {}
        with safe_open(path, framework="pt") as f:
            for n in f.keys():
                w = f.get_tensor(n)
                if w.ndim != 2 or "lm_head" in n:
                    continue
                g = group_of(n)
                m, k = meter(w.float())
                codes, scale, bits, err = pack_codes(w, sigma_law=m < 2.0)
                rows.append({"shard": i, "name": n, "group": g,
                             "shape": list(w.shape), "M": round(m, 3),
                             "kurt": round(k, 3), "bits": bits,
                             "law": "sigma" if m < 2.0 else "max",
                             "err": round(err, 5)})
                part[n + ".codes"] = codes.numpy()
                part[n + ".scale"] = scale.numpy().astype(np.float32)
                del w, codes
        np.savez_compressed(f"{OUT}/part-{i:05d}.npz", **part)
        with open(ATLAS, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.remove(os.path.realpath(path))
        n_exp = sum(1 for r in rows if r["group"] == "expert")
        me = [r["M"] for r in rows if r["group"] == "expert"]
        print(f"B0 shard {i}/{NSH}: {len(rows)} tensors "
              f"({n_exp} expert, M med "
              f"{sorted(me)[len(me) // 2] if me else float('nan'):.2f}) "
              f"| worst err {max(r['err'] for r in rows):.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
    print(f"B0 PASS COMPLETE: shards {start}-{end} in "
          f"{(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

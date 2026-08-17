"""QWEN-FAMILY-PROBE-0: cheapest acceptable RATE per non-FFN family.

Bridge item (GPT seat, banked 2026-08-17): for linear_attn (20% of
the model, the binding unknown), full_attn, embeddings, and lm_head,
measure a minimal RATE ladder — not just codec identity:

  S2@2    DP-optimal 4-level scalar,  2.0625 bpw payload
  W4@2    width-4 VQ, K=256, 1 stage, 2.0625 bpw
  S16@4   DP-optimal 16-level scalar, 4.0625 bpw
  W4x2@4  width-4 VQ, 2 stages,       4.0625 bpw

All per-block-128 E8M0 round-up, per-TENSOR codecs (role-table
discipline: each probed tensor trains its own alphabet/codebook).
Small numerically-special tensors (A_log, dt_bias, conv1d, norms)
are PASSTHROUGH by role and not probed.

Representative tensors (role census 2026-08-17):
  linear_attn L33: in_proj_qkv 10240x5120, in_proj_z 6144x5120,
                   out_proj 5120x6144
  full_attn   L3:  q_proj 12288x5120, k_proj 1024x5120,
                   v_proj 1024x5120, o_proj 5120x6144
  embeddings:      embed_tokens 248320x5120
  lm_head:         lm_head 248320x5120

Descriptive class; rankings and per-family rate curves only.

    SMOKE=1 python -u scratch/qwen_family_probe.py   (slices)
    python -u scratch/qwen_family_probe.py
"""
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

import torch  # noqa: E402
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("qp", "scratch/qwen_stream_probe.py")
qp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(qp)          # reuse loader, e8m0, kmeans, assign, T

DEV = qp.DEV
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = f"logs/qwenprobe/family{'_smoke' if SMOKE else ''}.jsonl"
SEED = 20260816
BLOCK = 128

TARGETS = [
    ("linear_attn", "model.language_model.layers.33.linear_attn.in_proj_qkv.weight"),
    ("linear_attn", "model.language_model.layers.33.linear_attn.in_proj_z.weight"),
    ("linear_attn", "model.language_model.layers.33.linear_attn.out_proj.weight"),
    ("full_attn", "model.language_model.layers.3.self_attn.q_proj.weight"),
    ("full_attn", "model.language_model.layers.3.self_attn.k_proj.weight"),
    ("full_attn", "model.language_model.layers.3.self_attn.v_proj.weight"),
    ("full_attn", "model.language_model.layers.3.self_attn.o_proj.weight"),
    ("embeddings", "model.language_model.embed_tokens.weight"),
    ("lm_head", "lm_head.weight"),
]


def dp_levels(count, ssum, ssq, K):
    """Exact optimal K-level scalar on the frozen binning (vectorized
    DP; K-parameterized generalization of the 0S 4-level solver,
    fp16-representable levels)."""
    n = len(count)
    cc = np.concatenate([[0.0], np.cumsum(count)])
    cs = np.concatenate([[0.0], np.cumsum(ssum)])
    cq = np.concatenate([[0.0], np.cumsum(ssq)])
    INF = float("inf")
    cost = np.full((K + 1, n + 1), INF)
    lev = np.zeros((K + 1, n + 1))
    cut = np.zeros((K + 1, n + 1), np.int64)
    cost[0, 0] = 0.0
    for k in range(1, K + 1):
        prev = cost[k - 1]
        for j in range(k, n + 1):
            i = np.arange(k - 1, j)
            c = cc[j] - cc[i]
            sm = cs[j] - cs[i]
            q = cq[j] - cq[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                L = np.where(c > 0, sm / np.maximum(c, 1e-300), 0.0)
            L = L.astype(np.float16).astype(np.float64)
            ic = np.where(c > 0, q - 2 * L * sm + L * L * c, 0.0)
            tot = prev[i] + ic
            b = int(np.argmin(tot))
            cost[k, j], cut[k, j], lev[k, j] = tot[b], i[b], L[b]
    levels, j = [], n
    for k in range(K, 0, -1):
        levels.append(lev[k, j])
        j = cut[k, j]
    return np.array(sorted(levels), np.float32)


def probe_tensor(fam, name):
    W = qp.load_tensor(name)
    if SMOKE:
        W = W[:max(W.shape[0] // 32, 8)]
    Wt = qp.T(W)
    Wb = Wt.reshape(Wt.shape[0], -1, BLOCK)
    sc = qp.e8m0(Wb)
    Wn = Wb / sc
    nb = Wn.shape[0] * Wn.shape[1]
    flat = Wn.reshape(-1)

    # scalar alphabets from this tensor's own histogram
    edges = qp.T(np.linspace(-1, 1, 4096 + 1).astype(np.float32))
    bi = torch.clamp(torch.bucketize(flat, edges, right=True) - 1, 0, 4095)
    h = [torch.bincount(bi, minlength=4096).double().cpu().numpy(),
         torch.bincount(bi, weights=flat.double(),
                        minlength=4096).cpu().numpy(),
         torch.bincount(bi, weights=flat.double() ** 2,
                        minlength=4096).cpu().numpy()]
    lv4 = qp.T(dp_levels(*h, 4))
    lv16 = qp.T(dp_levels(*h, 16))

    # vector codebooks on this tensor's own sample
    v = Wn.reshape(-1, 4)
    r = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 99991)
    idx = qp.T(r.choice(len(v), min(1 << 20, len(v)), replace=False))
    stack2 = qp.stack_train(v[idx], 2, f"{name}/w4x2")
    stack1 = stack2[:1]

    out = {}
    recon = {
        "S2@2": qp.nearest(Wn, lv4),
        "S16@4": qp.nearest(Wn, lv16),
        "W4@2": qp.vq(v, stack1).reshape(Wn.shape),
        "W4x2@4": qp.vq(v, stack2).reshape(Wn.shape),
    }
    for a, Rn in recon.items():
        R = (Rn * sc).reshape(Wt.shape)
        Dm = R - Wt
        frob = (float((Dm ** 2).sum()) / float((Wt ** 2).sum())) ** 0.5
        n2, d2 = qp.op_parts(Dm, Wt, SEED + 17)
        out[a] = {"frob": frob, "op": (n2 / max(d2, 1e-30)) ** 0.5}
    del Wt, Wb, Wn, recon
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return {"family": fam, "tensor": name.split("language_model.")[-1],
            "shape": list(W.shape), "arms": out,
            "s2_levels": [float(x) for x in lv4.cpu()],
            "mass_within_0p33": float((flat.abs() < 1 / 3).sum())
            / flat.numel()}


def main():
    os.makedirs("logs/qwenprobe", exist_ok=True)
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t0 = time.time()
    rows = []
    for fam, name in TARGETS:
        t = time.time()
        rec = probe_tensor(fam, name)
        rec["probe_s"] = round(time.time() - t, 1)
        rows.append(rec)
        ops = {a: round(v["op"], 4) for a, v in rec["arms"].items()}
        print(f"[fp] {rec['tensor']:45s} {ops}", flush=True)
    row = {"probe": "qwen-family-probe-0", "model": qp.MODEL,
           "revision": qp.REVISION, "smoke": SMOKE, "device": DEV,
           "code_commit": __import__("subprocess").check_output(
               ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
           "tensors": rows, "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[fp] -> {OUT} wall {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

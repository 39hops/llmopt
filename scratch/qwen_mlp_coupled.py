"""QWEN-MLP-COUPLED-0, stage 2: the complete gated MLP of layer 22 under
three weight sets on identical captured inputs, with the vendor's MLP
semantics (y = down(silu(gate x) * up x), mlx_lm Qwen3NextMLP; float32
in the tower) verified against PyTorch and MLX evaluations.

Weight sets (gate, up, down of layer 22):
  REFERENCE  the original vendor BF16 weights (shard 7, pin checked,
             exact in float32)
  CONTROL    the frozen arm C payloads through the canonical decoder
  CANDIDATE  gate_proj replaced by the E_awq representation
             (scratch/qwen_geometry_quant.candidate_w4 with how="awq":
             the arm C W4 layout, codebook and assignment weighted by
             the per-input-channel RMS of the DEV split's captured inputs,
             prefill and freerun together), up and down the unchanged arm
             C decodes; the candidate's storage is the control's (the
             same layout, no extra metadata)
The candidate is calibrated on the dev split ONLY; MODE=dev scores the
dev split (development), MODE=eval scores the eval split (the registered
comparison, read once after registration), always with the dev-fit
candidate (its codebook sha256 is recorded in both receipts and must
match).

Per prompt-block (prompt x distribution: prefill or freerun), in float64
from the float32 inputs: the three MLP outputs and the intermediate gate
outputs g = W_g x, gated products p = silu(g) * (W_u x), and the complete
outputs y = W_d p; sums of squared errors against the reference for the
control and the candidate at each stage; cosine similarity of y to the
reference; the per-position relative error of y and its tail; the cosine
between the candidate's and the control's output error vectors (error
direction); and the cancellation reading, ||W_d dp|| / ||dp|| relative to
||W_d||_F / sqrt(dim) (how much of the gated-product error the down
projection passes). Numerical floor: the reference MLP in float32 against
float64 (the tower's precision), and the same in torch float32 and mlx
float32 on a subset (semantics check: identical formula, differences at
rounding only). Primary: pooled complete-output MSE ratio candidate over
control, bootstrapped over prompt blocks paired (the same resampled
blocks for both arms), with breakdowns by split, category, distribution
and ood. Receipt logs/qwen_mlp/<MODE>.json (refuse-if-exists). Env:
CAPTURE, MODE (dev|eval), OUT, SEED (20261023), BOOT (2000), SMOKE=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from bf16_entropy_census import SHARDS, safetensors_header, tensor_view  # noqa: E402
from qwen_geometry_quant import candidate_w4, hadamard  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
MODE = os.environ.get("MODE", "dev")
CAPTURE = "logs/qwen_mlp/smoke/capture.json" if SMOKE else os.environ.get("CAPTURE", "logs/qwen_mlp/capture.json")
OUT = f"logs/qwen_mlp/smoke/{MODE}.json" if SMOKE else os.environ.get("OUT", f"logs/qwen_mlp/{MODE}.json")
SEED = int(os.environ.get("SEED", "20261023"))
BOOT = int(os.environ.get("BOOT", "2000"))
SHARD_DIR = os.path.expanduser(os.environ.get("SHARD_DIR", "~/qwen_vendor_shards/1d4bf0f2"))
SHARD = "model-00007-of-00018.safetensors"
LAYER = 22
NAMES = {k: f"model.language_model.layers.{LAYER}.mlp.{k}_proj.weight" for k in ("gate", "up", "down")}


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def silu64(g):
    return g / (1.0 + np.exp(-g))


def mlp64(Wg, Wu, Wd, X):
    g = X @ Wg.T
    u = X @ Wu.T
    p = silu64(g) * u
    return g, p, p @ Wd.T


def main() -> None:
    import torch
    import mlx.core as mx
    import mlx.nn as nn
    from llmopt.lab import qartifact
    from llmopt.lab.hash import require_pinned_file
    from llmopt.lab.qcodec import decode_entry
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    cap = json.load(open(CAPTURE))
    if cap.get("smoke") != SMOKE:
        raise SystemExit("REFUSING: capture smoke flag mismatch")
    split = "dev" if MODE == "dev" else "eval"
    rows = cap["prompts"] if SMOKE else [r for r in cap["prompts"] if r["split"] == split]
    dev_rows = cap["prompts"] if SMOKE else [r for r in cap["prompts"] if r["split"] == "dev"]
    if not rows or not dev_rows:
        raise SystemExit("REFUSING: empty split")
    # ---- weights ----
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), "logs/qwenwhole/artifact_digest_C.txt")
    MAN = q["manifest"]
    spath = os.path.join(SHARD_DIR, SHARD)
    pin = require_pinned_file(spath, SHARDS[SHARD]["sha256"], SHARDS[SHARD]["bytes"])
    hdr, base = safetensors_header(spath)
    ref, ctrl, meta = {}, {}, {}
    for k, name in NAMES.items():
        e = MAN[name]
        u16 = np.ascontiguousarray(np.asarray(tensor_view(spath, hdr, base, name)))
        ref[k] = bf16_to_f32(u16).reshape(e["shape"]).astype(np.float64)
        with open(os.path.join(art, e["shard"] + ".bin"), "rb") as f:
            f.seek(e["off"])
            buf = f.read(e["len"])
        ctrl[k] = decode_entry(buf, e).astype(np.float64)
        meta[k] = {"shape": e["shape"], "codec": e["codec"], "payload_sha256": hashlib.sha256(buf).hexdigest(),
                   "orig_bf16_sha256": hashlib.sha256(u16.tobytes()).hexdigest(), "payload_bytes": len(buf),
                   "armC_relative_frobenius": float(np.linalg.norm(ctrl[k] - ref[k]) / np.linalg.norm(ref[k]))}
        del u16
    # ---- candidate: E_awq gate on the DEV split's inputs (prefill + freerun) ----
    Xdev = np.concatenate([np.load(r["files"][d]["path"]) for r in dev_rows for d in ("prefill", "freerun")], 0).astype(np.float64)
    xw = np.sqrt((Xdev ** 2).mean(0))
    ndev = Xdev.shape[0]
    del Xdev
    C = ref["gate"].shape[1]
    tseed = SEED + int(hashlib.sha256(NAMES["gate"].encode()).hexdigest()[:8], 16) % 100000
    H = hadamard(128)
    t0 = time.perf_counter()
    cand_gate = candidate_w4(ref["gate"], "awq", tseed, np.ones(C), H, xw)
    t_fit = time.perf_counter() - t0
    Wg_c = cand_gate["W_hat"]
    cand = {"gate": Wg_c, "up": ctrl["up"], "down": ctrl["down"]}
    cand_meta = {"how": "awq", "fit_positions": int(ndev), "fit_split": "dev", "fit_s": t_fit,
                 "codebook_sha256": cand_gate["codebook_sha256"], "bits": cand_gate["bits"], "bits_per_weight": cand_gate["bits_per_weight"],
                 "gate_relative_frobenius": float(np.linalg.norm(Wg_c - ref["gate"]) / np.linalg.norm(ref["gate"])),
                 "xw_sha256": hashlib.sha256(xw.astype(np.float64).tobytes()).hexdigest(),
                 "xw_percentiles": {str(p): float(np.percentile(xw, p)) for p in (1, 50, 99, 100)}}
    # ---- semantics check on a subset: numpy f32, torch f32, mlx f32 against f64 ----
    X0 = np.load(rows[0]["files"]["prefill"]["path"])[: min(16, rows[0]["n_tokens"])]
    _, _, y64 = mlp64(ref["gate"], ref["up"], ref["down"], X0.astype(np.float64))
    Wg32, Wu32, Wd32 = (ref[k].astype(np.float32) for k in ("gate", "up", "down"))
    g32 = X0 @ Wg32.T
    y_np32 = (((g32 / (1.0 + np.exp(-g32))) * (X0 @ Wu32.T)) @ Wd32.T).astype(np.float64)
    tg, tu, td, tx = (torch.from_numpy(a) for a in (Wg32, Wu32, Wd32, X0))
    y_torch = (torch.nn.functional.silu(tx @ tg.T) * (tx @ tu.T)) @ td.T
    mg, mu, md, mxx = (mx.array(a) for a in (Wg32, Wu32, Wd32, X0))
    y_mlx = (nn.silu(mxx @ mg.T) * (mxx @ mu.T)) @ md.T
    mx.eval(y_mlx)
    den = np.linalg.norm(y64)
    sem = {"positions": int(X0.shape[0]), "formula": "y = W_down ( silu(W_gate x) * (W_up x) ), silu(g) = g / (1 + exp(-g)); mlx_lm Qwen3NextMLP",
           "numpy_f32_vs_f64_rel": float(np.linalg.norm(y_np32 - y64) / den),
           "torch_f32_vs_f64_rel": float(np.linalg.norm(y_torch.numpy().astype(np.float64) - y64) / den),
           "mlx_f32_vs_f64_rel": float(np.linalg.norm(np.asarray(y_mlx, dtype=np.float64) - y64) / den),
           "torch_vs_mlx_rel": float(np.linalg.norm(y_torch.numpy().astype(np.float64) - np.asarray(y_mlx, dtype=np.float64)) / den),
           "torch_version": torch.__version__, "mlx_version": mx.__version__}
    # ---- blocks ----
    blocks = []
    t_all = time.perf_counter()
    Wd_fro = np.linalg.norm(ref["down"])
    for r in rows:
        for dist in ("prefill", "freerun"):
            X = np.load(r["files"][dist]["path"]).astype(np.float64)
            g_r, p_r, y_r = mlp64(ref["gate"], ref["up"], ref["down"], X)
            g_c, p_c, y_c = mlp64(ctrl["gate"], ctrl["up"], ctrl["down"], X)
            g_k, p_k, y_k = mlp64(cand["gate"], cand["up"], cand["down"], X)
            # gate-only effect at the control's up/down: the candidate differs from the control in gate only
            def e2(a, b):
                return float(((a - b) ** 2).sum())
            yn = np.sqrt((y_r ** 2).sum(1))
            ec = np.sqrt(((y_c - y_r) ** 2).sum(1)) / np.maximum(yn, 1e-30)
            ek = np.sqrt(((y_k - y_r) ** 2).sum(1)) / np.maximum(yn, 1e-30)
            cos = lambda a, b: float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))  # noqa: E731
            dp_k = p_k - p_c                       # the candidate's change of the gated product against the control
            dy_k = y_k - y_c
            blk = {"id": r["id"], "category": r["category"], "split": r["split"], "ood_kind": r.get("ood_kind"), "dist": dist,
                   "n": int(X.shape[0]), "y2": e2(y_r, 0.0 * y_r),
                   "ctrl": {"gate_e2": e2(g_c, g_r), "prod_e2": e2(p_c, p_r), "out_e2": e2(y_c, y_r),
                            "cos_mean": float(np.mean((y_c * y_r).sum(1) / np.maximum(np.linalg.norm(y_c, axis=1) * yn, 1e-30))),
                            "pos_rel_p50": float(np.median(ec)), "pos_rel_p99": float(np.percentile(ec, 99)), "pos_rel_max": float(ec.max())},
                   "cand": {"gate_e2": e2(g_k, g_r), "prod_e2": e2(p_k, p_r), "out_e2": e2(y_k, y_r),
                            "cos_mean": float(np.mean((y_k * y_r).sum(1) / np.maximum(np.linalg.norm(y_k, axis=1) * yn, 1e-30))),
                            "pos_rel_p50": float(np.median(ek)), "pos_rel_p99": float(np.percentile(ek, 99)), "pos_rel_max": float(ek.max())},
                   "ref": {"gate2": e2(g_r, 0.0 * g_r), "prod2": e2(p_r, 0.0 * p_r)},
                   "error_direction_cos_cand_vs_ctrl": cos((y_k - y_r).ravel(), (y_c - y_r).ravel()),
                   "gate_change_direction_cos": cos((g_k - g_c).ravel(), (g_r - g_c).ravel()),
                   "down_gain_on_candidate_delta": float(np.linalg.norm(dy_k) / max(np.linalg.norm(dp_k), 1e-30) / (Wd_fro / np.sqrt(ref["down"].shape[0]))) if np.linalg.norm(dp_k) > 0 else None}
            blocks.append(blk)
        print(f"[mlp] {r['id']:10s} {r['category']:13s} out ratio prefill {blocks[-2]['cand']['out_e2']/blocks[-2]['ctrl']['out_e2']:.3f} "
              f"freerun {blocks[-1]['cand']['out_e2']/blocks[-1]['ctrl']['out_e2']:.3f}  gate ratio {blocks[-2]['cand']['gate_e2']/blocks[-2]['ctrl']['gate_e2']:.3f}", flush=True)

    def pooled(bs, stage):
        c = sum(b["ctrl"][stage] for b in bs)
        k = sum(b["cand"][stage] for b in bs)
        y = sum(b["y2"] if stage == "out_e2" else (b["ref"]["gate2"] if stage == "gate_e2" else b["ref"]["prod2"]) for b in bs)
        return {"ctrl_rel_mse": c / y, "cand_rel_mse": k / y, "ratio": k / c, "blocks": len(bs), "positions": sum(b["n"] for b in bs)}

    def boot(bs, stage, seed):
        rng = np.random.default_rng(seed)
        c = np.array([b["ctrl"][stage] for b in bs]); k = np.array([b["cand"][stage] for b in bs])
        n = len(bs)
        vals = np.sort([k[i].sum() / c[i].sum() for i in (rng.integers(0, n, n) for _ in range(BOOT))])
        return {"ratio": float(k.sum() / c.sum()), "ci95": [float(vals[int(0.025 * BOOT)]), float(vals[int(0.975 * BOOT) - 1])], "n_blocks": n}

    summary = {"primary_out": {**pooled(blocks, "out_e2"), **boot(blocks, "out_e2", SEED)},
               "gate": {**pooled(blocks, "gate_e2"), **boot(blocks, "gate_e2", SEED + 1)},
               "prod": {**pooled(blocks, "prod_e2"), **boot(blocks, "prod_e2", SEED + 2)},
               "by_dist": {d: {**pooled([b for b in blocks if b["dist"] == d], "out_e2"), **boot([b for b in blocks if b["dist"] == d], "out_e2", SEED + 3)} for d in ("prefill", "freerun")},
               "by_category": {c: {**pooled([b for b in blocks if b["category"] == c], "out_e2")} for c in sorted({b["category"] for b in blocks})},
               "in_domain": {**pooled([b for b in blocks if b["category"] != "ood"], "out_e2"), **boot([b for b in blocks if b["category"] != "ood"], "out_e2", SEED + 4)} if any(b["category"] != "ood" for b in blocks) else None,
               "ood": {**pooled([b for b in blocks if b["category"] == "ood"], "out_e2"), **boot([b for b in blocks if b["category"] == "ood"], "out_e2", SEED + 5)} if any(b["category"] == "ood" for b in blocks) else None,
               "max_category_ratio": max(pooled([b for b in blocks if b["category"] == c], "out_e2")["ratio"] for c in {b["category"] for b in blocks}),
               "cos_mean": {"ctrl": float(np.mean([b["ctrl"]["cos_mean"] for b in blocks])), "cand": float(np.mean([b["cand"]["cos_mean"] for b in blocks]))},
               "pos_rel_max": {"ctrl": max(b["ctrl"]["pos_rel_max"] for b in blocks), "cand": max(b["cand"]["pos_rel_max"] for b in blocks)},
               "pos_rel_p99_median_over_blocks": {"ctrl": float(np.median([b["ctrl"]["pos_rel_p99"] for b in blocks])), "cand": float(np.median([b["cand"]["pos_rel_p99"] for b in blocks]))},
               "error_direction_cos_median": float(np.median([b["error_direction_cos_cand_vs_ctrl"] for b in blocks])),
               "gate_change_direction_cos_median": float(np.median([b["gate_change_direction_cos"] for b in blocks])),
               "down_gain_median": float(np.median([b["down_gain_on_candidate_delta"] for b in blocks if b["down_gain_on_candidate_delta"] is not None])),
               "storage_bits_per_weight": {"control_gate": 8.0 * meta["gate"]["payload_bytes"] / ref["gate"].size, "candidate_gate": cand_gate["bits_per_weight"]}}
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "mode": MODE, "split": split, "capture": CAPTURE, "capture_commit": cap["commit"],
               "capture_collection_sha256": cap["collection_sha256"], "seed": SEED, "boot": BOOT, "pin_check": pin, "chain_rung0": q["report"]["rung0"],
               "weights": meta, "candidate": cand_meta, "semantics": sem, "prompts": [r["id"] for r in rows], "fit_prompts": [r["id"] for r in dev_rows],
               "blocks": blocks, "summary": summary, "wall_s": time.perf_counter() - t_all}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    s = summary
    print(f"[mlp] {MODE}: out ratio {s['primary_out']['ratio']:.4f} CI {s['primary_out']['ci95']} gate {s['gate']['ratio']:.4f} prod {s['prod']['ratio']:.4f} "
          f"prefill {s['by_dist']['prefill']['ratio']:.4f} freerun {s['by_dist']['freerun']['ratio']:.4f} maxcat {s['max_category_ratio']:.4f}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

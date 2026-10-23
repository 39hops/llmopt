"""QWEN-LOSSY-GEOMETRY-0, stage 2: local representation candidates and
their reconstruction fidelity, at arm C's effective bitrate, on the
frozen captures of scratch/qwen_geometry_capture.py. No kernel, no
model, no artifact is changed; everything here is numpy and torch on the
captured tensors and activations.

Candidates (W4 tensors, the 2-bit budget: one byte selects four weights
from a 256-entry codebook; block-128 power-of-two scale as one byte per
block; the codebook as fp16[256, 4]):
  A  control: the arm C payload through the canonical decoder
  B  refit: the same layout, the codebook refit by k-means++ and 30
     Lloyd iterations over every 4-vector of the tensor (arm C fit on a
     2^20 sample with its own iteration count), nearest assignment
  C  rht: a sign-randomised block-128 normalised Hadamard transform of
     the INPUT basis (W' = (W diag s) H_blk, x' = H_blk (s * x); exact in
     the reals, scratch/leancheck/InputRotation.lean), then B on W';
     stores the sign vector (one bit per input channel)
  D  higgs: C's transform, then a DATA-FREE codebook: 256 four-dimensional
     centroids fit by k-means on synthetic Gaussian vectors whose
     per-block standard deviation is spread as the power-of-two scale
     leaves it (the HIGGS pattern: incoherence, then an MSE-optimal grid
     for the resulting near-Gaussian weights, no tensor-specific
     codebook), with the block scale the power of two nearest the
     block's RMS
  E  awq: B's layout with the assignment and the codebook weighted by
     the development activations' per-input-channel RMS (minimising the
     activation-weighted weight error, the AWQ/GPTQ idea restricted to
     the assignment); fit on the DEV prompts only
S16 tensor (the 4-bit budget: one nibble per weight from 16 fp16 levels,
the same block scale): A control; B refit (Lloyd-Max 16 levels on the
tensor); C rht then B; D higgs (data-free Gaussian-optimal 16 levels after
the transform); E awq (dev-weighted Lloyd-Max and assignment).
Storage: every candidate's bits per weight count the indices or codes,
the block exponents, the codebook or levels, the sign vector and any
padding, and are compared with the control's.

Fidelity, on the split named by MODE (dev prompts while candidates are
designed; eval prompts once, after registration): relative Frobenius
weight error; pooled relative output MSE sum ||W_hat x - W x||^2 / sum
||W x||^2 over every captured position (float64 reference from the
original BF16 values), per prompt for the bootstrap over prompts; the
per-row relative error distribution; the per-position relative error
distribution with its tail; cancellation cases (positions in the lowest
5 percent of ||W x|| / (||W||_F ||x||)) against the rest; float32
evaluation of the same products (the evaluation-order effect of the
rotation against the quantisation error); and the HIGGS-style isotropic
prediction ||dW||_F^2 mean||x||^2 / C against the measured output error.
Receipt logs/qwen_geometry/<MODE>.json (refuse-if-exists). Env:
CAPTURE (logs/qwen_geometry/capture.json), MODE (dev|eval), OUT, SEED
(20261022), BOOT (2000), GEOM_DEVICE (cpu; the fits are deterministic on
the CPU and not on MPS), TENSORS (comma list of shorts, a smoke filter),
SMOKE=1 (the smoke capture, 2 prompts).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SMOKE = os.environ.get("SMOKE") == "1"
MODE = os.environ.get("MODE", "dev")
CAPTURE = "logs/qwen_geometry/smoke/capture.json" if SMOKE else os.environ.get("CAPTURE", "logs/qwen_geometry/capture.json")
OUT = (f"logs/qwen_geometry/smoke/{MODE}.json" if SMOKE else os.environ.get("OUT", f"logs/qwen_geometry/{MODE}.json"))
SEED = int(os.environ.get("SEED", "20261022"))
BOOT = int(os.environ.get("BOOT", "2000"))
BLOCK = 128
HAD = 128
K = 256
LLOYD = 30
DEV = torch.device(os.environ.get("GEOM_DEVICE", "cpu"))   # CPU by default: deterministic index_add_ and argmin (MPS is not)
TENSOR_FILTER = [t for t in os.environ.get("TENSORS", "").split(",") if t]


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def hadamard(b: int) -> np.ndarray:
    H = np.array([[1.0]])
    while H.shape[0] < b:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(b)


def rotate_rows(W: np.ndarray, s: np.ndarray, H: np.ndarray) -> np.ndarray:
    """(W diag s) H_blk along the last axis, float64."""
    R, C = W.shape
    return ((W * s).reshape(R, C // HAD, HAD) @ H).reshape(R, C)


def rotate_x(X: np.ndarray, s: np.ndarray, H: np.ndarray) -> np.ndarray:
    S, C = X.shape
    return ((X * s).reshape(S, C // HAD, HAD) @ H).reshape(S, C)


def e8m0_max(Wb: np.ndarray) -> np.ndarray:
    """arm C's block scale: the power of two at or above the block's max |w| (float64)."""
    m = np.abs(Wb).max(-1, keepdims=True)
    e = np.clip(np.ceil(np.log2(np.maximum(m, 2.0 ** -126))), -127, 127)
    return np.exp2(e)


def e8m0_rms(Wb: np.ndarray) -> np.ndarray:
    """the power of two nearest the block's RMS (the HIGGS-style normalisation under the one-byte scale)."""
    r = np.sqrt((Wb ** 2).mean(-1, keepdims=True))
    e = np.clip(np.round(np.log2(np.maximum(r, 2.0 ** -126))), -127, 127)
    return np.exp2(e)


def _assign(V: torch.Tensor, Cb: torch.Tensor, wdiag: torch.Tensor | None = None, chunk: int = 1 << 18) -> torch.Tensor:
    """nearest centroid (optionally under a diagonal metric wdiag [d])."""
    if wdiag is None:
        cn = (Cb * Cb).sum(1)
        out = torch.empty(len(V), dtype=torch.long, device=V.device)
        for i in range(0, len(V), chunk):
            out[i:i + chunk] = (cn[None] - 2.0 * (V[i:i + chunk] @ Cb.T)).argmin(1)
        return out
    Cw = Cb * wdiag[None]
    cn = (Cb * Cw).sum(1)
    out = torch.empty(len(V), dtype=torch.long, device=V.device)
    for i in range(0, len(V), chunk):
        out[i:i + chunk] = (cn[None] - 2.0 * (V[i:i + chunk] @ Cw.T)).argmin(1)
    return out


def kmeans(V: torch.Tensor, k: int, seed: int, iters: int, wdiag: torch.Tensor | None = None) -> torch.Tensor:
    """k-means++ on a 2^18 sample, then `iters` Lloyd iterations over all of V; centroids rounded to fp16."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(V)
    sub = V[torch.randperm(n, generator=g)[: min(1 << 18, n)].to(V.device)]
    cent = [sub[int(torch.randint(len(sub), (1,), generator=g))]]
    d2 = ((sub - cent[0]) ** 2).sum(1)
    for _ in range(k - 1):
        p = (d2 / torch.clamp(d2.sum(), min=1e-30)).cpu().double()
        idx = int(torch.multinomial(p, 1, generator=g))
        cent.append(sub[idx])
        d2 = torch.minimum(d2, ((sub - cent[-1]) ** 2).sum(1))
    Cb = torch.stack(cent)
    for _ in range(iters):
        a = _assign(V, Cb, wdiag)
        cnt = torch.bincount(a, minlength=k).float()
        S = torch.zeros_like(Cb)
        S.index_add_(0, a, V)
        nz = cnt > 0
        Cb[nz] = S[nz] / cnt[nz, None]
    return Cb.half().float()


def gaussian_codebook(d: int, k: int, seed: int, scale_spread: bool) -> torch.Tensor:
    """data-free: k-means on 2^20 synthetic d-vectors ~ N(0, sigma^2 I) with sigma = 2^u,
    u ~ Uniform(-0.5, 0.5) when scale_spread (the block RMS under a power-of-two scale sits
    anywhere in [2^-0.5, 2^0.5]); sigma = 1 otherwise."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 1 << 20
    Z = torch.randn(n, d, generator=g)
    if scale_spread:
        Z = Z * torch.exp2(torch.rand(n, 1, generator=g) - 0.5)
    return kmeans(Z.to(DEV), k, seed + 1, LLOYD)


def lloyd_max(v: torch.Tensor, levels: int, seed: int, iters: int, w: torch.Tensor | None = None) -> torch.Tensor:
    """16-level scalar quantiser by Lloyd iterations (optionally weighted); levels rounded to fp16."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.quantile(v[torch.randperm(len(v), generator=g)[: 1 << 20].to(v.device)].float().cpu(),
                       torch.linspace(0.5 / levels, 1 - 0.5 / levels, levels)).to(v.device)
    lv = q.clone()
    for _ in range(iters):
        a = (v[:, None] - lv[None]).abs().argmin(1)
        cnt = torch.bincount(a, minlength=levels, weights=w).float()
        S = torch.zeros(levels, device=v.device)
        S.index_add_(0, a, v * (w if w is not None else 1.0))
        nz = cnt > 0
        lv[nz] = S[nz] / cnt[nz]
        lv, _ = torch.sort(lv)
    return lv.half().float()


def quantize_vq(Wn: np.ndarray, Cb: np.ndarray, d: int, wdiag: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    V = torch.from_numpy(Wn.reshape(-1, d).astype(np.float32)).to(DEV)
    a = _assign(V, torch.from_numpy(Cb.astype(np.float32)).to(DEV),
                None if wdiag is None else torch.from_numpy(wdiag.astype(np.float32)).to(DEV))
    out = Cb[a.cpu().numpy()].reshape(Wn.shape)
    return out, a.cpu().numpy().astype(np.uint8)


def candidate_w4(W: np.ndarray, how: str, seed: int, s: np.ndarray | None, H: np.ndarray, xw: np.ndarray | None) -> dict:
    """returns reconstruction in the ORIGINAL basis (float64) plus storage and the rotated reconstruction if any."""
    R, C = W.shape
    n = W.size
    rot = how in ("rht", "higgs")
    Wd = rotate_rows(W, s, H) if rot else W
    Wb = Wd.reshape(R, C // BLOCK, BLOCK)
    sc = e8m0_rms(Wb) if how == "higgs" else e8m0_max(Wb)
    Wn = (Wb / sc).reshape(R, C)
    V = torch.from_numpy(Wn.reshape(-1, 4).astype(np.float32)).to(DEV)
    wdiag = None
    if how == "higgs":
        Cb = gaussian_codebook(4, K, seed, True).cpu().numpy().astype(np.float64)
    elif how == "awq":
        # per-input-channel weights, repeated per 4-vector position: channel j of the 4-vector is input j
        xw4 = xw.reshape(C // 4, 4)                                   # [C/4, 4] channel RMS by 4-group
        # the diagonal metric differs per 4-group along C; weight the k-means by the mean over groups
        wmean = torch.from_numpy((xw4 ** 2).mean(0).astype(np.float32)).to(DEV)
        Cb = kmeans(V, K, seed, LLOYD, wmean).cpu().numpy().astype(np.float64)
    else:
        Cb = kmeans(V, K, seed, LLOYD).cpu().numpy().astype(np.float64)
    if how == "awq":
        # exact weighted nearest assignment per 4-group: loop over the C/4 group positions in chunks of rows
        Vr = Wn.reshape(R, C // 4, 4)
        Cbt = torch.from_numpy(Cb.astype(np.float32)).to(DEV)
        out = np.empty_like(Vr)
        idx = np.empty((R, C // 4), np.uint8)
        xw4t = torch.from_numpy((xw.reshape(C // 4, 4) ** 2).astype(np.float32)).to(DEV)   # [C/4, 4]
        step = max(1, (1 << 26) // ((C // 4) * K * 4))
        for lo in range(0, R, step):
            Vt = torch.from_numpy(Vr[lo:lo + step].astype(np.float32)).to(DEV)             # [r, C/4, 4]
            # dist[r, g, k] = sum_j w[g,j] (V[r,g,j] - C[k,j])^2
            d = ((Vt[:, :, None, :] - Cbt[None, None, :, :]) ** 2 * xw4t[None, :, None, :]).sum(-1)
            a = d.argmin(-1)
            out[lo:lo + step] = Cb[a.cpu().numpy()]
            idx[lo:lo + step] = a.cpu().numpy().astype(np.uint8)
        Q = out.reshape(R, C)
    else:
        Q, idx = quantize_vq(Wn, Cb, 4)
    Qd = (Q.reshape(R, C // BLOCK, BLOCK) * sc).reshape(R, C)
    bits = {"indices": n * 2, "exponents": (n // BLOCK) * 8, "codebook": K * 4 * 16, "signs": C if rot else 0}
    rec = {"how": how, "rotated": rot, "W_hat_rot": Qd if rot else None, "W_hat": None,
           "bits": bits, "bits_per_weight": sum(bits.values()) / n, "codebook_sha256": hashlib.sha256(Cb.astype(np.float16).tobytes()).hexdigest()}
    if rot:
        # back to the original basis: W_hat = Q H_blk^T diag(s) (exact in the reals)
        Wo = (Qd.reshape(R, C // HAD, HAD) @ H.T).reshape(R, C) * s
        rec["W_hat"] = Wo
    else:
        rec["W_hat"] = Qd
    return rec


def candidate_s16(W: np.ndarray, how: str, seed: int, s: np.ndarray | None, H: np.ndarray, xw: np.ndarray | None) -> dict:
    R, C = W.shape
    n = W.size
    rot = how in ("rht", "higgs")
    Wd = rotate_rows(W, s, H) if rot else W
    Wb = Wd.reshape(R, C // BLOCK, BLOCK)
    sc = e8m0_rms(Wb) if how == "higgs" else e8m0_max(Wb)
    Wn = (Wb / sc).reshape(R, C)
    v = torch.from_numpy(Wn.reshape(-1).astype(np.float32)).to(DEV)
    if how == "higgs":
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(1 << 22, generator=g) * torch.exp2(torch.rand(1 << 22, generator=g) - 0.5)
        lv = lloyd_max(z.to(DEV), 16, seed + 1, LLOYD)
    elif how == "awq":
        w = torch.from_numpy(np.tile(xw ** 2, R).astype(np.float32)).to(DEV)
        lv = lloyd_max(v, 16, seed, LLOYD, w)
    else:
        lv = lloyd_max(v, 16, seed, LLOYD)
    a = (v[:, None] - lv[None]).abs().argmin(1)
    Q = lv[a].cpu().numpy().astype(np.float64).reshape(R, C)
    Qd = (Q.reshape(R, C // BLOCK, BLOCK) * sc).reshape(R, C)
    bits = {"codes": n * 4, "exponents": (n // BLOCK) * 8, "levels": 16 * 16, "signs": C if rot else 0}
    rec = {"how": how, "rotated": rot, "W_hat_rot": Qd if rot else None, "bits": bits, "bits_per_weight": sum(bits.values()) / n,
           "levels_sha256": hashlib.sha256(lv.cpu().numpy().astype(np.float16).tobytes()).hexdigest()}
    rec["W_hat"] = (Qd.reshape(R, C // HAD, HAD) @ H.T).reshape(R, C) * s if rot else Qd
    return rec


def fidelity(W: np.ndarray, rec: dict, Xs: list[np.ndarray], s, H, want32: bool) -> dict:
    """per-prompt sums in float64; row and position distributions pooled."""
    Wh = rec["W_hat"]
    dW = Wh - W
    R, C = W.shape
    frob = float(np.linalg.norm(dW) / np.linalg.norm(W))
    row_num = np.zeros(R)
    row_den = np.zeros(R)
    per_prompt = []
    pos_rel = []
    canc = []
    f32 = []
    sumx2 = 0.0
    npos = 0
    for X in Xs:
        X64 = X.astype(np.float64)
        y = X64 @ W.T                                   # [S, R]
        yh = X64 @ Wh.T
        err = yh - y
        per_prompt.append({"err2": float((err ** 2).sum()), "y2": float((y ** 2).sum()), "n": int(X.shape[0])})
        row_num += (err ** 2).sum(0)
        row_den += (y ** 2).sum(0)
        yn = np.sqrt((y ** 2).sum(1))
        en = np.sqrt((err ** 2).sum(1))
        pos_rel.append(en / np.maximum(yn, 1e-30))
        xn = np.sqrt((X64 ** 2).sum(1))
        canc.append(yn / np.maximum(np.linalg.norm(W) * xn, 1e-30))
        sumx2 += float((X64 ** 2).sum())
        npos += X.shape[0]
        if want32:
            X32 = X.astype(np.float32)
            if rec["rotated"]:
                Xr = rotate_x(X64, s, H).astype(np.float32)
                yh32 = Xr @ rec["W_hat_rot"].astype(np.float32).T
            else:
                yh32 = X32 @ Wh.astype(np.float32).T
            f32.append({"f32_vs_f64_same_candidate_rel": float(np.linalg.norm(yh32.astype(np.float64) - yh) / np.linalg.norm(yh)),
                        "f32_err_vs_ref_rel2": float(((yh32.astype(np.float64) - y) ** 2).sum() / (y ** 2).sum())})
    pos_rel = np.concatenate(pos_rel)
    canc = np.concatenate(canc)
    low = canc <= np.quantile(canc, 0.05)
    tot_err = sum(p["err2"] for p in per_prompt)
    tot_y = sum(p["y2"] for p in per_prompt)
    rowrel = np.sqrt(row_num / np.maximum(row_den, 1e-30))
    pred = float((dW ** 2).sum()) * (sumx2 / npos) / C / (tot_y / npos)
    out = {"relative_frobenius": frob, "output_rel_mse": tot_err / tot_y, "per_prompt": per_prompt,
           "row_rel_err_percentiles": {q: float(np.percentile(rowrel, q)) for q in (5, 25, 50, 75, 95, 99, 100)},
           "pos_rel_err_percentiles": {q: float(np.percentile(pos_rel, q)) for q in (5, 25, 50, 75, 95, 99, 99.9, 100)},
           "cancellation": {"threshold_ratio": float(np.quantile(canc, 0.05)), "low5_median_rel_err": float(np.median(pos_rel[low])),
                            "rest_median_rel_err": float(np.median(pos_rel[~low]))},
           "isotropic_prediction_rel_mse": pred, "measured_over_isotropic": (tot_err / tot_y) / pred if pred > 0 else None}
    if f32:
        out["float32"] = {"f32_vs_f64_same_candidate_rel_max": max(f["f32_vs_f64_same_candidate_rel"] for f in f32),
                          "f32_err_vs_ref_rel2_pooled": float(np.mean([f["f32_err_vs_ref_rel2"] for f in f32]))}
    return out


def bootstrap_ratio(cand: list[dict], ctrl: list[dict], boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(cand)
    c = np.array([p["err2"] for p in cand]); y = np.array([p["y2"] for p in cand]); k = np.array([p["err2"] for p in ctrl])
    point = float(c.sum() / k.sum())
    vals = []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        vals.append(c[i].sum() / k[i].sum())
    vals = np.sort(vals)
    return {"ratio_candidate_over_control": point, "boot_ci95": [float(vals[int(0.025 * boot)]), float(vals[int(0.975 * boot) - 1])],
            "n_prompts": n, "per_prompt_ratio": [float(a / b) for a, b in zip(c, k)]}


def main() -> None:
    from llmopt.lab import qartifact
    from llmopt.lab.qcodec import decode_entry
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    cap = json.load(open(CAPTURE))
    if cap.get("smoke") != SMOKE:
        raise SystemExit("REFUSING: capture smoke flag does not match")
    split = "dev" if MODE == "dev" else "eval"
    prompts = [r for r in cap["captures"]["per_prompt"] if r["split"] == split] if not SMOKE else cap["captures"]["per_prompt"]
    dev_prompts = [r for r in cap["captures"]["per_prompt"] if r["split"] == "dev"] if not SMOKE else cap["captures"]["per_prompt"]
    if not prompts:
        raise SystemExit("REFUSING: no prompts in the split")
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), "logs/qwenwhole/artifact_digest_C.txt")
    MAN = q["manifest"]
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "mode": MODE, "split": split, "capture": CAPTURE,
               "capture_commit": cap["commit"], "seed": SEED, "boot": BOOT, "device": str(DEV), "torch": torch.__version__,
               "prompts": [r["prompt"] for r in prompts], "fit_prompts_for_awq": [r["prompt"] for r in dev_prompts],
               "hadamard_block": HAD, "scale_block": BLOCK, "lloyd_iters": LLOYD, "tensors": {}}
    H = hadamard(HAD)
    for name, tinfo in cap["tensors"].items():
        short = name.split("layers.")[1].replace(".weight", "")
        if short.endswith("up_proj"):
            continue        # shares gate_proj's input; same codec; reported once through gate_proj
        if TENSOR_FILTER and short not in TENSOR_FILTER:
            continue
        e = MAN[name]
        W = bf16_to_f32(np.load(tinfo["orig_dump"])).astype(np.float64)
        R, C = W.shape
        with open(os.path.join(art, e["shard"] + ".bin"), "rb") as f:
            f.seek(e["off"])
            buf = f.read(e["len"])
        if hashlib.sha256(buf).hexdigest() != tinfo["payload_sha256"]:
            raise SystemExit(f"REFUSING: payload sha changed for {name}")
        cap_mod = tinfo["module"]
        Xs = [np.load(r["files"][cap_mod]["path"]) for r in prompts]
        Xdev = [np.load(r["files"][cap_mod]["path"]) for r in dev_prompts]
        xw = np.sqrt((np.concatenate(Xdev, 0).astype(np.float64) ** 2).mean(0))      # per-input-channel RMS on the DEV prompts
        tseed = SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100000
        rng = np.random.default_rng(tseed)
        s = rng.choice([-1.0, 1.0], size=C)
        t0 = time.perf_counter()
        ctrl = decode_entry(buf, e).astype(np.float64)
        rows = {}
        n = W.size
        if e["codec"] == "w4":
            ctrl_bits = {"indices": n * 2, "exponents": (n // BLOCK) * 8, "codebook": K * 4 * 16, "signs": 0}
            maker = candidate_w4
        else:
            ctrl_bits = {"codes": n * 4, "exponents": (n // BLOCK) * 8, "levels": 16 * 16, "signs": 0}
            maker = candidate_s16
        recs = {"A_control": {"how": "control", "rotated": False, "W_hat": ctrl, "W_hat_rot": None, "bits": ctrl_bits,
                              "bits_per_weight": sum(ctrl_bits.values()) / n}}
        for label, how in (("B_refit", "refit"), ("C_rht", "rht"), ("D_higgs", "higgs"), ("E_awq", "awq")):
            t1 = time.perf_counter()
            recs[label] = maker(W, how, tseed, s, H, xw)
            recs[label]["fit_s"] = time.perf_counter() - t1
        # exactness of the rotation itself on the ORIGINAL weights (no quantisation), float64 and float32
        Wr = rotate_rows(W, s, H)
        Xc = Xs[0].astype(np.float64)[: min(64, Xs[0].shape[0])]
        y_ref = Xc @ W.T
        y_rot64 = rotate_x(Xc, s, H) @ Wr.T
        y_rot32 = rotate_x(Xc, s, H).astype(np.float32) @ Wr.astype(np.float32).T
        y_plain32 = Xc.astype(np.float32) @ W.astype(np.float32).T
        ident = {"f64_rotated_vs_plain_rel": float(np.linalg.norm(y_rot64 - y_ref) / np.linalg.norm(y_ref)),
                 "f32_plain_vs_f64_rel": float(np.linalg.norm(y_plain32.astype(np.float64) - y_ref) / np.linalg.norm(y_ref)),
                 "f32_rotated_vs_f64_rel": float(np.linalg.norm(y_rot32.astype(np.float64) - y_ref) / np.linalg.norm(y_ref)),
                 "orthogonality_H_blk_max_dev": float(np.abs(H.T @ H - np.eye(HAD)).max()),
                 "rotated_frobenius_over_plain": float(np.linalg.norm(Wr) / np.linalg.norm(W)),
                 "incoherence_max_abs_over_rms": {"plain": float(np.abs(W).max() / np.sqrt((W ** 2).mean())),
                                                  "rotated": float(np.abs(Wr).max() / np.sqrt((Wr ** 2).mean()))}}
        for label, rec in recs.items():
            t1 = time.perf_counter()
            fid = fidelity(W, rec, Xs, s, H, want32=label in ("A_control", "C_rht", "D_higgs"))
            fid["bits"] = rec["bits"]
            fid["bits_per_weight"] = rec["bits_per_weight"]
            fid["fit_s"] = rec.get("fit_s")
            fid["eval_s"] = time.perf_counter() - t1
            for k in ("codebook_sha256", "levels_sha256"):
                if k in rec:
                    fid[k] = rec[k]
            rows[label] = fid
            print(f"[geom] {short} {label:10s} bits/w {rec['bits_per_weight']:.4f} relF {fid['relative_frobenius']:.4f} "
                  f"out relMSE {fid['output_rel_mse']:.3e} meas/iso {fid['measured_over_isotropic']:.2f} "
                  f"row p50 {fid['row_rel_err_percentiles'][50]:.3f} p99 {fid['row_rel_err_percentiles'][99]:.3f}", flush=True)
        boots = {label: bootstrap_ratio(rows[label]["per_prompt"], rows["A_control"]["per_prompt"], BOOT, tseed)
                 for label in rows if label != "A_control"}
        receipt["tensors"][name] = {"short": short, "codec": e["codec"], "shape": e["shape"], "capture_module": cap_mod,
                                    "sign_seed": tseed, "signs_sha256": hashlib.sha256(s.astype(np.int8).tobytes()).hexdigest(),
                                    "n_positions": sum(int(x.shape[0]) for x in Xs), "rotation_identity": ident,
                                    "candidates": rows, "bootstrap_vs_control": boots, "wall_s": time.perf_counter() - t0}
        del W, ctrl, recs, Xs, Xdev
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[geom] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""CUDA leg rung 2: one fused w4 decode+GEMV on a real projection.

y = W x for the rung-1 tensor (L33 in_proj_qkv, 10240 x 5120,
w4 payload 13.5 MB), decoded INSIDE the kernel — no fp16/fp32
weight materialization to DRAM (kernel_form: fused). fp32
accumulation.

Parity (declared BEFORE the run): against a float64-accumulated
reference on the canonically decoded weights, with fixed-seed
activation. Bar: max_i |y_i - ref_i| / max(1e-30, max_i |ref_i|)
<= 1e-5. The fp32 torch.mv of the decoded weights is reported
alongside as the same-error-class sanity (accumulation order
differs; bit-exactness is NOT expected for GEMV, unlike rung 1).

Timing protocol (registered): weights + x resident before t0;
cuda-event timing; 10 warmup iterations discarded; median and
spread of 50 reps. Reported metric = effective compressed-weight
bandwidth (payload bytes / median time). The dequant-then-mv wall
is reported for color, not comparison.

Receipt: logs/qwencuda/rung2.json (refuse-if-exists).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.qcodec import BLOCK, dec_w4  # noqa: E402

OUT = "logs/qwencuda/rung2.json"
ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
TENSOR = os.environ.get(
    "REAL_TENSOR",
    "model.language_model.layers.33.linear_attn.in_proj_qkv.weight")
REL_BAR = 1e-5


@triton.jit
def w4_gemv_kernel(idx_ptr, cb_ptr, scale_ptr, x_ptr, y_ptr,
                   C, nb_row, BLK_C: tl.constexpr):
    """One program per output row: decode the row's w4 blocks in
    registers and dot with x, fp32 accumulation. No weight bytes
    ever written back to DRAM."""
    r = tl.program_id(0)
    acc = tl.zeros((BLK_C,), tl.float32)
    for c0 in range(0, C, BLK_C):
        offs = c0 + tl.arange(0, BLK_C)
        flat = r * C + offs
        byte = tl.load(idx_ptr + flat // 4)
        lane = flat % 4
        val = tl.load(cb_ptr + byte.to(tl.int32) * 4
                      + lane).to(tl.float32)
        s = tl.load(scale_ptr + flat // 128)
        x = tl.load(x_ptr + offs)
        acc += val * s * x
    tl.store(y_ptr + r, tl.sum(acc, 0))


def main() -> int:
    if os.path.exists(OUT):
        raise SystemExit(f"refuse: {OUT} exists — new run, new path")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    from llmopt.lab import qartifact
    arm = os.path.basename(ART.rstrip("/"))
    chain = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "qwenwhole",
        f"artifact_digest_{arm}.txt")
    q = qartifact.qualify_artifact(
        ART, VDIR + "/model.safetensors.index.json",
        chain if os.path.exists(chain) else None,
        allow_unchained=os.environ.get("ALLOW_UNCHAINED") == "1")
    e = q["manifest"][TENSOR]
    assert e["codec"] == "w4"
    R, C = e["shape"]
    assert C % BLOCK == 0 and C % 4 == 0, "row-aligned contract"
    with open(os.path.join(ART, e["shard"] + ".bin"), "rb") as f:
        f.seek(e["off"])
        buf = f.read(e["len"])

    n = R * C
    nb = n // BLOCK
    exps = np.frombuffer(buf, np.uint8, nb, 0).astype(np.int32)
    scale = np.exp2(exps - 127).astype(np.float32)
    cb = np.frombuffer(buf, np.float16, 256 * 4, nb)
    idx = np.frombuffer(buf, np.uint8, n // 4, nb + 2048)

    d_idx = torch.from_numpy(idx.copy()).cuda()
    d_cb = torch.from_numpy(cb.copy()).cuda()
    d_scale = torch.from_numpy(scale).cuda()
    rng = np.random.default_rng(17)
    x = rng.standard_normal(C).astype(np.float32)
    d_x = torch.from_numpy(x).cuda()
    d_y = torch.empty(R, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    # references on the canonical decode
    W = dec_w4(buf, [R, C])
    ref64 = (W.astype(np.float64) @ x.astype(np.float64))
    ref32 = torch.mv(torch.from_numpy(W), torch.from_numpy(x)).numpy()

    grid = (R,)
    args = (d_idx, d_cb, d_scale, d_x, d_y, C, C // BLOCK)
    w4_gemv_kernel[grid](*args, BLK_C=512)
    torch.cuda.synchronize()
    y = d_y.cpu().numpy()
    denom = max(1e-30, float(np.abs(ref64).max()))
    rel_gpu = float(np.abs(y - ref64).max() / denom)
    rel_mv32 = float(np.abs(ref32 - ref64).max() / denom)
    passed = rel_gpu <= REL_BAR
    print(f"[r2] rel_gpu={rel_gpu:.3e} rel_mv32={rel_mv32:.3e} "
          f"bar={REL_BAR} passed={passed}", flush=True)

    # timing: warmup 10, 50 reps, cuda events
    for _ in range(10):
        w4_gemv_kernel[grid](*args, BLK_C=512)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        w4_gemv_kernel[grid](*args, BLK_C=512)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1) / 1e3)
    med = float(np.median(times))
    bw = e["len"] / med / 1e9

    # dequant-then-mv wall, for color (expanded operand = 4*n bytes)
    d_W = torch.from_numpy(W).cuda()
    torch.cuda.synchronize()
    dt = []
    for _ in range(50):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        torch.mv(d_W, d_x)
        t1.record()
        torch.cuda.synchronize()
        dt.append(t0.elapsed_time(t1) / 1e3)
    med_mv = float(np.median(dt))

    rec = {
        "rung": 2,
        "code_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "device_actual": torch.cuda.get_device_name(0),
        "artifact": arm, "tensor": TENSOR, "shape": [R, C],
        "kernel_form": "fused",
        "expanded_operand_bytes": 0,
        "compressed_payload_bytes": e["len"],
        "parity": {"rel_err_gpu_vs_f64": rel_gpu,
                   "rel_err_mv32_vs_f64": rel_mv32,
                   "bar": REL_BAR, "passed": passed},
        "timing": {"warmup": 10, "reps": 50,
                   "median_s": med,
                   "p10_s": float(np.percentile(times, 10)),
                   "p90_s": float(np.percentile(times, 90)),
                   "effective_compressed_bw_GBs": bw,
                   "resident_fp32_mv_median_s": med_mv},
        "env": {k: os.environ.get(k) for k in
                ("TORCH_DISABLE_NATIVE_JIT",
                 "PYTORCH_CUDA_ALLOC_CONF")},
    }
    with open(OUT, "x") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps(rec["timing"], indent=1), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""qcuda-tower qualification ladder, steps a-c (3080; spec
2026-08-19-qcuda-tower-runtime).

a. synthetic s16 parity incl. exponent edges (decode + gemv v
   canonical dec_s16),
b. REAL BLe s16 tensors: FusedS16Linear decode-rows + gemv v
   canonical dec_s16 @ x on representative qkv/z/out shapes,
c. microbench: dense-FP32 fallback matmul v fused s16 GEMV on the
   same real tensors.

    .venv/bin/python scratch/qcuda_tower_qualify.py
Receipt: logs/qcudatower/qualify_abc.json (append-refused).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab import qcuda_tower as qt  # noqa: E402
from llmopt.lab.qcodec import dec_s16, expected_len  # noqa: E402

import torch  # noqa: E402

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/BLe"))
# representative promoted-band shapes: qkv, z, out projections of an
# early-linear layer plus the io head row-space
REAL_TENSORS = [
    "model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.1.linear_attn.in_proj_z.weight",
    "model.language_model.layers.1.linear_attn.out_proj.weight",
]

# start-state provenance at process ENTRY (spec item 4; the RESIDUAL
# completion-commit deviation)
START = {"start_commit": subprocess.check_output(
             ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
         "start_tree_dirty": bool(subprocess.check_output(
             ["git", "status", "--porcelain"]).decode().strip()),
         "interpreter": sys.executable}


def synth_case(R, C, exp_lo, exp_hi, seed):
    rng = np.random.default_rng(seed)
    n = R * C
    exps = rng.integers(exp_lo, exp_hi, n // 128, dtype=np.uint8)
    lv = (rng.standard_normal(16) * 0.1).astype(np.float16)
    codes = rng.integers(0, 256, n // 2, dtype=np.uint8)
    return exps.tobytes() + lv.tobytes() + codes.tobytes()


def main():
    out_dir = "logs/qcudatower"
    os.makedirs(out_dir, exist_ok=True)
    rcpt = os.path.join(out_dir, "qualify_abc.json")
    if os.path.exists(rcpt):
        raise SystemExit(f"REFUSING: {rcpt} exists")
    assert torch.cuda.is_available() and qt.HAVE_TRITON
    res = {"start": START, "steps": {}}

    # ---- a: synthetic parity incl. exp edges ----
    a = []
    for (R, C, lo, hi, sd) in ((8, 256, 120, 135, 0),
                               (16, 512, 0, 3, 1),      # subnormal edge
                               (8, 384, 180, 200, 2)):  # huge-exp edge
        # (exp cap 200 = 2^73 scales: large but the fp32 REFERENCE
        # matmul stays finite; 250+ overflows the oracle, not the
        # kernel)
        buf = synth_case(R, C, lo, hi, sd)
        ref = dec_s16(buf, [R, C])
        pay = qt.S16Gpu(buf, [R, C])
        dec = qt.s16_decode_rows(pay, 0, R).cpu().numpy()
        bit = bool(np.array_equal(dec, ref))
        x = np.random.default_rng(sd + 10).standard_normal(C) \
            .astype(np.float32)
        y = pay.gemv(torch.from_numpy(x).cuda()).cpu().numpy()
        yref = ref @ x
        rel = float(np.abs(y - yref).max() /
                    max(np.abs(yref).max(), 1e-30))
        a.append({"shape": [R, C], "exp": [lo, hi],
                  "decode_bit_exact": bit, "gemv_rel_max": rel})
        assert bit, f"synthetic decode mismatch {R}x{C} exp[{lo},{hi})"
        assert rel <= 1e-5, f"synthetic gemv rel {rel}"
    res["steps"]["a_synthetic"] = a
    print(f"[qt] a: {len(a)} synthetic cases bit-exact + gemv <=1e-5",
          flush=True)

    # ---- b: real BLe s16 tensors ----
    man = json.load(open(os.path.join(ART, "manifest.json")))
    handles = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(ART, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    b = []
    mods = {}
    for nm in REAL_TENSORS:
        e = man[nm]
        assert e["codec"] == "s16", f"{nm} is {e['codec']}, want s16"
        assert e["len"] == expected_len("s16", e["shape"])
        buf = payload(e)
        ref = dec_s16(buf, e["shape"])
        mod = qt.FusedS16Linear(qt.S16Gpu(buf, e["shape"]))
        mods[nm] = (mod, ref)
        R, C = e["shape"]
        dec = qt.s16_decode_rows(mod.pay, 0, min(R, 4096)).cpu().numpy()
        bit = bool(np.array_equal(dec, ref[:min(R, 4096)]))
        x = np.random.default_rng(7).standard_normal(C) \
            .astype(np.float32)
        xt = torch.from_numpy(x).cuda()
        y = mod(xt.reshape(1, C))[0].cpu().numpy()
        yref = ref @ x
        rel = float(np.abs(y - yref).max() /
                    max(np.abs(yref).max(), 1e-30))
        b.append({"name": nm, "shape": e["shape"],
                  "decode_bit_exact": bit, "gemv_rel_max": rel})
        assert bit and rel <= 1e-5, (nm, bit, rel)
        print(f"[qt] b: {nm} {e['shape']} bit-exact, gemv rel "
              f"{rel:.2e}", flush=True)
    res["steps"]["b_real"] = b

    # ---- c: microbench dense-FP32 v fused s16 GEMV ----
    c = []
    for nm, (mod, ref) in mods.items():
        R, C = mod.pay.shape
        Wd = torch.from_numpy(ref).cuda()          # the abort's path
        x = torch.randn(C, device="cuda")
        for _ in range(3):
            mod.pay.gemv(x); Wd @ x
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            mod.pay.gemv(x)
        torch.cuda.synchronize()
        t_fused = (time.time() - t0) / 50
        t0 = time.time()
        for _ in range(50):
            Wd @ x
        torch.cuda.synchronize()
        t_dense = (time.time() - t0) / 50
        del Wd
        c.append({"name": nm, "fused_ms": t_fused * 1e3,
                  "dense_fp32_ms": t_dense * 1e3,
                  "dense_over_fused": t_dense / t_fused})
        print(f"[qt] c: {nm} fused {t_fused*1e3:.3f} ms v dense "
              f"{t_dense*1e3:.3f} ms", flush=True)
    res["steps"]["c_microbench"] = c

    res["completion_commit"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    res["art_dir"] = ART
    res["manifest_sha256"] = hashlib.sha256(
        open(os.path.join(ART, "manifest.json"), "rb").read()).hexdigest()
    with open(rcpt, "w") as f:
        f.write(json.dumps(res) + "\n")
    print(f"[qt] receipt -> {rcpt}", flush=True)


if __name__ == "__main__":
    main()

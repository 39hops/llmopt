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
# completion-commit deviation). r2: the literal status text and the
# blob shas of the critical runtime + driver files, not a bare bool.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fsha(rel):
    return hashlib.sha256(
        open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()


START = {"start_commit": subprocess.check_output(
             ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
         "start_status_porcelain": subprocess.check_output(
             ["git", "status", "--porcelain"]).decode(),
         "interpreter": sys.executable,
         "file_sha256": {p: _fsha(p) for p in (
             "llmopt/lab/qcuda.py", "llmopt/lab/qcuda_tower.py",
             "llmopt/lab/qcodec.py", "scratch/qcuda_tower_qualify.py")}}


def synth_case(R, C, exps, seed):
    """exps: explicit uint8 array (len n//128) — edge claims name
    their exponent exactly, never through an exclusive random bound
    (r1 wording bug: integers(180, 200) tops out at e=199 = 2^72)."""
    rng = np.random.default_rng(seed)
    n = R * C
    assert len(exps) == n // 128
    lv = (rng.standard_normal(16) * 0.1).astype(np.float16)
    codes = rng.integers(0, 256, n // 2, dtype=np.uint8)
    return (np.asarray(exps, np.uint8).tobytes()
            + lv.tobytes() + codes.tobytes())


def main():
    out_dir = "logs/qcudatower"
    os.makedirs(out_dir, exist_ok=True)
    rcpt = os.path.join(out_dir, "qualify_r2.json")
    if os.path.exists(rcpt):
        raise SystemExit(f"REFUSING: {rcpt} exists")
    assert torch.cuda.is_available() and qt.HAVE_TRITON
    res = {"start": START, "steps": {}}

    # ---- a: synthetic parity incl. EXPLICIT exp edges ----
    a = []
    rng0 = np.random.default_rng(99)
    cases = [
        ("typical", 8, 256,
         rng0.integers(120, 135, 8 * 256 // 128, dtype=np.uint8), 0),
        ("subnormal-e0", 16, 512,
         np.zeros(16 * 512 // 128, np.uint8), 1),         # exact e=0
        ("huge-e200", 8, 384,
         np.full(8 * 384 // 128, 200, np.uint8), 2),      # exact 2^73
    ]
    for (label, R, C, exps, sd) in cases:
        buf = synth_case(R, C, exps, sd)
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
        a.append({"case": label, "shape": [R, C],
                  "decode_bit_exact": bit, "gemv_rel_max": rel})
        assert bit, f"synthetic decode mismatch {label}"
        assert rel <= 1e-5, f"synthetic gemv rel {rel} ({label})"
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

    # ---- b2: the PREFILL branch through FusedS16Linear.forward ----
    # qkv [10240, 5120] crosses CHUNK=8192, so this exercises
    # s16_decode_rows chunking + concat ordering, which the [1,C]
    # rows above never touch.
    nm = REAL_TENSORS[0]
    mod, ref = mods[nm]
    R, C = mod.pay.shape
    assert R > qt.FusedS16Linear.CHUNK, (
        f"b2 needs a tensor crossing CHUNK; {nm} is {R}")
    X = np.random.default_rng(11).standard_normal((3, C)) \
        .astype(np.float32)
    Xt = torch.from_numpy(X).cuda()
    Y = mod(Xt).cpu().numpy()
    Yref = X @ ref.T
    rel2 = float(np.abs(Y - Yref).max() /
                 max(np.abs(Yref).max(), 1e-30))
    res["steps"]["b2_prefill"] = {
        "name": nm, "x_rows": 3, "chunks": -(-R // qt.FusedS16Linear.CHUNK),
        "rel_max": rel2}
    assert rel2 <= 1e-5, f"prefill branch rel {rel2}"
    print(f"[qt] b2: prefill {nm} X[3,{C}] over "
          f"{-(-R // qt.FusedS16Linear.CHUNK)} chunks rel {rel2:.2e}",
          flush=True)

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

    # manifest-derived route census (conservation preview for step d+:
    # counts derived, never hardcoded)
    want = qt.expected_compressed(man)
    census = {}
    for k, cod in want.items():
        census[cod] = census.get(cod, 0) + 1
    res["route_census"] = census
    print(f"[qt] route census (2D compressed): {census}", flush=True)

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

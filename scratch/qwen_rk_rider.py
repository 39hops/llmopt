"""QWEN-RK-CENSUS-0 descriptive rider (post-verdict, never a gate).

Two normalizations that separate intrinsic non-sparsity from
selection error, computed from the booked census receipt and the
frozen z captures:

- Q_recon = (E_random - E_arm) / (E_random - E_teacher_topz):
  arm-selected reconstruction placed on a 0 (random) to 1
  (teacher-top-|z| heuristic) scale. The denominator reference is
  a HEURISTIC selection, not the exact subset optimum.
- R_k_teacher: the teacher's own top-k |z| mass fraction (how much
  mass a perfect same-k selector could capture), so R_k (arm)
  normalizes as R_k / R_k_teacher, and random's expectation is
  k/C = 1024/17408 = 0.0588.

    .venv/bin/python scratch/qwen_rk_rider.py

Writes logs/qwenrouter/rk_rider.json (refuse-if-exists).
"""
import hashlib
import json
import os
import subprocess

import numpy as np

OUT = "logs/qwenrouter"
K = 1024


def main():
    rp = os.path.join(OUT, "rk_rider.json")
    if os.path.exists(rp):
        raise SystemExit(f"REFUSING: {rp} exists")
    cp = os.path.join(OUT, "rk_census.json")
    d = json.load(open(cp))
    meta = json.load(open(os.path.join(OUT, "capture_vendor_meta.json")))
    per = {}
    for li in d["layers"]:
        c = d["per_layer"][str(li)]["ks"][str(K)]
        zt = np.load(os.path.join(OUT, f"z_vendor_L{li}.npy"))
        zsha = hashlib.sha256(zt.tobytes()).hexdigest()
        rec = meta["z_sha256"][f"z_vendor_L{li}"]
        if zsha != rec:
            raise SystemExit(f"REFUSING: z_vendor_L{li} sha mismatch")
        a = np.abs(zt.astype(np.float64))
        topk = np.sort(a, axis=1)[:, -K:].sum(axis=1)
        r_teacher = float((topk / a.sum(axis=1)).mean())
        q = (c["recon_random"] - c["recon_arm"]) / (
            c["recon_random"] - c["recon_teacher_topz"])
        per[li] = {
            "r_k_arm": c["r_k"],
            "r_k_teacher_topk_selfmass": round(r_teacher, 4),
            "r_k_arm_over_teacher": round(c["r_k"] / r_teacher, 4),
            "r_k_random_expectation": round(K / zt.shape[1], 4),
            "q_recon_vs_topz_heuristic": round(q, 4)}
        print(f"[rider] L{li} r_teacher={r_teacher:.3f} "
              f"norm={per[li]['r_k_arm_over_teacher']:.3f} "
              f"Q_recon={q:.3f}", flush=True)
    rcpt = {"gate": "QWEN-RK-CENSUS-0 rider (descriptive)",
            "k": K, "per_layer": per,
            "census_receipt_sha256": hashlib.sha256(
                open(cp, "rb").read()).hexdigest(),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
            "tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain",
                 "-uno"]).decode().strip())}
    with open(rp, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[rider] -> {rp}", flush=True)


if __name__ == "__main__":
    main()

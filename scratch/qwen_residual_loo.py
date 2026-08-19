"""Leave-one-out family-pool cosine for RESIDUAL-STRUCTURE-0 (post-hoc
color registered in AMENDMENT -EXECUTION item 3; never feeds bar 5).

Rebuilds per-tensor code counts from arm A's w4 payloads (no vendor
read), then compares each saved conditional-DELTA table against its
family pool WITH THE TENSOR REMOVED.

    .venv/bin/python scratch/qwen_residual_loo.py
Receipt: logs/qwenresidual/loo_A.json (append-refused).
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
from llmopt.lab import qartifact  # noqa: E402
from llmopt.lab.qcodec import BLOCK, expected_len  # noqa: E402

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))


def main():
    t0 = time.time()
    out = "logs/qwenresidual/loo_A.json"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    chain = "logs/qwenwhole/artifact_digest_A.txt"
    q = qartifact.qualify_artifact(
        ART, os.path.join(VDIR, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    npz_path = "logs/qwenresidual/tables_A.npz"
    tables_sha = hashlib.sha256(open(npz_path, "rb").read()).hexdigest()
    z = np.load(npz_path)
    names = [k[len("tab_"):] for k in z.files if k.startswith("tab_")]

    handles, counts = {}, {}
    for nm in names:
        e = MAN[nm]
        assert e["codec"] == "w4", nm
        n = e["shape"][0] * e["shape"][1]
        nb = n // BLOCK
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(ART, sh + ".bin"), "rb")
        handles[sh].seek(e["off"] + nb + 2048)
        idx = np.frombuffer(handles[sh].read(n // 4), np.uint8)
        assert e["len"] == expected_len("w4", e["shape"])
        counts[nm] = np.bincount(idx, minlength=256).astype(np.int64)

    fam = lambda nm: ".".join(nm.split(".")[-3:-1])  # noqa: E731
    fams = sorted({fam(nm) for nm in names})
    num = {f: np.zeros((256, 4), np.float64) for f in fams}
    den = {f: np.zeros((256, 1), np.float64) for f in fams}
    for nm in names:
        num[fam(nm)] += z["tab_" + nm].astype(np.float64) \
            * counts[nm][:, None]
        den[fam(nm)] += counts[nm][:, None]

    loo = {}
    for nm in names:
        f = fam(nm)
        n2 = num[f] - z["tab_" + nm].astype(np.float64) * counts[nm][:, None]
        d2 = den[f] - counts[nm][:, None]
        if float(d2.sum()) == 0:      # singleton family (io tensors)
            loo[nm] = None
            continue
        pool = (n2 / np.maximum(d2, 1)).ravel()
        a = z["tab_" + nm].astype(np.float64).ravel()
        na, npn = float(np.linalg.norm(a)), float(np.linalg.norm(pool))
        loo[nm] = None if na == 0 or npn == 0 else \
            float(a @ pool) / (na * npn)

    vals = [v for v in loo.values() if v is not None]
    summ = {"arm": "A",
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
            "tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"]).decode().strip()),
            "qualification": q["report"],
            "chain_sha256": hashlib.sha256(
                open(chain, "rb").read()).hexdigest(),
            "tables_npz_sha256": tables_sha,
            "wall_s": round(time.time() - t0, 1),
            "n": len(names), "n_scored": len(vals),
            "n_singleton": sum(v is None for v in loo.values()),
            "median_loo_cosine": float(np.median(vals)),
            "loo_cosines": loo}
    with open(out, "w") as f:
        f.write(json.dumps(summ) + "\n")
    print(f"[loo] median {summ['median_loo_cosine']:.4f} "
          f"({summ['n_scored']} scored, {summ['n_singleton']} singleton) "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()

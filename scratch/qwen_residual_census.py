"""QWEN-RESIDUAL-STRUCTURE-0: weight-space census of R = W_vendor - decode(A).

Per w4 tensor of arm A (manifest consumed through llmopt.lab.qartifact,
decode through llmopt.lab.qcodec only):
  rel_l2            ||R||_F / ||W_vendor||_F
  cond-mean ceiling variance reduction from replacing codebook[idx]
                    with E[vendor_normalized | idx] (exact 256x4
                    conditional-mean table; riff level 1)
  sparse tail       energy fraction of the top 1% |R| entries
  table cosine      per-tensor conditional-DELTA table v its
                    family-pooled table (global-table learnability)
SVD (top-16/64/256 energy fractions of R; riff level 2) runs on the
fixed SVD_SAMPLE only.

STRUCTURE CENSUS ONLY: no quantity here is a capability score; the
weight-distance law forbids promoting any of it without X/K on a
held-out surface.

    ART_DIR=~/qwen_whole0t/A .venv/bin/python scratch/qwen_residual_census.py
    SMOKE=1 ...   # first 6 w4 tensors, *_smoke paths only

Receipt: logs/qwenresidual/census_A.json (+ tables_A.npz sidecar),
append-refused. Rows stream to census_rows_A.jsonl as tensors finish.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab import qartifact  # noqa: E402
from llmopt.lab.qcodec import BLOCK, dec_w4, expected_len  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SUF = "_smoke" if SMOKE else ""
ARM = "A"
ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
OUT = "logs/qwenresidual"
# fixed SVD sample: early/late layer per family (full-attn layers sit
# at 0,3,7,...; 4/40 are linear-band layers)
SVD_SAMPLE_SPEC = {
    "linear_attn.in_proj_qkv": (4, 40), "linear_attn.out_proj": (4, 40),
    "self_attn.q_proj": (3, 59), "self_attn.o_proj": (3, 59),
    "mlp.gate_proj": (4, 40), "mlp.down_proj": (4, 40)}


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def w4_parts(buf: bytes, shape):
    """Scales, codebook, and group indices of a w4 payload (layout
    frozen in llmopt.lab.qcodec's module docstring)."""
    n = shape[0] * shape[1]
    nb = n // BLOCK
    assert len(buf) == expected_len("w4", shape)
    exps = np.frombuffer(buf, np.uint8, nb, 0).astype(np.int32)
    scale = np.exp2(exps - 127).astype(np.float32)
    cb = np.frombuffer(buf, np.float16, 256 * 4, nb) \
        .reshape(256, 4).astype(np.float32)
    idx = np.frombuffer(buf, np.uint8, n // 4, nb + 2048)
    return scale, cb, idx


def cond_mean_stats(Wv: np.ndarray, buf: bytes, shape):
    """Level-1 ceiling: variance reduction from the exact per-code
    conditional mean of the normalized vendor groups, plus the
    conditional-DELTA table (E[vendor_n|idx] - codebook) used for the
    cross-layer cosine."""
    scale, cb, idx = w4_parts(buf, shape)
    n = shape[0] * shape[1]
    Vn = (Wv.reshape(n // BLOCK, BLOCK) / scale[:, None]).reshape(-1, 4)
    resid = Vn - cb[idx]                       # normalized residual
    var_before = float((resid ** 2).mean())
    sums = np.zeros((256, 4), np.float64)
    counts = np.zeros(256, np.int64)
    np.add.at(sums, idx, resid)
    np.add.at(counts, idx, 1)
    delta = np.where(counts[:, None] > 0,
                     sums / np.maximum(counts[:, None], 1), 0.0)
    var_after = float(((resid - delta[idx]) ** 2).mean())
    vr = 0.0 if var_before == 0 else 1.0 - var_after / var_before
    return vr, delta.astype(np.float32), counts


def tail_energy(R: np.ndarray, frac=0.01):
    a = (np.abs(R).ravel().astype(np.float64)) ** 2
    k = max(1, int(len(a) * frac))
    e = float(a.sum())
    return 0.0 if e == 0 else float(np.partition(a, -k)[-k:].sum()) / e


def svd_fracs(R: np.ndarray):
    s = np.linalg.svd(R.astype(np.float32), compute_uv=False)
    e = float((s ** 2).sum())
    return {f"top{k}": float((s[:k] ** 2).sum()) / e
            for k in (16, 64, 256)}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, f"census_{ARM}{SUF}.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    rows_path = os.path.join(OUT, f"census_rows_{ARM}{SUF}.jsonl")
    npz_path = os.path.join(OUT, f"tables_{ARM}{SUF}.npz")

    chain = f"logs/qwenwhole/artifact_digest_{ARM}.txt"
    q = qartifact.qualify_artifact(
        ART, os.path.join(VDIR, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    print(f"[rc] qualified {ARM}: {q['report']}", flush=True)

    from safetensors import safe_open
    idx_json = json.load(open(os.path.join(
        VDIR, "model.safetensors.index.json")))
    wmap = idx_json["weight_map"]

    w4_names = sorted(nm for nm, e in MAN.items() if e["codec"] == "w4")
    skipped = sorted({e["codec"] for e in MAN.values()} - {"w4"})
    if SMOKE:
        w4_names = w4_names[:6]
    print(f"[rc] {len(w4_names)} w4 tensors; skipped codecs {skipped}",
          flush=True)

    svd_want = {f"model.language_model.layers.{li}.{fam}.weight"
                for fam, lis in SVD_SAMPLE_SPEC.items() for li in lis}

    handles = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(ART, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    tables, tab_counts, fam_of = {}, {}, {}
    rows = []
    rf = open(rows_path, "w")
    for i, nm in enumerate(w4_names):
        e = MAN[nm]
        with safe_open(os.path.join(VDIR, wmap[nm]),
                       framework="pt", device="cpu") as h:
            Wv = h.get_tensor(nm).float().numpy()
        buf = payload(e)
        Wd = dec_w4(buf, e["shape"])
        R = Wv - Wd
        if not np.isfinite(R).all():
            raise SystemExit(f"REFUSING: non-finite residual {nm}")
        rel = float(np.linalg.norm(R) / np.linalg.norm(Wv))
        vr, delta, counts = cond_mean_stats(Wv, buf, e["shape"])
        tail = tail_energy(R)
        fam = ".".join(nm.split(".")[-3:-1])
        fam_of[nm] = fam
        tables[nm] = delta
        tab_counts[nm] = counts
        row = {"name": nm, "family": fam, "shape": e["shape"],
               "rel_l2": rel, "cond_mean_var_reduction": vr,
               "tail1pct_energy": tail}
        if nm in svd_want:
            row["svd"] = svd_fracs(R)
        rows.append(row)
        rf.write(json.dumps(row) + "\n")
        rf.flush()
        print(f"[rc] {i+1}/{len(w4_names)} {nm} rel {rel:.4f} "
              f"vr {vr:.4f} tail {tail:.4f}"
              + (f" svd {row.get('svd')}" if "svd" in row else ""),
              flush=True)
    rf.close()

    # cross-layer predictability: cosine v count-weighted family pool
    fams = sorted(set(fam_of.values()))
    pool = {}
    for f in fams:
        num = sum(tables[nm] * tab_counts[nm][:, None]
                  for nm in tables if fam_of[nm] == f)
        den = sum(tab_counts[nm][:, None] for nm in tables
                  if fam_of[nm] == f)
        pool[f] = (num / np.maximum(den, 1)).astype(np.float32)
    cosines = {}
    for nm, tab in tables.items():
        a = tab.ravel()
        b = pool[fam_of[nm]].ravel()
        na, nb_ = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        cosines[nm] = 0.0 if na == 0 or nb_ == 0 else \
            float(a @ b) / (na * nb_)
    np.savez_compressed(npz_path,
                        **{f"tab_{nm}": tables[nm] for nm in tables},
                        **{f"pool_{f}": pool[f] for f in fams})

    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    import subprocess
    summ = {
        "arm": ARM, "smoke": SMOKE, "n_w4": len(w4_names),
        "skipped_codecs": skipped,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        "tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"]).decode().strip()),
        "qualification": q["report"],
        "chain_sha256": fsha(chain),
        "median_rel_l2": med([r["rel_l2"] for r in rows]),
        "median_cond_mean_var_reduction":
            med([r["cond_mean_var_reduction"] for r in rows]),
        "median_tail1pct_energy": med([r["tail1pct_energy"] for r in rows]),
        "median_table_cosine": med(list(cosines.values())),
        "svd_sample": {r["name"]: r["svd"] for r in rows if "svd" in r},
        "table_cosines": cosines,
        "wall_s": round(time.time() - t0, 1)}
    for k in ("median_rel_l2", "median_cond_mean_var_reduction",
              "median_tail1pct_energy", "median_table_cosine"):
        if not np.isfinite(summ[k]):
            raise SystemExit(f"REFUSING: non-finite {k}")
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(summ) + "\n")
    print(f"[rc] receipt -> {rcpt_path} wall {summ['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

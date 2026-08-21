"""TOPSET-OVERLAP census (observation-only, gates nothing; the
banked spec of 2026-08-20, run per its own text): BLe v vendor
top-set geometry at the five frozen loop-state h, BEFORE any
alternate-token redesign is chosen. Reuses the frozen
qwen_alttok_derive machinery by import (never edited); logits
recomputed the identical chunked way.

Per locus, persisted primitives (the sufficient-statistics lesson —
this receipt makes the geometry machine-portable):
  - top-K recall and Jaccard between the two heads, K = 1..256
  - BOTH top-1024 rank lists with logits (ids_ble/logits_ble,
    ids_vendor/logits_vendor)
  - Spearman rank correlation over the union of the two top-1024s
  - the FIRST non-special BLe-ranked token outside vendor's
    top-256, and its logit gap to BLe top1 (the gap boundary the
    failed control ran into)
  - both recomputed argmaxes, flagged against the frozen tokens
  - exclusion attribution over BLe's top-256: how many are the
    frozen specials v vendor-top-256 v both v free

Receipt: logs/qwentopset/topset_census.json (refuse-if-exists).

    .venv/bin/python scratch/qwen_topset_census.py           (3080)
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

OUT = "logs/qwentopset"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def spearman(ra, rb):
    ra = np.asarray(ra, float)
    rb = np.asarray(rb, float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def main():
    ad = _load("qwen_alttok_derive", "scratch/qwen_alttok_derive.py")
    hs = _load("qwen_loop_state_headswap",
               "scratch/qwen_loop_state_headswap.py")
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "topset_census.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    START = start_provenance(
        ["scratch/qwen_topset_census.py",
         "scratch/qwen_alttok_derive.py",
         "scratch/qwen_loop_state_headswap.py", ad.HS_PARAMS],
        artifacts={"BLe": ad.ART, "vendor_slice": ad.VSLICE})
    inj = json.load(open(ad.HS_PARAMS))["injections"]
    rows = [json.loads(x) for x in
            open(os.path.join(ad.IN, "loopstate_rows.jsonl"))]
    byid = {r["id"]: r for r in rows}
    arrs, npz_shas = {}, {}
    for rid in sorted({j["item"] for j in inj}):
        arrs[rid], npz_shas[rid] = hs.load_npz(
            rid, byid[rid]["arrays_sha256"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ad.VDIR)
    specials = set(int(i) for i in tok.all_special_ids) \
        | set(ad.EOS_IDS)

    from safetensors import safe_open
    with safe_open(os.path.join(ad.VSLICE, "lm_head.safetensors"),
                   framework="pt", device="cpu") as f:
        Wv = f.get_tensor("lm_head.weight").float().numpy()
    Wb = hs.ble_head_rows()
    V = Wb.R

    H = []
    for j in inj:
        a = arrs[j["item"]]
        pos = {int(p): i for i, p in enumerate(a["positions"])}
        H.append(a["h"].astype(np.float32)[pos[j["pos"]]])
    H = np.stack(H)

    Z = np.empty((len(inj), V), np.float32)
    Zv = np.empty((len(inj), V), np.float32)
    for lo in range(0, V, 16384):
        hi = min(lo + 16384, V)
        Z[:, lo:hi] = H @ Wb.rows(lo, hi).T
        Zv[:, lo:hi] = H @ Wv[lo:hi].T

    loci = []
    for k, j in enumerate(inj):
        z, zv = Z[k], Zv[k]
        ob = np.argsort(-z, kind="stable")[:1024]
        ov = np.argsort(-zv, kind="stable")[:1024]
        rec = {}
        setb, setv = set(), set()
        for K in range(1, 257):
            setb.add(int(ob[K - 1]))
            setv.add(int(ov[K - 1]))
            inter = len(setb & setv)
            rec[K] = {"recall": round(inter / K, 4),
                      "jaccard": round(inter / len(setb | setv), 4)}
        keepK = {1, 2, 4, 8, 16, 32, 64, 128, 256}
        overlap_curve = {str(K): rec[K] for K in sorted(keepK)}
        union = sorted(set(int(i) for i in ob)
                       | set(int(i) for i in ov))
        rank_b = {int(t): r for r, t in enumerate(ob)}
        rank_v = {int(t): r for r, t in enumerate(ov)}
        ra = [rank_b.get(t, 1024) for t in union]
        rb = [rank_v.get(t, 1024) for t in union]
        rho = spearman(ra, rb)
        vtop256 = set(int(i) for i in ov[:256])
        first_out = None
        for r, t in enumerate(ob):
            t = int(t)
            if t not in vtop256 and t not in specials:
                first_out = {"token": t, "ble_rank": r,
                             "gap_to_ble_top1":
                                 round(float(z[ob[0]] - z[t]), 4)}
                break
        attr = {"specials_only": 0, "vtop256_only": 0,
                "both": 0, "free": 0}
        for t in (int(i) for i in ob[:256]):
            s, v256 = t in specials, t in vtop256
            key = ("both" if s and v256 else
                   "specials_only" if s else
                   "vtop256_only" if v256 else "free")
            attr[key] += 1
        loci.append({
            "item": j["item"], "pos": j["pos"],
            "frozen_ble_token": j["ble_token"],
            "frozen_vendor_token": j["vendor_token"],
            "ble_argmax": int(ob[0]), "vendor_argmax": int(ov[0]),
            "argmaxes_match_frozen":
                bool(int(ob[0]) == j["ble_token"]
                     and int(ov[0]) == j["vendor_token"]),
            "overlap_curve": overlap_curve,
            "spearman_rho_union_top1024": round(rho, 4),
            "first_ble_nonspecial_outside_vtop256": first_out,
            "ble_top256_attribution": attr,
            "ids_ble": [int(i) for i in ob],
            "logits_ble": [round(float(z[i]), 4) for i in ob],
            "ids_vendor": [int(i) for i in ov],
            "logits_vendor": [round(float(zv[i]), 4) for i in ov],
        })
        print(f"[ts] item{j['item']}@{j['pos']} "
              f"rec@8 {overlap_curve['8']['recall']} "
              f"rec@256 {overlap_curve['256']['recall']} "
              f"rho {rho:.3f} first_out {first_out}", flush=True)

    rcpt = {"note": "TOPSET-OVERLAP census (observation-only): BLe "
                    "v vendor top-set geometry at the five frozen "
                    "loop-state h, with portable top-1024 "
                    "sufficient statistics",
            "start": START, "completion_commit": completion_commit(),
            "input_npz_sha256": npz_shas,
            "n_specials": len(specials),
            "loci": loci}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[ts] receipt -> {rcpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

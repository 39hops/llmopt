"""QWEN-LOOP-STATE-1-HEADSWAP driver: vendor lm_head v BLe s16
lm_head on the IDENTICAL captured loop states of the booked
QWEN-LOOP-STATE-0 run (PRE-REG in docs/RESULTS.md; machine
projection docs/preregs/qwen-loop-state-1-headswap.json).

Offline, CPU-only, no tower residency. Precondition P1: the npz
input shas must equal the booked rows' arrays_sha256. The h are
BLe's own free-run states (off the vendor manifold): W_vendor @ h
isolates head representation conditional on the same damaged state
and is never teacher behavior.

    .venv/bin/python scratch/qwen_loop_state_headswap.py     (3080)
    SMOKE=1 ... (smoke path logs/qwenloopstate1_smoke, first 8
        positions per item only, P1 still enforced)

Receipts -> logs/qwenloopstate1/headswap_receipt.json +
headswap_observations.json (refuse-if-exists). The booked
logs/qwenloopstate/ receipts are read-only inputs here.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.lab.qcodec_fast import S16Rows  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
ART = os.path.expanduser(os.environ.get("ART_DIR",
                                        "~/qwen_whole0t/BLe"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR",
                                         "~/qwen_vendor"))
IN = "logs/qwenloopstate"
OUT = "logs/qwenloopstate1_smoke" if SMOKE else "logs/qwenloopstate1"
PARAMS = "docs/preregs/qwen-loop-state-0.params.json"
HOMOLOGOUS = {0: (88, range(1200, 1288)), 4: (242, range(1400, 1642))}


def load_npz(rid, booked_sha):
    path = os.path.join(IN, f"loopstate_arrays_id{rid}.npz")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != booked_sha:
        raise SystemExit(f"P1 FAIL: {path} sha {sha[:12]} != "
                         f"booked {booked_sha[:12]}")
    return np.load(path), sha


def vendor_head():
    """Vendor lm_head.weight, fp32 (the CHEAP-READOUT vendor-control
    load pattern)."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(
        VDIR, "model.safetensors.index.json")))
    # framework="pt" then .float(): the numpy framework cannot
    # represent bf16 shards
    import torch
    with safe_open(os.path.join(VDIR,
                                idx["weight_map"]["lm_head.weight"]),
                   framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").float().numpy()


def ble_head_rows():
    """BLe's s16 lm_head as a lazy row decoder from the artifact
    manifest (canonical qcodec_fast path)."""
    man = json.loads(open(os.path.join(ART, "manifest.json"),
                          "rb").read().decode())
    e = man["lm_head.weight"]
    fh = open(os.path.join(ART, e["shard"] + ".bin"), "rb")
    fh.seek(e["off"])
    return S16Rows(fh.read(e["len"]), e["shape"])


def vendor_top1(Wv, h):
    """argmax over the full vocab, chunked over head rows."""
    best_v = np.full(h.shape[0], -np.inf, np.float32)
    best_i = np.zeros(h.shape[0], np.int64)
    for lo in range(0, Wv.shape[0], 16384):
        z = h @ Wv[lo:lo + 16384].T
        m = z.max(1)
        upd = m > best_v
        best_v[upd] = m[upd]
        best_i[upd] = z.argmax(1)[upd] + lo
    return best_i


def main():
    pj = json.load(open(PARAMS))
    rows = [json.loads(x) for x in
            open(os.path.join(IN, "loopstate_rows.jsonl"))]
    byid = {r["id"]: r for r in rows}
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "headswap_receipt.json")
    obs_path = os.path.join(OUT, "headswap_observations.json")
    for p in (rcpt_path, obs_path):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/qwen_loop_state_headswap.py",
         "llmopt/lab/provenance.py", "llmopt/lab/qcodec_fast.py",
         PARAMS, "docs/preregs/qwen-loop-state-1-headswap.json"],
        artifacts={"BLe": ART, "vendor": VDIR})
    Wv = vendor_head()
    print(f"[hs] vendor head {Wv.shape}", flush=True)
    arrs, in_shas = {}, {}
    for rid in (0, 4, 3):
        arrs[rid], in_shas[rid] = load_npz(
            rid, byid[rid]["arrays_sha256"])
    # M1 + per-item vendor top1
    agree_n, total_n = 0, 0
    vtop = {}
    per_item = {}
    for rid in (0, 4, 3):
        a = arrs[rid]
        h = a["h"].astype(np.float32)
        if SMOKE:
            h = h[:8]
        vt = vendor_top1(Wv, h)
        bt = a["top256_ids"][:h.shape[0], 0].astype(np.int64)
        vtop[rid] = vt
        eq = int((vt == bt).sum())
        per_item[str(rid)] = {"n": int(h.shape[0]), "agree": eq}
        agree_n += eq
        total_n += h.shape[0]
        print(f"[hs] #{rid} vendor-v-BLe top1 {eq}/{h.shape[0]}",
              flush=True)
    m1 = agree_n / total_n
    # M2: vendor top1 agreement across item-3 attempt pairs
    m2 = None
    if not SMOKE:
        a = arrs[3]
        pos = {int(p): i for i, p in enumerate(a["positions"])}
        anchors = pj["capture"]["item3"]["anchor_positions"]
        w = pj["capture"]["item3"]["anchor_window"]
        vt = vtop[3]
        pairs = [(vt[pos[anchors[i] + o]] == vt[pos[anchors[i + 1]
                                                    + o]])
                 for i in range(len(anchors) - 1) for o in range(w)]
        m2 = float(np.mean(pairs))
        print(f"[hs] M2 vendor stuck-retry {m2:.4f} "
              f"({int(np.sum(pairs))}/{len(pairs)})", flush=True)
    # M3 descriptive: normalized boundary distance under BLe head
    m3 = {}
    if not SMOKE:
        S = ble_head_rows()

        def wrow(i):
            return S.rows(int(i), int(i) + 1)[0]

        for rid, (L, base) in HOMOLOGOUS.items():
            a = arrs[rid]
            pos = {int(p): i for i, p in enumerate(a["positions"])}
            h = a["h"].astype(np.float64)

            def cos(p, q):
                x, y = h[pos[p]], h[pos[q]]
                return float(x @ y / (np.linalg.norm(x)
                                      * np.linalg.norm(y)))

            cs1 = np.array([cos(p, p + L) for p in base])
            lo_mask = cs1 < 0.9
            d_ends = {"low": [], "rest": []}
            for j, p in enumerate(base):
                for q in (p, p + L):
                    k = pos[q]
                    t1, t2 = a["top256_ids"][k, 0], a["top256_ids"][k, 1]
                    z1, z2 = (float(a["top256_logits"][k, 0]),
                              float(a["top256_logits"][k, 1]))
                    dn = float(np.linalg.norm(wrow(t1) - wrow(t2)))
                    d = (z1 - z2) / dn
                    d_ends["low" if lo_mask[j] else "rest"].append(d)
            m3[f"item{rid}"] = {
                tag: {"n": len(v),
                      "median": float(np.median(v)),
                      "min": float(np.min(v))}
                for tag, v in d_ends.items() if v}
            print(f"[hs] #{rid} boundary-dist "
                  f"{json.dumps(m3[f'item{rid}'])}", flush=True)
    rcpt = {"start": START, "completion_commit": completion_commit(),
            "smoke": SMOKE, "input_npz_sha256":
            {str(k): v for k, v in in_shas.items()},
            "m1_total": {"agree": agree_n, "n": total_n,
                         "frac": m1},
            "m1_per_item": per_item, "m2": m2,
            "m3_boundary_distance_descriptive": m3}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    if SMOKE:
        print(f"[hs] smoke done m1={m1:.4f}", flush=True)
        return 0
    obs = {
        "note": "offline head swap on the booked loop-state npz; "
                "P1 sha equality enforced at load; h are off-manifold "
                "free-run states, no teacher-behavior claim",
        "measurement_valid": True,
        "arms": {"BLe": {"admissible": True,
                         "reason": f"npz shas match booked rows "
                                   f"({len(in_shas)}/3); artifact "
                                   "identity recorded for BLe+vendor"}},
        "measurements": {
            "1": {"value": m2,
                  "metric": "vendor_top1_agreement_attempt_pairs",
                  "population": "item:3 successive-attempt pairs x "
                                "64 offsets (4x64)",
                  "aggregation": "fraction",
                  "provenance": "vendor fp32 full-vocab argmax on "
                                "captured h; pairs per the frozen "
                                "anchor set"},
            "2": {"value": m1,
                  "metric": "vendor_v_ble_top1_agreement",
                  "population": "all captured positions (1409)",
                  "aggregation": "fraction",
                  "provenance": f"agree {agree_n}/{total_n}; BLe "
                                "top1 from fixture-certified stored "
                                "top256"}}}
    with open(obs_path, "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation,
                                   load as load_prereg)
    doc = load_prereg("docs/preregs/qwen-loop-state-1-headswap.json")
    outcomes = adjudicate_prereg(doc, obs)
    ref = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"bars": {o.bar_id: o.outcome
                               for o in outcomes},
                      "refutation": ref, "m1": m1, "m2": m2},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

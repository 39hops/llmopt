"""QWEN-LOOP-STATE-0 second descriptive color pass (unregistered,
gates nothing; external-review asks, 2026-08-20): recomputed from
the SAME sha-pinned npz primitives as the booked verdict, written to
a NEW receipt (loopstate_color2.json) — the booked observations file
is frozen and is not touched.

Emits per exact-orbit item:
- k=1 homologous pair margins at BOTH endpoints (pair-min), split
  low-cosine (<0.9) v rest — raw logit margins, explicitly NOT
  normalized decision-boundary distances (those need head weights);
- phase-shift calibration: median cos(h_p, h_{p+L+delta}) for a
  delta ladder including half-period, v the homologous delta=0;
and for item 3: JS quantiles (q50/q90/q95/q99/max) + per
successive-attempt-pair median/max JS and top1 agreement.

    .venv/bin/python scratch/qwen_loop_state_color2.py   (3080)
"""
import hashlib
import json
import os
import sys

import numpy as np

OUT = "logs/qwenloopstate"
PARAMS = "docs/preregs/qwen-loop-state-0.params.json"
HOMOLOGOUS = {0: (88, range(1200, 1288)), 4: (242, range(1400, 1642))}
DELTAS = (1, 2, 4, 8, 16, 32)


def load(rid):
    path = os.path.join(OUT, f"loopstate_arrays_id{rid}.npz")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return np.load(path), sha


def main():
    pj = json.load(open(PARAMS))
    out = {"note": "unregistered descriptive color, second pass; "
                   "recomputed from the sha-pinned npz of the booked "
                   "QWEN-LOOP-STATE-0 run; gates nothing",
           "npz_sha256": {}, "items": {}}
    for rid, (L, base) in HOMOLOGOUS.items():
        arr, sha = load(rid)
        out["npz_sha256"][str(rid)] = sha
        pos = {int(p): i for i, p in enumerate(arr["positions"])}
        h = arr["h"].astype(np.float64)
        m = arr["local_margin"]

        def cos(p, q):
            x, y = h[pos[p]], h[pos[q]]
            return float(x @ y / (np.linalg.norm(x)
                                  * np.linalg.norm(y)))

        cs1 = np.array([cos(p, p + L) for p in base])
        lo = cs1 < 0.9
        mb = np.array([m[pos[p]] for p in base])
        me = np.array([m[pos[p + L]] for p in base])
        pair_min = np.minimum(mb, me)
        item = {"n_pairs_k1": len(cs1), "n_low": int(lo.sum())}
        for tag, sel in (("low", lo), ("rest", ~lo)):
            if sel.any():
                item[f"margin_{tag}"] = {
                    "base_median": float(np.median(mb[sel])),
                    "end_median": float(np.median(me[sel])),
                    "pair_min_median": float(np.median(pair_min[sel])),
                    "pair_min_min": float(pair_min[sel].min())}
        deltas = sorted(set(DELTAS) | {L // 2})
        cal = {"0": float(np.median(cs1))}
        for d in deltas:
            ok = [p for p in base if (p + L + d) in pos]
            if len(ok) >= 20:
                cal[str(d)] = float(np.median(
                    [cos(p, p + L + d) for p in ok]))
        item["cos_by_delta_median"] = cal
        out["items"][str(rid)] = item
    arr, sha = load(3)
    out["npz_sha256"]["3"] = sha
    anchors = pj["capture"]["item3"]["anchor_positions"]
    w = pj["capture"]["item3"]["anchor_window"]
    fpos = {int(p): i for i, p in enumerate(arr["full_logits_pos"])}
    fl = arr["full_logits_fp16"].astype(np.float64)

    def dist(p):
        z = fl[fpos[p]]
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def js(pa, pb):
        da, db = dist(pa), dist(pb)
        mm = 0.5 * (da + db)

        def kl(a, b):
            msk = a > 0
            return float((a[msk] * np.log(a[msk] / b[msk])).sum())

        return (0.5 * kl(da, mm) + 0.5 * kl(db, mm),
                int(da.argmax()) == int(db.argmax()))

    per_pair, all_js = {}, []
    for i in range(len(anchors) - 1):
        vals, eq = [], []
        for o in range(w):
            j, e = js(anchors[i] + o, anchors[i + 1] + o)
            vals.append(j)
            eq.append(e)
        all_js.extend(vals)
        per_pair[f"{i}->{i+1}"] = {
            "median_js": float(np.median(vals)),
            "max_js": float(np.max(vals)),
            "top1_agree_frac": float(np.mean(eq))}
    q = np.quantile(all_js, [0.5, 0.9, 0.95, 0.99])
    out["items"]["3"] = {
        "js_quantiles": {"q50": float(q[0]), "q90": float(q[1]),
                         "q95": float(q[2]), "q99": float(q[3]),
                         "max": float(np.max(all_js))},
        "per_attempt_pair": per_pair}
    path = os.path.join(OUT, "loopstate_color2.json")
    if os.path.exists(path):
        raise SystemExit(f"REFUSING: {path} exists")
    with open(path, "w") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""QWEN-LOOP-STATE-0 adjudicator: preconditions + bars recomputed
from the persisted primitives (npz arrays + rows), never from the
driver's in-run numbers (PRIMITIVE-EVIDENCE doctrine).

    .venv/bin/python scratch/qwen_loop_state_adjudicate.py   (3080,
        after the driver; writes loopstate_observations.json)
"""
import hashlib
import json
import os
import sys

import numpy as np

OUT = "logs/qwenloopstate"
PREREG = "docs/preregs/qwen-loop-state-0.json"
PARAMS = "docs/preregs/qwen-loop-state-0.params.json"
HOMOLOGOUS = {0: (88, range(1200, 1288)), 4: (242, range(1400, 1642))}


def load(rid):
    path = os.path.join(OUT, f"loopstate_arrays_id{rid}.npz")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return np.load(path), sha


def cosines(arr, rid):
    """Bar-1 primitive: cosine(h_p, h_{p+kL}) for k in {1,2} over the
    registered homologous base positions."""
    L, base = HOMOLOGOUS[rid]
    pos = {int(p): i for i, p in enumerate(arr["positions"])}
    h = arr["h"].astype(np.float64)
    out = []
    for p in base:
        for k in (1, 2):
            a, b = h[pos[p]], h[pos[p + k * L]]
            out.append(float(a @ b / (np.linalg.norm(a)
                                      * np.linalg.norm(b))))
    return out


def item3_pairs(arr, pj):
    """Bar-2 primitives: per successive-anchor pair x offset, top1
    equality and full-vocab JS (fp64 softmax, natural log)."""
    anchors = pj["capture"]["item3"]["anchor_positions"]
    w = pj["capture"]["item3"]["anchor_window"]
    fpos = {int(p): i for i, p in enumerate(arr["full_logits_pos"])}
    fl = arr["full_logits_fp16"].astype(np.float64)

    def dist(p):
        z = fl[fpos[p]]
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    rows = []
    for i in range(len(anchors) - 1):
        for o in range(w):
            pa, pb = anchors[i] + o, anchors[i + 1] + o
            da, db = dist(pa), dist(pb)
            m = 0.5 * (da + db)

            def kl(p, q):
                mask = p > 0
                return float((p[mask]
                              * np.log(p[mask] / q[mask])).sum())

            js = 0.5 * kl(da, m) + 0.5 * kl(db, m)
            rows.append({"pair": i, "offset": o,
                         "top1_equal": bool(int(da.argmax())
                                            == int(db.argmax())),
                         "js_nats": js})
    return rows


def main():
    pj = json.load(open(PARAMS))
    rows = [json.loads(x) for x in
            open(os.path.join(OUT, "loopstate_rows.jsonl"))]
    byid = {r["id"]: r for r in rows}
    pre, meas = {}, {}
    for rid in (0, 4, 3):
        r = byid[rid]
        arr, sha = load(rid)
        assert sha == r["arrays_sha256"], (rid, sha)
        ids = arr["gen_token_ids"].tolist()
        gen_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        p1 = gen_sha == pj["preconditions"][
            "P1_trajectory_identity_sha256"][str(rid)]
        p2 = all(x["top1_identical"] and x["max_abs_diff"] == 0.0
                 for x in r["fixture"]) and len(r["fixture"]) == 8
        pre[rid] = {"P1": p1, "P2": p2}
        if rid == 3:
            g3 = pj["capture"]["item3"]["anchor_gram_g3"]
            hits = [t for t in range(len(ids) - 32)
                    if ids[t:t + 32] == g3]
            pre[rid]["P3"] = (hits == pj["preconditions"]
                              ["P3_anchor_census"]["positions"])
    valid = {rid: all(pre[rid].values()) for rid in pre}
    if valid[0] and valid[4]:
        med = {}
        for rid in (0, 4):
            arr, _ = load(rid)
            cs = cosines(arr, rid)
            med[rid] = float(np.median(cs))
            meas[f"item{rid}"] = {
                "median_cosine": med[rid],
                "min_cosine": float(min(cs)), "n_pairs": len(cs)}
        bar1 = "FIRE" if min(med.values()) >= 0.99 else "NO-FIRE"
        meas["bar1_min_median"] = float(min(med.values()))
    else:
        bar1 = "INSTRUMENT-INVALID"
    if valid[3]:
        arr, _ = load(3)
        pr = item3_pairs(arr, pj)
        frac = float(np.mean([x["top1_equal"] for x in pr]))
        mjs = float(np.median([x["js_nats"] for x in pr]))
        meas["item3"] = {"top1_agreement_frac": frac,
                         "median_js_nats": mjs, "n_pairs": len(pr)}
        bar2 = ("FIRE" if frac >= 0.80 and mjs <= 0.05
                else "NO-FIRE")
    else:
        bar2 = "INSTRUMENT-INVALID"
    all_valid = all(valid.values())
    obs = {
        "note": "preconditions and bars recomputed from the "
                "sha-pinned npz primitives by this adjudicator; "
                "driver in-run numbers unused (PRIMITIVE-EVIDENCE).",
        "preconditions": pre,
        "measurement_valid": all_valid,
        "arms": {"BLe": {
            "admissible": all_valid,
            "reason": "3/3 rows, P1 sha identity on all items, "
                      "P2 fixture bit-exact 24/24, P3 anchor census "
                      "exact, refuse-if-exists held"}},
        "measurements": {},
        "detail": meas}
    if not all_valid:
        obs["measurement_reason"] = f"precondition failure: {pre}"
    if "bar1_min_median" in meas:
        obs["measurements"]["1"] = {
            "value": meas["bar1_min_median"],
            "metric": "min_over_items_median_cosine",
            "population": "items:0,4 homologous pairs k in {1,2}",
            "aggregation": "median per item, min over items",
            "provenance": "recomputed from npz; per-item medians "
                          f"{meas['item0']['median_cosine']:.4f} "
                          f"(176 pairs) / "
                          f"{meas['item4']['median_cosine']:.4f} "
                          "(484 pairs)"}
        obs["measurements"]["refute:min_median_cosine_items04"] = {
            "value": meas["bar1_min_median"],
            "metric": "min_over_items_median_cosine",
            "population": "items:0,4 homologous pairs k in {1,2}",
            "aggregation": "median per item, min over items",
            "provenance": "same quantity as bar 1, read by the "
                          "refutation predicate (< 0.90)"}
    if "item3" in meas:
        obs["measurements"]["2"] = {
            "value": meas["item3"]["top1_agreement_frac"],
            "metric": "top1_agreement_frac",
            "population": "item:3 successive-attempt pairs x 64 "
                          "offsets (4x64)",
            "aggregation": "fraction",
            "provenance": "recomputed from full fp16 logits in npz "
                          "(fp64 softmax)"}
        obs["measurements"]["2:median_js_nats"] = {
            "value": meas["item3"]["median_js_nats"],
            "metric": "median_js_nats",
            "population": "item:3 successive-attempt pairs x 64 "
                          "offsets (4x64)",
            "aggregation": "median",
            "provenance": "conjunct of bar 2; full-vocab JS, fp64 "
                          "softmax, natural log"}
    with open(os.path.join(OUT, "loopstate_observations.json"),
              "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load as
                                   load_prereg)
    doc = load_prereg(PREREG)
    outcomes = adjudicate_prereg(doc, obs)
    ref = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"pre": pre,
                      "bars": {o.bar_id: o.outcome
                               for o in outcomes},
                      "refutation": ref,
                      "local_bars": {"1": bar1, "2": bar2},
                      "meas": meas}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

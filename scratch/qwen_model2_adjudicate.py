"""QWEN-MODEL2-ALLOC-1 observations builder + adjudication.

Held-out-surface receipts only (logs/qwenmodel2/score_{PX,PK,FLe,
C}.json). Bar 1 (sanity) = compose-admissibility violations over
PX/PK (reusing the LBAND fail-closed checks with base FLe) plus
cpu/traversal on all four receipts; bars 2-3 signed crossover
floor-multiples; bars 4-5 transport deviations in nats v the
registered F-conditioned marginals. Refutation predicate = min of
the two signed crossover multiples; precedence (bar 1, sanity) is
read from the registered JSON by the engine — this is the first
rung where alarm->refutation precedence executes from the
registration, not adjudicator code.

    .venv/bin/python scratch/qwen_model2_adjudicate.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

SC = "logs/qwenmodel2"
AT = "logs/qwenattrib"
WH = "logs/qwenwhole"
ARMS = ("PX", "PK", "FLe", "C")
BAND_BYTES = 461276672
BAND_LAYERS = {"PX": sorted(range(21, 42)), "PK": sorted(range(42, 63))}
# registered F-conditioned marginals (PRE-REG QWEN-MODEL2-ALLOC-1)
REG_DX_MID = 0.128
REG_DK_LATE = 0.0401


def _m(value, metric, population, aggregation, provenance=""):
    return {"value": float(value), "metric": metric,
            "population": population, "aggregation": aggregation,
            "provenance": provenance}


def _frozen_chain_sha(arm):
    p = os.path.join(WH, f"artifact_digest_{arm}.txt")
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def compose_violations(a, c, rc_a, frozen):
    reasons = []
    if c is None:
        return ["compose receipt missing"]
    if c.get("name") != a:
        reasons.append(f"compose name {c.get('name')}")
    rec = c.get("recipe", {})
    if rec.get("base") != "FLe":
        reasons.append(f"base {rec.get('base')} != FLe")
    if rec.get("donor") != "C":
        reasons.append(f"donor {rec.get('donor')} != C")
    if rec.get("mark") != ".linear_attn.":
        reasons.append(f"mark {rec.get('mark')}")
    if sorted(rec.get("layers", [])) != BAND_LAYERS[a]:
        reasons.append("layers != frozen band")
    if c.get("promoted_keys") != 48:
        reasons.append(f"promoted_keys {c.get('promoted_keys')}")
    if c.get("bytes_added") != BAND_BYTES:
        reasons.append(f"bytes_added {c.get('bytes_added')}")
    q = rc_a.get("qualification", {})
    if c.get("out_chain_sha256") != q.get("chain_sha256"):
        reasons.append("compose out_chain != score chain")
    for role, src in (("base", "FLe"), ("donor", "C")):
        want = frozen.get(src)
        got = c.get(role, {}).get("chain_sha256")
        if want is None or got != want:
            reasons.append(f"{role} chain != frozen {src} digest")
    return reasons


def build_observations(rc, comp):
    frozen = {s: _frozen_chain_sha(s) for s in ("FLe", "C")}
    viol = []
    for a in ("PX", "PK"):
        viol += [f"{a}: {r}" for r in
                 compose_violations(a, comp.get(a), rc[a], frozen)]
    for a in ARMS:
        r = rc[a]
        if r.get("smoke"):
            viol.append(f"{a}: smoke receipt")
        if r.get("device_actual") != "cpu":
            viol.append(f"{a}: device {r.get('device_actual')}")
        tv = r.get("traversal", {})
        if (tv.get("linear_attn"), tv.get("full_attn")) != (48, 16):
            viol.append(f"{a}: traversal {tv}")
    valid, vreason = True, ""
    for k in ("ce_teacher_nats", "f_X", "v_live"):
        if len({rc[a][k] for a in ARMS}) != 1:
            valid, vreason = False, f"{k} differs across receipts"
    for a in ARMS:
        if rc[a]["teacher"] != rc["FLe"]["teacher"]:
            valid, vreason = False, f"teacher identity differs: {a}"
    X = {a: rc[a]["X"] for a in ARMS}
    K = {a: rc[a]["K"] for a in ARMS}
    f_X = max(rc[a]["f_X"] for a in ARMS)
    f_K = max(rc[a]["f_K"] for a in ARMS)
    gx = (X["PK"] - X["PX"]) / f_X
    gk = (K["PX"] - K["PK"]) / f_K
    meas = {
        "1": _m(len(viol), "compose_violation_count",
                "arms:recomposed", "count",
                "; ".join(viol) if viol else "clean"),
        "2": _m(gx, "crossover_gap_floor_multiple_x",
                "positions:corpus-m2", "ratio",
                "(X_PK - X_PX)/f_X, signed"),
        "3": _m(gk, "crossover_gap_floor_multiple_k",
                "positions:prefixes-m2", "ratio",
                "(K_PX - K_PK)/f_K, signed"),
        "4": _m(abs((X["FLe"] - X["PX"]) - REG_DX_MID),
                "transport_dev_x_nats", "positions:corpus-m2",
                "mean", f"dX(PX|FLe) v registered {REG_DX_MID}"),
        "5": _m(abs((K["FLe"] - K["PK"]) - REG_DK_LATE),
                "transport_dev_k_nats", "positions:prefixes-m2",
                "mean", f"dK(PK|FLe) v registered {REG_DK_LATE}"),
        "refuted:crossover_min_signed_floor_multiple": _m(
            min(gx, gk), "crossover_min_signed_floor_multiple",
            "contrasts:crossover-m2", "ratio",
            "min over the corpus-X and prefix-K signed multiples "
            "(composite objective, not a single-surface read)"),
    }
    obs = {"measurement_valid": valid,
           "arms": {a: {"admissible": True} for a in
                    ("PX", "PK", "FLe", "C")},
           "measurements": meas, "contrasts": {},
           "X": X, "K": K, "f_X": f_X, "f_K": f_K,
           "dX_PX_given_FLe": X["FLe"] - X["PX"],
           "dX_PK_given_FLe": X["FLe"] - X["PK"],
           "dK_PX_given_FLe": K["FLe"] - K["PX"],
           "dK_PK_given_FLe": K["FLe"] - K["PK"]}
    if not valid:
        obs["measurement_reason"] = vreason
    return obs


def main():
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    prereg = load("docs/preregs/qwen-model2-alloc-1.json")
    rc, comp = {}, {}
    for a in ARMS:
        p = os.path.join(SC, f"score_{a}.json")
        if not os.path.exists(p):
            raise SystemExit(f"REFUSING: missing receipt {p}")
        rc[a] = json.load(open(p))
        if rc[a].get("smoke"):
            raise SystemExit(f"REFUSING: smoke receipt {p}")
        # consumer-side nonfinite refusal, independent of the
        # scorer's write-time invariant
        import math
        for k in ("X", "K", "ce_teacher_nats"):
            if not math.isfinite(rc[a][k]):
                raise SystemExit(f"REFUSING: non-finite {k} in {p}")
        for k in ("f_X", "f_K"):
            if not (math.isfinite(rc[a][k]) and rc[a][k] > 0):
                raise SystemExit(f"REFUSING: bad floor {k} in {p}")
    for a in ("PX", "PK"):
        p = os.path.join(AT, f"compose_{a}.json")
        comp[a] = json.load(open(p)) if os.path.exists(p) else None
    obs = build_observations(rc, comp)
    obs["provenance"] = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        "tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"]).decode().strip()),
        "receipt_sha256": {
            a: hashlib.sha256(open(os.path.join(
                SC, f"score_{a}.json"), "rb").read()).hexdigest()
            for a in ARMS}}
    outs = adjudicate_prereg(prereg, obs)
    lines = []
    for o in outs:
        line = f"BAR {o.bar_id} {o.bar_name}: {o.outcome}"
        if o.reasons:
            line += " [" + "; ".join(o.reasons) + "]"
        lines.append(line)
        print(line, flush=True)
    ref = adjudicate_refutation(prereg, obs, bar_outcomes=outs)
    lines.append(f"REGISTERED-PRIOR(crossover-transport): {ref}")
    lines.append(f"dX|FLe: PX {obs['dX_PX_given_FLe']:.5f} "
                 f"PK {obs['dX_PK_given_FLe']:.5f}  "
                 f"dK|FLe: PX {obs['dK_PX_given_FLe']:.5f} "
                 f"PK {obs['dK_PK_given_FLe']:.5f}")
    lines.append(f"PRODUCER {obs['provenance']['code_commit']}"
                 f" dirty={obs['provenance']['tree_dirty']}")
    for line in lines[-3:]:
        print(line, flush=True)
    with open(os.path.join(AT, "model2_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    with open(os.path.join(AT, "model2_verdict.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[m2] -> {AT}/model2_verdict.txt", flush=True)


if __name__ == "__main__":
    main()

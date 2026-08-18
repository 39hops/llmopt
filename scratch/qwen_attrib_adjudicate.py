"""QWEN-ATTN-ATTRIB-1 observations builder + deterministic reading.

Consumes the frozen B/C scorer receipts plus the new F/L/Q ones
(same scorer, scratch/qwen_model1_score.py), emits observations for
docs/preregs/qwen-attn-attrib-1.json, adjudicates, and applies the
REGISTERED resolution rule with zero discretion:

  L-dominance   iff bars 2-5 all FIRE
  F-dominance   iff bars 2 and 4 both read below zero while the
                floor bars 3 and 5 FIRE
  MIXED/UNRESOLVED otherwise (signs quoted)
  bar 1 miss -> INSTRUMENT-ALARM (recomposer), no attribution claim

Also books report-only extras: U_X/U_K per byte (from the compose
receipts' bytes_added) and the raw recovery fractions.

    .venv/bin/python scratch/qwen_attrib_adjudicate.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

SC = "logs/qwenmodel1"
AT = "logs/qwenattrib"
GiB = 2 ** 30


def _m(value, metric, population, aggregation, provenance=""):
    return {"value": float(value), "metric": metric,
            "population": population, "aggregation": aggregation,
            "provenance": provenance}


def build_observations(rc: dict, comp: dict) -> dict:
    """rc: {arm: score receipt} for B,C,F,L,Q; comp: {arm: compose
    receipt} for F,L,Q. Pure, no I/O."""
    arms = {}
    for a in ("F", "L", "Q"):
        r = rc[a]
        reasons = []
        if r.get("smoke"):
            reasons.append("smoke receipt")
        if r.get("device_actual") != "cpu":
            reasons.append(f"device {r.get('device_actual')}")
        tv = r.get("traversal", {})
        if (tv.get("linear_attn"), tv.get("full_attn")) != (48, 16):
            reasons.append(f"traversal {tv}")
        c = comp.get(a)
        if c is None:
            reasons.append("no compose receipt")
        elif c.get("name") != a:
            reasons.append("compose receipt arm mismatch")
        arms[a] = {"admissible": not reasons}
        if reasons:
            arms[a]["reason"] = "; ".join(reasons)

    valid, vreason = True, ""
    base_t = rc["B"]["teacher"]
    for a in rc:
        if rc[a]["teacher"] != base_t:
            valid, vreason = False, f"teacher identity differs: {a}"
    for k in ("ce_teacher_nats", "f_X", "v_live"):
        if len({rc[a][k] for a in rc}) != 1:
            valid, vreason = False, f"{k} differs across receipts"

    X = {a: rc[a]["X"] for a in rc}
    K = {a: rc[a]["K"] for a in rc}
    f_X = max(rc[a]["f_X"] for a in rc)
    f_K = max(rc[a]["f_K"] for a in rc)
    meas, contrasts = {}, {}

    viol = 0
    for v, lo, hi, fl in (
            (X["F"], X["C"], X["B"], f_X), (X["L"], X["C"], X["B"], f_X),
            (K["F"], K["C"], K["B"], f_K), (K["L"], K["C"], K["B"], f_K)):
        if not (lo - 5 * fl <= v <= hi + 5 * fl):
            viol += 1
    meas["1"] = _m(viol, "bracket_violation_count", "arms:recomposed",
                   "count", "X/K of F,L v [C-5f, B+5f]")

    dX = X["B"] - X["C"]
    dK = K["B"] - K["C"]
    if dX <= 0:
        for b in ("2", "3", "6", "10", "11"):
            contrasts[b] = {"admissible": False,
                            "reason": f"degenerate: X_B - X_C = {dX}"}
    else:
        RXF = (X["B"] - X["F"]) / dX
        RXL = (X["B"] - X["L"]) / dX
        meas["2"] = _m(RXL - RXF, "recovery_fraction_gap_x",
                       "positions:corpus", "mean", "R_X(L)-R_X(F)")
        meas["3"] = _m(abs(X["F"] - X["L"]) / f_X,
                       "x_gap_floor_multiple", "positions:corpus",
                       "ratio", "")
        meas["6"] = _m(abs(RXF + RXL - 1), "interaction_abs_x",
                       "positions:corpus", "mean", "")
        meas["10"] = _m((X["Q"] - X["B"]) / f_X,
                        "iso_x_gap_floor_multiple_io",
                        "positions:corpus", "ratio", "")
        meas["11"] = _m((X["B"] - X["Q"]) / f_X,
                        "iso_x_gap_floor_multiple_attn",
                        "positions:corpus", "ratio", "")
    if dK <= 0:
        for b in ("4", "5", "7", "8", "9"):
            contrasts[b] = {"admissible": False,
                            "reason": f"degenerate: K_B - K_C = {dK}"}
    else:
        RKF = (K["B"] - K["F"]) / dK
        RKL = (K["B"] - K["L"]) / dK
        meas["4"] = _m(RKL - RKF, "recovery_fraction_gap_k",
                       "positions:prefixes", "mean", "R_K(L)-R_K(F)")
        meas["5"] = _m(abs(K["F"] - K["L"]) / f_K,
                       "k_gap_floor_multiple", "positions:prefixes",
                       "ratio", "")
        meas["7"] = _m(abs(RKF + RKL - 1), "interaction_abs_k",
                       "positions:prefixes", "mean", "")
        meas["8"] = _m((K["Q"] - K["B"]) / f_K,
                       "iso_k_gap_floor_multiple_io",
                       "positions:prefixes", "ratio", "")
        meas["9"] = _m((K["B"] - K["Q"]) / f_K,
                       "iso_k_gap_floor_multiple_attn",
                       "positions:prefixes", "ratio", "")

    extras = {}
    for a in ("F", "L"):
        gb = comp[a]["bytes_added"] / GiB if comp.get(a) else None
        if gb:
            extras[a] = {"gib_added": round(gb, 4),
                         "U_X_nat_per_gib": (X["B"] - X[a]) / gb,
                         "U_K_nat_per_gib": (K["B"] - K[a]) / gb}
    obs = {"measurement_valid": valid, "arms": arms,
           "measurements": meas, "contrasts": contrasts,
           "X": X, "K": K, "f_X": f_X, "f_K": f_K,
           "per_byte": extras}
    if not valid:
        obs["measurement_reason"] = vreason
    return obs


def resolution(outcomes: dict, meas: dict) -> tuple:
    if outcomes[1] != "FIRE":
        return ("INSTRUMENT-ALARM",
                "bracket violation — recomposer alarm, no claim")
    st = {i: outcomes[i] for i in (2, 3, 4, 5)}
    if any(s == "UNRESOLVED" for s in st.values()):
        return ("UNRESOLVED", "dominance bars unresolved")
    if all(s == "FIRE" for s in st.values()):
        return ("L-DOMINANT", "bars 2-5 all fire")
    g2 = meas.get("2", {}).get("value")
    g4 = meas.get("4", {}).get("value")
    if (g2 is not None and g4 is not None and g2 < 0 and g4 < 0
            and outcomes[3] == "FIRE" and outcomes[5] == "FIRE"):
        return ("F-DOMINANT", "both gaps below zero past floors")
    return ("MIXED/UNRESOLVED",
            f"signs: gap_x={g2}, gap_k={g4}, floors "
            f"3={outcomes[3]} 5={outcomes[5]}")


def main():
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    prereg = load("docs/preregs/qwen-attn-attrib-1.json")
    rc, comp = {}, {}
    for a in ("B", "C", "F", "L", "Q"):
        p = os.path.join(SC, f"score_{a}.json")
        if not os.path.exists(p):
            raise SystemExit(f"REFUSING: missing receipt {p}")
        rc[a] = json.load(open(p))
        if rc[a].get("smoke"):
            raise SystemExit(f"REFUSING: smoke receipt {p}")
    for a in ("F", "L", "Q"):
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
            for a in rc}}
    outcomes, lines = {}, []
    for o in adjudicate_prereg(prereg, obs):
        outcomes[o.bar_id] = o.outcome
        line = f"BAR {o.bar_id} {o.bar_name}: {o.outcome}"
        if o.reasons:
            line += " [" + "; ".join(o.reasons) + "]"
        lines.append(line)
        print(line, flush=True)
    res, why = resolution(outcomes, obs["measurements"])
    # refutation predicate emits only when the floor resolves the gap
    if res not in ("INSTRUMENT-ALARM",) and \
            obs["measurements"].get("3", {}).get("value", 0) > 5:
        obs["measurements"]["refuted:recovery_fraction_gap_x_floored"] \
            = _m(obs["measurements"]["2"]["value"],
                 "recovery_fraction_gap_x_floored", "positions:corpus",
                 "mean", "emitted only past the 5 f_X floor")
        ref = adjudicate_refutation(prereg, obs)
    else:
        ref = "UNADJUDICATED (alarm or gap inside floor)"
    lines.append(f"RESOLUTION {res}: {why}")
    lines.append(f"REGISTERED-PRIOR(i): {ref}")
    lines.append(f"PRODUCER {obs['provenance']['code_commit']}"
                 f" dirty={obs['provenance']['tree_dirty']}")
    for line in lines[-3:]:
        print(line, flush=True)
    os.makedirs(AT, exist_ok=True)
    with open(os.path.join(AT, "attrib_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    with open(os.path.join(AT, "attrib_verdict.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[attrib] -> {AT}/attrib_verdict.txt", flush=True)


if __name__ == "__main__":
    main()

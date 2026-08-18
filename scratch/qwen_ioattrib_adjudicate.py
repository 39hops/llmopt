"""QWEN-IO-ATTRIB-1 observations builder + registered resolution.

Mirror of scratch/qwen_attrib_adjudicate.py on the A->B axis:
rec_X(Y) = (X_A - X_Y)/(X_A - X_B), Y in {D, E}; floors
per-contrast max over {A, B, D, E}. Resolution rule (frozen in the
prose pre-reg): D-dominance iff bars 2-5 all FIRE; E-dominance iff
both gaps read below zero while floor bars fire; else
MIXED/UNRESOLVED; bar-1 miss -> INSTRUMENT-ALARM (recomposer).

    .venv/bin/python scratch/qwen_ioattrib_adjudicate.py
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


def _m(value, metric, population, aggregation, provenance=""):
    return {"value": float(value), "metric": metric,
            "population": population, "aggregation": aggregation,
            "provenance": provenance}


def build_observations(rc: dict, comp: dict) -> dict:
    arms = {}
    for a in ("D", "E"):
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
        if c is None or c.get("name") != a:
            reasons.append("compose receipt missing/mismatched")
        arms[a] = {"admissible": not reasons}
        if reasons:
            arms[a]["reason"] = "; ".join(reasons)
    valid, vreason = True, ""
    base_t = rc["A"]["teacher"]
    for a in rc:
        if rc[a]["teacher"] != base_t:
            valid, vreason = False, f"teacher identity differs: {a}"
    for k in ("ce_teacher_nats", "f_X", "v_live"):
        if len({rc[a][k] for a in rc}) != 1:
            valid, vreason = False, f"{k} differs across receipts"
    X = {a: rc[a]["X"] for a in rc}
    K = {a: rc[a]["K"] for a in rc}
    f_X = max(rc[a]["f_X"] for a in ("A", "B", "D", "E"))
    f_K = max(rc[a]["f_K"] for a in ("A", "B", "D", "E"))
    meas, contrasts = {}, {}
    viol = 0
    for v, lo, hi, fl in (
            (X["D"], X["B"], X["A"], f_X), (X["E"], X["B"], X["A"], f_X),
            (K["D"], K["B"], K["A"], f_K), (K["E"], K["B"], K["A"], f_K)):
        if not (lo - 5 * fl <= v <= hi + 5 * fl):
            viol += 1
    meas["1"] = _m(viol, "bracket_violation_count", "arms:recomposed",
                   "count", "D/E X and K v [B-5f, A+5f]")
    dX = X["A"] - X["B"]
    dK = K["A"] - K["B"]
    if dX <= 0:
        for b in ("2", "3", "6"):
            contrasts[b] = {"admissible": False,
                            "reason": f"degenerate: X_A - X_B = {dX}"}
    else:
        RD = (X["A"] - X["D"]) / dX
        RE = (X["A"] - X["E"]) / dX
        meas["2"] = _m(RD - RE, "recovery_fraction_gap_x",
                       "positions:corpus", "mean",
                       "rec_X(D) - rec_X(E), A->B axis")
        meas["3"] = _m(abs(X["D"] - X["E"]) / f_X,
                       "x_gap_floor_multiple", "positions:corpus",
                       "ratio", "")
        meas["6"] = _m(abs(RD + RE - 1), "interaction_abs_x",
                       "positions:corpus", "mean", "")
    if dK <= 0:
        for b in ("4", "5", "7"):
            contrasts[b] = {"admissible": False,
                            "reason": f"degenerate: K_A - K_B = {dK}"}
    else:
        RDk = (K["A"] - K["D"]) / dK
        REk = (K["A"] - K["E"]) / dK
        meas["4"] = _m(RDk - REk, "recovery_fraction_gap_k",
                       "positions:prefixes", "mean", "")
        meas["5"] = _m(abs(K["D"] - K["E"]) / f_K,
                       "k_gap_floor_multiple", "positions:prefixes",
                       "ratio", "")
        meas["7"] = _m(abs(RDk + REk - 1), "interaction_abs_k",
                       "positions:prefixes", "mean", "")
    obs = {"measurement_valid": valid, "arms": arms,
           "measurements": meas, "contrasts": contrasts,
           "X": X, "K": K, "f_X": f_X, "f_K": f_K}
    if not valid:
        obs["measurement_reason"] = vreason
    return obs


def resolution(outcomes: dict, meas: dict) -> tuple:
    if outcomes[1] != "FIRE":
        return ("INSTRUMENT-ALARM", "bracket violation")
    st = {i: outcomes[i] for i in (2, 3, 4, 5)}
    if any(s == "UNRESOLVED" for s in st.values()):
        return ("UNRESOLVED", "dominance bars unresolved")
    if all(s == "FIRE" for s in st.values()):
        return ("D-DOMINANT", "bars 2-5 all fire")
    g2 = meas.get("2", {}).get("value")
    g4 = meas.get("4", {}).get("value")
    if (g2 is not None and g4 is not None and g2 < 0 and g4 < 0
            and outcomes[3] == "FIRE" and outcomes[5] == "FIRE"):
        return ("E-DOMINANT", "both gaps below zero past floors")
    return ("MIXED/UNRESOLVED",
            f"signs: gap_x={g2}, gap_k={g4}")


def main():
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    prereg = load("docs/preregs/qwen-io-attrib-1.json")
    rc, comp = {}, {}
    for a in ("A", "B", "D", "E"):
        p = os.path.join(SC, f"score_{a}.json")
        if not os.path.exists(p):
            raise SystemExit(f"REFUSING: missing receipt {p}")
        rc[a] = json.load(open(p))
        if rc[a].get("smoke"):
            raise SystemExit(f"REFUSING: smoke receipt {p}")
    for a in ("D", "E"):
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
    if res != "INSTRUMENT-ALARM" and \
            obs["measurements"].get("3", {}).get("value", 0) > 5:
        obs["measurements"]["refuted:recovery_fraction_gap_x_floored"] \
            = _m(obs["measurements"]["2"]["value"],
                 "recovery_fraction_gap_x_floored", "positions:corpus",
                 "mean", "emitted only past the 5 f_X floor")
        ref = adjudicate_refutation(prereg, obs)
    else:
        ref = "UNADJUDICATED (alarm or gap inside floor)"
    lines.append(f"RESOLUTION {res}: {why}")
    lines.append(f"REGISTERED-PRIOR(D>E): {ref}")
    lines.append(f"PRODUCER {obs['provenance']['code_commit']}"
                 f" dirty={obs['provenance']['tree_dirty']}")
    for line in lines[-3:]:
        print(line, flush=True)
    with open(os.path.join(AT, "ioattrib_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    with open(os.path.join(AT, "ioattrib_verdict.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[io] -> {AT}/ioattrib_verdict.txt", flush=True)


if __name__ == "__main__":
    main()

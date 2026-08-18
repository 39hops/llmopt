"""QWEN-LBAND-1 observations builder + registered resolution.

Six arms (BLe/BLm/BLl from base B, FLe/FLm/FLl from base F), band
recovery on the B->C axis. Floor-bar semantics pinned in the
machine projection: every *_floor_multiple metric is a RAW X-or-K
gap in nats over the per-contrast floor (max over the nine
receipts in play); the ONE rec-unit consumer is the refutation
predicate (best-minus-second from base B in rec_X units).
Resolution is bar-wise: bar 1 miss -> INSTRUMENT-ALARM; otherwise
the reading is the conjunction of STRUCTURE (bars 2-4, 7) and
CONDITIONING (bars 5-6) outcomes, no discretion.

    .venv/bin/python scratch/qwen_lband_adjudicate.py
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
WH = "logs/qwenwhole"
B_ARMS = ("BLe", "BLm", "BLl")
F_ARMS = ("FLe", "FLm", "FLl")
BASES = {"BLe": "B", "BLm": "B", "BLl": "B",
         "FLe": "F", "FLm": "F", "FLl": "F"}
ALL = ("B", "C", "F") + B_ARMS + F_ARMS
# frozen by PRE-REG QWEN-LBAND-1: 16/16/16 ascending linear-attn split
BAND_LAYERS = {"e": sorted(range(0, 21)), "m": sorted(range(21, 42)),
               "l": sorted(range(42, 63))}


def _frozen_chain_sha(arm: str):
    """Chain identity of a frozen source arm, derived from its
    committed digest file — never a literal."""
    p = os.path.join(WH, f"artifact_digest_{arm}.txt")
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def compose_admissibility(a, c, rc_a, frozen) -> list:
    """Fail-closed checks that compose_<a>.json records the exact
    registered treatment. Returns a list of failure reasons."""
    reasons = []
    if c is None:
        return ["compose receipt missing"]
    if c.get("name") != a:
        reasons.append(f"compose name {c.get('name')}")
    rec = c.get("recipe", {})
    if rec.get("base") != BASES[a]:
        reasons.append(f"recipe base {rec.get('base')} != {BASES[a]}")
    if rec.get("donor") != "C":
        reasons.append(f"recipe donor {rec.get('donor')} != C")
    if rec.get("mark") != ".linear_attn.":
        reasons.append(f"recipe mark {rec.get('mark')}")
    want_layers = BAND_LAYERS[a[-1]]
    if sorted(rec.get("layers", [])) != want_layers:
        reasons.append(f"recipe layers != frozen band {a[-1]}")
    if c.get("promoted_keys") != 48:
        reasons.append(f"promoted_keys {c.get('promoted_keys')} != 48")
    q = rc_a.get("qualification", {})
    if c.get("out_chain_sha256") != q.get("chain_sha256"):
        reasons.append("compose out_chain != score qualification chain")
    for role, src in (("base", BASES[a]), ("donor", "C")):
        want = frozen.get(src)
        got = c.get(role, {}).get("chain_sha256")
        if want is None:
            reasons.append(f"no frozen digest for {src}")
        elif got != want:
            reasons.append(f"{role} chain != frozen {src} digest")
    return reasons


def _m(value, metric, population, aggregation, provenance=""):
    return {"value": float(value), "metric": metric,
            "population": population, "aggregation": aggregation,
            "provenance": provenance}


def build_observations(rc: dict, comp: dict) -> dict:
    arms = {}
    frozen = {s: _frozen_chain_sha(s) for s in ("B", "C", "F")}
    bytes_added = {a: (comp.get(a) or {}).get("bytes_added")
                   for a in B_ARMS + F_ARMS}
    iso = len(set(bytes_added.values())) == 1 \
        and None not in bytes_added.values()
    for a in B_ARMS + F_ARMS:
        r = rc[a]
        reasons = []
        if r.get("smoke"):
            reasons.append("smoke receipt")
        if r.get("device_actual") != "cpu":
            reasons.append(f"device {r.get('device_actual')}")
        tv = r.get("traversal", {})
        if (tv.get("linear_attn"), tv.get("full_attn")) != (48, 16):
            reasons.append(f"traversal {tv}")
        reasons += compose_admissibility(a, comp.get(a), r, frozen)
        if not iso:
            reasons.append(f"bytes_added not equal across six: "
                           f"{bytes_added}")
        arms[a] = {"admissible": not reasons}
        if reasons:
            arms[a]["reason"] = "; ".join(reasons)
    valid, vreason = True, ""
    base_t = rc["B"]["teacher"]
    for a in rc:
        if rc[a]["teacher"] != base_t:
            valid, vreason = False, f"teacher identity differs: {a}"
    for key in ("ce_teacher_nats", "f_X", "v_live"):
        if len({rc[a][key] for a in rc}) != 1:
            valid, vreason = False, f"{key} differs across receipts"
    X = {a: rc[a]["X"] for a in rc}
    K = {a: rc[a]["K"] for a in rc}
    f_X = max(rc[a]["f_X"] for a in ALL)
    f_K = max(rc[a]["f_K"] for a in ALL)
    meas, contrasts = {}, {}
    viol = 0
    for a in B_ARMS + F_ARMS:
        base = BASES[a]
        for v, lo, hi, fl in ((X[a], X["C"], X[base], f_X),
                              (K[a], K["C"], K[base], f_K)):
            if not (lo - 5 * fl <= v <= hi + 5 * fl):
                viol += 1
    meas["1"] = _m(viol, "bracket_violation_count", "arms:recomposed",
                   "count", "six arms' X and K v [C-5f, base+5f]")
    # marginal band values (raw nats): dX(band|base) = X_base - X_arm
    dX = {a: X[BASES[a]] - X[a] for a in B_ARMS + F_ARMS}
    dK = {a: K[BASES[a]] - K[a] for a in B_ARMS + F_ARMS}
    # signed conditioning terms (descriptive; registered bars stay
    # absolute): I_b = dX(band|F) - dX(band|B). Negative = the band
    # buys less on top of F (redundant with F); positive = synergy.
    I_X = {b: dX[f"FL{b}"] - dX[f"BL{b}"] for b in "eml"}
    I_K = {b: dK[f"FL{b}"] - dK[f"BL{b}"] for b in "eml"}
    obs_extra = {"dX": dX, "dK": dK,
                 "conditioning_signed_x": I_X,
                 "conditioning_signed_k": I_K,
                 "bytes_added_per_arm": bytes_added}

    def band_gap(vals, arm_set):
        s = sorted((vals[a] for a in arm_set), reverse=True)
        return s[0] - s[1]

    meas["2"] = _m(band_gap(dX, B_ARMS) / f_X,
                   "band_gap_floor_multiple_x_baseB",
                   "positions:corpus", "ratio",
                   "(best - second) band dX from B, over f_X")
    meas["3"] = _m(band_gap(dX, F_ARMS) / f_X,
                   "band_gap_floor_multiple_x_baseF",
                   "positions:corpus", "ratio", "")
    meas["4"] = _m(band_gap(dK, B_ARMS) / f_K,
                   "band_gap_floor_multiple_k_baseB",
                   "positions:prefixes", "ratio", "")
    meas["7"] = _m(band_gap(dK, F_ARMS) / f_K,
                   "band_gap_floor_multiple_k_baseF",
                   "positions:prefixes", "ratio", "")
    best_bx = max(B_ARMS, key=lambda a: dX[a])
    best_bk = max(B_ARMS, key=lambda a: dK[a])
    pair = {"BLe": "FLe", "BLm": "FLm", "BLl": "FLl"}
    meas["5"] = _m(abs(dX[pair[best_bx]] - dX[best_bx]) / f_X,
                   "conditioning_gap_floor_multiple_x",
                   "positions:corpus", "ratio",
                   f"best B-band on X = {best_bx}")
    meas["6"] = _m(abs(dK[pair[best_bk]] - dK[best_bk]) / f_K,
                   "conditioning_gap_floor_multiple_k",
                   "positions:prefixes", "ratio",
                   f"best B-band on K = {best_bk}")
    dBC = X["B"] - X["C"]
    if dBC <= 0:
        contrasts["rec"] = {"admissible": False,
                            "reason": f"degenerate: X_B - X_C = {dBC}"}
    else:
        obs_extra["rec_X_baseB"] = {
            a: (X["B"] - X[a]) / dBC for a in B_ARMS}
        obs_extra["band_gap_rec_units_baseB"] = \
            band_gap(dX, B_ARMS) / dBC
    obs = {"measurement_valid": valid, "arms": arms,
           "measurements": meas, "contrasts": contrasts,
           "X": X, "K": K, "f_X": f_X, "f_K": f_K}
    obs.update(obs_extra)
    if not valid:
        obs["measurement_reason"] = vreason
    return obs


def resolution(outcomes: dict) -> tuple:
    if outcomes[1] != "FIRE":
        return ("INSTRUMENT-ALARM", "bracket violation")
    if any(outcomes[i] == "UNRESOLVED" for i in (2, 3, 4, 5, 6, 7)):
        return ("UNRESOLVED", "bars unresolved")
    sx = "STRUCTURE" if outcomes[2] == "FIRE" else "FLAT"
    parts = [f"B-X {sx}",
             f"F-X {'STRUCTURE' if outcomes[3] == 'FIRE' else 'FLAT'}",
             f"B-K {'STRUCTURE' if outcomes[4] == 'FIRE' else 'FLAT'}",
             f"F-K {'STRUCTURE' if outcomes[7] == 'FIRE' else 'FLAT'}",
             f"COND-X {'FIRES' if outcomes[5] == 'FIRE' else 'QUIET'}",
             f"COND-K {'FIRES' if outcomes[6] == 'FIRE' else 'QUIET'}"]
    return ("; ".join(parts), "bar-wise reading, no discretion")


def main():
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    prereg = load("docs/preregs/qwen-lband-1.json")
    rc, comp = {}, {}
    for a in ALL:
        p = os.path.join(SC, f"score_{a}.json")
        if not os.path.exists(p):
            raise SystemExit(f"REFUSING: missing receipt {p}")
        rc[a] = json.load(open(p))
        if rc[a].get("smoke"):
            raise SystemExit(f"REFUSING: smoke receipt {p}")
    for a in B_ARMS + F_ARMS:
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
    res, why = resolution(outcomes)
    if (res != "INSTRUMENT-ALARM" and outcomes.get(2) == "FIRE"
            and "band_gap_rec_units_baseB" in obs):
        obs["measurements"]["refuted:band_gap_rec_units_baseB"] = _m(
            obs["band_gap_rec_units_baseB"],
            "band_gap_rec_units_baseB", "positions:corpus", "mean",
            "emitted only when BAND-STRUCTURE-B-X fires")
        ref = adjudicate_refutation(prereg, obs)
    else:
        ref = "UNADJUDICATED (alarm or B-X structure bar quiet)"
    lines.append("SIGNED-CONDITIONING I_X "
                 f"{ {b: round(v, 5) for b, v in obs['conditioning_signed_x'].items()} } "
                 "I_K "
                 f"{ {b: round(v, 5) for b, v in obs['conditioning_signed_k'].items()} }")
    lines.append(f"RESOLUTION {res}: {why}")
    lines.append(f"REGISTERED-PRIOR(flatness): {ref}")
    lines.append(f"PRODUCER {obs['provenance']['code_commit']}"
                 f" dirty={obs['provenance']['tree_dirty']}")
    for line in lines[-4:]:
        print(line, flush=True)
    with open(os.path.join(AT, "lband_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    with open(os.path.join(AT, "lband_verdict.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[lband] -> {AT}/lband_verdict.txt", flush=True)


if __name__ == "__main__":
    main()

"""MATH-CYBER-1-RENDER-ATLAS-ROLE-CENSUS-0 — bank G of L65753 (prereg
RESULTS L66095): the role-position / order-grammar census of the twelve
booked render atlases. Model-free conditioning of the booked T, B,
A0_correct and B0_correct on where each of the six structural term roles
sits in the rendering: role x position means, W-position and
negative-bearing-role censuses, precedence and adjacency effects, an
additive role-position least-squares fit, and the local probe of the 120
Cayley edges moving W from position 0 to position 1. Discovery and fresh
cohorts separately, never pooled. Graph builder, gates and pins imported
from the frozen scratch/mathworld1_cayley.py. No model, no logit; torch is
never imported.

Usage:
    .venv/bin/python scratch/mathworld1_rolecensus.py
    RC_SMOKE=1 .venv/bin/python scratch/mathworld1_rolecensus.py   # one checkpoint, own directory
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathworld1_cayley as C  # noqa: E402  (frozen instrument, imported not copied)
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402

gate, fsha = C.gate, C.fsha
SMOKE = os.environ.get("RC_SMOKE") == "1"
OUTDIR = Path("logs/mathworld1/rolecensus_smoke" if SMOKE else "logs/mathworld1/rolecensus")
ROLES = list(C.ROLES)
FIELDS = ("T", "B", "A0_correct", "B0_correct")
NEG = {"SIN_LOW": "HI_D", "COS_LOW": "LO_D"}


def load_roles():
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    roles = {m["atlas_index"]: list(m["roles"]) for m in man}
    gate(len(roles) == 720 and all(sorted(r) == sorted(ROLES) for r in roles.values()), "MANIFEST ROLES")
    pos = {i: {r: roles[i].index(r) for r in ROLES} for i in roles}
    for r in ROLES:
        for p in range(6):
            gate(sum(pos[i][r] == p for i in roles) == 120, f"CELL {r} {p}")
    # negative-bearing roles from the model-blind census
    for row in map(json.loads, open("logs/mathworld1/prband2atlas/atlas_policies.jsonl")):
        mp = row["ceilings"]["minus_pos"]["values"]
        for theta, role in NEG.items():
            at = [int(k) for k, v in mp.items() if theta in v]
            gate(len(at) == 1 and at[0] == pos[row["atlas_index"]][role], f"NEGATIVE ROLE {theta} {row['atlas_index']}")
    return roles, pos


def design(pos, w_only=False):
    X = np.zeros((720, (6 if w_only else 36) + 1))
    for i in range(720):
        X[i, -1] = 1.0
        if w_only:
            X[i, pos[i]["W"]] = 1.0
        else:
            for k, r in enumerate(ROLES):
                X[i, k * 6 + pos[i][r]] = 1.0
    return X


def r2(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return (1.0 - float((res ** 2).sum()) / sst) if sst > 0 else None


def analyze_ck(ck, rows, roles, pos, edges):
    x = {f: np.array([rows[i][ck][f] for i in range(720)], dtype=float) for f in FIELDS}
    out = {"role_position_mean": {}, "precedence": {}, "adjacency": {}, "fit": {}, "neg_position": {}}
    for f in FIELDS:
        tab = {r: [float(np.mean([x[f][i] for i in range(720) if pos[i][r] == p])) for p in range(6)] for r in ROLES}
        out["role_position_mean"][f] = tab
        prec = {}
        for a, b in itertools.combinations(ROLES, 2):
            ab = np.mean([x[f][i] for i in range(720) if pos[i][a] < pos[i][b]])
            ba = np.mean([x[f][i] for i in range(720) if pos[i][a] > pos[i][b]])
            prec[f"{a}<{b}"] = float(ab - ba)
        dom = max(prec.items(), key=lambda kv: (abs(kv[1]), kv[0]))
        out["precedence"][f] = {"effects": prec, "dominant_pair": dom[0], "dominant_effect": dom[1],
                                "dominant_first_role": dom[0].split("<")[0] if dom[1] >= 0 else dom[0].split("<")[1],
                                "dominant_involves_W": "W" in dom[0].split("<")}
        adjd = {}
        for a, b in itertools.combinations(ROLES, 2):
            near = [x[f][i] for i in range(720) if abs(pos[i][a] - pos[i][b]) == 1]
            far = [x[f][i] for i in range(720) if abs(pos[i][a] - pos[i][b]) != 1]
            gate(len(near) == 240 and len(far) == 480, "ADJACENCY COUNTS")
            adjd[f"{a}|{b}"] = float(np.mean(near) - np.mean(far))
        doma = max(adjd.items(), key=lambda kv: (abs(kv[1]), kv[0]))
        out["adjacency"][f] = {"effects": adjd, "dominant_pair": doma[0], "dominant_effect": doma[1],
                               "dominant_involves_W": "W" in doma[0].split("|")}
        out["fit"][f] = {"r2_36": r2(design(pos), x[f]), "r2_W_only": r2(design(pos, w_only=True), x[f])}
    for theta, role in NEG.items():
        f = "A0_correct" if theta == "SIN_LOW" else "B0_correct"
        row = out["role_position_mean"][f][role]
        out["neg_position"][theta] = {"role": role, "field": f, "means_by_position": row,
                                      "effect_size": max(row) - min(row), "best_position": int(np.argmax(row)), "worst_position": int(np.argmin(row))}
    wrow = out["role_position_mean"]["B"]["W"]
    frow = {r: out["role_position_mean"]["B"][r][0] for r in ROLES}
    lrow = {r: out["role_position_mean"]["B"][r][5] for r in ROLES}
    # local probe: W-first to W-second edges
    probe = []
    for u, v in edges:
        if pos[u]["W"] == 0 and pos[v]["W"] == 1 or pos[v]["W"] == 0 and pos[u]["W"] == 1:
            a, b = (u, v) if pos[u]["W"] == 0 else (v, u)
            probe.append({"W_first": a, "W_second": b, "dT": int(x["T"][b] - x["T"][a]), "dB": int(x["B"][b] - x["B"][a])})
    gate(len(probe) == 120, "PROBE EDGES")
    wfirst = [i for i in range(720) if pos[i]["W"] == 0]
    wsecond = [i for i in range(720) if pos[i]["W"] == 1]
    fam = lambda S: sum(1 for i in S if x["T"][i] == 48 and x["B"][i] == 0)
    dts = sorted(p["dT"] for p in probe)
    dbs = sorted(p["dB"] for p in probe)
    out["W_position_B"] = wrow
    out["W_best_position_B"] = int(np.argmax(wrow))
    out["first_role_B"] = frow
    out["best_first_role_B"] = max(frow.items(), key=lambda kv: (kv[1], kv[0]))[0]
    out["last_role_B"] = lrow
    out["best_last_role_B"] = max(lrow.items(), key=lambda kv: (kv[1], kv[0]))[0]
    out["probe"] = {"edges": probe, "dT_median": (dts[59] + dts[60]) / 2, "dB_median": (dbs[59] + dbs[60]) / 2,
                    "dT_max": dts[-1], "dB_max": dbs[-1], "dT_min": dts[0], "dB_min": dbs[0],
                    "n_dT_zero": sum(d == 0 for d in dts), "n_dB_zero": sum(d == 0 for d in dbs),
                    "W_first_family_T48_B0": fam(wfirst), "W_second_family_T48_B0": fam(wsecond)}
    return out


def main():
    gate("torch" not in sys.modules, "TORCH IMPORTED")
    for p, h in C.PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    START = start_provenance(["scratch/mathworld1_rolecensus.py", "scratch/mathworld1_cayley.py", "llmopt/lab/provenance.py"])
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for f in ("rolecensus_receipt.json", "DISCOVERY.json", "FRESH.json"):
        gate(not (OUTDIR / f).exists(), f"REFUSE OVERWRITE {f}")
    _, adj, edges, _ = C.build_graph()
    roles, pos = load_roles()
    results = {}
    for cohort in (["DISCOVERY"] if SMOKE else list(C.COHORTS)):
        spec = C.COHORTS[cohort]
        rows = {r["atlas_index"]: r for r in map(json.loads, open(spec["table"]))}
        gate(len(rows) == 720, "TABLE")
        cks = spec["cks"][:1] if SMOKE else spec["cks"]
        res = {"cohort": cohort, "checkpoints": cks, "checkpoint": {ck: analyze_ck(ck, rows, roles, pos, edges) for ck in cks}}
        g = lambda fn: {ck: fn(res["checkpoint"][ck]) for ck in cks}
        bars = {"B1": {"pass": True, "note": "120 per role x position cell (36 / 36) and negative-bearing-role identity 720 / 720 are entry gates"}}
        b2 = g(lambda r: r["W_best_position_B"])
        bars["B2"] = {"W_best_position_B": b2, "fires": all(v == 5 for v in b2.values())}
        b3 = g(lambda r: r["best_first_role_B"])
        bars["B3"] = {"best_first_role_B": b3, "fires": len(set(b3.values())) == 1, "role": (list(b3.values())[0] if len(set(b3.values())) == 1 else None)}
        b4 = g(lambda r: (r["precedence"]["B"]["dominant_pair"], round(r["precedence"]["B"]["dominant_effect"], 4), r["precedence"]["B"]["dominant_involves_W"]))
        bars["B4"] = {"dominant_precedence_B": b4, "fires": all(v[2] for v in b4.values()),
                      "identical_pair_and_sign": len({(v[0], v[1] >= 0) for v in b4.values()}) == 1}
        b5 = g(lambda r: r["fit"]["B"]["r2_36"])
        bars["B5"] = {"r2_36_B": b5, "r2_W_only_B": g(lambda r: r["fit"]["B"]["r2_W_only"]), "fires": all(v is not None and v >= 0.5 for v in b5.values())}
        b6 = g(lambda r: r["probe"]["W_second_family_T48_B0"])
        bars["B6"] = {"W_second_family_T48_B0": b6, "W_first_family_T48_B0": g(lambda r: r["probe"]["W_first_family_T48_B0"]), "fires": all(v >= 100 for v in b6.values())}
        b7 = g(lambda r: (r["neg_position"]["SIN_LOW"]["effect_size"], r["neg_position"]["COS_LOW"]["effect_size"]))
        bars["B7"] = {"effect_SIN_HI_D_v_COS_LO_D": b7, "fires": all(v[1] > v[0] for v in b7.values())}
        b8 = g(lambda r: (r["adjacency"]["B"]["dominant_pair"], round(r["adjacency"]["B"]["dominant_effect"], 4), r["adjacency"]["B"]["dominant_involves_W"]))
        bars["B8"] = {"dominant_adjacency_B": b8, "fires": all(v[2] for v in b8.values())}
        prior5 = g(lambda r: (r["fit"]["B"]["r2_W_only"] is not None and r["fit"]["B"]["r2_36"] is not None and r["fit"]["B"]["r2_W_only"] >= 0.5 * r["fit"]["B"]["r2_36"]))
        bars["prior5_W_only_half_of_r2"] = {"per_ck": prior5, "all": all(prior5.values())}
        res["bars"] = bars
        (OUTDIR / f"{cohort}.json").write_text(json.dumps(res, indent=1))
        results[cohort] = res
        print(f"[{cohort}] bars {json.dumps(bars, default=str)[:2500]}", flush=True)
    receipt = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-ROLE-CENSUS-0" + ("-SMOKE" if SMOKE else ""), "smoke": SMOKE,
               "pins": {p: fsha(p) for p in C.PINS}, "graph_edges": len(edges),
               "cohorts": {c: {"sha256": fsha(str(OUTDIR / f"{c}.json")), "checkpoints": results[c]["checkpoints"], "bars": results[c]["bars"]} for c in results},
               "semantic_beyond_all_surface_identifiable": False,
               "wall_s": round(time.time() - t0, 1), "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "rolecensus_receipt.json").write_text(json.dumps(receipt, indent=1))
    print("wall", receipt["wall_s"])


if __name__ == "__main__":
    main()

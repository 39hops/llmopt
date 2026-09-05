"""Independent verifier for MATH-CYBER-1-RENDER-ATLAS-ROLE-CENSUS-0 (prereg
RESULTS L66095). Shares no code with scratch/mathworld1_rolecensus.py:
own manifest read, own role-position means, precedence and adjacency
effects, own least squares through the normal equations with a
pseudo-inverse, own W-first-to-second edge list by inversion distance,
own family censuses, own bar recomputation; pins the instrument source
against its start provenance and every input against the instrument's
entry pins; refuses to overwrite its receipt.

Usage:
    .venv/bin/python scratch/mathworld1_rolecensusverify.py
"""
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

OUTDIR = Path("logs/mathworld1/rolecensus")
ROLES = ["HI_D", "HI_L", "LO_D", "LO_L", "K", "W"]
FIELDS = ("T", "B", "A0_correct", "B0_correct")
D = []


def chk(c, m):
    if not c:
        D.append(m)


def inv_dist(p, q):
    pos = {r: i for i, r in enumerate(q)}
    s = [pos[r] for r in p]
    return sum(1 for i in range(6) for j in range(i + 1, 6) if s[i] > s[j])


def main():
    chk(not (OUTDIR / "verify_receipt.json").exists(), "REFUSE OVERWRITE verify_receipt.json")
    if D:
        raise SystemExit(D[-1])
    inst = json.load(open(OUTDIR / "rolecensus_receipt.json"))
    chk(inst.get("smoke") is False and inst["prereg"] == "MATH-CYBER-1-RENDER-ATLAS-ROLE-CENSUS-0", "instrument receipt identity")
    src_sha = hashlib.sha256(open("scratch/mathworld1_rolecensus.py", "rb").read()).hexdigest()
    chk(src_sha == inst["start"]["file_sha256"]["scratch/mathworld1_rolecensus.py"], "instrument source sha v receipt")
    for pth, h in inst["pins"].items():
        chk(hashlib.sha256(open(pth, "rb").read()).hexdigest() == h, f"pin drift {pth}")
    chk(inst["start"]["start_commit"] == inst["completion_commit"], "instrument start v completion commit")
    if D:
        raise SystemExit(D[-1])
    man = [json.loads(l) for l in open("logs/mathworld1/prband2atlas/atlas_manifest.jsonl")]
    roles = {m["atlas_index"]: tuple(m["roles"]) for m in man}
    pos = {i: {r: roles[i].index(r) for r in ROLES} for i in roles}
    # B1's second half, recomputed: the minus sign sits at HI_D under SIN_LOW and at LO_D under COS_LOW
    n_neg = 0
    for row in map(json.loads, open("logs/mathworld1/prband2atlas/atlas_policies.jsonl")):
        mp = row["ceilings"]["minus_pos"]["values"]
        sin = [int(k) for k, v in mp.items() if "SIN_LOW" in v]
        cos = [int(k) for k, v in mp.items() if "COS_LOW" in v]
        n_neg += (sin == [pos[row["atlas_index"]]["HI_D"]] and cos == [pos[row["atlas_index"]]["LO_D"]])
    chk(n_neg == 720, f"negative-bearing role identity {n_neg} / 720")
    chk(all(sum(pos[i][r] == p for i in roles) == 120 for r in ROLES for p in range(6)), "120 per cell")
    probe_edges = [(i, j) for i in range(720) for j in range(720)
                   if pos[i]["W"] == 0 and pos[j]["W"] == 1 and inv_dist(roles[i], roles[j]) == 1]
    chk(len(probe_edges) == 120, "probe edges")
    inputs = {}
    tables = {"DISCOVERY": "logs/mathworld1/prband2atlasscore/policy_table.jsonl", "FRESH": "logs/mathworld1/prband2atlasfresh/policy_table.jsonl"}
    for cohort in ("DISCOVERY", "FRESH"):
        res = json.load(open(OUTDIR / f"{cohort}.json"))
        inputs[cohort] = hashlib.sha256((OUTDIR / f"{cohort}.json").read_bytes()).hexdigest()
        chk(inst["cohorts"][cohort]["sha256"] == inputs[cohort], f"{cohort} cohort sha v instrument receipt")
        rows = {r["atlas_index"]: r for r in map(json.loads, open(tables[cohort]))}
        for ck in res["checkpoints"]:
            R = res["checkpoint"][ck]
            x = {f: np.array([rows[i][ck][f] for i in range(720)], dtype=float) for f in FIELDS}
            for f in FIELDS:
                for r in ROLES:
                    for p in range(6):
                        sel = [i for i in range(720) if pos[i][r] == p]
                        chk(len(sel) == 120 and abs(R["role_position_mean"][f][r][p] - x[f][sel].mean()) < 1e-9, f"{ck} {f} cell {r} {p}")
                prec = {}
                for a, b in itertools.combinations(ROLES, 2):
                    ab = [i for i in range(720) if pos[i][a] < pos[i][b]]
                    ba = [i for i in range(720) if pos[i][a] > pos[i][b]]
                    prec[f"{a}<{b}"] = x[f][ab].mean() - x[f][ba].mean()
                    chk(abs(R["precedence"][f]["effects"][f"{a}<{b}"] - prec[f"{a}<{b}"]) < 1e-9, f"{ck} {f} prec {a} {b}")
                dom = max(prec.items(), key=lambda kv: (abs(kv[1]), kv[0]))
                chk(R["precedence"][f]["dominant_pair"] == dom[0] and R["precedence"][f]["dominant_involves_W"] == ("W" in dom[0].split("<")), f"{ck} {f} dominant prec")
                adjd = {}
                for a, b in itertools.combinations(ROLES, 2):
                    near = [i for i in range(720) if abs(pos[i][a] - pos[i][b]) == 1]
                    far = [i for i in range(720) if abs(pos[i][a] - pos[i][b]) != 1]
                    adjd[f"{a}|{b}"] = x[f][near].mean() - x[f][far].mean()
                    chk(abs(R["adjacency"][f]["effects"][f"{a}|{b}"] - adjd[f"{a}|{b}"]) < 1e-9, f"{ck} {f} adj {a} {b}")
                doma = max(adjd.items(), key=lambda kv: (abs(kv[1]), kv[0]))
                chk(R["adjacency"][f]["dominant_pair"] == doma[0], f"{ck} {f} dominant adj")
                # least squares via normal equations + pseudo-inverse
                for key, cols in (("r2_36", [(r, p) for r in ROLES for p in range(6)]), ("r2_W_only", [("W", p) for p in range(6)])):
                    X = np.zeros((720, len(cols) + 1))
                    X[:, -1] = 1
                    for i in range(720):
                        for k, (r, p) in enumerate(cols):
                            if pos[i][r] == p:
                                X[i, k] = 1
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ x[f]
                    res_ = x[f] - X @ beta
                    sst = ((x[f] - x[f].mean()) ** 2).sum()
                    r2 = 1 - (res_ ** 2).sum() / sst if sst > 0 else None
                    got = R["fit"][f][key]
                    chk((got is None and r2 is None) or (got is not None and r2 is not None and abs(got - r2) < 1e-6), f"{ck} {f} {key} {got} v {r2}")
            for theta, role, f in (("SIN_LOW", "HI_D", "A0_correct"), ("COS_LOW", "LO_D", "B0_correct")):
                row = [x[f][[i for i in range(720) if pos[i][role] == p]].mean() for p in range(6)]
                chk(abs(R["neg_position"][theta]["effect_size"] - (max(row) - min(row))) < 1e-9 and R["neg_position"][theta]["role"] == role, f"{ck} neg {theta}")
            wrow = [x["B"][[i for i in range(720) if pos[i]["W"] == p]].mean() for p in range(6)]
            chk(R["W_best_position_B"] == int(np.argmax(wrow)), f"{ck} W best position")
            frow = {r: x["B"][[i for i in range(720) if pos[i][r] == 0]].mean() for r in ROLES}
            chk(R["best_first_role_B"] == max(frow.items(), key=lambda kv: (kv[1], kv[0]))[0], f"{ck} best first role")
            dT = sorted(int(x["T"][b] - x["T"][a]) for a, b in probe_edges)
            dB = sorted(int(x["B"][b] - x["B"][a]) for a, b in probe_edges)
            got = sorted((p["W_first"], p["W_second"]) for p in R["probe"]["edges"])
            chk(got == sorted(probe_edges), f"{ck} probe edge set")
            chk(R["probe"]["dT_median"] == (dT[59] + dT[60]) / 2 and R["probe"]["dB_median"] == (dB[59] + dB[60]) / 2, f"{ck} probe medians")
            chk(R["probe"]["n_dB_zero"] == sum(d == 0 for d in dB) and R["probe"]["dB_max"] == dB[-1] and R["probe"]["dT_max"] == dT[-1], f"{ck} probe counts")
            fam = lambda P: sum(1 for i in range(720) if pos[i]["W"] == P and x["T"][i] == 48 and x["B"][i] == 0)
            chk(R["probe"]["W_first_family_T48_B0"] == fam(0) and R["probe"]["W_second_family_T48_B0"] == fam(1), f"{ck} families")
            print(ck, "checked; discrepancies so far", len(D), flush=True)
        b = res["bars"]
        cks = res["checkpoints"]
        chk(b["B2"]["fires"] == all(res["checkpoint"][ck]["W_best_position_B"] == 5 for ck in cks), f"{cohort} B2")
        chk(b["B3"]["fires"] == (len({res["checkpoint"][ck]["best_first_role_B"] for ck in cks}) == 1), f"{cohort} B3")
        chk(b["B4"]["fires"] == all(res["checkpoint"][ck]["precedence"]["B"]["dominant_involves_W"] for ck in cks), f"{cohort} B4")
        chk(b["B5"]["fires"] == all(res["checkpoint"][ck]["fit"]["B"]["r2_36"] >= 0.5 for ck in cks), f"{cohort} B5")
        chk(b["B6"]["fires"] == all(res["checkpoint"][ck]["probe"]["W_second_family_T48_B0"] >= 100 for ck in cks), f"{cohort} B6")
        chk(b["B7"]["fires"] == all(res["checkpoint"][ck]["neg_position"]["COS_LOW"]["effect_size"] > res["checkpoint"][ck]["neg_position"]["SIN_LOW"]["effect_size"] for ck in cks), f"{cohort} B7")
        chk(b["B8"]["fires"] == all(res["checkpoint"][ck]["adjacency"]["B"]["dominant_involves_W"] for ck in cks), f"{cohort} B8")
    rec = {"prereg": "MATH-CYBER-1-RENDER-ATLAS-ROLE-CENSUS-0", "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "discrepancies": D[:40], "n_discrepancies": len(D), "inputs": inputs, "instrument_source_sha256": src_sha,
           "instrument_receipt_sha256": hashlib.sha256((OUTDIR / "rolecensus_receipt.json").read_bytes()).hexdigest(),
           "B1_recomputed": {"negative_bearing_role_identity": n_neg, "cells_of_120": 36},
           "instrument_run_commit": inst["start"]["start_commit"],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
           "verifier_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:1500])


if __name__ == "__main__":
    main()

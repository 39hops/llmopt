"""ATOM-DIET-TRAJECTORY-1 census (pre-reg RESULTS L66546, sealed by
AMENDMENT -SEAL L66822). Loads every snapshot of every birth, computes
the literal 59-key P8 / C8 / centroid law against the birth's own
step_0, the paired treatment displacements T_s and the six within-arm
displacements at every snapshot, the S1 / S1b / S2 / S3 bars at snapshot
15,420, F1 / F2 from the post-hoc gate rows, the descriptive lag tables
(delta_h(t) = W[t+h] - W[t], lag_h(t) = cos(delta_h(t), delta_h(t+h)) on
grid snapshots) and the per-tensor IPR / effective-rank tables. Writes
logs/atomtraj1/census.json (bar inputs and per-snapshot tables) and
logs/atomtraj1/census_tensors.json (per-tensor tables, sha256 recorded
in census.json). Refuses to overwrite. Bars are evaluated exactly as
written in the pre-reg; nothing here is adaptive.

Env: SMOKE=1 runs the plumbing on the last smoke birth in
logs/atomtraj1/smoke.jsonl and appends a kind=census row there.

Usage: .venv/bin/python scratch/atomtraj_census.py
"""
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch  # noqa: E402

from atomtraj_pins import BLOCK2D, CLASSES, state_digest  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
LOGS = Path("logs/atomtraj1")
SEEDS = (5, 6, 7)
FINAL = 15_420
GRID = 1_028
LAGS = (1_028, 2_056, 3_084, 5_140)
GATE_STEPS = (2_056, 5_140, 10_280, 15_420)


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def objects(sd, sd0):
    """P8, C8 (dict), centroid, per-tensor stats for the 40 block 2-D tensors."""
    e = []
    tens = {}
    for l in range(8):
        el = 0.0
        for t in BLOCK2D:
            k = f"blocks.{l}.{t}"
            D = (sd[k].double() - sd0[k].double())
            en = float((D * D).sum())
            el += en
            S = torch.linalg.svdvals(D)
            rank = int((S > 0).sum())
            if rank > 0:
                U = torch.linalg.svd(D, full_matrices=False)[0]
                kk = min(16, rank)
                ipr = float((U[:, :kk] ** 4).sum(0).mean()) * D.shape[0]
                p = S[:rank] / S[:rank].sum()
                eff = float(torch.exp(-(p * p.log()).sum()))
            else:
                ipr, eff = None, None
            tens[k] = {"energy": en, "ipr_top16_x_rows": ipr, "effrank": eff, "rank": rank}
        e.append(el)
    tot = sum(e)
    ce = {c: sum(float(((sd[k].double() - sd0[k].double()) ** 2).sum()) for k in ks) for c, ks in CLASSES.items()}
    ct = sum(ce.values())
    if tot == 0 or ct == 0:
        # zero delta (step 0 against itself): the normalized objects are undefined
        return {"P8": None, "C8": None, "centroid": None, "E2d": tot, "E_all": ct}, tens
    p8 = [x / tot for x in e]
    c8 = {c: v / ct for c, v in ce.items()}
    return {"P8": p8, "C8": c8, "centroid": sum(i * x for i, x in enumerate(p8)), "E2d": tot, "E_all": ct}, tens


def flat2d(sd):
    return torch.cat([(sd[f"blocks.{l}.{t}"]).double().reshape(-1) for l in range(8) for t in BLOCK2D])


def separation(paired, within):
    return bool(paired) and bool(within) and min(paired.values()) > max(within.values())


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if SMOKE:
        rows = [json.loads(l) for l in (LOGS / "smoke.jsonl").open()]
        births = [r for r in rows if r.get("kind") == "birth" and r.get("emit") and r.get("device") != "cpu"][-1:]
        gates = [r for r in rows if r.get("kind") == "gate"]
    else:
        assert not (LOGS / "census.json").exists() and not (LOGS / "census_tensors.json").exists(), "REFUSING: census exists"
        births = [json.loads(l) for l in (LOGS / "births.jsonl").open()]
        gates = [json.loads(l) for l in (LOGS / "gates.jsonl").open()]
        assert len(births) == 6 and all(not r["smoke"] for r in births)
        assert len(gates) == 24 and all(not g["smoke"] for g in gates)
    per = {}      # (arm, seed) -> step -> objects
    tens_out = {}
    for r in births:
        key = f"{r['arm']}_s{r['seed']}"
        outdir = Path(r["outdir"])
        sd0 = torch.load(outdir / "step_00000.pt", map_location="cpu")
        assert state_digest(sd0) == r["snapshots"]["0"]["state_digest"], f"{key} step_0 digest"
        per[key] = {}
        prev_flat = {}
        for s in sorted(int(x) for x in r["snapshots"]):
            sd = torch.load(outdir / f"step_{s:05d}.pt", map_location="cpu")
            assert state_digest(sd) == r["snapshots"][str(s)]["state_digest"], f"{key} step {s} digest"
            ob, tens = objects(sd, sd0)
            per[key][s] = ob
            tens_out[f"{key}/{s}"] = tens
            if s % GRID == 0 and s > 0:
                prev_flat[s] = flat2d(sd)
        # lag tables on grid snapshots
        lag = {}
        grid = sorted(prev_flat)
        for h in LAGS:
            lag[str(h)] = {}
            for t in grid:
                if t + h in prev_flat and t + 2 * h in prev_flat:
                    d1 = prev_flat[t + h] - prev_flat[t]
                    d2 = prev_flat[t + 2 * h] - prev_flat[t + h]
                    n1, n2 = float(d1.norm()), float(d2.norm())
                    lag[str(h)][str(t)] = float((d1 @ d2) / (n1 * n2)) if n1 > 0 and n2 > 0 else None
        per[key]["lag"] = lag
        prev_flat.clear()
        print(f"[census] {key}: {len(r['snapshots'])} snapshots", flush=True)

    out = {"prereg": "ATOM-DIET-TRAJECTORY-1", "smoke": SMOKE, "commit": commit, "births": sorted(per),
           "snapshot_objects": {k: {str(s): v for s, v in d.items() if s != "lag"} for k, d in per.items()},
           "lag": {k: d["lag"] for k, d in per.items()}}
    steps_all = sorted({s for d in per.values() for s in d if s != "lag" and d[s]["P8"] is not None})
    if not SMOKE:
        bars_by_step = {}
        for s in steps_all:
            P = {(a, sd_): per[f"{a}_s{sd_}"][s] for a in ("stock", "atoms") for sd_ in SEEDS}
            T = {str(sd_): l2(P[("atoms", sd_)]["P8"], P[("stock", sd_)]["P8"]) for sd_ in SEEDS}
            Tc = {str(sd_): l2(list(P[("atoms", sd_)]["C8"].values()), list(P[("stock", sd_)]["C8"].values())) for sd_ in SEEDS}
            W = {f"{a}_{x}_{y}": l2(P[(a, x)]["P8"], P[(a, y)]["P8"]) for a in ("stock", "atoms") for x, y in itertools.combinations(SEEDS, 2)}
            Wc = {f"{a}_{x}_{y}": l2(list(P[(a, x)]["C8"].values()), list(P[(a, y)]["C8"].values())) for a in ("stock", "atoms") for x, y in itertools.combinations(SEEDS, 2)}
            dp = {sd_: [x - y for x, y in zip(P[("atoms", sd_)]["P8"], P[("stock", sd_)]["P8"])] for sd_ in SEEDS}
            cosines = {f"{x}_{y}": cos(dp[x], dp[y]) for x, y in itertools.combinations(SEEDS, 2)}
            nonzero = all(any(v != 0 for v in dp[sd_]) for sd_ in SEEDS)
            s1b = nonzero and all(c is not None and c > 0 for c in cosines.values())
            csign = {str(sd_): (P[("atoms", sd_)]["centroid"] > P[("stock", sd_)]["centroid"]) - (P[("atoms", sd_)]["centroid"] < P[("stock", sd_)]["centroid"]) for sd_ in SEEDS}
            s3 = len(set(csign.values())) == 1 and 0 not in csign.values()
            bars_by_step[str(s)] = {"T": T, "W": W, "T_C8": Tc, "W_C8": Wc, "treatment_vectors": {str(k): v for k, v in dp.items()},
                                    "cosines": cosines, "centroid_sign": csign,
                                    "S1": separation(T, W), "S1b": bool(s1b), "S2": separation(Tc, Wc), "S3": bool(s3),
                                    "ratio_minT_maxW": (min(T.values()) / max(W.values())) if max(W.values()) > 0 else None}
        fin = bars_by_step[str(FINAL)]
        g = {(r["arm"], r["seed"], r["step"]): r for r in gates}
        deltas = {str(sd_): g[("atoms", sd_, FINAL)]["total"] - g[("stock", sd_, FINAL)]["total"] for sd_ in SEEDS}
        l4 = {str(sd_): (g[("atoms", sd_, FINAL)]["solves"]["4"], g[("stock", sd_, FINAL)]["solves"]["4"]) for sd_ in SEEDS}
        f1 = all(d > 0 for d in deltas.values()) and (sum(deltas.values()) / 3) >= 5
        f2 = all(a >= b for a, b in l4.values())
        structural = fin["S1"] and fin["S1b"]
        out["bars"] = {"S1": fin["S1"], "S1b": fin["S1b"], "S2": fin["S2"], "S3": fin["S3"], "F1": bool(f1), "F2": bool(f2),
                       "STRUCTURAL": bool(structural), "FUNCTION": bool(f1), "JOINT": bool(structural and f1),
                       "paired_deltas": deltas, "mean_paired_delta": sum(deltas.values()) / 3, "L4_atoms_stock": l4,
                       "first_step_S1_holds": next((s for s in steps_all if bars_by_step[str(s)]["S1"]), None)}
        out["bars_by_step"] = bars_by_step
        out["gates"] = {f"{r['arm']}_s{r['seed']}/{r['step']}": {"solves": r["solves"], "total": r["total"], "valid_pct": r["valid_pct"]} for r in gates}
    tens_json = json.dumps(tens_out, indent=0)
    out["census_tensors_sha256"] = hashlib.sha256(tens_json.encode()).hexdigest()
    out["census_tensors_bytes"] = len(tens_json)
    if SMOKE:
        with (LOGS / "smoke.jsonl").open("a") as f:
            f.write(json.dumps({"kind": "census", **out, "census_tensors": tens_out}) + "\n")
        print("[census] smoke row appended", flush=True)
    else:
        (LOGS / "census_tensors.json").write_text(tens_json)
        (LOGS / "census.json").write_text(json.dumps(out, indent=1))
        print("[census] written; bars:", json.dumps({k: out["bars"][k] for k in ("S1", "S1b", "S2", "S3", "F1", "F2", "STRUCTURAL", "FUNCTION", "JOINT")}), flush=True)


if __name__ == "__main__":
    main()

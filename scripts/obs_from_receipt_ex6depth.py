"""Observations adapter for PRE-REG EX6-DEPTH-0.

Derives every registered measurement from the run's receipts in
committed code: bar 1 anchors from qual.jsonl, bar 2 from
census.json (5 arms x 48 modules), bars 3-5 and the refutation
predicate from treatment.jsonl + qual.jsonl pooled deltas. Also
emits the registered alignment read inputs (demand-excess band
from demand.json) as provenance color. Refuses smoke rows
(n_eval != 120), partial populations, and a missing census.

    .venv/bin/python scripts/obs_from_receipt_ex6depth.py \
        logs/ex6depth > logs/ex6depth/ex6depth_observations.json
    .venv/bin/python scripts/adjudicate.py \
        docs/preregs/ex6-depth-0.json \
        logs/ex6depth/ex6depth_observations.json
"""
import json
import sys
from pathlib import Path

SEEDS = (7001, 8002, 9003)
N_EVAL = 120
ANCHORS = {"dep_NONE": {7001: 64, 8002: 61, 9003: 66},
           "dep_Z1_ALL": {7001: 74, 8002: 66, 9003: 72}}
BANDS = ("Z1_EARLY", "Z1_MID", "Z1_LATE")


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if "meta" in r:
            continue
        if r["n_eval"] != N_EVAL:
            raise SystemExit(f"REFUSING smoke/partial row: {r}")
        out[(r["arm"], r["seed"])] = r["gate_ok"]
    return out


def demand_excess_band(demand):
    """Per band: z1 outside-fraction minus mean(z2,z3) fraction."""
    frac = {}
    for key, (tot, out) in demand.items():
        bp, ph = key.split(":")
        band = int(bp) // 16
        t0, o0 = frac.get((band, ph), (0, 0))
        frac[(band, ph)] = (t0 + tot, o0 + out)
    excess = {}
    for band in (0, 1, 2):
        f = {ph: frac[(band, ph)][1] / frac[(band, ph)][0]
             for ph in ("z1", "z2", "z3")}
        excess[band] = f["z1"] - (f["z2"] + f["z3"]) / 2
    return excess


def main(run_dir):
    d = Path(run_dir)
    qual = rows(d / "qual.jsonl")
    cen = json.loads((d / "census.json").read_text())["verdicts"]
    n_cells = sum(v["n_ok"] for v in cen.values())
    for arm in ANCHORS:
        for s in SEEDS:
            if (arm, s) not in qual:
                raise SystemExit(f"REFUSING: missing qual cell {arm}/{s}")
    n_exact = sum(1 for arm in ANCHORS for s in SEEDS
                  if qual[(arm, s)] == ANCHORS[arm][s])
    treat = rows(d / "treatment.jsonl") if (d / "treatment.jsonl").exists() \
        else {}
    obs = {"measurement_valid": True,
           "arms": {a: {"admissible": True} for a in
                    ("none", "z1_all", "z1_early", "z1_mid",
                     "z1_late")},
           "measurements": {}}
    ms = obs["measurements"]
    ms["1"] = {"value": n_exact, "metric": "n_exact_reproductions",
               "population": "anchors:none+z1all:3seeds",
               "aggregation": "count_of_6"}
    ms["2"] = {"value": n_cells, "metric": "n_module_arm_cells_ok",
               "population": "seed7001:problem2:all5arms",
               "aggregation": "count_of_240"}
    census_pass = all(v["pass"] for v in cen.values())
    if n_exact == 6 and census_pass:
        for band in BANDS:
            for s in SEEDS:
                if (f"dep_{band}", s) not in treat:
                    raise SystemExit(
                        f"REFUSING: missing treatment cell {band}/{s}")
        dz = {b: sum(treat[(f"dep_{b}", s)] - qual[("dep_NONE", s)]
                     for s in SEEDS) for b in BANDS}
        de, dm, dl = dz["Z1_EARLY"], dz["Z1_MID"], dz["Z1_LATE"]
        d_all = sum(qual[("dep_Z1_ALL", s)] - qual[("dep_NONE", s)]
                    for s in SEEDS)
        mx = max(de, dm, dl)
        ms["3"] = {"value": mx, "metric": "max_pooled_delta",
                   "population": "bands_v_none:3seeds",
                   "aggregation": "pooled_sum"}
        ms["4"] = dict(ms["3"], metric="max_pooled_delta")
        ms["4:delta_early"] = {"value": de, "metric": "pooled_delta",
                               "population": "z1early_v_none:3seeds",
                               "aggregation": "pooled_sum"}
        ms["4:delta_mid"] = {"value": dm, "metric": "pooled_delta",
                             "population": "z1mid_v_none:3seeds",
                             "aggregation": "pooled_sum"}
        ms["4:delta_late"] = {"value": dl, "metric": "pooled_delta",
                              "population": "z1late_v_none:3seeds",
                              "aggregation": "pooled_sum"}
        ms["5"] = {"value": abs(d_all - (de + dm + dl)),
                   "metric": "abs_additivity_residual",
                   "population": "all_v_bandsum:3seeds",
                   "aggregation": "pooled_sum"}
        ms["refutation:max_band_delta"] = ms["3"]
        dem = json.loads((d / "demand.json").read_text())["demand"]
        obs["alignment_read"] = {
            "demand_excess_by_band": demand_excess_band(dem),
            "band_deltas": {"early": de, "mid": dm, "late": dl},
            "delta_all": d_all}
    else:
        for a in ("z1_early", "z1_mid", "z1_late"):
            obs["arms"][a] = {
                "admissible": False,
                "reason": "qualification bridge failed "
                          f"(anchors {n_exact}/6, census cells "
                          f"{n_cells}/240) — treatment sealed, "
                          "no verdict on the bands"}
    print(json.dumps(obs, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/ex6depth")

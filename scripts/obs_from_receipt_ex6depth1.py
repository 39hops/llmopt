"""Observations adapter for PRE-REG EX6-DEPTH-1.

Bar 1 anchors from qual.jsonl, bar 2 from census.json (6 arms x
48 modules), bar 3 from inert.jsonl (gate == 64 AND recall ==
1.0), bars 4-6 and the refutation predicate from treatment.jsonl
+ qual.jsonl pooled deltas. Refuses smoke rows and partial
populations.

    .venv/bin/python scripts/obs_from_receipt_ex6depth1.py \
        logs/ex6depth1 > logs/ex6depth1/ex6depth1_observations.json
    .venv/bin/python scripts/adjudicate.py \
        docs/preregs/ex6-depth-1.json \
        logs/ex6depth1/ex6depth1_observations.json
"""
import json
import sys
from pathlib import Path

SEEDS = (7001, 8002, 9003)
N_EVAL = 120
ANCHORS = {"dep_NONE": {7001: 64, 8002: 61, 9003: 66},
           "dep_Z1_LATE": {7001: 70, 8002: 69, 9003: 71}}
SINGLES = ("Z1_B43", "Z1_B46", "Z1_B43_46")


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if "meta" in r:
            continue
        if r["n_eval"] != N_EVAL:
            raise SystemExit(f"REFUSING smoke/partial row: {r}")
        out[(r["arm"], r["seed"])] = r
    return out


def main(run_dir):
    d = Path(run_dir)
    qual = rows(d / "qual.jsonl")
    cen = json.loads((d / "census.json").read_text())["verdicts"]
    n_cells = sum(v["n_ok"] for v in cen.values())
    inert = rows(d / "inert.jsonl")
    ir = inert.get(("dep_Z1_L32_39", 7001))
    if ir is None:
        raise SystemExit("REFUSING: missing inertness cell")
    n_inert = int(ir["gate_ok"] == 64) + \
        int(ir["masked_recall_named80"] == 1.0)
    for arm in ANCHORS:
        for s in SEEDS:
            if (arm, s) not in qual:
                raise SystemExit(f"REFUSING: missing qual cell {arm}/{s}")
    n_exact = sum(1 for arm in ANCHORS for s in SEEDS
                  if qual[(arm, s)]["gate_ok"] == ANCHORS[arm][s])
    treat = rows(d / "treatment.jsonl") if (d / "treatment.jsonl").exists() \
        else {}
    obs = {"measurement_valid": True,
           "arms": {a: {"admissible": True} for a in
                    ("none", "z1_late", "z1_l32_39", "z1_b43",
                     "z1_b46", "z1_b43_46")},
           "measurements": {}}
    ms = obs["measurements"]
    ms["1"] = {"value": n_exact, "metric": "n_exact_reproductions",
               "population": "anchors:none+z1late:3seeds",
               "aggregation": "count_of_6"}
    ms["2"] = {"value": n_cells, "metric": "n_module_arm_cells_ok",
               "population": "seed7001:problem2:all6arms",
               "aggregation": "count_of_288"}
    ms["3"] = {"value": n_inert, "metric": "n_inertness_checks_ok",
               "population": "seed7001:z1_l32_39",
               "aggregation": "count_of_2"}
    census_pass = all(v["pass"] for v in cen.values())
    if n_exact == 6 and census_pass and n_inert == 2:
        for arm in SINGLES:
            for s in SEEDS:
                if (f"dep_{arm}", s) not in treat:
                    raise SystemExit(
                        f"REFUSING: missing treatment cell {arm}/{s}")
        dz = {a: [treat[(f"dep_{a}", s)]["gate_ok"]
                  - qual[("dep_NONE", s)]["gate_ok"] for s in SEEDS]
              for a in SINGLES}
        d43, d46 = sum(dz["Z1_B43"]), sum(dz["Z1_B46"])
        dpair = sum(dz["Z1_B43_46"])
        ms["4"] = {"value": dpair, "metric": "pooled_delta",
                   "population": "b43_46_v_none:3seeds",
                   "aggregation": "pooled_sum"}
        ms["4:signs"] = {"value": sum(1 for x in dz["Z1_B43_46"]
                                      if x > 0),
                         "metric": "n_positive_seeds",
                         "population": "b43_46_v_none:3seeds",
                         "aggregation": "count_of_3"}
        ms["5"] = {"value": max(d43, d46),
                   "metric": "max_pooled_delta",
                   "population": "b43+b46_v_none:3seeds",
                   "aggregation": "pooled_sum"}
        ms["6"] = {"value": abs(dpair - (d43 + d46)),
                   "metric": "abs_additivity_residual",
                   "population": "pair_v_singlesum:3seeds",
                   "aggregation": "pooled_sum"}
        ms["refutation:delta_pair"] = ms["4"]
        obs["color"] = {"delta_b43": d43, "delta_b46": d46,
                        "delta_pair": dpair}
    else:
        for a in ("z1_b43", "z1_b46", "z1_b43_46"):
            obs["arms"][a] = {
                "admissible": False,
                "reason": "qualification bridge failed "
                          f"(anchors {n_exact}/6, census "
                          f"{n_cells}/288, inert {n_inert}/2) — "
                          "treatment sealed, no verdict on singles"}
    print(json.dumps(obs, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/ex6depth1")

"""Observations adapter for PRE-REG EX6-TEMPORAL-0.

Derives every registered measurement from the run's receipts in
committed code (never hand-authored): bar 1 anchor reproductions
from qual.jsonl, bar 2 from census.json, bars 3-5 and the
refutation predicate from treatment.jsonl + qual.jsonl pooled
deltas. Refuses smoke rows (n_eval != 120), partial seed
populations, and a missing/failed census.

    .venv/bin/python scripts/obs_from_receipt_ex6temporal.py \
        logs/ex6temporal > logs/ex6temporal/ex6temporal_observations.json
    .venv/bin/python scripts/adjudicate.py \
        docs/preregs/ex6-temporal-0.json \
        logs/ex6temporal/ex6temporal_observations.json
"""
import json
import sys
from pathlib import Path

SEEDS = (7001, 8002, 9003)
N_EVAL = 120
ANCHORS = {"tmp_NONE": {7001: 64, 8002: 61, 9003: 66},
           "tmp_Z1": {7001: 74, 8002: 66, 9003: 72}}


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


def main(run_dir):
    d = Path(run_dir)
    qual = rows(d / "qual.jsonl")
    census = json.loads((d / "census.json").read_text())["verdict"]
    for arm in ("tmp_NONE", "tmp_Z1"):
        for s in SEEDS:
            if (arm, s) not in qual:
                raise SystemExit(f"REFUSING: missing qual cell {arm}/{s}")
    n_exact = sum(1 for arm in ANCHORS for s in SEEDS
                  if qual[(arm, s)] == ANCHORS[arm][s])
    treat = rows(d / "treatment.jsonl") if (d / "treatment.jsonl").exists() \
        else {}
    obs = {"measurement_valid": True,
           "arms": {a: {"admissible": True}
                    for a in ("none", "z1", "z2", "z3")},
           "measurements": {}}
    ms = obs["measurements"]
    ms["1"] = {"value": n_exact, "metric": "n_exact_reproductions",
               "population": "anchors:none+z1:3seeds",
               "aggregation": "count_of_6"}
    ms["2"] = {"value": census["n_ok"],
               "metric": "n_modules_temporal_law_ok",
               "population": "seed7001:problem2:none",
               "aggregation": "count_of_48"}
    if n_exact == 6 and census.get("pass"):
        for arm in ("tmp_Z2", "tmp_Z3"):
            for s in SEEDS:
                if (arm, s) not in treat:
                    raise SystemExit(
                        f"REFUSING: missing treatment cell {arm}/{s}")
        dz = {k: [treat[(f"tmp_{k}", s)] - qual[("tmp_NONE", s)]
                  for s in SEEDS] for k in ("Z2", "Z3")}
        d2, d3 = sum(dz["Z2"]), sum(dz["Z3"])
        ms["3"] = {"value": d2, "metric": "pooled_delta",
                   "population": "z2_v_none:3seeds",
                   "aggregation": "pooled_sum"}
        ms["3:signs"] = {"value": sum(1 for x in dz["Z2"] if x > 0),
                         "metric": "n_positive_seeds",
                         "population": "z2_v_none:3seeds",
                         "aggregation": "count_of_3"}
        ms["4"] = {"value": max(d2, d3), "metric": "max_pooled_delta",
                   "population": "z2z3_v_none:3seeds",
                   "aggregation": "pooled_sum"}
        ms["4:delta_z2"] = {"value": d2, "metric": "pooled_delta",
                            "population": "z2_v_none:3seeds",
                            "aggregation": "pooled_sum"}
        ms["4:delta_z3"] = {"value": d3, "metric": "pooled_delta",
                            "population": "z3_v_none:3seeds",
                            "aggregation": "pooled_sum"}
        ms["5"] = {"value": d2 - d3, "metric": "delta_gap",
                   "population": "z2_v_z3:3seeds",
                   "aggregation": "pooled_sum"}
        ms["refutation:delta_z2"] = ms["3"]
        # DOSE CENSUS (AMENDMENT -DOSE): a perprob row without the
        # 'recall' key had zero masked calls on that problem — the
        # count-k call never happened (completion too short). The
        # contrast is admissible only under <= 36/360 (10%) per arm.
        zero_dose = {"Z2": 0, "Z3": 0}
        for line in (d / "treatment_perprob.jsonl").read_text() \
                .splitlines():
            r = json.loads(line)
            arm = r["frac"].removeprefix("tmp_")
            if arm in zero_dose and "recall" not in r:
                zero_dose[arm] += 1
        obs["dose_census"] = zero_dose
        obs["contrasts"] = {}
        for bar, arms_involved in (("3", ("Z2",)), ("4", ("Z2", "Z3")),
                                   ("5", ("Z2", "Z3"))):
            bad = [a for a in arms_involved if zero_dose[a] > 36]
            obs["contrasts"][bar] = (
                {"admissible": True} if not bad else
                {"admissible": False,
                 "reason": f"zero-dose problems over 10% pooled: "
                           f"{ {a: zero_dose[a] for a in bad} }"})
    else:
        obs["measurement_valid"] = True
        for a in ("z2", "z3"):
            obs["arms"][a] = {
                "admissible": False,
                "reason": "qualification bridge failed "
                          f"(anchors {n_exact}/6, census "
                          f"{census['n_ok']}/48) — treatment sealed, "
                          "no verdict on Z2/Z3"}
    print(json.dumps(obs, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/ex6temporal")

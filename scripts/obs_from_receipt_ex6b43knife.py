"""Observations adapter for PRE-REG EX6-B43-KNIFE-0.

Bar 1 anchors from qual.jsonl, bar 2 from census.json (5 arms x
48 modules), bars 3-5 and the refutation predicate from
treatment.jsonl + qual.jsonl pooled deltas (NATIVE pooled anchor
191, FULL pooled anchor 211). Refuses smoke rows and partial
populations. The refutation predicate knife_inversion is 1 iff
Delta_D_ONLY <= +9 AND Delta_NO_D >= +13, else 0 (prereg
direction 'above 0').

    .venv/bin/python scripts/obs_from_receipt_ex6b43knife.py \
        logs/ex6b43knife > logs/ex6b43knife/ex6b43knife_observations.json
    .venv/bin/python scripts/adjudicate.py \
        docs/preregs/ex6-b43-knife-0.json \
        logs/ex6b43knife/ex6b43knife_observations.json
"""
import json
import sys
from pathlib import Path

SEEDS = (7001, 8002, 9003)
N_EVAL = 120
ANCHORS = {"kn_NATIVE": {7001: 64, 8002: 61, 9003: 66},
           "kn_FULL_DIRECT": {7001: 70, 8002: 70, 9003: 71}}
KNIVES = ("D_ONLY", "NO_D", "FULL_SUM")
FULL_POOLED = 211


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
                    ("native", "full_direct", "d_only", "no_d",
                     "full_sum")},
           "measurements": {}}
    ms = obs["measurements"]
    ms["1"] = {"value": n_exact, "metric": "n_exact_reproductions",
               "population": "anchors:native+full_direct:3seeds",
               "aggregation": "count_of_6"}
    ms["2"] = {"value": n_cells, "metric": "n_module_arm_cells_ok",
               "population": "seed7001:problem2:all5arms",
               "aggregation": "count_of_240"}
    census_pass = all(v["pass"] for v in cen.values())
    if n_exact == 6 and census_pass:
        for arm in KNIVES:
            for s in SEEDS:
                if (f"kn_{arm}", s) not in treat:
                    raise SystemExit(
                        f"REFUSING: missing treatment cell {arm}/{s}")
        dz = {a: [treat[(f"kn_{a}", s)]["gate_ok"]
                  - qual[("kn_NATIVE", s)]["gate_ok"] for s in SEEDS]
              for a in KNIVES}
        d_d, d_nod = sum(dz["D_ONLY"]), sum(dz["NO_D"])
        full_sum_pooled = sum(treat[("kn_FULL_SUM", s)]["gate_ok"]
                              for s in SEEDS)
        ms["3"] = {"value": d_d, "metric": "pooled_delta",
                   "population": "d_only_v_native:3seeds",
                   "aggregation": "pooled_sum"}
        ms["3:signs"] = {"value": sum(1 for x in dz["D_ONLY"]
                                      if x > 0),
                         "metric": "n_positive_seeds",
                         "population": "d_only_v_native:3seeds",
                         "aggregation": "count_of_3"}
        ms["4"] = {"value": d_nod, "metric": "pooled_delta",
                   "population": "no_d_v_native:3seeds",
                   "aggregation": "pooled_sum"}
        ms["5"] = {"value": abs(full_sum_pooled - FULL_POOLED),
                   "metric": "abs_pooled_gap_to_full_anchor",
                   "population": "full_sum_v_booked_211:3seeds",
                   "aggregation": "pooled_sum"}
        ms["refutation:knife_inversion"] = {
            "value": int(d_d <= 9 and d_nod >= 13),
            "metric": "knife_inversion_indicator",
            "population": "d_only+no_d_v_native:3seeds",
            "aggregation": "pooled_sum"}
        obs["color"] = {"delta_d_only": d_d, "delta_no_d": d_nod,
                        "pooled_full_sum": full_sum_pooled,
                        "knife_interaction":
                            full_sum_pooled - 191 - (d_d + d_nod)}
    else:
        for a in ("d_only", "no_d", "full_sum"):
            obs["arms"][a] = {
                "admissible": False,
                "reason": "qualification bridge failed "
                          f"(anchors {n_exact}/6, census "
                          f"{n_cells}/240) — treatment sealed, no "
                          "verdict on knife arms"}
    print(json.dumps(obs, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/ex6b43knife")

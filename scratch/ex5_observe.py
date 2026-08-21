"""EX5-LAYERMATCH-0 observations builder: derives every registered
measurement from the streamed receipts (logs/ex5/qual.jsonl +
logs/ex5/ex5.jsonl) and writes the observations document the machine
adjudicator consumes. All definitions are the pre-reg's: Delta_m =
sum over seeds 4001/5002/6003 of (gate_m - gate_full128); a mask
CLEARS +21 when Delta >= 21 AND per-seed deltas same-sign 3/3.

Receipt: logs/ex5/ex5_observations.json (refuse-if-exists).

    .venv/bin/python scratch/ex5_observe.py                (Mac desk)
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SEEDS = [4001, 5002, 6003]
RANK = [f"ex5_del_rank{j}" for j in range(3)]
LAYER = [f"ex5_del_layer{j}" for j in range(3)]
OUT = Path("logs/ex5/ex5_observations.json")
QUAL_EXPECT = {"ex1_full128": (59, {"1": 22, "2": 19, "3": 18}),
               "ex3_del_rand0": (70, {"1": 26, "2": 24, "3": 20})}


def main():
    import os
    if OUT.exists() and os.environ.get("OBS_OVERWRITE") != "1":
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/ex5_observe.py", "scratch/ex5_run.sh",
         "scratch/gt7_run.py", "scratch/moe_gt1_arm2.py",
         "scratch/ex5_build.py"])
    qual = {r["arm"]: r for r in
            map(json.loads, open("logs/ex5/qual.jsonl"))}
    n_exact = sum(1 for arm, (tot, lv) in QUAL_EXPECT.items()
                  if qual[arm]["gate_ok"] == tot
                  and qual[arm]["gate_per_level"] == lv
                  and qual[arm]["seed"] == 1001)
    rows = list(map(json.loads, open("logs/ex5/ex5.jsonl")))
    assert len(rows) == 24, len(rows)
    gate = {(r["seed"], r["arm"]): r["gate_ok"] for r in rows}

    def deltas(arm):
        return [gate[(s, arm)] - gate[(s, "ex1_full128")]
                for s in SEEDS]

    def clears(ds, thresh=21):
        return sum(ds) >= thresh and (all(d > 0 for d in ds)
                                      or all(d < 0 for d in ds))

    named_d = deltas("ex3_del_invp")
    rank_d = {m: deltas(m) for m in RANK}
    layer_d = {m: deltas(m) for m in LAYER}
    rank_sums = [sum(d) for d in rank_d.values()]
    layer_sums = [sum(d) for d in layer_d.values()]
    med_rank = statistics.median(rank_sums)
    med_layer = statistics.median(layer_sums)

    def premium(m):
        return [gate[(s, "ex3_del_invp")] - gate[(s, m)]
                for s in SEEDS]

    prem = {m: premium(m) for m in RANK}

    def M(v, metric, pop, agg, prov):
        return {"value": v, "metric": metric, "population": pop,
                "aggregation": agg, "provenance": prov}

    import hashlib
    reused = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in ("checkpoints/ex1_full128.json",
                        "checkpoints/ex3_del_invp.json",
                        "checkpoints/ex3_del_rand0.json")}
    obs = {
        "measurement_valid": True,
        "start": START, "completion_commit": completion_commit(),
        "audit_notes": {
            "battery_field": "every receipt row carries "
                "battery='gt7', a literal inherited from the frozen "
                "gt7_run.py driver (results-cited, never edited); "
                "EX5 membership is defined by receipt path "
                "logs/ex5/ + seed sets {4001,5002,6003}/{1001}, "
                "NEVER by the battery field — grouping the corpus "
                "by battery=='gt7' pools EX3/EX4/EX5 cells",
            "answers_log_seed": "ex5_answers.jsonl rows carry "
                "(arm, idx) only; the three seed passes append in "
                "launch order and are separable only by order — "
                "use ex5_perprob.jsonl (carries seed) for any "
                "per-seed answer analysis",
            "model_identity": "moe_gt1_arm2.py loads by repo name "
                "with no revision pin; identity established by the "
                "single-snapshot cache + refs/main == d388dead... "
                "+ per-file shas in logs/ex5/model_manifest.json "
                "(manifest built from the remote listing; "
                ".gitattributes and README are not loaded by the "
                "run)",
            "perprob_frac_field": "the perprob 'frac' field holds "
                "the arm NAME (legacy field name from the "
                "fraction-sweep era)",
        },
        "reused_keepset_sha256": reused,
        "arms": {a: {"admissible": True} for a in
                 ["full128", "named80", "rank0", "rank1", "rank2",
                  "layer0", "layer1", "layer2"]},
        "detail": {
            "full_per_seed": {str(s): gate[(s, "ex1_full128")]
                              for s in SEEDS},
            "named_deltas": named_d,
            "rank_deltas": {m: rank_d[m] for m in RANK},
            "layer_deltas": {m: layer_d[m] for m in LAYER},
            "rank_sums": rank_sums, "layer_sums": layer_sums,
            "premium_deltas": prem,
        },
        "measurements": {
            "1": M(n_exact, "n_exact_reproductions",
                   "seed1001:full128+ex3_del_rand0", "count_of_2",
                   "logs/ex5/qual.jsonl v booked cells"),
            "2": M(sum(1 for d in layer_d.values() if clears(d)),
                   "n_masks_clearing_plus21", "layer_family:3masks",
                   "count_of_3",
                   f"sums {layer_sums}"),
            "3": M(sum(1 for d in rank_d.values() if clears(d)),
                   "n_masks_clearing_plus21", "rank_family:3masks",
                   "count_of_3",
                   f"sums {rank_sums}"),
            "4": M(med_rank - med_layer, "median_delta_gap",
                   "rank_v_layer:3v3masks", "median_over_masks",
                   f"median {med_rank} - {med_layer}"),
            "4:nonoverlap_margin": M(
                min(rank_sums) - max(layer_sums), "median_delta_gap",
                "rank_v_layer:3v3masks", "median_over_masks",
                f"min(rank) {min(rank_sums)} - max(layer) "
                f"{max(layer_sums)}"),
            "5": M(sum(named_d), "pooled_delta",
                   "named80_v_full:3seeds", "pooled_sum",
                   f"deltas {named_d}"),
            "5:n_positive_seeds": M(
                sum(1 for d in named_d if d > 0), "pooled_delta",
                "named80_v_full:3seeds", "pooled_sum",
                f"signs of {named_d}"),
            "6": M(sum(1 for m in RANK if clears(prem[m])),
                   "n_masks_clearing_premium",
                   "named_v_rank_family:3masks", "count_of_3",
                   f"premium sums "
                   f"{[sum(prem[m]) for m in RANK]}"),
            "7": M(abs(med_rank - 28), "abs_median_dev_from_28",
                   "rank_family:3masks", "median_over_masks",
                   f"|{med_rank} - 28|"),
            "8": M(max(max(rank_sums) - min(rank_sums),
                       max(layer_sums) - min(layer_sums)),
                   "max_family_range", "both_families:worst",
                   "range_over_masks",
                   f"rank range {max(rank_sums)-min(rank_sums)}, "
                   f"layer range {max(layer_sums)-min(layer_sums)}"),
            "refutation:separation_median": M(
                med_rank - med_layer, "separation_median",
                "rank_v_layer:3v3masks", "median_over_masks",
                f"{med_rank} - {med_layer}"),
        },
    }
    OUT.write_text(json.dumps(obs, indent=1) + "\n")
    for k in sorted(obs["measurements"], key=str):
        m = obs["measurements"][k]
        print(f"{k}: {m['value']}  ({m['provenance']})")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

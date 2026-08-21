"""EX6-PHASE-0 observations builder: derives every registered
measurement from logs/ex6/qual.jsonl (v2 rows) + logs/ex6/ex6.jsonl
for the machine adjudicator. Delta_mode = sum over seeds
7001/8002/9003 of (gate_mode - gate_none), per the pre-reg.

Receipt: logs/ex6/ex6_observations.json (refuse-if-exists).

    .venv/bin/python scratch/ex6_observe.py                (Mac desk)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SEEDS = [7001, 8002, 9003]
OUT = Path("logs/ex6/ex6_observations.json")
QUAL_EXPECT = {"ex6_qual_none_v2": (59, {"1": 22, "2": 19, "3": 18}),
               "ex6_qual_all_v2": (78, {"1": 28, "2": 26, "3": 24})}


def main():
    import os
    if OUT.exists() and os.environ.get("OBS_OVERWRITE") != "1":
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/ex6_observe.py", "scratch/ex6_phase.py",
         "scratch/ex6_run.sh", "scratch/moe_gt1_arm2.py"])
    qual = {r["arm"]: r for r in
            map(json.loads, open("logs/ex6/qual.jsonl"))
            if r["arm"].endswith("_v2")}
    n_exact = sum(1 for arm, (tot, lv) in QUAL_EXPECT.items()
                  if qual[arm]["gate_ok"] == tot
                  and qual[arm]["gate_per_level"] == lv
                  and qual[arm]["seed"] == 1001)
    rows = list(map(json.loads, open("logs/ex6/ex6.jsonl")))
    assert len(rows) == 12, len(rows)
    gate = {(r["seed"], r["mode"]): r["gate_ok"] for r in rows}

    def deltas(mode):
        return [gate[(s, mode)] - gate[(s, "NONE")] for s in SEEDS]

    d_all, d_prompt, d_decode = (deltas(m) for m in
                                 ("ALL", "PROMPT", "DECODE"))

    def M(v, metric, pop, agg, prov):
        return {"value": v, "metric": metric, "population": pop,
                "aggregation": agg, "provenance": prov}

    def signs(ds):
        return sum(1 for d in ds if d > 0)

    import hashlib
    keepset_sha = hashlib.sha256(
        Path("checkpoints/ex3_del_invp.json").read_bytes()).hexdigest()
    obs = {
        "measurement_valid": True,
        "start": START, "completion_commit": completion_commit(),
        "keepset_sha256": {"checkpoints/ex3_del_invp.json":
                           keepset_sha},
        "arms": {a: {"admissible": True}
                 for a in ("none", "all", "prompt", "decode")},
        "detail": {"none_per_seed": {str(s): gate[(s, "NONE")]
                                     for s in SEEDS},
                   "deltas": {"all": d_all, "prompt": d_prompt,
                              "decode": d_decode}},
        "measurements": {
            "1": M(n_exact, "n_exact_reproductions",
                   "seed1001:none_v_full128+all_v_named80",
                   "count_of_2", "logs/ex6/qual.jsonl v2 rows v "
                   "booked cells"),
            "2": M(sum(d_all), "pooled_delta", "all_v_none:3seeds",
                   "pooled_sum", f"deltas {d_all}"),
            "2:signs": M(signs(d_all), "n_positive_seeds",
                         "all_v_none:3seeds", "count_of_3",
                         f"signs of {d_all}"),
            "3": M(sum(d_decode), "pooled_delta",
                   "decode_v_none:3seeds", "pooled_sum",
                   f"deltas {d_decode}"),
            "3:signs": M(signs(d_decode), "n_positive_seeds",
                         "decode_v_none:3seeds", "count_of_3",
                         f"signs of {d_decode}"),
            "4": M(abs(sum(d_all) - sum(d_decode)), "abs_delta_gap",
                   "all_v_decode:3seeds", "pooled_sum",
                   f"|{sum(d_all)} - {sum(d_decode)}|"),
            "5": M(sum(d_prompt), "pooled_delta",
                   "prompt_v_none:3seeds", "pooled_sum",
                   f"deltas {d_prompt}"),
            "5:signs": M(signs(d_prompt), "n_positive_seeds",
                         "prompt_v_none:3seeds", "count_of_3",
                         f"signs of {d_prompt}"),
            "6": M(sum(d_decode) - sum(d_prompt), "delta_gap",
                   "decode_v_prompt:3seeds", "pooled_sum",
                   f"{sum(d_decode)} - {sum(d_prompt)}"),
            "7": M(abs(sum(d_all) - (sum(d_prompt) + sum(d_decode))),
                   "abs_additivity_residual", "all_phases:3seeds",
                   "pooled_sum",
                   f"|{sum(d_all)} - ({sum(d_prompt)} + "
                   f"{sum(d_decode)})|"),
            "refutation:delta_decode": M(
                sum(d_decode), "delta_decode",
                "decode_v_none:3seeds", "pooled_sum",
                f"pooled {d_decode}"),
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

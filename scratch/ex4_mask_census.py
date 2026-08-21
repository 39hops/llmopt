"""EX4 mask-composition census (desk, observation-only, zero GPU).

One receipt describing every deletion arm already in the ledger, so
the next EX-ANAT registration can cite composition facts instead of
recomputing them: for each arm the 48-layer deletion vector, its L1
distance to the named-80 layer profile, summed pooled demand and
demand share, in-layer demand-rank distribution of the deleted
slots, and carrier-class membership (overlap with the named 80).

Arms: ex3_del_invp (named 80), ex3_del_rand0/1 (layer+rank matched),
ex4_del_unif0/1 (bank-wide uniform), ex4_del_top80 (top demand).
Demand instrument = pooled arm0 counts (checkpoints/moe_gt1_arm0.json),
the same table that ordered the carriers' rank windows.

Receipt: logs/ex4/mask_census.json (refuse-if-exists).

    .venv/bin/python scratch/ex4_mask_census.py            (Mac desk)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

N_LAYERS, N_EXPERTS = 48, 128
ARMS = ["ex3_del_invp", "ex3_del_rand0", "ex3_del_rand1",
        "ex4_del_unif0", "ex4_del_unif1", "ex4_del_top80"]
ARMS += [a for a in os.environ.get("EXTRA_ARMS", "").split(",") if a]
OUT = Path(os.environ.get("CENSUS_OUT", "logs/ex4/mask_census.json"))


def deleted_slots(keepset):
    return sorted((l, e) for l in range(N_LAYERS)
                  for e in range(N_EXPERTS)
                  if e not in set(keepset[str(l)]))


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/ex4_mask_census.py", "scratch/ex3_build.py",
         "scratch/ex4_build.py"])
    arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
    pooled = {int(l): r for l, r in arm0["counts"].items()}
    total_demand = sum(sum(pooled[l]) for l in range(N_LAYERS))
    named = {tuple(x) for x in json.loads(
        Path("checkpoints/ex3_inv_pooled.json").read_text())}
    # in-layer demand order, ties ascending id (the frozen carrier rule)
    order = {l: sorted(range(N_EXPERTS),
                       key=lambda x: (-pooled[l][x], x))
             for l in range(N_LAYERS)}
    rank_of = {l: {e: i for i, e in enumerate(order[l])}
               for l in range(N_LAYERS)}

    profiles = {}
    rows = {}
    for arm in ARMS:
        ks = json.loads(Path(f"checkpoints/{arm}.json").read_text())
        dele = deleted_slots(ks)
        vec = [sum(1 for (l, _) in dele if l == li)
               for li in range(N_LAYERS)]
        profiles[arm] = vec
        ranks = sorted(rank_of[l][e] for (l, e) in dele)
        dem = sum(pooled[l][e] for (l, e) in dele)
        rows[arm] = {
            "n_deleted": len(dele),
            "layer_deletion_vector": vec,
            "n_layers_touched": sum(1 for v in vec if v),
            "max_per_layer": max(vec),
            "summed_pooled_demand": dem,
            "demand_share": round(dem / total_demand, 6),
            "in_layer_rank_min": ranks[0],
            "in_layer_rank_median": ranks[len(ranks) // 2],
            "in_layer_rank_max": ranks[-1],
            "n_named_carrier_slots": sum(1 for s in dele
                                         if tuple(s) in named),
        }
    ref = profiles["ex3_del_invp"]
    for arm in ARMS:
        rows[arm]["l1_to_named_layer_profile"] = sum(
            abs(a - b) for a, b in zip(profiles[arm], ref))

    rcpt = {"note": "EX4 mask-composition census (desk, observation-"
                    "only): layer profiles, demand, in-layer ranks, "
                    "carrier membership per deletion arm",
            "start": START, "completion_commit": completion_commit(),
            "demand_instrument": "checkpoints/moe_gt1_arm0.json "
                                 "pooled counts (in-layer order = "
                                 "-count, ascending id)",
            "total_pooled_demand": total_demand,
            "arms": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    for arm in ARMS:
        r = rows[arm]
        print(f"{arm}: L1-to-named {r['l1_to_named_layer_profile']:3d}"
              f"  demand_share {r['demand_share']:.4f}"
              f"  rank med/max {r['in_layer_rank_median']}/"
              f"{r['in_layer_rank_max']}"
              f"  named-slots {r['n_named_carrier_slots']}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

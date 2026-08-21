"""EX5-TRAJ-ANATOMY-0 prompt-normalized rider (desk, zero-cost,
frozen by the commit carrying this file before any value is read).

The booked -0.0352 decode failure enrichment is selection/row
weighted: long failing generations contribute more rows, so it
cannot distinguish BROAD failure recruitment (named slots engage
more on most failing prompts) from REPEATED recruitment on a
subset. Per group and per prompt, over decode rows of
logs/opus/moe_gt1_traj_v2.jsonl only:

  rate(p)      = group selections in decode rows of p
                 / (decode rows of p * 8)     [opportunities]
  mean_rate    compared solved v failed prompts (unweighted over
               prompts — the prompt-normalized read)
  incidence    fraction of prompts with >= 1 group decode
               selection, solved v failed (reported only if
               non-saturated: some group x outcome cell < 0.95)

DISCLOSED: this artifact predates the prompt_tail phase split, so
its decode rows include the one chat-template tail step per
prompt x layer (<= 48 rows/prompt, <= ~4.3% of decode rows).

Receipt: logs/ex5/traj_rider.json (refuse-if-exists).

    .venv/bin/python scratch/ex5_traj_rider.py             (Mac desk)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

TRAJ = "logs/opus/moe_gt1_traj_v2.jsonl"
OUT = Path("logs/ex5/traj_rider.json")


def mask_slots(name):
    ks = json.loads(Path(f"checkpoints/{name}.json").read_text())
    return {(l, e) for l in range(48) for e in range(128)
            if e not in set(ks[str(l)])}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex5_traj_rider.py",
                              "scratch/ex5_traj_census.py"])
    groups = {"named80": {tuple(x) for x in json.loads(
        Path("checkpoints/ex3_inv_pooled.json").read_text())}}
    for fam in ("rank", "layer"):
        for j in range(3):
            groups[f"{fam}{j}"] = mask_slots(f"ex5_del_{fam}{j}")

    n_dec = defaultdict(int)          # prompt -> decode rows
    sel = defaultdict(int)            # (group, prompt) -> selections
    inc = defaultdict(set)            # group -> prompts with >=1
    ok_of = {}
    with open(TRAJ) as f:
        for line in f:
            r = json.loads(line)
            if not isinstance(r["prompt"], int) \
                    or r["phase"] != "decode":
                continue
            p, l = r["prompt"], r["layer"]
            n_dec[p] += 1
            ok_of[p] = bool(r["ok"])
            for e in r["topk"]:
                for g, slots in groups.items():
                    if (l, e) in slots:
                        sel[(g, p)] += 1
                        inc[g].add(p)

    prompts = sorted(n_dec)
    solved = [p for p in prompts if ok_of[p]]
    failed = [p for p in prompts if not ok_of[p]]
    out = {}
    for g in groups:
        def rates(ps):
            return [sel.get((g, p), 0) / (n_dec[p] * 8) for p in ps]
        rs, rf = rates(solved), rates(failed)
        inc_s = sum(1 for p in solved if p in inc[g]) / max(len(solved), 1)
        inc_f = sum(1 for p in failed if p in inc[g]) / max(len(failed), 1)
        out[g] = {
            "mean_rate_solved": round(sum(rs) / max(len(rs), 1), 5),
            "mean_rate_failed": round(sum(rf) / max(len(rf), 1), 5),
            "rate_gap_failed_minus_solved": round(
                sum(rf) / max(len(rf), 1) - sum(rs) / max(len(rs), 1), 5),
            "incidence_solved": round(inc_s, 4),
            "incidence_failed": round(inc_f, 4),
            "incidence_saturated": bool(min(inc_s, inc_f) >= 0.95),
        }
    rcpt = {"note": "EX5 TRAJ rider: prompt-normalized decode "
                    "recruitment, solved v failed (hypothesis "
                    "generation only)",
            "start": START, "completion_commit": completion_commit(),
            "n_prompts": len(prompts), "n_solved": len(solved),
            "n_failed": len(failed),
            "groups": out}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"prompts {len(prompts)} (solved {len(solved)} / "
          f"failed {len(failed)})")
    for g, r in out.items():
        print(f"{g:8s} rate S {r['mean_rate_solved']:.5f} "
              f"F {r['mean_rate_failed']:.5f} "
              f"gap {r['rate_gap_failed_minus_solved']:+.5f} "
              f"inc S/F {r['incidence_solved']}/"
              f"{r['incidence_failed']}"
              f"{' SATURATED' if r['incidence_saturated'] else ''}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

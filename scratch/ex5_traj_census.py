"""EX5 carrier TRAJ anatomy census (desk, observation-only,
hypothesis generation ONLY — outcome-conditioned desk findings gate
nothing and register nothing; GPT-seat feature list, house freeze).

FEATURES ARE FROZEN HERE, BEFORE ANY GROUP VALUE IS READ (the
commit carrying this file predates the receipt). Per slot group
(named80 = ex3_inv_pooled; ex5_del_rank0/1/2; ex5_del_layer0/1/2;
OTHER = all remaining (layer, expert) slots), aggregated over
logs/opus/moe_gt1_traj_v2.jsonl (the pooled math trajectory that
built the demand table; 'selected' = slot in the row's top-8):

  n_sel, sel_per_slot       raw selection mass and per-slot mean
  decode_frac               fraction of selections in decode phase
  mean_H                    router entropy at selection
  mean_score                router score of the slot when selected
  mean_rank8                rank within the top-8 (1 = strongest)
  low_margin_frac           fraction of selections at rank 7-8 (the
                            near-boundary slots of the top-8)
  mean_normpos              pos / max_pos(prompt) at selection
  first_touch               mean over (slot, prompt) of the earliest
                            normalized position of first selection
  ok_enrich_decode          P(ok | selected, decode) - P(ok | decode)
  core_corout               fraction of selections whose top-8 also
                            contains a gt3_core_keep expert of the
                            same layer (excluding the slot itself)
  demand_share              pooled arm0 demand share (covariate)
  domain_breadth            mean over slots of the number of side
                            trajectories (gt2_code, gt2_phys,
                            gt3_proofs, gt3_prose, gt4_dialog) in
                            which the slot appears in top-8 >= 100
                            times (T_DOMAIN = 100, frozen)

Receipt: logs/ex5/traj_census.json (refuse-if-exists).

    .venv/bin/python scratch/ex5_traj_census.py            (Mac desk)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

TRAJ = "logs/opus/moe_gt1_traj_v2.jsonl"
DOMS = {"code": "logs/opus/gt2_code_traj.jsonl",
        "phys": "logs/opus/gt2_phys_traj.jsonl",
        "proofs": "logs/opus/gt3_proofs_traj.jsonl",
        "prose": "logs/opus/gt3_prose_traj.jsonl",
        "dialog": "logs/opus/gt4_dialog_traj.jsonl"}
T_DOMAIN = 100
OUT = Path("logs/ex5/traj_census.json")


def mask_slots(name):
    ks = json.loads(Path(f"checkpoints/{name}.json").read_text())
    return {(l, e) for l in range(48) for e in range(128)
            if e not in set(ks[str(l)])}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex5_traj_census.py",
                              "scratch/ex3_build.py",
                              "scratch/ex5_build.py"])
    named = {tuple(x) for x in json.loads(
        Path("checkpoints/ex3_inv_pooled.json").read_text())}
    groups = {"named80": named}
    for fam in ("rank", "layer"):
        for j in range(3):
            groups[f"{fam}{j}"] = mask_slots(f"ex5_del_{fam}{j}")
    member = defaultdict(list)
    for g, slots in groups.items():
        for s in slots:
            member[s].append(g)
    all_named = set().union(*groups.values())
    core = {int(l): set(v) for l, v in json.load(
        open("checkpoints/gt3_core_keep.json")).items()}
    arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
    pooled = {int(l): r for l, r in arm0["counts"].items()}
    total_demand = sum(sum(pooled[l]) for l in range(48))

    # pass 1: per-prompt max pos + per-(prompt,phase) ok stats
    maxpos = {}
    decode_rows = ok_decode_rows = 0
    with open(TRAJ) as f:
        for line in f:
            r = json.loads(line)
            key = r["prompt"]
            if r["pos"] > maxpos.get(key, 0):
                maxpos[key] = r["pos"]
            if r["layer"] == 0 and r["phase"] == "decode":
                decode_rows += 1
                ok_decode_rows += bool(r["ok"])
    base_ok = ok_decode_rows / max(decode_rows, 1)

    # pass 2: per-slot accumulation (only slots in any group + a
    # global OTHER pool accumulated directly)
    acc = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0])
    # [n_sel, n_decode, sum_H, sum_score, sum_rank, n_lowmargin,
    #  sum_normpos, n_ok_decode, n_corout]
    first = {}
    other = [0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0]
    n_other_slots = 48 * 128 - len(all_named)
    ft_other = {}
    with open(TRAJ) as f:
        for line in f:
            r = json.loads(line)
            l, mp = r["layer"], max(maxpos[r["prompt"]], 1)
            np_ = r["pos"] / mp
            dec = r["phase"] == "decode"
            core_l = core.get(l, set())
            tk = r["topk"]
            has_core = [e for e in tk if e in core_l]
            for i, e in enumerate(tk):
                rank = len(tk) - i
                s = (l, e)
                a = acc[s] if s in member else other
                a[0] += 1
                a[1] += dec
                a[2] += r["H"]
                a[3] += r["scores"][i]
                a[4] += rank
                a[5] += rank >= 7
                a[6] += np_
                a[7] += bool(r["ok"]) if dec else 0
                a[8] += bool([c for c in has_core if c != e])
                fd = first if s in member else ft_other
                k = (s, r["prompt"]) if s in member else (l, e, r["prompt"])
                if k not in fd or np_ < fd[k]:
                    fd[k] = np_

    doms = {}
    for dname, path in DOMS.items():
        cnt = defaultdict(int)
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                l = r["layer"]
                for e in r["topk"]:
                    if (l, e) in member:
                        cnt[(l, e)] += 1
        doms[dname] = {s: c for s, c in cnt.items() if c >= T_DOMAIN}

    def agg(slots, a_pool=None, ft_pool=None):
        if a_pool is None:
            rows = [acc[s] for s in slots if acc[s][0]]
            n_slots = len(slots)
            tot = [sum(r[i] for r in rows) for i in range(9)]
            fts = [v for (s, _), v in first.items() if s in slots]
        else:
            tot, n_slots = a_pool, n_other_slots
            fts = list(ft_pool.values())
        n = max(tot[0], 1)
        nd = max(tot[1], 1)
        return {
            "n_slots": n_slots, "n_sel": tot[0],
            "sel_per_slot": round(tot[0] / max(n_slots, 1), 1),
            "decode_frac": round(tot[1] / n, 4),
            "mean_H": round(tot[2] / n, 4),
            "mean_score": round(tot[3] / n, 4),
            "mean_rank8": round(tot[4] / n, 3),
            "low_margin_frac": round(tot[5] / n, 4),
            "mean_normpos": round(tot[6] / n, 4),
            "ok_enrich_decode": round(tot[7] / nd - base_ok, 4),
            "core_corout": round(tot[8] / n, 4),
            "first_touch": round(sum(fts) / max(len(fts), 1), 4),
        }

    out_groups = {}
    for g, slots in groups.items():
        row = agg(slots)
        row["demand_share"] = round(
            sum(pooled[l][e] for (l, e) in slots) / total_demand, 5)
        row["domain_breadth"] = round(
            sum(sum(1 for d in doms.values() if s in d)
                for s in slots) / max(len(slots), 1), 3)
        out_groups[g] = row
    orow = agg(set(), a_pool=other, ft_pool=ft_other)
    orow["demand_share"] = round(
        sum(pooled[l][e] for l in range(48) for e in range(128)
            if (l, e) not in all_named) / total_demand, 5)
    orow["domain_breadth"] = None
    out_groups["OTHER"] = orow

    rcpt = {"note": "EX5 carrier TRAJ anatomy census (desk, "
                    "hypothesis generation only; features frozen in "
                    "the committed script before any value was "
                    "read)",
            "start": START, "completion_commit": completion_commit(),
            "base_ok_decode": round(base_ok, 4),
            "t_domain": T_DOMAIN,
            "groups": out_groups}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    hdr = ["group", "sel/slot", "decF", "H", "score", "rank8",
           "lowM", "npos", "okEnr", "core", "firstT", "demand",
           "breadth"]
    print(" | ".join(hdr))
    for g, r in out_groups.items():
        print(f"{g:8s} | {r['sel_per_slot']:8.1f} | "
              f"{r['decode_frac']:.3f} | {r['mean_H']:.3f} | "
              f"{r['mean_score']:.4f} | {r['mean_rank8']:.2f} | "
              f"{r['low_margin_frac']:.3f} | {r['mean_normpos']:.3f}"
              f" | {r['ok_enrich_decode']:+.4f} | "
              f"{r['core_corout']:.3f} | {r['first_touch']:.3f} | "
              f"{r['demand_share']:.4f} | {r['domain_breadth']}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

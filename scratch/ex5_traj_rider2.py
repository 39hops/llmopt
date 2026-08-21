"""EX5 TRAJ rider v2 — prompt_tail-corrected (desk, frozen by this
commit before any value is read; hypothesis generation only).

The v1 rider counted every historical phase=="decode" row, but
moe_gt1_traj_v2.jsonl predates the prompt_tail split: the FIRST
one-token row per (prompt, layer) is the chat-template tail step
mislabeled decode. named80 is uniquely prompt-heavy, so that <=4.3%
contamination can disproportionately affect its small solved/failed
recruitment gap. This rider reclassifies that first old-decode row
per (prompt, layer) as prompt_tail and reports TRUE-DECODE and
PROMPT-TAIL separately, per group:

  ok_enrich          selection-weighted P(ok | selected) - P(ok)
                     within the phase class
  rate gap           per-prompt normalized recruitment
                     (selections / opportunities), mean over
                     prompts, failed minus solved
  rate gap dispersion  per-prompt paired spread is NOT computable
                     (solved and failed are different prompts);
                     reported instead: sd of rate over prompts
                     within each outcome class, so the gap can be
                     read against between-prompt spread

Receipt: logs/ex5/traj_rider2.json (refuse-if-exists).

    .venv/bin/python scratch/ex5_traj_rider2.py            (Mac desk)
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

TRAJ = "logs/opus/moe_gt1_traj_v2.jsonl"
OUT = Path("logs/ex5/traj_rider2.json")


def mask_slots(name):
    ks = json.loads(Path(f"checkpoints/{name}.json").read_text())
    return {(l, e) for l in range(48) for e in range(128)
            if e not in set(ks[str(l)])}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex5_traj_rider2.py",
                              "scratch/ex5_traj_rider.py"])
    groups = {"named80": {tuple(x) for x in json.loads(
        Path("checkpoints/ex3_inv_pooled.json").read_text())}}
    for fam in ("rank", "layer"):
        for j in range(3):
            groups[f"{fam}{j}"] = mask_slots(f"ex5_del_{fam}{j}")

    seen_tail = set()                 # (prompt, layer) tail consumed
    # per phase-class accumulators
    n_rows = {"true_decode": 0, "prompt_tail": 0}
    n_ok_rows = {"true_decode": 0, "prompt_tail": 0}
    n_dec_rows_p = defaultdict(int)   # prompt -> true-decode rows
    n_tail_rows_p = defaultdict(int)
    sel = defaultdict(int)            # (phase, group, prompt) -> n
    sel_ok = defaultdict(int)         # (phase, group) -> ok sels
    sel_n = defaultdict(int)          # (phase, group) -> sels
    ok_of = {}
    with open(TRAJ) as f:
        for line in f:
            r = json.loads(line)
            if not isinstance(r["prompt"], int) \
                    or r["phase"] != "decode":
                continue
            p, l = r["prompt"], r["layer"]
            key = (p, l)
            if key not in seen_tail:
                seen_tail.add(key)
                ph = "prompt_tail"
                n_tail_rows_p[p] += 1
            else:
                ph = "true_decode"
                n_dec_rows_p[p] += 1
            n_rows[ph] += 1
            n_ok_rows[ph] += bool(r["ok"])
            ok_of[p] = bool(r["ok"])
            for e in r["topk"]:
                for g, slots in groups.items():
                    if (l, e) in slots:
                        sel[(ph, g, p)] += 1
                        sel_n[(ph, g)] += 1
                        sel_ok[(ph, g)] += bool(r["ok"])

    prompts = sorted(set(n_dec_rows_p) | set(n_tail_rows_p))
    solved = [p for p in prompts if ok_of[p]]
    failed = [p for p in prompts if not ok_of[p]]
    out = {}
    for ph, nrp in (("true_decode", n_dec_rows_p),
                    ("prompt_tail", n_tail_rows_p)):
        base = n_ok_rows[ph] / max(n_rows[ph], 1)
        phase_out = {"n_rows": n_rows[ph],
                     "base_ok_rowweighted": round(base, 4)}
        for g in groups:
            def rates(ps):
                return [sel.get((ph, g, p), 0) / (nrp[p] * 8)
                        for p in ps if nrp.get(p)]
            rs, rf = rates(solved), rates(failed)
            ms = sum(rs) / max(len(rs), 1)
            mf = sum(rf) / max(len(rf), 1)
            phase_out[g] = {
                "ok_enrich": round(
                    sel_ok[(ph, g)] / max(sel_n[(ph, g)], 1) - base, 4),
                "mean_rate_solved": round(ms, 5),
                "mean_rate_failed": round(mf, 5),
                "rate_gap": round(mf - ms, 5),
                "rate_sd_solved": round(
                    statistics.pstdev(rs) if len(rs) > 1 else 0.0, 5),
                "rate_sd_failed": round(
                    statistics.pstdev(rf) if len(rf) > 1 else 0.0, 5),
            }
        out[ph] = phase_out

    rcpt = {"note": "EX5 TRAJ rider v2: prompt_tail-corrected phase "
                    "split (first old-decode row per (prompt,layer) "
                    "reclassified); hypothesis generation only",
            "start": START, "completion_commit": completion_commit(),
            "n_prompts": len(prompts), "n_solved": len(solved),
            "n_failed": len(failed),
            "phases": out}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    for ph, po in out.items():
        print(f"== {ph}: rows {po['n_rows']} base_ok "
              f"{po['base_ok_rowweighted']}")
        for g in groups:
            r = po[g]
            print(f"  {g:8s} enr {r['ok_enrich']:+.4f}  "
                  f"rate S {r['mean_rate_solved']:.5f} "
                  f"(sd {r['rate_sd_solved']:.5f}) "
                  f"F {r['mean_rate_failed']:.5f} "
                  f"(sd {r['rate_sd_failed']:.5f}) "
                  f"gap {r['rate_gap']:+.5f}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

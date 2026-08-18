"""QWEN-RK-CENSUS-0 observations builder + adjudication.

Reads logs/qwenrouter/rk_census.json (committed producer:
scratch/qwen_rk_census.py MODE=analyze, whose fail-closed sidecar
protocol has already gated the inputs), builds the single-bar
observations for docs/preregs/qwen-rk-census-0.json, adjudicates,
and writes rk_observations.json + rk_verdict.txt. Provenance is
derived: census receipt sha, capture-sidecar commits and dirty
flags are read from the artifacts, never typed in.

    .venv/bin/python scratch/qwen_rk_adjudicate.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

OUT = "logs/qwenrouter"


def main():
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation, load)
    prereg = load("docs/preregs/qwen-rk-census-0.json")
    cp = os.path.join(OUT, "rk_census.json")
    d = json.load(open(cp))
    if d.get("gate") != "QWEN-RK-CENSUS-0":
        raise SystemExit(f"REFUSING: gate {d.get('gate')}")
    metas = {}
    for tag in ("vendor", "arm"):
        mp = os.path.join(OUT, f"capture_{tag}_meta.json")
        if not os.path.exists(mp):
            raise SystemExit(f"REFUSING: no sidecar {mp}")
        metas[tag] = json.load(open(mp))
    vals = {li: d["per_layer"][str(li)]["ks"]["1024"]["r_k"]
            for li in d["layers"]}
    mn = min(vals.values())
    m = {"value": mn, "metric": "r_k_min_over_layers",
         "population": "positions:corpus+prefixes",
         "aggregation": "mean",
         "provenance": f"min over layers {vals}"}
    obs = {"measurement_valid": True,
           "arms": {"A": {"admissible": True}},
           "measurements": {"1": dict(m),
                            "refuted:r_k_min_over_layers": dict(m)},
           "contrasts": {},
           "per_layer_r_k_at_1024": vals,
           "provenance": {
               "code_commit": subprocess.check_output(
                   ["git", "rev-parse", "--short",
                    "HEAD"]).decode().strip(),
               "tree_dirty": bool(subprocess.check_output(
                   ["git", "status", "--porcelain",
                    "-uno"]).decode().strip()),
               "census_receipt_sha256": hashlib.sha256(
                   open(cp, "rb").read()).hexdigest(),
               "census_code_commit": d["code_commit"],
               "capture_commits": {
                   t: metas[t]["code_commit"] for t in metas},
               "capture_tree_dirty": {
                   t: metas[t]["tree_dirty"] for t in metas}}}
    lines = []
    for o in adjudicate_prereg(prereg, obs):
        line = f"BAR {o.bar_id} {o.bar_name}: {o.outcome}"
        if o.reasons:
            line += " [" + "; ".join(o.reasons) + "]"
        lines.append(line)
    ref = adjudicate_refutation(prereg, obs)
    lines.append(f"REGISTERED-PRIOR(faithful-router): {ref}")
    lines.append(f"per-layer r_k@1024: {vals}")
    lines.append(f"PRODUCER {obs['provenance']['code_commit']}"
                 f" dirty={obs['provenance']['tree_dirty']}"
                 f" capture_dirty={obs['provenance']['capture_tree_dirty']}")
    for line in lines:
        print(line, flush=True)
    with open(os.path.join(OUT, "rk_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    with open(os.path.join(OUT, "rk_verdict.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[rk] -> {OUT}/rk_verdict.txt", flush=True)


if __name__ == "__main__":
    main()

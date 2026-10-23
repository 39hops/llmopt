"""Observations adapter for PRE-REG QWEN-THINKING-BATTERY-0: derives the
registered measurements from the three battery receipts (the frozen
no-thinking baseline logs/qwen_battery/results_v2.json and the two
thinking configurations written by scratch/qwen_battery_run.py under
THINK=1 with BUDGET 384 and 768) in committed code. The honest count is
pass rows over all 46 tasks (error and hold rows are non-passes).

  bar 1  think384_pass: honest passes at thinking, 384 tokens
  bar 2  think768_pass: honest passes at thinking, 768 tokens
  bar 3  think768_arith_json_pass: arithmetic plus JSON passes at 768
  bar 4  think768_other_pass: code, instruction, two-turn and termination
         passes at 768
Validity: three clean-tree non-smoke receipts on the same collection
sha256 (46 tasks each), the baseline with enable_thinking False at 384,
the two thinking receipts with enable_thinking True at their budgets.
Usage: qwen_think_obs.py <baseline.json> <think384.json> <think768.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402


def counts(r):
    by = collections.defaultdict(int)
    for t in r["tasks"]:
        if t["verdict"]["status"] == "pass":
            by[t["kind"]] += 1
    return by


def main(base_path, t384_path, t768_path, prereg_path):
    base, t384, t768 = (json.load(open(p)) for p in (base_path, t384_path, t768_path))
    prereg = load(prereg_path)
    reasons = []
    for r, nm, think, budget in ((base, "baseline", False, 384), (t384, "think384", True, 384), (t768, "think768", True, 768)):
        if r.get("smoke") or r.get("tree_dirty"):
            reasons.append(f"{nm}: smoke run or dirty tree")
        if len(r.get("tasks", [])) != 46:
            reasons.append(f"{nm}: {len(r.get('tasks', []))} tasks, not 46")
        cfg = r.get("execution_config", {"enable_thinking": r["rules"]["enable_thinking"], "max_tokens_per_turn": r["rules"]["max_tokens_per_turn"]})
        if cfg.get("enable_thinking") is not think or cfg.get("max_tokens_per_turn") != budget:
            reasons.append(f"{nm}: execution config {cfg} is not thinking={think} budget={budget}")
    if len({r["collection_sha256"] for r in (base, t384, t768)}) != 1:
        reasons.append("the three receipts are not on the same collection")
    valid = not reasons
    meas = {}
    if valid:
        cb, c3, c7 = counts(base), counts(t384), counts(t768)
        tot = lambda c: sum(c.values())  # noqa: E731
        prov = f"baseline by kind {dict(cb)} total {tot(cb)}; think384 {dict(c3)} total {tot(c3)}; think768 {dict(c7)} total {tot(c7)}"
        meas["1"] = {"value": tot(c3), "metric": "think384_pass", "population": "tasks:46", "aggregation": "count", "provenance": prov}
        meas["2"] = {"value": tot(c7), "metric": "think768_pass", "population": "tasks:46", "aggregation": "count", "provenance": prov}
        meas["3"] = {"value": c7["arith"] + c7["json"], "metric": "think768_arith_json_pass", "population": "tasks:18", "aggregation": "count",
                     "provenance": f"baseline arith+json {cb['arith'] + cb['json']}; think384 {c3['arith'] + c3['json']}"}
        meas["4"] = {"value": c7["code"] + c7["instruct"] + c7["twoturn"] + c7["degen"], "metric": "think768_other_pass", "population": "tasks:28", "aggregation": "count",
                     "provenance": f"baseline other {cb['code'] + cb['instruct'] + cb['twoturn'] + cb['degen']}; think384 {c3['code'] + c3['instruct'] + c3['twoturn'] + c3['degen']}"}
        meas["think768_arith_json_pass"] = dict(meas["3"])
    obs = {"note": f"Mechanical projection of {base_path}, {t384_path}, {t768_path} (commits {base.get('commit')}, {t384.get('commit')}, {t768.get('commit')}); adapter scratch/qwen_think_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"BASELINE": {"admissible": True, "reason": "the frozen v2 no-thinking receipt of QWEN-INFERENCE-BASELINE-0, unchanged"},
                    "THINK384": {"admissible": valid, "reason": "enable_thinking True through the pinned template, 384 tokens per turn, one run"},
                    "THINK768": {"admissible": valid, "reason": "enable_thinking True through the pinned template, 768 tokens per turn, one run"}},
           "measurements": meas}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

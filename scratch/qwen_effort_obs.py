"""Observations adapter for PRE-REG QWEN-EFFORT-0 (study A of the GO
QWEN-EFFORT-AND-DEEPCHAT-0): derives the registered measurements from the
two effort receipts (scratch/qwen_battery_run.py under THINK=1 BUDGET=768
with EFFORT low and medium) against the historical controls (the frozen
no-thinking baseline logs/qwen_battery/results_v2.json and the xhigh 768
receipt logs/qwen_battery/results_v2_think768.json), in committed code.
Every thinking receipt's verdicts are recomputed from its stored replies
and token ids with the committed amended scorer (qbattery.score_thinking);
the baseline's are read as stored. The honest count is pass rows over all
46 tasks (error and hold rows are non-passes). Reasoning tokens count to
the LAST </think>; an unterminated turn is one with no </think> id.

Bars (per arm, LOW and MEDIUM; keys <bar id> as registered):
  1/6  arith_json_pass_<arm>      arithmetic plus JSON passes of 18
  2/7  code_pass_<arm>            code passes of 8
  3/8  other_pass_<arm>           instruction, two-turn and termination passes of 20
  4/9  unterminated_turns_<arm>   turns whose think block never closed, of 52
  5/10 generated_tokens_<arm>     generated tokens over the 52 turns, with the
       conjunct <id>:time task_time_s_<arm> the sum of per-turn total time
Validity: four clean-tree non-smoke receipts on the committed collection's
sha256 (46 tasks each); the arms' execution configs enable_thinking True,
768 tokens, reasoning_effort low and medium; the controls as registered.
Usage: qwen_effort_obs.py <baseline.json> <xhigh768.json> <low.json> <medium.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab import qbattery  # noqa: E402
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

COLLECTION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalsets", "qwen_battery_v2.json")
THINK_CLOSE_ID = 248069
OTHER = ("instruct", "twoturn", "degen")


def rescored(r, tasks_by_id, thinking):
    status, why, changed = {}, {}, []
    for t in r["tasks"]:
        if not thinking:
            status[t["id"]], why[t["id"]] = t["verdict"]["status"], t["verdict"]["why"]
            continue
        task = tasks_by_id[t["id"]]
        replies = [res["text"] for res in t["results"]]
        if len(replies) < len(task["turns"]):
            v = {"status": "error", "why": t["verdict"].get("why", "incomplete turns")}
        else:
            try:
                v = qbattery.score_thinking(task, replies, t["results"])
            except Exception as e:  # noqa: BLE001
                v = {"status": "error", "why": f"validator {type(e).__name__}: {e}"}
        status[t["id"]], why[t["id"]] = v["status"], v["why"]
        if v["status"] != t["verdict"]["status"]:
            changed.append({"id": t["id"], "stored": t["verdict"]["status"], "recomputed": v["status"]})
    return status, why, changed


def per_task(r, tasks_by_id, status, why, thinking):
    out = {}
    for t in r["tasks"]:
        res = t["results"][-1]
        ids = res["generated_ids"]
        close = (len(ids) - ids[::-1].index(THINK_CLOSE_ID)) if (thinking and THINK_CLOSE_ID in ids) else None
        final = qbattery.split_final(res["text"])[1] if thinking else res["text"]
        out[t["id"]] = {"status": status[t["id"]], "why": why[t["id"]][:120], "final_answer_present": bool(final),
                        "thinking_terminated": (close is not None) if thinking else None, "generated": sum(len(x["generated_ids"]) for x in t["results"]),
                        "reasoning_tokens": close, "answer_tokens": (len(ids) - close) if close is not None else (len(ids) if not thinking else None),
                        "stop_reason": res["stop_reason"], "first_token_s": round(res["timings"]["first_token_s"], 2),
                        "total_s": round(sum(x["timings"]["total_s"] for x in t["results"]), 1),
                        "format_ok": qbattery.format_ok(tasks_by_id[t["id"]], final) if final else None}
    return out


def summary(r, tasks_by_id, status, thinking):
    turns = [res for t in r["tasks"] for res in t["results"]]
    unterminated = sum(1 for res in turns if THINK_CLOSE_ID not in res["generated_ids"]) if thinking else 0
    reasoning = sum((len(res["generated_ids"]) - res["generated_ids"][::-1].index(THINK_CLOSE_ID)) if THINK_CLOSE_ID in res["generated_ids"] else len(res["generated_ids"])
                    for res in turns) if thinking else 0
    by_kind = collections.defaultdict(collections.Counter)
    for i, s in status.items():
        by_kind[tasks_by_id[i]["kind"]][s] += 1
    return {"passes": sum(s == "pass" for s in status.values()), "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "arith_json": sum(1 for i, s in status.items() if s == "pass" and tasks_by_id[i]["kind"] in ("arith", "json")),
            "code": sum(1 for i, s in status.items() if s == "pass" and tasks_by_id[i]["kind"] == "code"),
            "other": sum(1 for i, s in status.items() if s == "pass" and tasks_by_id[i]["kind"] in OTHER),
            "unterminated_turns": unterminated, "generated_tokens": sum(len(res["generated_ids"]) for res in turns), "reasoning_tokens": reasoning,
            "answer_tokens": sum(len(res["generated_ids"]) for res in turns) - reasoning,
            "stop_reasons": dict(collections.Counter(res["stop_reason"] for res in turns)),
            "first_token_s_median": round(statistics.median(res["timings"]["first_token_s"] for res in turns), 2),
            "task_time_s": round(sum(res["timings"]["total_s"] for res in turns), 1), "wall_s": round(r.get("wall_s", 0), 1),
            "peak_bytes_max": max(res["memory"]["peak_bytes"] for res in turns), "active_bytes_max": max(res["memory"]["active_bytes"] for res in turns),
            "errors": [i for i, s in status.items() if s == "error"], "context_refusals": sum("ContextLimitError" in t["verdict"].get("why", "") for t in r["tasks"])}


def main(base_path, xhigh_path, low_path, med_path, prereg_path):
    recs = {"BASELINE": json.load(open(base_path)), "XHIGH768": json.load(open(xhigh_path)), "LOW": json.load(open(low_path)), "MEDIUM": json.load(open(med_path))}
    prereg = load(prereg_path)
    coll = json.load(open(COLLECTION))
    coll_sha = hashlib.sha256(open(COLLECTION, "rb").read()).hexdigest()
    tasks_by_id = {t["id"]: t for t in coll["tasks"]}
    want = {"BASELINE": (False, 384, None), "XHIGH768": (True, 768, "xhigh"), "LOW": (True, 768, "low"), "MEDIUM": (True, 768, "medium")}
    reasons = []
    for nm, r in recs.items():
        think, budget, effort = want[nm]
        if r.get("smoke") or r.get("tree_dirty"):
            reasons.append(f"{nm}: smoke run or dirty tree")
        if len(r.get("tasks", [])) != 46:
            reasons.append(f"{nm}: {len(r.get('tasks', []))} tasks, not 46")
        if r.get("collection_sha256") != coll_sha:
            reasons.append(f"{nm}: collection sha differs from the committed collection")
        cfg = r.get("execution_config", {"enable_thinking": r["rules"]["enable_thinking"], "max_tokens_per_turn": r["rules"]["max_tokens_per_turn"]})
        eff = cfg.get("reasoning_effort_effective", "xhigh" if cfg.get("enable_thinking") else None)
        if cfg.get("enable_thinking") is not think or cfg.get("max_tokens_per_turn") != budget or eff != effort:
            reasons.append(f"{nm}: execution config {cfg} is not thinking={think} budget={budget} effort={effort}")
    valid = not reasons
    meas, diag = {}, {}
    if valid:
        st, tasks = {}, {}
        for nm, r in recs.items():
            think = want[nm][0]
            st[nm], why, changed = rescored(r, tasks_by_id, think)
            diag[nm] = summary(r, tasks_by_id, st[nm], think)
            diag[nm]["rows_changed_by_amended_scorer"] = changed
            tasks[nm] = per_task(r, tasks_by_id, st[nm], why, think)
        diag["per_task"] = {i: {nm: tasks[nm][i] for nm in recs} for i in tasks_by_id}
        diag["transitions"] = {i: " > ".join(tasks[nm][i]["status"][0] for nm in ("BASELINE", "XHIGH768", "LOW", "MEDIUM")) for i in tasks_by_id}
        for arm, base_id in (("LOW", 1), ("MEDIUM", 6)):
            d, a = diag[arm], arm.lower()
            ctrl = f"baseline {diag['BASELINE']['arith_json']}/{diag['BASELINE']['code']}/{diag['BASELINE']['other']}, xhigh768 {diag['XHIGH768']['arith_json']}/{diag['XHIGH768']['code']}/{diag['XHIGH768']['other']} (arith+json/code/other)"
            meas[str(base_id)] = {"value": d["arith_json"], "metric": f"arith_json_pass_{a}", "population": "tasks:18", "aggregation": "count", "unit": "tasks", "provenance": ctrl}
            meas[str(base_id + 1)] = {"value": d["code"], "metric": f"code_pass_{a}", "population": "tasks:8", "aggregation": "count", "unit": "tasks", "provenance": ctrl}
            meas[str(base_id + 2)] = {"value": d["other"], "metric": f"other_pass_{a}", "population": "tasks:20", "aggregation": "count", "unit": "tasks", "provenance": ctrl}
            meas[str(base_id + 3)] = {"value": d["unterminated_turns"], "metric": f"unterminated_turns_{a}", "population": "turns:52", "aggregation": "count", "unit": "turns",
                                      "provenance": f"xhigh768 {diag['XHIGH768']['unterminated_turns']}"}
            meas[str(base_id + 4)] = {"value": d["generated_tokens"], "metric": f"generated_tokens_{a}", "population": "turns:52", "aggregation": "sum", "unit": "tokens",
                                      "provenance": f"xhigh768 {diag['XHIGH768']['generated_tokens']}; baseline {diag['BASELINE']['generated_tokens']}"}
            meas[f"{base_id + 4}:time"] = {"value": d["task_time_s"], "metric": f"task_time_s_{a}", "population": "turns:52", "aggregation": "sum", "unit": "s",
                                           "provenance": f"xhigh768 {diag['XHIGH768']['task_time_s']}; baseline {diag['BASELINE']['task_time_s']}"}
        meas["arith_json_pass_low"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {base_path}, {xhigh_path}, {low_path}, {med_path}; adapter scratch/qwen_effort_obs.py; thinking verdicts recomputed under the amended scorer",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"BASELINE": {"admissible": True, "reason": "the frozen v2 no-thinking receipt, verdicts as stored (historical control)"},
                    "XHIGH768": {"admissible": valid, "reason": "the frozen xhigh 768 receipt of QWEN-THINKING-BATTERY-0 (historical control)"},
                    "LOW": {"admissible": valid, "reason": "enable_thinking True, reasoning_effort low, 768 tokens, one run"},
                    "MEDIUM": {"admissible": valid, "reason": "enable_thinking True, reasoning_effort medium, 768 tokens, one run"}},
           "measurements": meas, "diagnostics": diag}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

"""Observations adapter for PRE-REG QWEN-THINKING-BATTERY-0 as amended by
AMENDMENT QWEN-THINKING-BATTERY-0-RULES: derives the registered
measurements from the three battery receipts (the frozen no-thinking
baseline logs/qwen_battery/results_v2.json and the two thinking
configurations written by scratch/qwen_battery_run.py under THINK=1 with
BUDGET 384 and 768) in committed code. The baseline's stored verdicts are
the booked ones and are read as stored; each thinking receipt's verdicts
are RECOMPUTED from its stored replies and token ids with the committed
amended scorer (qbattery.score_thinking: the final answer after the last
</think>, a reopened block no final answer, the two-turn kind scored on
its second turn), the code tasks re-run in the sandbox, and every row
whose status differs from the stored one is listed. The honest count is
pass rows over all 46 tasks (error and hold rows are non-passes).

  bar 1  think384_pass: honest passes at thinking, 384 tokens
  bar 2  think768_pass: honest passes at thinking, 768 tokens
  bar 3  think768_arith_json_pass: arithmetic plus JSON passes at 768
  bar 4  think768_other_pass: code, instruction, two-turn and termination
         passes at 768
Diagnostics (free-form, not adjudicated): per-configuration format
compliance of the final answer on the arithmetic and JSON kinds (the
baseline's derived from its stored replies), reasoning and answer tokens
(to the LAST close), unterminated blocks, budget exhaustion and stop
rates, generated tokens, first-token latency, task time, memory, errors.
Validity: three clean-tree non-smoke receipts on the same collection
sha256 as the committed collection (46 tasks each), the baseline with
enable_thinking False at 384, the two thinking receipts with
enable_thinking True at their budgets.
Usage: qwen_think_obs.py <baseline.json> <think384.json> <think768.json> <prereg.json> > observations.json
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


def rescored(r, tasks_by_id, thinking):
    """(status by task id, changed rows) with the thinking receipts recomputed under the amended scorer."""
    status, changed = {}, []
    for t in r["tasks"]:
        if not thinking:
            status[t["id"]] = t["verdict"]["status"]
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
        status[t["id"]] = v["status"]
        if v["status"] != t["verdict"]["status"]:
            changed.append({"id": t["id"], "stored": t["verdict"]["status"], "recomputed": v["status"], "why": v["why"][:160]})
    return status, changed


def diagnostics(r, tasks_by_id, status, thinking):
    rows = r["tasks"]
    turns = [res for t in rows for res in t["results"]]
    fmt = collections.Counter()
    for t in rows:
        if t["kind"] not in ("arith", "json"):
            continue
        txt = t["results"][-1]["text"]
        final = qbattery.split_final(txt)[1] if thinking else txt
        fmt[str(qbattery.format_ok(tasks_by_id[t["id"]], final)) if final else "no_final"] += 1
    reasoning, answer, unterminated = 0, 0, 0
    for res in turns:
        ids = res["generated_ids"]
        if thinking:
            if THINK_CLOSE_ID in ids:
                close = len(ids) - ids[::-1].index(THINK_CLOSE_ID)
                reasoning += close
                answer += len(ids) - close
            else:
                unterminated += 1
                reasoning += len(ids)
        else:
            answer += len(ids)
    by_kind = collections.defaultdict(collections.Counter)
    for t in rows:
        by_kind[t["kind"]][status[t["id"]]] += 1
    ttft = sorted(res["timings"]["first_token_s"] for res in turns)
    return {"passes": sum(s == "pass" for s in status.values()), "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "errors": [i for i, s in status.items() if s == "error"], "format_final_answer_arith_json": dict(fmt),
            "generated_tokens": sum(len(res["generated_ids"]) for res in turns), "reasoning_tokens": reasoning, "answer_tokens": answer,
            "unterminated_thinking_turns": unterminated, "turns": len(turns),
            "stop_reasons": dict(collections.Counter(res["stop_reason"] for res in turns)),
            "first_token_s_median": statistics.median(ttft), "task_total_s_sum": round(sum(res["timings"]["total_s"] for res in turns), 1),
            "peak_bytes_max": max(res["memory"]["peak_bytes"] for res in turns), "active_bytes_max": max(res["memory"]["active_bytes"] for res in turns),
            "context_refusals": sum(1 for t in rows if "ContextLimitError" in t["verdict"].get("why", "")),
            "wall_s": round(r.get("wall_s", 0), 1)}


def main(base_path, t384_path, t768_path, prereg_path):
    base, t384, t768 = (json.load(open(p)) for p in (base_path, t384_path, t768_path))
    prereg = load(prereg_path)
    coll = json.load(open(COLLECTION))
    coll_sha = hashlib.sha256(open(COLLECTION, "rb").read()).hexdigest()
    tasks_by_id = {t["id"]: t for t in coll["tasks"]}
    reasons = []
    for r, nm, think, budget in ((base, "baseline", False, 384), (t384, "think384", True, 384), (t768, "think768", True, 768)):
        if r.get("smoke") or r.get("tree_dirty"):
            reasons.append(f"{nm}: smoke run or dirty tree")
        if len(r.get("tasks", [])) != 46:
            reasons.append(f"{nm}: {len(r.get('tasks', []))} tasks, not 46")
        cfg = r.get("execution_config", {"enable_thinking": r["rules"]["enable_thinking"], "max_tokens_per_turn": r["rules"]["max_tokens_per_turn"]})
        if cfg.get("enable_thinking") is not think or cfg.get("max_tokens_per_turn") != budget:
            reasons.append(f"{nm}: execution config {cfg} is not thinking={think} budget={budget}")
        if r.get("collection_sha256") != coll_sha:
            reasons.append(f"{nm}: collection sha differs from the committed collection")
    if t384.get("commit") != t768.get("commit"):
        reasons.append("the two thinking receipts are from different commits")
    valid = not reasons
    meas, diag = {}, {}
    if valid:
        st = {}
        for r, nm, think in ((base, "baseline", False), (t384, "think384", True), (t768, "think768", True)):
            st[nm], changed = rescored(r, tasks_by_id, think)
            diag[nm] = diagnostics(r, tasks_by_id, st[nm], think)
            diag[nm]["rows_changed_by_amended_scorer"] = changed
        kinds = lambda nm, ks: sum(1 for i, s in st[nm].items() if s == "pass" and tasks_by_id[i]["kind"] in ks)  # noqa: E731
        prov = f"baseline {diag['baseline']['by_kind']} total {diag['baseline']['passes']}; think384 {diag['think384']['by_kind']} total {diag['think384']['passes']}; think768 {diag['think768']['by_kind']} total {diag['think768']['passes']}"
        meas["1"] = {"value": diag["think384"]["passes"], "metric": "think384_pass", "population": "tasks:46", "aggregation": "count", "unit": "tasks", "provenance": prov}
        meas["2"] = {"value": diag["think768"]["passes"], "metric": "think768_pass", "population": "tasks:46", "aggregation": "count", "unit": "tasks", "provenance": prov}
        meas["3"] = {"value": kinds("think768", ("arith", "json")), "metric": "think768_arith_json_pass", "population": "tasks:18", "aggregation": "count", "unit": "tasks",
                     "provenance": f"baseline arith+json {kinds('baseline', ('arith', 'json'))}; think384 {kinds('think384', ('arith', 'json'))}"}
        meas["4"] = {"value": kinds("think768", ("code", "instruct", "twoturn", "degen")), "metric": "think768_other_pass", "population": "tasks:28", "aggregation": "count", "unit": "tasks",
                     "provenance": f"baseline other {kinds('baseline', ('code', 'instruct', 'twoturn', 'degen'))}; think384 {kinds('think384', ('code', 'instruct', 'twoturn', 'degen'))}"}
        meas["think768_arith_json_pass"] = dict(meas["3"])
    obs = {"note": f"Mechanical projection of {base_path}, {t384_path}, {t768_path} (commits {base.get('commit')}, {t384.get('commit')}, {t768.get('commit')}); adapter scratch/qwen_think_obs.py; thinking verdicts recomputed under the amended scorer",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"BASELINE": {"admissible": True, "reason": "the frozen v2 no-thinking receipt of QWEN-INFERENCE-BASELINE-0, verdicts as stored"},
                    "THINK384": {"admissible": valid, "reason": "enable_thinking True through the pinned template, 384 tokens per turn, one run, recomputed under the amended scorer"},
                    "THINK768": {"admissible": valid, "reason": "enable_thinking True through the pinned template, 768 tokens per turn, one run, recomputed under the amended scorer"}},
           "measurements": meas, "diagnostics": diag}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

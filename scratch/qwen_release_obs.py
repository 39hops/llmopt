"""Observations adapter for PRE-REG QWEN-LOW-RELEASE-0: derives the nine
registered measurements from the three v3 battery receipts (arms A, B, C
written by scratch/qwen_battery_run.py with COLLECTION=qwen_battery_v3.json)
and the conversation receipt (scratch/qwen_lowchat.py), in committed code.
Every battery verdict is RECOMPUTED from the stored replies with the
committed validators (qbattery.score for A and B; qbattery.score_thinking
for C, the final answer after the last </think>), the code tasks re-run
in the sandbox; rows whose status differs from the stored verdict are
listed. The honest count is pass rows over all 60 tasks (error and hold
rows are non-passes; a hold invalidates the measurement). Evaluator
errors are listed apart from model failures.

  1  arith_json_gain_c_minus_b   (arith+json passes of C) - (of B), 24 tasks
  2  code_loss_b_minus_c         (code passes of B) - (of C), 12 tasks
  3  other_loss_b_minus_c        (instruct+twoturn+degen passes of B) - (of C), 24 tasks
  4  unterminated_turns_c        C turns with no </think>, of 68; 4:cat the largest count in one kind
  5  generated_tokens_c          C generated tokens over 68 turns; 5:time C task time
  6  longest_empty_run_low       longest run of consecutive empty completions in a low conversation
  7  addressed_turns_low         low turns passing their registered addressing check, of 32
  8  travel_empty_turns_low      empty completions on the travel transcript under low
  9  later_fallbacks_low         later low turns not taken incrementally; 9:replay incremental-vs-replay id mismatches
Validity: four clean-tree non-smoke receipts; 60 tasks and 68 turns per
battery receipt on the committed v3 collection's sha256; the execution
configs A (thinking False, 384), B (thinking False, 768), C (thinking True,
low, 768); one artifact chain and one tokenizer across all receipts; the
conversation receipt with four conversations of eight turns under both
configurations at 768 tokens.
Usage: qwen_release_obs.py <A.json> <B.json> <C.json> <lowchat.json> <prereg.json> > observations.json
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

COLLECTION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalsets", "qwen_battery_v3.json")
THINK_CLOSE_ID = 248069
OTHER = ("instruct", "twoturn", "degen")
WANT = {"A": (False, 384, None), "B": (False, 768, None), "C": (True, 768, "low")}


def rescore(r, tasks_by_id, thinking):
    status, why, changed, errors = {}, {}, [], []
    for t in r["tasks"]:
        task = tasks_by_id[t["id"]]
        replies = [res["text"] for res in t["results"]]
        if len(replies) < len(task["turns"]):
            v = {"status": "error", "why": t["verdict"].get("why", "incomplete turns")}
        else:
            try:
                v = (qbattery.score_thinking if thinking else qbattery.score)(task, replies, t["results"])
            except Exception as e:  # noqa: BLE001
                v = {"status": "error", "why": f"validator {type(e).__name__}: {e}"}
        status[t["id"]], why[t["id"]] = v["status"], v["why"]
        if v["status"] == "error":
            errors.append((t["id"], v["why"][:120]))
        if v["status"] != t["verdict"]["status"]:
            changed.append({"id": t["id"], "stored": t["verdict"]["status"], "recomputed": v["status"]})
    return status, why, changed, errors


def battery_summary(r, tasks_by_id, status, thinking):
    turns = [res for t in r["tasks"] for res in t["results"]]
    unterm_by_kind = collections.Counter()
    reasoning = 0
    for t in r["tasks"]:
        for res in t["results"]:
            ids = res["generated_ids"]
            if thinking:
                if THINK_CLOSE_ID in ids:
                    reasoning += len(ids) - ids[::-1].index(THINK_CLOSE_ID)
                else:
                    unterm_by_kind[t["kind"]] += 1
                    reasoning += len(ids)
    by_kind = collections.defaultdict(collections.Counter)
    for i, s in status.items():
        by_kind[tasks_by_id[i]["kind"]][s] += 1
    passes = lambda ks: sum(1 for i, s in status.items() if s == "pass" and tasks_by_id[i]["kind"] in ks)  # noqa: E731
    final_present = sum(1 for t in r["tasks"] if (qbattery.split_final(t["results"][-1]["text"])[1] if thinking else t["results"][-1]["text"]))
    return {"passes": passes(("arith", "json", "code") + OTHER), "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "arith_json": passes(("arith", "json")), "code": passes(("code",)), "other": passes(OTHER),
            "unterminated_turns": sum(unterm_by_kind.values()), "unterminated_by_kind": dict(unterm_by_kind),
            "final_answer_present": final_present, "generated_tokens": sum(len(res["generated_ids"]) for res in turns), "reasoning_tokens": reasoning,
            "stop_reasons": dict(collections.Counter(res["stop_reason"] for res in turns)),
            "first_token_s_median": round(statistics.median(res["timings"]["first_token_s"] for res in turns), 2),
            "task_time_s": round(sum(res["timings"]["total_s"] for res in turns), 1), "wall_s": round(r.get("wall_s", 0), 1),
            "peak_bytes_max": max(res["memory"]["peak_bytes"] for res in turns), "context_refusals": sum("ContextLimitError" in t["verdict"].get("why", "") for t in r["tasks"])}


def main(a_path, b_path, c_path, chat_path, prereg_path):
    recs = {"A": json.load(open(a_path)), "B": json.load(open(b_path)), "C": json.load(open(c_path))}
    chat = json.load(open(chat_path))
    prereg = load(prereg_path)
    coll = json.load(open(COLLECTION))
    coll_sha = hashlib.sha256(open(COLLECTION, "rb").read()).hexdigest()
    tasks_by_id = {t["id"]: t for t in coll["tasks"]}
    reasons = []
    for nm, r in recs.items():
        think, budget, effort = WANT[nm]
        if r.get("smoke") or r.get("tree_dirty"):
            reasons.append(f"{nm}: smoke run or dirty tree")
        if len(r.get("tasks", [])) != 60 or sum(len(t["results"]) for t in r.get("tasks", [])) != 68:
            reasons.append(f"{nm}: not 60 tasks and 68 turns")
        if r.get("collection_sha256") != coll_sha:
            reasons.append(f"{nm}: collection sha differs from the committed v3 collection")
        cfg = r.get("execution_config", {})
        if cfg.get("enable_thinking") is not think or cfg.get("max_tokens_per_turn") != budget or cfg.get("reasoning_effort_effective") != effort:
            reasons.append(f"{nm}: execution config {cfg} is not thinking={think} budget={budget} effort={effort}")
    if chat.get("smoke") or chat.get("tree_dirty"):
        reasons.append("chat: smoke run or dirty tree")
    if [c["name"] for c in chat.get("conversations", [])] != ["travel", "planning", "programming", "factual"] or chat.get("budget") != 768:
        reasons.append("chat: conversations or budget are not the registered ones")
    if any(len(c["configs"][k]["incremental"]) != 8 for c in chat.get("conversations", []) for k in ("NOTHINK", "LOW")):
        reasons.append("chat: a conversation is incomplete")
    chains = {r.get("engine", {}).get("chain_sha256") for r in list(recs.values()) + [chat]}
    if len(chains) != 1 or None in chains:
        reasons.append("the receipts are not on one artifact chain")
    tsh = {json.dumps(r.get("engine", {}).get("tokenizer_files_sha256"), sort_keys=True) for r in list(recs.values()) + [chat]}
    if len(tsh) != 1 or "null" in tsh:
        reasons.append("the receipts do not record one tokenizer")
    valid = not reasons
    meas, diag = {}, {}
    if valid:
        st = {}
        for nm, r in recs.items():
            st[nm], why, changed, errors = rescore(r, tasks_by_id, WANT[nm][0])
            diag[nm] = battery_summary(r, tasks_by_id, st[nm], WANT[nm][0])
            diag[nm]["rows_changed_by_recompute"] = changed
            diag[nm]["evaluator_errors"] = errors
            if any(s == "hold" for s in st[nm].values()):
                reasons.append(f"{nm}: a code row recomputed as hold (sandbox unavailable at adapter time)")
            if errors:
                reasons.append(f"{nm}: evaluator or plumbing errors {errors[:3]} (an error row is never a model failure; the measurement is invalid until the run is repaired)")
        diag["paired"] = {i: " > ".join(st[nm][i][0] for nm in ("A", "B", "C")) for i in tasks_by_id}
        diag["paired_changes"] = {i: v for i, v in diag["paired"].items() if len(set(v.split(" > "))) > 1}
        cs = chat["summary"]
        diag["chat"] = cs
        if reasons:
            valid = False
        else:
            dA, dB, dC = diag["A"], diag["B"], diag["C"]
            prov = f"A {dA['arith_json']}/{dA['code']}/{dA['other']} total {dA['passes']}; B {dB['arith_json']}/{dB['code']}/{dB['other']} total {dB['passes']}; C {dC['arith_json']}/{dC['code']}/{dC['other']} total {dC['passes']} (arith+json/code/other)"
            meas["1"] = {"value": dC["arith_json"] - dB["arith_json"], "metric": "arith_json_gain_c_minus_b", "population": "tasks:24", "aggregation": "count_difference", "unit": "tasks", "provenance": prov}
            meas["2"] = {"value": dB["code"] - dC["code"], "metric": "code_loss_b_minus_c", "population": "tasks:12", "aggregation": "count_difference", "unit": "tasks", "provenance": prov}
            meas["3"] = {"value": dB["other"] - dC["other"], "metric": "other_loss_b_minus_c", "population": "tasks:24", "aggregation": "count_difference", "unit": "tasks", "provenance": prov}
            meas["4"] = {"value": dC["unterminated_turns"], "metric": "unterminated_turns_c", "population": "turns:68", "aggregation": "count", "unit": "turns", "provenance": f"by kind {dC['unterminated_by_kind']}"}
            meas["4:cat"] = {"value": max(dC["unterminated_by_kind"].values(), default=0), "metric": "unterminated_max_per_kind_c", "population": "kinds:6", "aggregation": "max_over_kinds", "unit": "turns"}
            meas["5"] = {"value": dC["generated_tokens"], "metric": "generated_tokens_c", "population": "turns:68", "aggregation": "sum", "unit": "tokens",
                         "provenance": f"A {dA['generated_tokens']} B {dB['generated_tokens']}; ratios C/A {dC['generated_tokens'] / dA['generated_tokens']:.2f} C/B {dC['generated_tokens'] / dB['generated_tokens']:.2f}"}
            meas["5:time"] = {"value": dC["task_time_s"], "metric": "task_time_s_c", "population": "turns:68", "aggregation": "sum", "unit": "s",
                              "provenance": f"A {dA['task_time_s']} B {dB['task_time_s']} s; ratios C/A {dC['task_time_s'] / dA['task_time_s']:.2f} C/B {dC['task_time_s'] / dB['task_time_s']:.2f}"}
            lo, nt = cs["LOW"], cs["NOTHINK"]
            meas["6"] = {"value": lo["longest_empty_run"], "metric": "longest_empty_run_low", "population": "conversations:4", "aggregation": "max_over_conversations", "unit": "turns",
                         "provenance": f"low empty turns {lo['empty_turns']}; nothink longest run {nt['longest_empty_run']} empty turns {nt['empty_turns']}"}
            meas["7"] = {"value": lo["addressed_turns"], "metric": "addressed_turns_low", "population": "chat-turns:32", "aggregation": "count", "unit": "turns",
                         "provenance": f"low unaddressed {lo['unaddressed']}; nothink addressed {nt['addressed_turns']} of {nt['turns']}"}
            meas["8"] = {"value": lo["travel_empty_turns"], "metric": "travel_empty_turns_low", "population": "travel-turns:8", "aggregation": "count", "unit": "turns",
                         "provenance": f"nothink travel empty turns {nt['travel_empty_turns']}"}
            meas["9"] = {"value": lo["later_fallbacks_after_closed"], "metric": "later_fallbacks_low", "population": "chat-later-turns:28", "aggregation": "count", "unit": "turns",
                         "provenance": f"fallbacks after a closed previous turn; fallbacks after a budget-cut turn (by template design) {lo['later_fallbacks_after_cut']}; all later fallbacks {lo['later_fallbacks']}"}
            meas["9:replay"] = {"value": lo["replay_token_mismatch_turns"], "metric": "replay_token_mismatch_turns_low", "population": "chat-turns:32", "aggregation": "count", "unit": "turns"}
            meas["arith_json_gain_c_minus_b"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {a_path}, {b_path}, {c_path}, {chat_path}; adapter scratch/qwen_release_obs.py; every battery verdict recomputed with the committed validators",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"A": {"admissible": valid, "reason": "no thinking, 384, on v3"}, "B": {"admissible": valid, "reason": "no thinking, 768, on v3"},
                    "C": {"admissible": valid, "reason": "thinking low, 768, on v3"},
                    "CHAT-NOTHINK": {"admissible": valid, "reason": "the four conversations, thinking off, 768"},
                    "CHAT-LOW": {"admissible": valid, "reason": "the four conversations, thinking low, 768, incremental and replay"}},
           "measurements": meas, "diagnostics": diag}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

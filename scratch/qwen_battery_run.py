"""QWEN-INFERENCE-BASELINE-0, phase E: score the frozen battery
(scratch/evalsets/qwen_battery_v2.json by default; COLLECTION names
another file; v1 is the superseded first freeze kept as a record) ONCE on the local arm C engine
(llmopt.lab.qserve) under the collection's frozen rules (greedy,
enable_thinking False, no system prompt, 384 output tokens per turn,
context cap 8,192), each task in a fresh replay session, every turn's
prompt ids, generated ids, text, stop reason, timings and memory
recorded, every verdict by the committed validators
(llmopt.lab.qbattery.score; code tasks in the sandbox, HOLD when the
sandbox is unavailable). Plumbing failures (an exception, a context
refusal, a sandbox failure) are recorded as "error" apart from model
failures. Receipt logs/qwen_battery/results.json (refuse-if-exists; OUT
overrides). THINK=1 enables the pinned template's thinking control and
scores the final answer after the last </think> (qbattery.score_thinking;
an unclosed block is a fail with no final answer); BUDGET=N overrides the
per-turn budget; both are recorded in execution_config and the frozen
rules are copied unchanged. SMOKE=1 scores the first task of each kind into
logs/qwen_battery/smoke/results.json.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SMOKE = os.environ.get("SMOKE") == "1"
THINK = os.environ.get("THINK") == "1"                       # QWEN-THINKING-BATTERY-0: enable_thinking True through the pinned template
BUDGET = int(os.environ["BUDGET"]) if os.environ.get("BUDGET") else None   # overrides the frozen per-turn budget
assert BUDGET in (None, 384, 768), "only the registered budgets"
assert BUDGET is None or THINK, "a budget override belongs to a thinking configuration" 
OUT = os.environ.get("OUT", "logs/qwen_battery/smoke/results.json") if SMOKE else os.environ.get("OUT", "logs/qwen_battery/results.json")
assert not SMOKE or "/smoke/" in OUT, "a smoke receipt stays under a smoke/ path"
COLLECTION = os.path.join(HERE, "evalsets", os.environ.get("COLLECTION", "qwen_battery_v2.json"))


THINK_CLOSE_ID = 248069      # </think> in the pinned tokenizer (verified against the loaded tokenizer at run time)


def main() -> None:
    from llmopt.lab import qbattery, qserve
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    coll = json.load(open(COLLECTION))
    coll_sha = hashlib.sha256(open(COLLECTION, "rb").read()).hexdigest()
    rules = coll["rules"]
    tasks = coll["tasks"]
    thinking = True if THINK else rules["enable_thinking"]
    budget = BUDGET or rules["max_tokens_per_turn"]
    exec_cfg = {"name": f"v{coll['version']}-think-{budget}" if THINK else f"v{coll['version']}-frozen", "enable_thinking": thinking,
                "max_tokens_per_turn": budget, "scoring": "score_thinking (final answer after the last </think>)" if THINK else "score"}
    if SMOKE:
        seen, keep = set(), []
        for t in tasks:
            if t["kind"] not in seen:
                seen.add(t["kind"])
                keep.append(t)
        tasks = keep
    eng = qserve.Engine(context_cap=rules["context_cap"])
    assert eng.tokenizer.convert_tokens_to_ids("</think>") == THINK_CLOSE_ID
    rows = []
    t_all = time.perf_counter()
    for task in tasks:
        sess = qserve.Session(eng, system=rules["system"], mode="replay", enable_thinking=thinking)
        replies, results, err = [], [], None
        try:
            for turn in task["turns"]:
                r = sess.ask(turn, max_tokens=budget)
                replies.append(r.text)
                close = (len(r.generated_ids) - r.generated_ids[::-1].index(THINK_CLOSE_ID)) if THINK_CLOSE_ID in r.generated_ids else None   # the LAST close
                results.append({"prompt_ids": r.prompt_ids, "generated_ids": r.generated_ids, "text": r.text, "stop_reason": r.stop_reason,
                                "stop_token_id": r.stop_token_id, "n_prompt": r.n_prompt, "n_generated": r.n_generated,
                                "cache_tokens": r.cache_tokens, "cache_offset_measured": r.cache_offset_measured, "timings": r.timings, "memory": r.memory,
                                "reasoning_tokens": close, "answer_tokens": (r.n_generated - close) if close is not None else None})
        except Exception as e:  # noqa: BLE001 - plumbing failures are the point of recording
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()[-600:]
        if err is None:
            try:
                verdict = (qbattery.score_thinking if THINK else qbattery.score)(task, replies, results)
                final = (verdict.get("final_answers") or [None])[-1] if THINK else replies[-1]
                verdict["format_ok"] = qbattery.format_ok(task, final) if final else None   # None: no final answer to judge
            except Exception as e:  # noqa: BLE001
                verdict = {"status": "error", "why": f"validator {type(e).__name__}: {e}"}
        else:
            verdict = {"status": "error", "why": err, "traceback": tb}
        row = {"id": task["id"], "kind": task["kind"], "turns": task["turns"], "expected": task.get("expected"), "verdict": verdict, "results": results}
        rows.append(row)
        print(f"[battery] {task['id']:12s} {verdict['status']:5s} {verdict['why'][:90]!r}  | {(replies[-1] if replies else '')[:60]!r}", flush=True)
    by_kind = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        by_kind[r["kind"]][r["verdict"]["status"]] += 1
    n_pass = sum(1 for r in rows if r["verdict"]["status"] == "pass")
    n_scored = sum(1 for r in rows if r["verdict"]["status"] in ("pass", "fail"))
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "collection": os.path.relpath(COLLECTION, os.path.dirname(HERE)), "execution_config": exec_cfg, "collection_sha256": coll_sha,
               "collection_version": coll["version"], "rules": rules, "sandbox_available": qbattery.sandbox_available(),
               "engine": {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"),
                          "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {}),
                          "stop_ids": sorted(eng.stop_ids), "context_cap": eng.context_cap, "cache_limit_bytes": eng.cfg.cache_limit_bytes},
               "summary": {"pass": n_pass, "scored": n_scored, "tasks": len(rows), "pass_rate_of_scored": (n_pass / n_scored) if n_scored else None,
                           "by_kind": {k: dict(v) for k, v in by_kind.items()},
                           "errors": [r["id"] for r in rows if r["verdict"]["status"] == "error"], "holds": [r["id"] for r in rows if r["verdict"]["status"] == "hold"],
                           "stop_reasons": dict(collections.Counter(res["stop_reason"] for r in rows for res in r["results"])),
                           "generated_tokens_total": sum(res["n_generated"] for r in rows for res in r["results"]),
                           "reasoning_tokens_total": sum(res["reasoning_tokens"] or 0 for r in rows for res in r["results"]),
                           "unterminated_thinking_turns": sum(1 for r in rows for res in r["results"] if THINK and res["reasoning_tokens"] is None),
                           "format_ok": dict(collections.Counter(str(r["verdict"].get("format_ok")) for r in rows if r["kind"] in ("arith", "json"))),
                           "first_token_s_median": sorted(res["timings"]["first_token_s"] for r in rows for res in r["results"])[len([1 for r in rows for _ in r["results"]]) // 2],
                           "task_total_s_sum": sum(res["timings"]["total_s"] for r in rows for res in r["results"])},
               "wall_s": time.perf_counter() - t_all, "tasks": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[battery] pass {n_pass}/{n_scored} scored of {len(rows)} tasks; by kind {receipt['summary']['by_kind']}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

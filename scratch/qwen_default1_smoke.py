"""QWEN-LOW-DEFAULT-1: the fresh integration smoke of the corrected
operational cache contract under the promoted defaults (Session and CLI:
enable_thinking True, reasoning_effort low, 768 tokens, incremental reuse
under the exact-prefix check with replay fallback, greedy, generation 5).

Stages, all on the real tower and the pinned tokenizer:
  tests     the release suites in fresh subprocesses (tests/test_qdefaults.py
            on the stub; tests/test_qtemplate.py and tests/test_qrelease.py on
            the pinned tokenizer, the live ones on the tower), pass and fail
            counts recorded from pytest's own summary
  fixture   the planning-turn-7 tokenizer mismatch: a session holding exactly
            the fed ids of the fixture's six turns asks turn 7; recorded: the
            rendering does not extend the fed ids, the mode taken, the tokens
            against a replay session and against the frozen receipt, offsets
  smoke     two bounded conversations under the default (the programming and
            factual conversations of scratch/qwen_lowchat.py, first four turns
            each) as an incremental and a replay session: per turn the mode,
            the fallback reason if any, token equality, offsets, first-token
            latency, total time, memory
  contract  a mid-session effort change and a mid-session thinking change:
            the next turn must fall back (never reuse); a budget stop; a
            context-cap refusal and recovery; a reset
Summary: unsafe reuses (a turn taken incrementally whose rendered prompt did
not have the fed ids as an exact prefix: computed from the recorded ids),
replay token mismatches, offset mismatches, fallbacks with reasons and their
latency cost, the fixture's outcome, test counts. Receipt OUT (default
logs/qwen_release/default1_smoke.json; refuses to overwrite).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.environ.get("OUT", "logs/qwen_release/default1_smoke.json")
FIXTURE = "tests/fixtures/qwen_planning_turn7.json"
_spec = importlib.util.spec_from_file_location("qwen_lowchat", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_lowchat.py"))
_lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lc)


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def run_tests():
    out = {}
    for name, cmd in (("defaults_stub", ["tests/test_qdefaults.py"]), ("template", ["tests/test_qtemplate.py"]), ("release", ["tests/test_qrelease.py"])):
        env = dict(os.environ, LLMOPT_QWEN_LIVE="1")
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *cmd], capture_output=True, text=True, timeout=3600, env=env)
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)}
        out[name] = {"returncode": p.returncode, "summary": tail, **m}
        print(f"[default1] tests {name}: {tail}", flush=True)
    return out


def session_rows(eng, sess, turns):
    from llmopt.lab import qserve
    rows = []
    for text, budget in turns:
        fed_before = list(sess.fed_ids)
        try:
            r = sess.ask(text, max_tokens=budget)
        except qserve.ContextLimitError as e:
            rows.append({"user": text[:40], "refused": str(e)[:80]})
            continue
        # the contract check from the recorded ids: an incremental turn's new ids must continue the fed prefix exactly
        full_prefix_ok = (r.mode != "incremental") or (sess.fed_ids[: len(fed_before)] == fed_before)
        rows.append({"user": text[:40], "mode": r.mode, "note": r.note, "n_prompt": r.n_prompt, "n_generated": r.n_generated, "stop_reason": r.stop_reason,
                     "generated_sha256": sha(r.generated_ids), "generated_ids": r.generated_ids, "cache_tokens": r.cache_tokens, "cache_offset_measured": r.cache_offset_measured,
                     "fed_prefix_preserved": full_prefix_ok, "first_token_s": r.timings["first_token_s"], "total_s": r.timings["total_s"],
                     "active_bytes": r.memory["active_bytes"], "peak_bytes": r.memory["peak_bytes"], "think_closed": 248069 in r.generated_ids})
    return rows


def main() -> None:
    from llmopt.lab import qserve
    import mlx.core as mx
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t0 = time.perf_counter()
    tests = run_tests()
    eng = qserve.Engine()
    assert eng.tokenizer.convert_tokens_to_ids("</think>") == 248069
    # ---- fixture
    fx = json.load(open(FIXTURE))
    msgs = []
    for t in fx["turns"][:6]:
        msgs.append({"role": "user", "content": t["user"]})
        reasoning, content = qserve.split_thinking(t["text"])
        msgs.append({"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})})
    p6 = eng.render_chat(msgs[:11], True, "low")
    fed = p6 + fx["turns"][5]["generated_ids"]
    warm = eng.generate_ids(fed, max_tokens=1)
    a = qserve.Session(eng)
    a.messages, a.fed_ids, a.cache = list(msgs), list(fed), warm.cache
    b = qserve.Session(eng, mode="replay")
    b.messages = list(msgs)
    p7 = eng.render_chat(msgs + [{"role": "user", "content": fx["turns"][6]["user"]}], True, "low")
    ra, rb = a.ask(fx["turns"][6]["user"], 768), b.ask(fx["turns"][6]["user"], 768)
    fixture = {"fed_len": len(fed), "fed_equals_receipt_cache_tokens": len(fed) == fx["turns"][5]["cache_tokens"], "warm_offset": warm.cache_offset_measured,
               "rendering_extends_fed": len(p7) > len(fed) and p7[: len(fed)] == fed, "break_offset_in_generation": next((i for i, (x, y) in enumerate(zip(fed, p7)) if x != y), None) - len(p6),
               "mode": ra.mode, "note": ra.note, "generated_equal_replay": ra.generated_ids == rb.generated_ids, "generated_equal_frozen_receipt": ra.generated_ids == fx["replay_turn7_generated_ids"],
               "n_generated": ra.n_generated, "offsets_ok": ra.cache_offset_measured == ra.cache_tokens == len(a.fed_ids) and rb.cache_offset_measured == rb.cache_tokens,
               "prompt_len": ra.n_prompt, "first_token_s_fallback": ra.timings["first_token_s"], "first_token_s_replay": rb.timings["first_token_s"]}
    print(f"[default1] fixture mode {ra.mode} extends {fixture['rendering_extends_fed']} equal_replay {fixture['generated_equal_replay']} equal_receipt {fixture['generated_equal_frozen_receipt']} offsets {fixture['offsets_ok']}", flush=True)
    del a, b, warm, ra, rb
    mx.clear_cache()
    # ---- bounded multi-turn smoke under the default
    convs = []
    for conv in _lc.CONVERSATIONS:
        if conv["name"] not in ("programming", "factual"):
            continue
        turns = [(t, 768) for t, _ in conv["turns"][:4]]
        inc = session_rows(eng, qserve.Session(eng), turns)
        rep = session_rows(eng, qserve.Session(eng, mode="replay"), turns)
        convs.append({"name": conv["name"], "incremental": inc, "replay": rep,
                      "token_equal": [x["generated_sha256"] == y["generated_sha256"] for x, y in zip(inc, rep)]})
        print(f"[default1] smoke {conv['name']} modes {[x['mode'][:4] for x in inc]} equal {convs[-1]['token_equal']} ttft inc {[round(x['first_token_s'], 2) for x in inc]} rep {[round(x['first_token_s'], 2) for x in rep]}", flush=True)
        mx.clear_cache()
    # ---- contract: incompatible changes, budget, cap, reset
    s = qserve.Session(eng)
    c_rows = session_rows(eng, s, [("Which planet is known as the red planet? One word.", 128)])
    s.reasoning_effort = "xhigh"
    c_rows += session_rows(eng, s, [("And the largest planet? One word.", 128)])
    s.enable_thinking, s.reasoning_effort = False, None
    c_rows += session_rows(eng, s, [("And the smallest? One word.", 64)])
    c_rows += session_rows(eng, s, [("Write one long sentence about rivers.", 5)])
    c_rows += session_rows(eng, s, [("x " * 5000, 64)])
    cap_dropped_cache = s.cache is None
    c_rows += session_rows(eng, s, [("Reply with just: ok.", 64)])
    s.reset()
    reset_ok = s.messages == [] and s.fed_ids == [] and s.cache is None
    contract = {"rows": c_rows, "effort_change_mode": c_rows[1]["mode"], "thinking_change_mode": c_rows[2]["mode"], "budget_stop": c_rows[3]["stop_reason"],
                "cap_refused": "refused" in c_rows[4], "cap_dropped_cache": cap_dropped_cache, "recovery_mode": c_rows[5]["mode"], "reset_ok": reset_ok}
    print(f"[default1] contract {json.dumps({k: v for k, v in contract.items() if k != 'rows'})}", flush=True)
    all_rows = [x for c in convs for x in c["incremental"] + c["replay"]] + [x for x in c_rows if "mode" in x]
    summary = {"tests_failed": sum(t.get("failed", 0) + t.get("error", 0) for t in tests.values()), "tests_passed": sum(t.get("passed", 0) for t in tests.values()),
               "tests_returncodes": {k: v["returncode"] for k, v in tests.items()},
               "fixture_detected_and_replayed": fixture["rendering_extends_fed"] is False and fixture["mode"] == "replay-fallback",
               "fixture_tokens_equal_replay_and_receipt": fixture["generated_equal_replay"] and fixture["generated_equal_frozen_receipt"], "fixture_offsets_ok": fixture["offsets_ok"],
               "smoke_turns": sum(len(c["incremental"]) for c in convs), "smoke_token_mismatch_turns": sum(1 for c in convs for eq in c["token_equal"] if not eq),
               "smoke_later_fallbacks": sum(1 for c in convs for x in c["incremental"][1:] if x["mode"] != "incremental"),
               "unsafe_reuse_turns": sum(1 for x in all_rows if x["mode"] == "incremental" and not x["fed_prefix_preserved"]),
               "offset_mismatch_turns": sum(1 for x in all_rows if x["cache_offset_measured"] != x["cache_tokens"]),
               "incompatible_change_reused": int(contract["effort_change_mode"] == "incremental") + int(contract["thinking_change_mode"] == "incremental"),
               "fallback_first_token_s": [round(x["first_token_s"], 2) for c in convs for x in c["incremental"][1:] if x["mode"] != "incremental"],
               "incremental_later_first_token_s": [round(x["first_token_s"], 2) for c in convs for x in c["incremental"][1:] if x["mode"] == "incremental"],
               "replay_later_first_token_s": [round(x["first_token_s"], 2) for c in convs for x in c["replay"][1:]],
               "smoke_generated_tokens": sum(x["n_generated"] for c in convs for x in c["incremental"]), "smoke_total_s": round(sum(x["total_s"] for c in convs for x in c["incremental"]), 1),
               "peak_bytes_max": max(x["peak_bytes"] for x in all_rows), "think_closed_turns": sum(1 for c in convs for x in c["incremental"] if x["think_closed"])}
    receipt = {"commit": head, "tree_dirty": dirty, "defaults": {"enable_thinking": qserve.DEFAULT_THINKING, "reasoning_effort": qserve.DEFAULT_EFFORT, "max_tokens": qserve.DEFAULT_MAX_TOKENS, "mode": "incremental"},
               "engine": {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"), "stop_ids": sorted(eng.stop_ids),
                          "context_cap": eng.context_cap, "cache_limit_bytes": eng.cfg.cache_limit_bytes, "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {})},
               "tests": tests, "fixture": fixture, "conversations": convs, "contract": contract, "summary": summary, "wall_s": time.perf_counter() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[default1] summary {json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

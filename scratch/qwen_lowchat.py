"""QWEN-LOW-RELEASE-0, phase 4: the deep-conversation reliability gate.
Four eight-turn conversations with fixed user turns, each run under two
configurations at identical 768-token budgets: NOTHINK (enable_thinking
False) and LOW (enable_thinking True, reasoning_effort low). The
conversations: the previously exposed travel transcript (a regression
diagnostic, its user turns unchanged from scratch/qwen_incr_perf.py) and
three new preregistered ones (planning with revisions, a programming
discussion, follow-up factual questions). Every turn carries a
mechanical check of whether the reply ADDRESSES the turn (a required
word or number, a line or sentence count, a format), so success is never
non-empty text alone.

Each configuration runs as an incremental Session (the default) and,
for LOW only, as a replay Session for token equality. Per turn: reply
length in tokens and words, stop reason and exact stop token, the
first-token stop margin (from the Engine's top-16 record), reasoning
termination (LOW), the final answer (LOW: after the last </think>), the
addressed check, the prompt sha256 and markers, mode and note, accounted
and measured cache offsets, first-token latency, total time, memory.
Summary: per configuration the empty-completion turns, the longest run
of consecutive empty completions in any conversation, addressed turns,
unterminated blocks, token totals and time. Receipt OUT (default
logs/qwen_serve/lowchat_v1.json; refuses to overwrite). SMOKE=1 runs the
first two turns of the planning conversation into
logs/qwen_serve/smoke/lowchat.json.
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
from llmopt.lab import qbattery  # noqa: E402

_spec = importlib.util.spec_from_file_location("qwen_incr_perf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_incr_perf.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TRAVEL_TURNS = [t for t, _ in _mod.CONVERSATIONS[0]["turns"]]        # the exposed transcript's user turns, verbatim

SMOKE = os.environ.get("SMOKE") == "1"
OUT = "logs/qwen_serve/smoke/lowchat.json" if SMOKE else os.environ.get("OUT", "logs/qwen_serve/lowchat_v1.json")
BUDGET = 768
MARKERS = {"im_start": 248045, "im_end": 248046, "think": 248068, "think_close": 248069}

# per-turn checks: ("any", [words]) at least one substring present; ("all", [words]); ("number", n) the integer n present;
# ("lines", n) exactly n non-empty lines; ("max_words", n); ("min_words", n); ("re", [patterns]) any word-bounded pattern;
# ("exact", phrase) the whole reply is the phrase up to punctuation; ("yes_no", "yes") a word-bounded yes without a no
CONVERSATIONS = [
    {"name": "travel", "turns": list(zip(TRAVEL_TURNS, [
        ("any", ["day 1", "day one", "day 2", "day 3"]), ("any", ["sintra"]), ("lines", 5), ("min_words", 10),
        ("any", ["restaurant", "tasca", "dish", "bacalhau", "sardine", "pastel", "cuisine", "seafood", "pastéis", "grilled", "cafe", "café", "bistro"]), ("lines", 2),
        ("re", [r"\bjour\b", r"\blisbonne\b", r"\bvoyage\b", r"\bles\b", r"\bdu\b", r"\bet\b", r"\bune\b", r"\bdes\b"]), ("exact", "safe travels")]))},
    {"name": "planning", "turns": [
        ("I am organising a small team offsite for 8 people over two days. Propose a schedule: one line per half-day, four lines.", ("lines", 4)),
        ("Move the workshop to the first morning and keep everything else. Reprint the four lines.", ("lines", 4)),
        ("We have one vegetarian and one person who cannot drive. What two things change in the plan? Two sentences.", ("min_words", 12)),
        ("Give me a budget line for food only, assuming 25 per person per day. Reply with the total number only.", ("number", 400)),
        ("Add a 10 percent buffer to that food budget. Reply with the new total number only.", ("number", 440)),
        ("Write the subject line of the invitation email, under ten words.", ("max_words", 9)),
        ("List three things each person should bring, one per line, nothing else.", ("lines", 3)),
        ("Reply with just: plan locked.", ("exact", "plan locked"))]},
    {"name": "programming", "turns": [
        ("I have a Python list of dictionaries with keys 'name' and 'age'. How do I sort it by age descending? One line of code.", ("any", ["sorted(", ".sort("])),
        ("Now sort by age descending and then by name ascending for ties. One line of code.", ("any", ["sorted(", ".sort("])),
        ("What happens if some dictionaries lack the 'age' key? One sentence.", ("any", ["keyerror", "error", "missing", "exception", "raise", ".get("])),
        ("Show a version that treats a missing age as 0. One line of code.", ("any", [".get(", "get("])),
        ("Write a one-line pytest assertion that the first element after sorting has the largest age.", ("any", ["assert"])),
        ("Is the sort stable? Answer yes or no, then one sentence.", ("yes_no", "yes")),
        ("Give the function a name and a one-line docstring. Show only the def line and the docstring.", ("any", ["def ", '"""', "'''"])),
        ("Reply with just: done.", ("exact", "done"))]},
    {"name": "factual", "turns": [
        ("Which element has the chemical symbol Fe? One word.", ("any", ["iron"])),
        ("What is its atomic number? Number only.", ("number", 26)),
        ("Name one common alloy that contains it. One line.", ("any", ["steel", "cast iron", "wrought", "stainless", "pig iron"])),
        ("What is the capital of Portugal? One word.", ("any", ["lisbon", "lisboa"])),
        ("Which ocean borders that country? One word.", ("any", ["atlantic"])),
        ("Roughly how many days does Earth take to orbit the Sun? Number only.", ("number", 365)),
        ("Which planet is known as the red planet? One word.", ("any", ["mars"])),
        ("Reply with just: thanks.", ("exact", "thanks"))]},
]
CONFIGS = [("NOTHINK", {"enable_thinking": False}), ("LOW", {"enable_thinking": True, "reasoning_effort": "low"})]


def addressed(check, final: str) -> tuple:
    t = (final or "").strip()
    low = t.lower()
    kind = check[0]
    if not t:
        return False, "empty reply"
    if kind == "any":
        hits = [w for w in check[1] if w in low]
        return bool(hits), f"any of {check[1][:4]}: {hits[:3]}"
    if kind == "all":
        hits = [w for w in check[1] if w in low]
        return len(hits) == len(check[1]), f"all of {check[1]}: {hits}"
    if kind == "number":
        nums = [int(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*", t)]
        return check[1] in nums, f"number {check[1]} in {nums[:6]}"
    if kind == "lines":
        n = len([ln for ln in t.splitlines() if ln.strip()])
        return n == check[1], f"{n} lines, wanted {check[1]}"
    if kind == "max_words":
        n = len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])
        return 0 < n <= check[1], f"{n} words, max {check[1]}"
    if kind == "min_words":
        n = len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])
        return n >= check[1], f"{n} words, min {check[1]}"
    if kind == "re":
        hits = [pat for pat in check[1] if re.search(pat, low)]
        return bool(hits), f"patterns {hits[:3]}"
    if kind == "exact":
        return re.sub(r"[^a-z ]", "", low).strip() == check[1], f"exact {check[1]!r}: got {t[:30]!r}"
    if kind == "yes_no":
        yes, no = re.search(r"\byes\b", low) is not None, re.search(r"\bno\b", low) is not None
        return (yes and not no) if check[1] == "yes" else (no and not yes), f"yes {yes} no {no}"
    return True, "nonempty"


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def margin(rec, stop_ids):
    ids, lg = rec["ids"], rec["logits"]
    best_stop = max(rec["stop_logits"].values())
    non = [v for i, v in zip(ids, lg) if i not in stop_ids]
    chosen_stop = ids[0] in stop_ids
    return {"first_argmax": ids[0], "first_is_stop": chosen_stop, "stop_margin": (lg[0] - max(non)) if (chosen_stop and non) else (lg[0] - best_stop)}


def run(eng, conv, cfg, mode):
    from llmopt.lab import qserve
    import mlx.core as mx
    sess = qserve.Session(eng, mode=mode, **cfg)
    rows = []
    for k, (text, check) in enumerate(conv["turns"]):
        r = sess.ask(text, max_tokens=BUDGET)
        thinking = cfg.get("enable_thinking") is not False
        final = qbattery.split_final(r.text)[1] if thinking else r.text
        terminated = (248069 in r.generated_ids) if thinking else None
        ok, why = addressed(check, final if thinking else r.text)
        rows.append({"turn": k + 1, "user": text, "check": list(check), "mode": r.mode, "note": r.note, "prompt_len": r.n_prompt, "prompt_sha256": sha(r.prompt_ids),
                     "markers": {nm: r.prompt_ids.count(v) for nm, v in MARKERS.items()}, "generated_ids": r.generated_ids, "text": r.text, "final": final,
                     "n_generated": r.n_generated, "n_words_final": len(re.findall(r"[A-Za-z0-9']+", final or "")), "stop_reason": r.stop_reason, "stop_token_id": r.stop_token_id,
                     **margin(r.first_logits_topk, eng.stop_ids), "empty": (final or "").strip() == "", "thinking_terminated": terminated,
                     "addressed": ok, "addressed_why": why, "cache_tokens": r.cache_tokens, "cache_offset_measured": r.cache_offset_measured,
                     "first_token_s": r.timings["first_token_s"], "total_s": r.timings["total_s"], "active_bytes": r.memory["active_bytes"],
                     "footprint_bytes": r.memory["footprint_bytes"], "peak_bytes": r.memory["peak_bytes"]})
        print(f"[lowchat] {conv['name']:11s} {cfg.get('reasoning_effort') or 'nothink':7s} {mode[:4]} t{k + 1} gen {r.n_generated:3d} {r.stop_reason[:4]} "
              f"empty {rows[-1]['empty']!s:5s} addressed {ok!s:5s} term {terminated} margin {rows[-1]['stop_margin']:+.2f} {r.mode[:4]}", flush=True)
    del sess, r
    mx.clear_cache()
    return rows


def longest_empty_run(rows):
    best = cur = 0
    for x in rows:
        cur = cur + 1 if x["empty"] else 0
        best = max(best, cur)
    return best


def main() -> None:
    from llmopt.lab import qserve
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    convs = CONVERSATIONS if not SMOKE else [dict(CONVERSATIONS[1], turns=CONVERSATIONS[1]["turns"][:2])]
    eng = qserve.Engine()
    t0 = time.perf_counter()
    out = []
    for conv in convs:
        entry = {"name": conv["name"], "configs": {}}
        for cname, cfg in CONFIGS:
            inc = run(eng, conv, cfg, "incremental")
            rep = run(eng, conv, cfg, "replay") if cname == "LOW" else None
            entry["configs"][cname] = {"incremental": inc, "replay": rep,
                                       "replay_equal_tokens": ([a["generated_ids"] == b["generated_ids"] for a, b in zip(inc, rep)] if rep else None)}
        out.append(entry)
    summary = {}
    for cname, _ in CONFIGS:
        rows = [(e["name"], x) for e in out for x in e["configs"][cname]["incremental"]]
        summary[cname] = {"turns": len(rows), "empty_turns": [(n, x["turn"]) for n, x in rows if x["empty"]],
                          "longest_empty_run": max(longest_empty_run(e["configs"][cname]["incremental"]) for e in out),
                          "addressed_turns": sum(x["addressed"] for _, x in rows), "unaddressed": [(n, x["turn"], x["addressed_why"]) for n, x in rows if not x["addressed"]],
                          "unterminated_turns": sum(1 for _, x in rows if x["thinking_terminated"] is False),
                          "travel_empty_turns": sum(1 for n, x in rows if n == "travel" and x["empty"]),
                          "generated_tokens": sum(x["n_generated"] for _, x in rows), "total_s": round(sum(x["total_s"] for _, x in rows), 1),
                          "first_token_s_median": sorted(x["first_token_s"] for _, x in rows)[len(rows) // 2],
                          "later_fallbacks": sum(1 for _, x in rows if x["turn"] > 1 and x["mode"] != "incremental"),
                          # the template re-renders an UNCLOSED thinking turn with a close the model never emitted, so the turn after a
                          # budget cut must fall back by design; the gate counts fallbacks after a CLOSED (or non-thinking) previous turn
                          "later_fallbacks_after_closed": sum(1 for e in out for k, x in enumerate(e["configs"][cname]["incremental"])
                                                              if k > 0 and x["mode"] != "incremental" and e["configs"][cname]["incremental"][k - 1]["thinking_terminated"] is not False),
                          "later_fallbacks_after_cut": sum(1 for e in out for k, x in enumerate(e["configs"][cname]["incremental"])
                                                           if k > 0 and x["mode"] != "incremental" and e["configs"][cname]["incremental"][k - 1]["thinking_terminated"] is False),
                          "offset_mismatch_turns": sum(1 for _, x in rows if x["cache_offset_measured"] != x["cache_tokens"]),
                          "peak_bytes_max": max(x["peak_bytes"] for _, x in rows), "active_bytes_last": [e["configs"][cname]["incremental"][-1]["active_bytes"] for e in out],
                          "stop_tokens": {str(x["stop_token_id"]): sum(1 for _, y in rows if y["stop_token_id"] == x["stop_token_id"]) for _, x in rows if x["stop_token_id"] is not None},
                          "min_stop_margin_on_empty": min((x["stop_margin"] for _, x in rows if x["empty"]), default=None)}
    summary["LOW"]["replay_token_mismatch_turns"] = sum(1 for e in out for eq in e["configs"]["LOW"]["replay_equal_tokens"] if not eq)
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "budget": BUDGET, "engine": {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"),
               "chain_sha256": eng.info.get("chain_sha256"), "stop_ids": sorted(eng.stop_ids), "context_cap": eng.context_cap, "cache_limit_bytes": eng.cfg.cache_limit_bytes,
               "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {})}, "conversations": out, "summary": summary, "wall_s": time.perf_counter() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[lowchat] summary {json.dumps({c: {k: v for k, v in s.items() if not isinstance(v, (list, dict))} for c, s in summary.items()})}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

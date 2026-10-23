"""QWEN-EFFORT-AND-DEEPCHAT-0, study B: a bounded, read-only diagnostic of
the empty completions seen in the eight-turn thinking-off travel
conversation of VERDICT QWEN-CHAT-DEFAULT-0 (three consecutive immediate
stops on turns 5 to 7 in both modes). The frozen conversations are the
travel transcript and two fixed controls (debug, recipe) exactly as in
scratch/qwen_incr_perf.CONVERSATIONS; no prompt is adapted to an output.

Per turn, in a replay session and in an incremental session: the rendered
prompt ids (sha256, length, the counts and positions of the template
markers <|im_start|> 248045, <|im_end|> 248046, <think> 248068, </think>
248069), the prefill's top-16 first-token logits with the two stop ids'
logits, the argmax, the stop margin (the chosen stop's logit minus the
best non-stop logit when a stop is chosen; else the best logit minus the
best stop logit), generated count, stop reason, budget, mode and note,
accounted and measured cache offsets, memory, and the engine's kernel
generation and chain sha. The independent reconstruction: the transcript
re-rendered OUTSIDE Session (the tokenizer's chat template applied to the
replay session's stored messages) and compared id for id with the
session's prompt. Numerical: on every turn whose replay first token was a
stop id, the same prompt re-prefilled through the fused (non-GEMM) path
and the top-16 logits compared (max absolute difference, argmax equality);
and the incremental session's first-token logits compared with replay's
(two kernel paths for the earlier tokens). COUNTERFACTUALS, labelled and
never a baseline: (CF1) the same user turn asked as a fresh single-turn
conversation; (CF2) the transcript with the empty assistant turns removed
up to that point (stored as counterfactual rows). Receipt OUT (default
logs/qwen_serve/deepchat_v1.json; refuses to overwrite). SMOKE=1 runs the
first three turns of travel into logs/qwen_serve/smoke/deepchat.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("qwen_incr_perf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_incr_perf.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CONVERSATIONS = _mod.CONVERSATIONS      # the frozen conversations of VERDICT QWEN-CHAT-DEFAULT-0, unchanged

SMOKE = os.environ.get("SMOKE") == "1"
OUT = "logs/qwen_serve/smoke/deepchat.json" if SMOKE else os.environ.get("OUT", "logs/qwen_serve/deepchat_v1.json")
MARKERS = {"im_start": 248045, "im_end": 248046, "think": 248068, "think_close": 248069}


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def markers(ids):
    return {k: {"count": ids.count(v), "positions": [i for i, t in enumerate(ids) if t == v][-6:]} for k, v in MARKERS.items()}


def margin(rec, stop_ids):
    ids, lg = rec["ids"], rec["logits"]
    best_stop = max(rec["stop_logits"].values())
    non_stop = [v for i, v in zip(ids, lg) if i not in stop_ids]
    best_non_stop = max(non_stop) if non_stop else float("-inf")
    chosen_stop = ids[0] in stop_ids
    return {"argmax": ids[0], "argmax_is_stop": chosen_stop, "margin": (lg[0] - best_non_stop) if chosen_stop else (lg[0] - best_stop),
            "best_non_stop": (next(i for i in ids if i not in stop_ids) if non_stop else None)}


def row_of(r, eng):
    m = margin(r.first_logits_topk, eng.stop_ids)
    return {"mode": r.mode, "note": r.note, "prompt_len": r.n_prompt, "prompt_sha256": sha(r.prompt_ids), "prompt_tail": r.prompt_ids[-12:], "markers": markers(r.prompt_ids),
            "first_logits_topk": r.first_logits_topk, **m, "generated_ids": r.generated_ids, "text": r.text, "n_generated": r.n_generated,
            "stop_reason": r.stop_reason, "stop_token_id": r.stop_token_id, "cache_tokens": r.cache_tokens, "cache_offset_measured": r.cache_offset_measured,
            "first_token_s": r.timings["first_token_s"], "active_bytes": r.memory["active_bytes"], "footprint_bytes": r.memory["footprint_bytes"], "peak_bytes": r.memory["peak_bytes"]}


def fused_first_logits(eng, ids):
    """the same prompt through the fused (non-GEMM) prefill path: top-16 on the final position."""
    import mlx.core as mx
    from llmopt.lab import qprefill
    cache = eng.model.make_cache()
    lg = qprefill.prefill_chunked(eng.model, ids, cache, chunk=eng.cfg.chunk, head="final", matmul="fused")
    mx.eval(lg)
    row = lg[0, -1].astype(mx.float32)
    top = mx.argpartition(-row, 16)[:16]
    order = mx.argsort(-row[top])
    out = {"ids": [int(i) for i in top[order].tolist()], "logits": [float(v) for v in row[top][order].tolist()],
           "stop_logits": {int(t): float(row[t].item()) for t in sorted(eng.stop_ids)}}
    del cache
    mx.clear_cache()
    return out


def compare_topk(a, b):
    da = dict(zip(a["ids"], a["logits"]))
    db = dict(zip(b["ids"], b["logits"]))
    common = set(da) & set(db)
    return {"argmax_equal": a["ids"][0] == b["ids"][0], "common_top16": len(common),
            "max_abs_diff_common": max(abs(da[i] - db[i]) for i in common) if common else None,
            "stop_logit_abs_diff": max(abs(a["stop_logits"][k] - b["stop_logits"][k]) for k in a["stop_logits"])}


def main() -> None:
    from llmopt.lab import qserve
    import mlx.core as mx
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    specs = [c for c in CONVERSATIONS if c["name"] in ("travel", "debug", "recipe")]
    if SMOKE:
        specs = [dict(specs[0], turns=specs[0]["turns"][:3])]
    eng = qserve.Engine()
    t0 = time.perf_counter()
    convs = []
    for spec in specs:
        rep_sess = qserve.Session(eng, mode="replay", enable_thinking=spec["thinking"])
        inc_sess = qserve.Session(eng, mode="incremental", enable_thinking=spec["thinking"])
        turns = []
        for k, (text, budget) in enumerate(spec["turns"]):
            msgs_before = list(rep_sess.messages)
            rr = rep_sess.ask(text, max_tokens=budget)
            ri = inc_sess.ask(text, max_tokens=budget)
            # independent reconstruction of the prompt, outside Session
            ids_ind = eng.tokenizer.apply_chat_template(msgs_before + [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, enable_thinking=spec["thinking"])
            ids_ind = [int(t) for t in (ids_ind["input_ids"] if hasattr(ids_ind, "input_ids") else ids_ind)]
            row = {"turn": k + 1, "user": text, "budget": budget, "replay": row_of(rr, eng), "incremental": row_of(ri, eng),
                   "reconstruction_equal_to_replay_prompt": ids_ind == rr.prompt_ids, "reconstruction_len": len(ids_ind),
                   "replay_vs_incremental_generated_equal": rr.generated_ids == ri.generated_ids,
                   "replay_vs_incremental_first_logits": compare_topk(rr.first_logits_topk, ri.first_logits_topk),
                   "empty_assistant_turns_in_history": sum(1 for m in msgs_before if m["role"] == "assistant" and not m["content"].strip())}
            if rr.first_logits_topk["ids"][0] in eng.stop_ids:
                fused = fused_first_logits(eng, rr.prompt_ids)
                row["fused_first_logits_topk"] = fused
                row["gemm_vs_fused_first_logits"] = compare_topk(rr.first_logits_topk, fused)
                # counterfactual 1: the user turn alone, a fresh single-turn conversation
                cf1 = eng.chat([{"role": "user", "content": text}], max_tokens=budget, enable_thinking=spec["thinking"])
                row["counterfactual_single_turn"] = {"label": "COUNTERFACTUAL", **{k2: v for k2, v in row_of(cf1, eng).items() if k2 in ("argmax", "argmax_is_stop", "margin", "n_generated", "stop_reason", "text")}}
                # counterfactual 2: the history with its empty assistant turns removed (pairs with their user turns kept)
                pruned = [m for m in msgs_before if not (m["role"] == "assistant" and not m["content"].strip())]
                if len(pruned) != len(msgs_before):
                    cf2 = eng.chat(pruned + [{"role": "user", "content": text}], max_tokens=budget, enable_thinking=spec["thinking"])
                    row["counterfactual_empties_removed"] = {"label": "COUNTERFACTUAL", "removed": len(msgs_before) - len(pruned),
                                                              **{k2: v for k2, v in row_of(cf2, eng).items() if k2 in ("argmax", "argmax_is_stop", "margin", "n_generated", "stop_reason", "text", "prompt_len")}}
            turns.append(row)
            print(f"[deepchat] {spec['name']:7s} turn {k + 1} replay gen {rr.n_generated:3d} stop {rr.stop_reason:10s} argmax {rr.first_logits_topk['ids'][0]} margin {row['replay']['margin']:+.3f} "
                  f"inc {ri.mode[:4]} gen {ri.n_generated:3d} recon {row['reconstruction_equal_to_replay_prompt']} empties {row['empty_assistant_turns_in_history']}", flush=True)
        convs.append({"name": spec["name"], "thinking": spec["thinking"], "turns": turns})
        del rep_sess, inc_sess, rr, ri
        mx.clear_cache()
    empties = [(c["name"], t["turn"]) for c in convs for t in c["turns"] if t["replay"]["n_generated"] == 0]
    stop_rows = [t for c in convs for t in c["turns"] if t["replay"]["argmax_is_stop"]]
    summary = {"empty_replay_turns": empties, "travel_empty_turns": sum(1 for n, _ in empties if n == "travel"),
               "stop_tokens_chosen": {str(t["replay"]["argmax"]): sum(1 for u in stop_rows if u["replay"]["argmax"] == t["replay"]["argmax"]) for t in stop_rows},
               "reconstruction_mismatch_turns": sum(1 for c in convs for t in c["turns"] if not t["reconstruction_equal_to_replay_prompt"]),
               "incremental_first_argmax_mismatch_turns": sum(1 for c in convs for t in c["turns"] if not t["replay_vs_incremental_first_logits"]["argmax_equal"]),
               "incremental_generated_mismatch_turns": sum(1 for c in convs for t in c["turns"] if not t["replay_vs_incremental_generated_equal"]),
               "offset_mismatch_turns": sum(1 for c in convs for t in c["turns"] for m in ("replay", "incremental") if t[m]["cache_offset_measured"] != t[m]["cache_tokens"]),
               "gemm_vs_fused_max_abs_diff": max((t["gemm_vs_fused_first_logits"]["max_abs_diff_common"] or 0) for t in stop_rows) if stop_rows else None,
               "gemm_vs_fused_argmax_mismatch_turns": sum(1 for t in stop_rows if not t["gemm_vs_fused_first_logits"]["argmax_equal"]),
               "replay_vs_incremental_max_abs_diff": max(t["replay_vs_incremental_first_logits"]["max_abs_diff_common"] or 0 for c in convs for t in c["turns"]),
               "stop_margins": [(t["user"][:30], round(t["replay"]["margin"], 3)) for t in stop_rows],
               "min_stop_margin": min((t["replay"]["margin"] for t in stop_rows), default=None),
               "counterfactual_single_turn_nonempty": sum(1 for t in stop_rows if t.get("counterfactual_single_turn", {}).get("n_generated", 0) > 0),
               "counterfactual_empties_removed_nonempty": sum(1 for t in stop_rows if t.get("counterfactual_empties_removed", {}).get("n_generated", 0) > 0),
               "counterfactual_empties_removed_rows": sum(1 for t in stop_rows if "counterfactual_empties_removed" in t)}
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "engine": {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"),
               "chain_sha256": eng.info.get("chain_sha256"), "stop_ids": sorted(eng.stop_ids), "context_cap": eng.context_cap, "cache_limit_bytes": eng.cfg.cache_limit_bytes,
               "matmul": eng.cfg.matmul, "chunk": eng.cfg.chunk, "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {})},
               "conversations": convs, "summary": summary, "wall_s": time.perf_counter() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[deepchat] summary {json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)})}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

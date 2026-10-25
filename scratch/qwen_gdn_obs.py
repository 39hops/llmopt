"""Observations adapter for PRE-REG QWEN-GDN-KERNEL-0: derives the registered
measurements from the five stage receipts of scratch/qwen_gdn_kernel_qual.py
in committed code (never hand-written).

Measurements (bar: metric)
  1   census_argmax_mismatches        positions of 4,722 where kernel argmax != loop argmax
  2   census_near_tie_flips           of those, the 15 frozen near ties (frozen metal_margin < 0.02)
  3   state_max_abs_over_rms          layers stage: kernel from the loop's own operands and state-in,
                                      and kernel carried across chunks, max over 6 layers x 4 chunks (+ T = 1 step)
  3:rel  state_max_rel_on_large       the element-wise relative error on large elements, same population
  3:prop propagated_state_max_abs_over_rms  final states after the whole tower: census 48 layers x 24 prompts
                                      and the decode prompts (sanity bound)
  3:finite nonfinite_arrays           compared arrays that were not finite
  4   decode_stream_mismatch_prompts  prompts whose 256-step free-run streams differ
  4:logit teacher_forced_max_abs_logit_diff   max over 2 x 256 teacher-forced steps
  4:repeat kernel_repeat_unequal_prompts      kernel run twice, streams unequal
  5   cache_history_unequal_turns     turns where kernel incremental != kernel replay or != loop incremental
  6   peak_ratio_max                  max over the three held-out lengths of the peak-memory ratio
  6:decode decode16_s_ratio_max       max over lengths of the median 16-step decode time ratio
  6:ttft512 ttft_ratio_512            median TTFT ratio at 512
  7/8 ttft_ratio_1024 / _2048         median TTFT ratios
  9   decode_step_ratio_long_1024     median steady step ratio over 256 steps at 1,024 (4 against 4)
  10  e2e_ratio_long_1024             prefill plus 256 steps, ratio at 1,024 (4 against 4)
Validity: five non-smoke clean-tree receipts from ONE commit, one vendor
provenance (mlx, mlx-lm, source shas, Metal GPU as the default device), one
selector source, the registered artifact chain and serving configuration; the
tower in its default training mode in every receipt; DISPATCH PROOF: every
LOOP selection executed ops only and every KERNEL selection executed the
kernel only (the receipts' executed counts, per label, per held-out run and
for the layers stage's offline calls); the frozen census and teacher token
files at their pinned sha256s; the census covers 4,722 positions with exactly
15 near ties and at least one sequence longer than a chunk; the layers stage
ran six ordinals over four chunks of an unread 1,000 to 1,024-token prompt;
the held-out prompts carry the frozen sha256s with 6 runs per arm per length
and both arms' ABBA spreads within 3 percent on TTFT and 16-step decode; the
loop arm self-consistent on argmax and decode; every eligible chat turn reused
the cache in both arms with no fallback. The long 4-against-4 runs with a
spread above 5 percent mark the contrast behind bars 9 and 10 inadmissible
(UNRESOLVED) without invalidating the rest.
Usage: qwen_gdn_obs.py <receipt dir> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

STAGES = ("layers", "census", "decode", "heldout", "cachehist")
PINS = {"frozen_census_sha256": "b2ec44f16916476abd891dd6c0601ed1982732f7ff5b185de23233e305646287",      # logs/qwen_metal_qual/C_v5/positions.jsonl
        "rollout_tokens_sha256": "5cf8d21469fdc7eba1a00c0317f22a262a965c017417935b718477cf64bb7fa3"}     # logs/qwenteacher_v2/rollout_tokens.jsonl
LAYER_ORDINALS = [0, 9, 19, 29, 39, 47]
SPREAD_PCT = 3.0        # paired ABBA spread (max minus min over the median, per arm) admissible under fences of 0.01 to 0.03
LONG_SPREAD_PCT = 5.0   # the long 4-against-4 runs behind bars 9 and 10
WANT = {"LOOP": "ops", "KERNEL": "kernel", "loop": "ops", "kernel": "kernel"}


def dispatch_ok(d):
    """One selector receipt: the selected implementation executed, the other never, at least one call."""
    want = WANT.get(d.get("path"))
    ex = d.get("executed", {})
    other = "kernel" if want == "ops" else "ops"
    return bool(want) and ex.get(want, 0) > 0 and ex.get(other, 0) == 0 and d.get("update_calls", 0) > 0


def main(rdir, prereg_path):
    prereg = load(prereg_path)
    R = {}
    reasons = []
    for st in STAGES:
        p = os.path.join(rdir, f"{st}.json")
        if not os.path.exists(p):
            reasons.append(f"missing receipt {p}")
            continue
        R[st] = json.load(open(p))
    if reasons:
        return emit(prereg, False, reasons, {}, {}, rdir)
    commits = {r.get("commit") for r in R.values()}
    if len(commits) != 1:
        reasons.append(f"receipts from several commits {sorted(commits)}")
    for st, r in R.items():
        if r.get("smoke") or r.get("tree_dirty"):
            reasons.append(f"{st}: smoke run or dirty tree")
        if r.get("model_training_flag") is not True:
            reasons.append(f"{st}: the tower was not in its default training mode")
        e = r.get("engine", {})
        if not e.get("chain_sha256") or e.get("kernel_version") != 5 or e.get("cache_limit_bytes") != 2 ** 31 or e.get("matmul") != "gemm" or e.get("chunk") != 256:
            reasons.append(f"{st}: engine is not the qualified serving configuration")
        if e.get("families") != {"linear_attn": 48, "full_attn": 16}:
            reasons.append(f"{st}: layer families {e.get('families')}")
        for d in r.get("dispatch", []):
            if not dispatch_ok(d):
                reasons.append(f"{st}: dispatch proof failed for {d.get('label')}: {d.get('executed')}")
    chains = {r["engine"].get("chain_sha256") for r in R.values()}
    if len(chains) != 1:
        reasons.append("receipts on different artifact chains")
    vendors = {json.dumps({k: r.get("vendor", {}).get(k) for k in ("mlx_version", "mlx_lm_version", "qwen3_5_sha256", "gated_delta_sha256", "default_device")}, sort_keys=True) for r in R.values()}
    if len(vendors) != 1:
        reasons.append("receipts under different vendor provenance (mlx, mlx-lm, source sha or device)")
    for st, r in R.items():
        if "gpu" not in str(r.get("vendor", {}).get("default_device", "")) or not r.get("vendor", {}).get("metal_available"):
            reasons.append(f"{st}: default device is not the Metal GPU (the kernel path would silently fall back to the loop)")
    selectors = {r.get("selector_sha256") for r in R.values()}
    if len(selectors) != 1 or None in selectors:
        reasons.append("receipts under different (or unrecorded) selector sources")
    c, s = R["census"], R["census"]["summary"]
    if s.get("n_positions") != 4722:
        reasons.append(f"census covers {s.get('n_positions')} positions, not 4722")
    if s.get("n_near_ties") != 15:
        reasons.append(f"census found {s.get('n_near_ties')} near ties, not 15")
    if c.get("frozen_census", {}).get("n") != 4722 or c.get("frozen_census", {}).get("sha256") != PINS["frozen_census_sha256"]:
        reasons.append("frozen census reference is not the pinned 4,722-row file")
    if c.get("teacher", {}).get("rollout_tokens_sha256") != PINS["rollout_tokens_sha256"] or R["decode"].get("teacher", {}).get("rollout_tokens_sha256") != PINS["rollout_tokens_sha256"]:
        reasons.append("teacher rollout tokens are not the pinned file")
    if not any(p.get("tokens", 0) > 256 for p in c.get("per_prompt", [])) or not any(d.get("max_T") == 256 for d in c.get("dispatch", [])):
        reasons.append("no census sequence spans more than one chunk, so the kernel never carried state across a chunk boundary inside the census")
    L0 = R["layers"]
    if L0.get("capture", {}).get("chunks") != 4 or L0.get("capture", {}).get("ordinals") != LAYER_ORDINALS or not (1000 <= L0.get("prompt", {}).get("tokens", 0) <= 1024):
        reasons.append(f"layers stage population is not six ordinals {LAYER_ORDINALS} over four chunks of a 1,000 to 1,024-token prompt: {L0.get('capture')}, {L0.get('prompt')}")
    if L0.get("prompt", {}).get("seed") == "gdn-dev-1-1024":
        reasons.append("layers stage ran on the pre-registration calibration prompt")
    for path, want in (("loop", "ops"), ("kernel", "kernel")):
        od = L0.get("offline_dispatch", {}).get(path, {})
        other = "kernel" if want == "ops" else "ops"
        if not od.get(want) or od.get(other):
            reasons.append(f"layers stage offline {path} dispatch proof failed: {od}")
    h = R["heldout"]
    frozen = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalsets", "qwen_prefill_heldout_prompts.json")))["prompts"]
    for t in ("512", "1024", "2048"):
        if h.get("prompts", {}).get(t, {}).get("sha256") != frozen[t]["sha256"]:
            reasons.append(f"held-out prompt {t} differs from the frozen sha")
        for arm in ("LOOP", "KERNEL"):
            n = sum(1 for x in h.get("runs", []) if x.get("arm") == arm and str(x.get("target")) == t)
            if n != 6:
                reasons.append(f"heldout {arm} at {t}: {n} runs, not 6")
        if h.get("summary", {}).get(t, {}).get("loop_self_mismatches", 1):
            reasons.append(f"heldout {t}: the loop arm disagrees with itself on argmax or decode")
        hs_t = h.get("summary", {}).get(t, {})
        for k in ("loop_spread_pct", "kernel_spread_pct", "loop_decode16_spread_pct", "kernel_decode16_spread_pct"):
            if hs_t.get(k, 99) > SPREAD_PCT:
                reasons.append(f"heldout {t}: {k} {hs_t.get(k, 99):.1f} above {SPREAD_PCT} percent; the fences cannot be read against this noise")
    for x in h.get("runs", []) + h.get("long_runs", []) + h.get("instrumented", []) + h.get("runs_4096", []):
        d = x.get("dispatch", {})
        if x.get("arm") in WANT and (d.get("path") != WANT[x["arm"]].replace("ops", "loop") or not dispatch_ok(d)):
            reasons.append(f"heldout run ({x.get('arm')}, {x.get('tokens')} tokens): dispatch proof failed {d.get('executed')}")
            break
    contrasts = {}
    lg0 = h.get("summary", {}).get("long:1024", {})
    if lg0.get("steps") != 256 or lg0.get("n") != 4 or lg0.get("loop_self_equal") != 4:
        reasons.append("the long generation at 1,024 is not two ABBA blocks of 256 steps with a self-consistent loop arm")
    elif any(v > LONG_SPREAD_PCT for v in lg0.get("spread_pct", {}).values()):
        # the arms stay admissible; the CONTRAST behind bars 9 and 10 is not readable at their fences
        contrasts = {"9": {"admissible": False, "reason": f"long-run spread {lg0['spread_pct']} above {LONG_SPREAD_PCT} percent"}, "10": {"admissible": False, "reason": f"long-run spread {lg0['spread_pct']} above {LONG_SPREAD_PCT} percent"}}
    d = R["decode"]["summary"]
    if d.get("n_prompts") != 2:
        reasons.append(f"decode covers {d.get('n_prompts')} prompts, not 2")
    ch = R["cachehist"]["summary"]
    if ch.get("n_turns") != 9:
        reasons.append(f"cachehist covers {ch.get('n_turns')} turns, not 9")
    if ch.get("incremental_reused_turns") != {"LOOP": 6, "KERNEL": 6} or ch.get("fallback_turns") != {"LOOP": 0, "KERNEL": 0}:
        reasons.append(f"cachehist did not exercise incremental reuse on every eligible turn: reused {ch.get('incremental_reused_turns')}, fallbacks {ch.get('fallback_turns')}")
    L = R["layers"]["summary"]
    if not L.get("same_state_in", {}).get("all_finite", False):
        reasons.append("layers stage: non-finite arrays")
    valid = not reasons
    meas, diag = {}, {}
    if valid:
        hs = h["summary"]
        pop3 = "layers-stage:6-dnet-layers-x-4-chunks+kernel-carried+decode-step"
        pop_prop = "final-states:census-48-layers-x-24-prompts+decode-2-prompts"
        state_max = max(L["same_state_in"]["state_max_abs_over_rms"], L["kernel_carried"]["state_max_abs_over_rms"])
        rel_max = max(L["same_state_in"]["state_max_rel_on_large"], L["kernel_carried"]["state_max_rel_on_large"])
        prop_max = max(s["state_max_abs_over_rms"], d["final_state_max_abs_over_rms"])
        nonfinite = R["layers"]["nonfinite_arrays"] + R["census"]["nonfinite_arrays"] + d["nonfinite_arrays"]
        meas["1"] = {"value": s["n_mismatches"], "metric": "census_argmax_mismatches", "population": "census-positions:4722", "aggregation": "count", "unit": "positions",
                     "provenance": json.dumps({"mismatches": s["argmax_mismatches"], "full_vocab_max_abs_diff": s["full_vocab_max_abs_diff"], "top16_max_abs_diff": s["top16_max_abs_diff"], "teacher_agree": s["teacher_agree"], "loop_vs_frozen_metal_identical": s["loop_vs_frozen_metal_identical"]})}
        meas["2"] = {"value": s["near_tie_flips"], "metric": "census_near_tie_flips", "population": "census-near-ties:15", "aggregation": "count", "unit": "positions", "provenance": json.dumps(s["near_ties"])}
        meas["3"] = {"value": state_max, "metric": "state_max_abs_over_rms", "population": pop3, "aggregation": "max", "unit": "ratio",
                     "provenance": json.dumps({"same_state_in": L["same_state_in"]["state_max_abs_over_rms"], "kernel_carried": L["kernel_carried"]["state_max_abs_over_rms"], "y_same_state_in": L["same_state_in"]["y_max_abs_over_rms"], "kernel_repeat_max_abs": L["same_state_in"]["kernel_repeat_max_abs"], "per_call_ms": L["per_call_ms"]})}
        meas["3:rel"] = {"value": rel_max, "metric": "state_max_rel_on_large", "population": pop3, "aggregation": "max", "unit": "ratio"}
        meas["3:prop"] = {"value": prop_max, "metric": "propagated_state_max_abs_over_rms", "population": pop_prop, "aggregation": "max", "unit": "ratio",
                          "provenance": json.dumps({"census_final": s["state_max_abs_over_rms"], "census_conv": s["conv_state_max_abs_over_rms"], "census_rel_on_large": s["state_max_rel_on_large"], "decode_final": d["final_state_max_abs_over_rms"]})}
        meas["3:finite"] = {"value": nonfinite, "metric": "nonfinite_arrays", "population": "all-compared-arrays", "aggregation": "count", "unit": "arrays"}
        meas["4"] = {"value": d["stream_mismatch_prompts"], "metric": "decode_stream_mismatch_prompts", "population": "decode-prompts:2", "aggregation": "count", "unit": "prompts",
                     "provenance": json.dumps([{k: p["free_run"][k] for k in ("streams_equal", "first_divergence", "margins_at_divergence", "step_median_s")} for p in R["decode"]["prompts"]])}
        meas["4:logit"] = {"value": d["teacher_forced_max_abs_logit_diff"], "metric": "teacher_forced_max_abs_logit_diff", "population": "decode-steps:512", "aggregation": "max", "unit": "logits",
                           "provenance": json.dumps({"argmax_mismatches_teacher_forced": d["teacher_forced_argmax_mismatches"], "final_state_max_abs_over_rms": d["final_state_max_abs_over_rms"], "state_by_step_max": d.get("state_by_step_max")})}
        meas["4:repeat"] = {"value": d["kernel_repeat_unequal_prompts"], "metric": "kernel_repeat_unequal_prompts", "population": "decode-prompts:2", "aggregation": "count", "unit": "prompts"}
        meas["5"] = {"value": ch["unequal_turns"], "metric": "cache_history_unequal_turns", "population": "chat-turns:9", "aggregation": "count", "unit": "turns",
                     "provenance": json.dumps({k: ch[k] for k in ("kernel_inc_vs_replay_unequal", "kernel_vs_loop_unequal", "loop_inc_vs_replay_unequal", "incremental_reused_turns", "fallback_turns")})}
        lens = ("512", "1024", "2048")
        meas["6"] = {"value": max(hs[t]["peak_after_decode_ratio"] for t in lens), "metric": "peak_ratio_max", "population": "heldout-lengths:3", "aggregation": "max_ratio", "unit": "ratio",
                     "provenance": json.dumps({t: {"peak": round(hs[t]["peak_ratio"], 4), "peak_after_decode": round(hs[t]["peak_after_decode_ratio"], 4), "footprint": round(hs[t]["footprint_ratio"], 4), "transient": round(hs[t]["transient_ratio"], 4), "parity_mismatches": hs[t]["parity_mismatches"], "top16_max_abs_diff": hs[t]["top16_max_abs_diff"]} for t in lens})}
        meas["6:decode"] = {"value": max(hs[t]["decode16_s_ratio"] for t in lens), "metric": "decode16_s_ratio_max", "population": "heldout-lengths:3", "aggregation": "max_median_ratio", "unit": "ratio",
                            "provenance": json.dumps({t: round(hs[t]["decode16_s_ratio"], 4) for t in lens})}
        meas["6:ttft512"] = {"value": hs["512"]["ttft_ratio"], "metric": "ttft_ratio_512", "population": "heldout-512:6v6", "aggregation": "median_ratio", "unit": "ratio"}
        for t, bid in (("1024", "7"), ("2048", "8")):
            meas[bid] = {"value": hs[t]["ttft_ratio"], "metric": f"ttft_ratio_{t}", "population": f"heldout-{t}:6v6", "aggregation": "median_ratio", "unit": "ratio",
                         "provenance": json.dumps({"ttft_median_s": hs[t]["ttft_median_s"], "loop_spread_pct": round(hs[t]["loop_spread_pct"], 2), "kernel_spread_pct": round(hs[t]["kernel_spread_pct"], 2), "instrumented": hs[t].get("instrumented")})}
        lg = hs["long:1024"]
        meas["9"] = {"value": lg["decode_step_ratio"], "metric": "decode_step_ratio_long_1024", "population": "long-1024:4v4", "aggregation": "median_ratio", "unit": "ratio", "provenance": json.dumps({"decode_step_ms": lg["decode_step_ms"], "streams_equal_to_loop": lg["streams_equal_to_loop"], "loop_self_equal": lg["loop_self_equal"]})}
        meas["10"] = {"value": lg["e2e_ratio"], "metric": "e2e_ratio_long_1024", "population": "long-1024:4v4", "aggregation": "median_ratio", "unit": "ratio", "provenance": json.dumps({"e2e_s": lg["e2e_s"]})}
        meas["census_argmax_mismatches"] = dict(meas["1"])
        diag = {"heldout": hs, "layers": L, "decode": d, "cachehist": ch, "census": {k: s[k] for k in ("argmax_identical", "teacher_agree", "full_vocab_max_abs_diff", "top16_max_abs_diff", "state_max_abs_over_rms", "wall_s")},
                "dispatch_counts": {st: [{"label": x["label"], "path": x["path"], "executed": x["executed"], "update_calls": x["update_calls"]} for x in R[st].get("dispatch", [])] for st in STAGES}}
    return emit(prereg, valid, reasons, meas, diag, rdir, commit=next(iter(commits)) if len(commits) == 1 else None, contrasts=contrasts)


def emit(prereg, valid, reasons, meas, diag, rdir, commit=None, contrasts=None):
    obs = {"note": f"Mechanical projection of the five stage receipts under {rdir} (commit {commit}); adapter scratch/qwen_gdn_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"LOOP": {"admissible": True, "reason": "the released path's recurrence (gated_delta_ops), selected by name; dispatch proven by the executed counts"},
                    "KERNEL": {"admissible": valid, "reason": "the Metal recurrence kernel selected by name; dispatch proven by the executed counts" if valid else "; ".join(reasons)}},
           "measurements": meas, "diagnostics": diag}
    if contrasts:
        obs["contrasts"] = contrasts
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(*sys.argv[1:]))

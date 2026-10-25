"""Observations adapter for PRE-REG QWEN-GDN-PROMOTE-0: derives the registered
measurements from the five stage receipts of scratch/qwen_gdn_promote.py in
committed code (never hand-written).

Measurements (bar: metric)
  1   default_path_ops_executions   gated_delta_ops executions under any DEFAULT-arm counted block (heldout, cache, battery, the cli classification);
                                     a non-exclusive block is COUNTED here (bar 1 reads NO-FIRE), never a validity failure
  1:loop loop_arm_kernel_executions  gated_delta_kernel executions under any LOOP-arm block
  1:sel  default_selection_ok        1 when every engine block records recurrence kernel on every DeltaNet layer (48), the training flag True,
                                     every receipt records DEFAULT_RECURRENCE "kernel" in code, and every CLI receipt names kernel/loop on 48 with the
                                     Metal GPU as the default device; else 0
  2   tests_failed                   failed or errored tests plus non-zero suite return codes without a parsed failure
  3   heldout_parity_mismatches      over the 18 DEFAULT runs, two comparisons each (36): final argmax or 16 decode ids differing from the in-process
                                     LOOP arm, or from the frozen LOOP rows of VERDICT QWEN-GDN-KERNEL-0's held-out receipt (the in-process loop
                                     equal to the frozen rows is a VALIDITY floor: loop drift is not the kernel's fault)
  4   cache_inc_vs_replay_unequal_turns   DEFAULT incremental against DEFAULT replay, 9 turns
  4:loop cache_unexplained_divergences    first DEFAULT-against-LOOP divergence per conversation not classified as a near tie
  5   battery_pass                   passes of the v3 collection on the DEFAULT arm (60 scored)
  5:div  battery_unexplained_divergences  tasks whose ids differ from the frozen loop receipt at a position that is not a near tie
  5:err  battery_errors               plumbing errors
  6   cli_unexplained_divergences    fresh CLI tasks whose DEFAULT and --recurrence loop streams differ at a position that is not a near tie
  6:rec  cli_receipt_recurrence_ok     1 when every CLI receipt names its selection (kernel on 48 layers by default; loop on 48 with the flag)
  6:err  cli_errors                   CLI process failures plus error verdicts in either arm
  7   ttft_ratio_1024 (7:2048, 7:512) median TTFT ratios, DEFAULT over LOOP, 6 against 6 per length
  8   peak_ratio_max                 max over lengths of the peak-after-decode memory ratio; 8:decode decode16_s_ratio_max
Near tie: at the first divergent position the LOOP arm's top-1 margin is under 0.02 logits, the kernel's choice is the loop's second token, and each
arm reproduces its own choice when re-read. Validity: five non-smoke clean-tree receipts from ONE commit; one vendor provenance; the model pin
(HF revision, chain sha256 recomputed from logs/qwenwhole/artifact_digest_C.txt, config sha256) identical across the engine stages; the frozen
references and prompts at their pinned sha256s; every dispatch block exclusive; 6 runs per arm per length with both arms' ABBA spreads within
3 percent on TTFT and 16-step decode (else the contrasts behind bars 7 and 8 are inadmissible, UNRESOLVED, without invalidating the rest);
the two live suites ran (LLMOPT_QWEN_LIVE=1 with the artifact and tokenizer present recorded as preconditions, return code 0, at least 25 passed in
tests/test_qserve.py with at most its one known stub-trajectory skip, at least 6 passed and none skipped in tests/test_qrelease.py; AMENDMENT
QWEN-GDN-PROMOTE-0-ADAPTER: the first form of this floor read "none skipped" and would have invalidated the run on a stub-model skip that is not a live test);
9 cache turns with every eligible DEFAULT turn reused (else bar 4's contrast is inadmissible); 60 battery tasks compared; 16 CLI tasks in both arms.
Usage: qwen_gdn_promote_obs.py <receipt dir> <prereg.json> > observations.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

STAGES = ("tests", "heldout", "cache", "battery", "cli")
HF_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CHAIN = "logs/qwenwhole/artifact_digest_C.txt"
# the live suites' expected shape at the registration commit: (minimum passed, maximum skipped); the one admissible test_qserve skip is the
# stub model's "no stop trajectory" parametrization (tests/test_qserve.py, a pytest.skip inside a stub-only test), never a live test
LIVE_SUITE_FLOORS = {"tests/test_qserve.py": (25, 1), "tests/test_qrelease.py": (6, 0)}
PINS = {"logs/qwen_gdn/heldout.json": "288d8fd85c5c855b12eebe8a7c0623ac0764165ad2775cde1d655f1603971382",
        "logs/qwen_battery/results_v3_C.json": "c47c2d42cd58443f7a510615c6eed101fe7b7557483c8663f4000f8077b1a5d0"}
SPREAD_PCT = 3.0
LENS = ("512", "1024", "2048")
N_DNET = 48


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def engine_ok(e):
    return bool(e) and e.get("recurrence") == "kernel" and e.get("recurrence_layers") == N_DNET and e.get("model_training_flag") is True \
        and e.get("recurrence_of") == {"loop": 0, "kernel": N_DNET, "unselected": 0}


def main(rdir, prereg_path):
    prereg = load(prereg_path)
    R, reasons, contrasts = {}, [], {}
    for st in STAGES:
        p = os.path.join(rdir, f"{st}.json")
        if not os.path.exists(p):
            reasons.append(f"missing receipt {p}")
            continue
        R[st] = json.load(open(p))
        if R[st].get("smoke"):
            reasons.append(f"{st}: smoke receipt")
        if R[st].get("tree_dirty"):
            reasons.append(f"{st}: dirty tree")
        if R[st].get("default_recurrence_in_code") != "kernel":
            reasons.append(f"{st}: DEFAULT_RECURRENCE in code is {R[st].get('default_recurrence_in_code')!r}")
    if reasons:
        return emit(prereg, False, reasons, {}, {}, rdir)
    commits = {R[st]["commit"] for st in STAGES}
    if len(commits) != 1:
        reasons.append(f"receipts from several commits {sorted(commits)}")
    vendors = {json.dumps(R[st]["vendor"], sort_keys=True) for st in STAGES}
    if len(vendors) != 1:
        reasons.append("vendor provenance differs between receipts")
    v = R["heldout"]["vendor"]
    if not v.get("metal_available") or "gpu" not in v.get("default_device", ""):
        reasons.append("the default device is not the Metal GPU")
    def stable_pin(pin):
        """The pin without the qualify report's preflight block (free memory at load time, a per-process reading, not an identity)."""
        pin = json.loads(json.dumps(pin))
        pin.get("artifact", {}).get("chain_report", {}).pop("preflight", None)
        return json.dumps(pin, sort_keys=True)
    pins = {stable_pin(R[st]["model_pin"]) for st in ("heldout", "cache", "battery") if "model_pin" in R[st]}
    if len(pins) != 1:
        reasons.append(f"model pin differs between the engine stages ({len(pins)} distinct)")
    pin = R["heldout"].get("model_pin", {})
    sw = pin.get("source_weights", {})
    if sw.get("hf_revision") != HF_REVISION or sw.get("revision_source") != "logs/qwenteacher_v2/teacher_manifest.json":
        reasons.append(f"the source-weights revision is not the pinned one ({sw.get('hf_revision')} from {sw.get('revision_source')})")
    if os.path.exists(CHAIN) and pin.get("artifact", {}).get("chain_sha256") != fsha(CHAIN):
        reasons.append("the artifact chain sha256 in the receipt is not the chain file's")
    cfg = pin.get("config", {})
    if cfg.get("model_type") != "qwen3_5" or cfg.get("num_hidden_layers") != 64 or cfg.get("layer_types", {}).get("linear_attention") != N_DNET or cfg.get("vocab_size") != 248320:
        reasons.append(f"the configuration summary is not the pinned model's: {cfg}")
    for path, sha in PINS.items():
        if not os.path.exists(path) or fsha(path) != sha:
            reasons.append(f"frozen reference {path} missing or changed")
    for st in ("heldout", "cache", "battery"):
        if not engine_ok(R[st].get("engine")) or not engine_ok(R[st].get("engine_after")):
            reasons.append(f"{st}: the engine was not on the kernel default on every DeltaNet layer at entry and exit")
    # dispatch counts, every block (an empty block is a validity failure: nothing was measured; a NON-EXCLUSIVE block is counted and read by bar 1)
    ops_default = kern_loop = 0
    for st in ("heldout", "cache", "battery", "cli"):
        for d in R[st].get("dispatch", []):
            if d.get("update_calls", 0) == 0:
                reasons.append(f"{st}: dispatch block {d.get('label')} measured no recurrence call")
            if d["arm"] == "DEFAULT":
                ops_default += d["executed"]["ops"]
            else:
                kern_loop += d["executed"]["kernel"]
    T = R["tests"]["summary"]
    if not T.get("live") or T.get("suites") != 9:
        reasons.append(f"tests: {T.get('suites')} suites, live {T.get('live')} (9 suites with the two live ones registered)")
    pre = R["tests"].get("live_preconditions", {})
    if not (pre.get("LLMOPT_QWEN_LIVE") == "1" and pre.get("artifact_manifest_present") and pre.get("vendor_tokenizer_present")):
        reasons.append(f"tests: live preconditions not met {pre}")
    for suite, (min_passed, max_skipped) in LIVE_SUITE_FLOORS.items():
        c = T.get("live_suites", {}).get(suite)
        if not c or c.get("rc") != 0 or c.get("passed", 0) < min_passed or c.get("skipped", 0) > max_skipped:
            reasons.append(f"tests: live suite {suite} {c} outside its floor (at least {min_passed} passed, at most {max_skipped} skipped: the live tests did not run)")
    H = R["heldout"]
    if H.get("frozen_reference", {}).get("sha256") != PINS["logs/qwen_gdn/heldout.json"]:
        reasons.append("heldout: the frozen reference is not the pinned receipt")
    hs = H.get("summary", {})
    for t in LENS:
        s = hs.get(t)
        if not s or s["n"] != 6:
            reasons.append(f"heldout {t}: {s and s['n']} DEFAULT runs, not 6")
            continue
        if s["loop_self_mismatches"]:
            reasons.append(f"heldout {t}: the loop arm is not self-consistent")
        if s["loop_runs_not_equal_frozen_loop"]:
            reasons.append(f"heldout {t}: the in-process loop differs from the frozen loop rows ({s['loop_runs_not_equal_frozen_loop']} runs): loop drift, not a kernel reading")
        if max(s["loop_spread_pct"], s["default_spread_pct"]) > SPREAD_PCT:
            contrasts["7"] = {"admissible": False, "reason": f"TTFT spread at {t} above {SPREAD_PCT} percent: loop {s['loop_spread_pct']:.2f}, default {s['default_spread_pct']:.2f}"}
        if max(s["loop_decode16_spread_pct"], s["default_decode16_spread_pct"]) > SPREAD_PCT:
            contrasts["8"] = {"admissible": False, "reason": f"16-step decode spread at {t} above {SPREAD_PCT} percent: loop {s['loop_decode16_spread_pct']:.2f}, default {s['default_decode16_spread_pct']:.2f}"}
    C = R["cache"]["summary"]
    if C.get("n_turns") != 9:
        reasons.append(f"cache: {C.get('n_turns')} turns, not 9")
    if C.get("offset_mismatches"):
        reasons.append("cache: a KV offset disagreed with the accounting")
    if C.get("incremental_reused_turns", {}).get("DEFAULT") != C.get("eligible_turns") or C.get("fallback_turns", {}).get("DEFAULT"):
        contrasts["4"] = {"admissible": False, "reason": f"the DEFAULT incremental session did not reuse the cache on every eligible turn: {C.get('incremental_reused_turns')}, fallbacks {C.get('fallback_turns')}"}
    B = R["battery"]["summary"]
    if B.get("tasks") != 60 or B.get("tasks_compared") != 60 or R["battery"].get("frozen_reference", {}).get("sha256") != PINS["logs/qwen_battery/results_v3_C.json"]:
        reasons.append(f"battery: {B.get('tasks')} tasks, {B.get('tasks_compared')} compared, frozen reference {R['battery'].get('frozen_reference', {}).get('sha256', '')[:12]}")
    K = R["cli"]["summary"]
    if K.get("tasks") != 16 or len(R["cli"].get("processes", [])) != 32:
        reasons.append(f"cli: {K.get('tasks')} tasks, {len(R['cli'].get('processes', []))} processes (16 and 32 registered)")
    sel_ok = all(engine_ok(R[st].get("engine")) and engine_ok(R[st].get("engine_after")) for st in ("heldout", "cache", "battery")) and bool(K.get("receipt_recurrence_ok"))
    valid = not reasons
    meas, diag = {}, {}
    if valid:
        meas["1"] = {"value": ops_default, "metric": "default_path_ops_executions", "population": "default-arm-blocks:all", "aggregation": "count", "unit": "calls",
                     "provenance": json.dumps({st: [{"label": d["label"], "arm": d["arm"], "executed": d["executed"], "update_calls": d["update_calls"]} for d in R[st].get("dispatch", [])][:12] for st in ("heldout", "cache", "battery", "cli")})}
        meas["1:loop"] = {"value": kern_loop, "metric": "loop_arm_kernel_executions", "population": "loop-arm-blocks:all", "aggregation": "count", "unit": "calls"}
        meas["2"] = None
        meas["1:sel"] = {"value": 1 if sel_ok else 0, "metric": "default_selection_ok", "population": "engine-blocks:6+cli-receipts:32", "aggregation": "indicator", "unit": "indicator",
                         "provenance": json.dumps({"engines": {st: R[st]["engine"] for st in ("heldout", "cache", "battery")}, "cli_receipt_recurrence_ok": K.get("receipt_recurrence_ok"), "non_exclusive_blocks": [f"{st}:{d['label']}" for st in ("heldout", "cache", "battery", "cli") for d in R[st].get("dispatch", []) if not d.get("exclusive")]})}
        failed = T["failed"] + sum(1 for s in R["tests"]["suites"] if s["returncode"] != 0 and not (s["counts"].get("failed") or s["counts"].get("error") or s["counts"].get("errors")))
        meas["2"] = {"value": failed, "metric": "tests_failed", "population": "suites:9", "aggregation": "count", "unit": "tests",
                     "provenance": json.dumps([{"suite": s["suite"], "rc": s["returncode"], "summary": s["summary"]} for s in R["tests"]["suites"]])}
        par = sum(hs[t]["parity_mismatches_vs_loop"] + hs[t]["default_runs_not_equal_frozen_loop"] for t in LENS)
        meas["3"] = {"value": par, "metric": "heldout_parity_mismatches", "population": "heldout-default-run-comparisons:36", "aggregation": "count", "unit": "comparisons",
                     "provenance": json.dumps({t: {"vs_loop": hs[t]["parity_mismatches_vs_loop"], "vs_frozen_loop": hs[t]["default_runs_not_equal_frozen_loop"], "loop_vs_frozen_loop": hs[t]["loop_runs_not_equal_frozen_loop"], "top16_max_abs_diff": hs[t]["top16_max_abs_diff_vs_loop"]} for t in LENS})}
        meas["4"] = {"value": C["default_inc_vs_replay_unequal"], "metric": "cache_inc_vs_replay_unequal_turns", "population": "chat-turns:9", "aggregation": "count", "unit": "turns",
                     "provenance": json.dumps({k: C[k] for k in ("default_vs_loop_inc_unequal", "default_vs_loop_replay_unequal", "loop_inc_vs_replay_unequal", "incremental_reused_turns", "fallback_turns", "generated_tokens")})}
        meas["4:loop"] = {"value": C["unexplained_divergences"], "metric": "cache_unexplained_divergences", "population": "conversations:3", "aggregation": "count", "unit": "divergences",
                          "provenance": json.dumps(R["cache"]["divergences"])}
        meas["5"] = {"value": B["pass"], "metric": "battery_pass", "population": "v3-tasks:60", "aggregation": "count", "unit": "tasks",
                     "provenance": json.dumps({k: B[k] for k in ("scored", "by_kind", "frozen_pass", "tasks_all_equal_to_frozen", "divergent_tasks", "explained_near_tie_divergences", "verdict_changes_vs_frozen", "unterminated_thinking_turns")})}
        meas["5:div"] = {"value": B["unexplained_divergences"], "metric": "battery_unexplained_divergences", "population": "v3-tasks:60", "aggregation": "count", "unit": "divergences",
                         "provenance": json.dumps([{k: d.get(k) for k in ("label", "position", "explained", "near_tie", "flip_between_the_two", "arm_gap_difference", "reason", "LOOP", "DEFAULT")} for d in R["battery"]["divergences"]])}
        meas["5:err"] = {"value": len(B["errors"]), "metric": "battery_errors", "population": "v3-tasks:60", "aggregation": "count", "unit": "tasks"}
        meas["6"] = {"value": K["unexplained_divergences"], "metric": "cli_unexplained_divergences", "population": "cli-tasks:16", "aggregation": "count", "unit": "divergences",
                     "provenance": json.dumps({"pass": K["pass"], "scored": K["scored"], "tasks_all_equal": K["tasks_all_equal"], "explained": K["explained_near_tie_divergences"], "divergences": [{k: d.get(k) for k in ("label", "position", "explained", "near_tie", "reason")} for d in R["cli"]["divergences"]]})}
        meas["6:rec"] = {"value": 1 if K["receipt_recurrence_ok"] else 0, "metric": "cli_receipt_recurrence_ok", "population": "cli-receipts:32", "aggregation": "indicator", "unit": "indicator"}
        meas["6:err"] = {"value": K["process_failures"] + K["errors"]["DEFAULT"] + K["errors"]["LOOP"], "metric": "cli_errors", "population": "cli-processes:32", "aggregation": "count", "unit": "processes"}
        meas["7"] = {"value": hs["1024"]["ttft_ratio"], "metric": "ttft_ratio_1024", "population": "heldout-1024:6v6", "aggregation": "median_ratio", "unit": "ratio",
                     "provenance": json.dumps({t: {"ttft_median_s": hs[t]["ttft_median_s"], "loop_spread_pct": round(hs[t]["loop_spread_pct"], 2), "default_spread_pct": round(hs[t]["default_spread_pct"], 2)} for t in LENS})}
        meas["7:2048"] = {"value": hs["2048"]["ttft_ratio"], "metric": "ttft_ratio_2048", "population": "heldout-2048:6v6", "aggregation": "median_ratio", "unit": "ratio"}
        meas["7:512"] = {"value": hs["512"]["ttft_ratio"], "metric": "ttft_ratio_512", "population": "heldout-512:6v6", "aggregation": "median_ratio", "unit": "ratio"}
        meas["8"] = {"value": max(hs[t]["peak_after_decode_ratio"] for t in LENS), "metric": "peak_ratio_max", "population": "heldout-lengths:3", "aggregation": "max_ratio", "unit": "ratio",
                     "provenance": json.dumps({t: {"peak": round(hs[t]["peak_ratio"], 4), "peak_after_decode": round(hs[t]["peak_after_decode_ratio"], 4), "footprint": round(hs[t]["footprint_ratio"], 4), "transient": round(hs[t]["transient_ratio"], 4)} for t in LENS})}
        meas["8:decode"] = {"value": max(hs[t]["decode16_s_ratio"] for t in LENS), "metric": "decode16_s_ratio_max", "population": "heldout-lengths:3", "aggregation": "max_median_ratio", "unit": "ratio",
                            "provenance": json.dumps({t: {"decode16": round(hs[t]["decode16_s_ratio"], 4), "step": round(hs[t]["decode_step_ratio"], 4), "e2e16": round(hs[t]["e2e16_ratio"], 4)} for t in LENS})}
        meas["heldout_parity_mismatches"] = dict(meas["3"])
        diag = {"heldout": hs, "cache": C, "battery": {k: B[k] for k in B if k not in ("by_kind",)}, "cli": K, "tests": T, "model_pin": pin,
                "walls_s": {st: R[st].get("wall_s") for st in STAGES}}
    return emit(prereg, valid, reasons, meas, diag, rdir, commit=next(iter(commits)) if len(commits) == 1 else None, contrasts=contrasts)


def emit(prereg, valid, reasons, meas, diag, rdir, commit=None, contrasts=None):
    obs = {"note": f"Mechanical projection of the five stage receipts under {rdir} (commit {commit}); adapter scratch/qwen_gdn_promote_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"LOOP": {"admissible": True, "reason": "the Python reference loop re-selected by name on the loaded default engine; dispatch proven by the executed counts"},
                    "DEFAULT": {"admissible": valid, "reason": "the promoted default (kernel selected by name per layer at load); dispatch proven by the executed counts" if valid else "; ".join(reasons)}},
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

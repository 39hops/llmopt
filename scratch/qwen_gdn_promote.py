"""QWEN-GDN-PROMOTE-0: the acceptance run of the promoted serving default
for the arm C GatedDeltaNet recurrence (llmopt.lab.qserve.DEFAULT_RECURRENCE
"kernel", selected by name per layer through llmopt.lab.qtower.
select_recurrence; the Python loop stays selectable as "loop"). VERDICT
QWEN-GDN-KERNEL-0 qualified the kernel against the loop under the
qualification instrument (llmopt.lab.qgdn.recurrence_path forcing a path
process-wide); this run measures the PROMOTED CODE (the per-instance
selection the Engine and the CLI actually use) and leaves the default in
place only if every registered gate fires (any other outcome reverts the
default to "loop" in the HOLD commit).

Arms. DEFAULT: an Engine loaded with no recurrence argument (what every
user of qserve and scripts/qwen_chat.py gets). LOOP: the same loaded
engine re-selected to "loop" (Engine.select_recurrence), paired in one
process; the default is restored and asserted after every loop block. The
qualification instrument is used in its counting mode only
(recurrence_path("vendor"), which forces nothing) to PROVE which
implementation executed under each selection.

Model pin (every receipt): the vendor configuration files' sha256s and
the configuration summary (architectures, model_type qwen3_5 / text
qwen3_5_text, 64 layers of which 48 linear_attention and 16 full_attention,
hidden 5,120, DeltaNet 16 key and 48 value heads of 128, vocabulary
248,320), the pinned source weights Qwen/Qwen3.8-27B at revision 1d4bf0f2
(the teacher manifest and the compile log carry the same revision), the
arm C artifact under its digest chain (logs/qwenwhole/artifact_digest_C.txt)
and the installed mlx-lm qwen3_5 ARCHITECTURE module's sha256 (mlx-lm names
the shared Qwen3.5-family architecture qwen3_5; the weights are Qwen3.8-27B).

Stages (STAGE env; each writes logs/qwen_gdn_promote/<stage>.json and
refuses to overwrite; SMOKE=1 shortens every stage into
logs/qwen_gdn_promote/smoke/):
  tests     the release suites in fresh subprocesses (pass, fail, skip counts
            from pytest's own summary and the return codes): selector and
            isolation tests, the adoption guard, the stub defaults, the
            pinned template, prefill, battery validators, and the live serve
            and release tests on the tower (LLMOPT_QWEN_LIVE=1)
  heldout   the three frozen held-out prompts (512, 1,024, 2,048 by target):
            DEFAULT against LOOP, one warm-up per arm per length, then ABBA
            three times with 16 greedy decode steps (TTFT, peak and transient
            memory, decode time, parity of the final argmax and the 16 ids
            against the in-process LOOP arm and against the frozen LOOP rows
            of VERDICT QWEN-GDN-KERNEL-0's held-out receipt)
  cache     three fresh three-turn conversations under the chat default
            (thinking low, 768 tokens, incremental) in both arms, incremental
            and replay: per turn the ids, mode and note; DEFAULT incremental
            against DEFAULT replay (the continuation of the recurrent state
            across a closed turn) and against LOOP incremental; a divergence
            is classified by re-reading both arms' logits at the first
            divergent position (a near tie is a loop margin under 0.02
            logits with the kernel's choice the loop's second token)
  battery   the frozen v3 collection (60 tasks) under the released
            configuration (thinking low, 768) on the DEFAULT arm, scored by
            the committed validators; every turn's ids compared with the
            frozen receipt logs/qwen_battery/results_v3_C.json (the loop path,
            VERDICT QWEN-LOW-RELEASE-0), divergences classified as above
  cli       the fresh collection scratch/evalsets/qwen_battery_v4_cli.json
            through scripts/qwen_chat.py in fresh subprocesses, default flags
            (DEFAULT) and --recurrence loop (LOOP), one process per task and
            arm; the CLI's own receipts (recurrence, layers, vendor) collected,
            ids compared, both arms scored; divergences classified afterwards
            by an in-process engine loaded only once every subprocess has
            exited (one resident tower at a time)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE = os.environ.get("STAGE", "tests")
SMOKE = os.environ.get("SMOKE") == "1"
OUTDIR = os.environ.get("OUTDIR", f"logs/qwen_gdn_promote/{'smoke/' if SMOKE else ''}")
OUT = os.path.join(OUTDIR, f"{STAGE}.json")
if SMOKE != ("smoke" in OUTDIR):
    raise SystemExit(f"REFUSING: SMOKE={int(SMOKE)} with OUTDIR {OUTDIR!r}; smoke receipts live under a path containing 'smoke' and real ones never do")
ARMS = {"DEFAULT": "kernel", "LOOP": "loop"}
HF_REPO = "Qwen/Qwen3.8-27B"
TEACHER_MANIFEST = "logs/qwenteacher_v2/teacher_manifest.json"   # the source-weights revision is READ from here (the teacher capture's own record), never a literal
FROZEN_HELDOUT = "logs/qwen_gdn/heldout.json"                 # VERDICT QWEN-GDN-KERNEL-0, stage 4 (its LOOP rows are the cross-session parity reference)
FROZEN_HELDOUT_SHA = "288d8fd85c5c855b12eebe8a7c0623ac0764165ad2775cde1d655f1603971382"
FROZEN_V3 = "logs/qwen_battery/results_v3_C.json"             # VERDICT QWEN-LOW-RELEASE-0 arm C: v3, thinking low, 768, the loop path
FROZEN_V3_SHA = "c47c2d42cd58443f7a510615c6eed101fe7b7557483c8663f4000f8077b1a5d0"
HELDOUT_PROMPTS = "scratch/evalsets/qwen_prefill_heldout_prompts.json"
V3 = "scratch/evalsets/qwen_battery_v3.json"
V4 = "scratch/evalsets/qwen_battery_v4_cli.json"
NEAR_TIE_MARGIN = 0.02
ARM_DIFF_BOUND = 5e-3      # |(kernel_tok - loop_tok logit gap)_DEFAULT - (same)_LOOP| at the divergent position; the qualification's worst full-vocabulary arm difference was 9.3e-4
TARGETS = [512, 1024, 2048] if not SMOKE else [256]
DECODE_STEPS = 16
REPS = 3 if not SMOKE else 1
CHAT_BUDGET = 768 if not SMOKE else 48
CONVERSATIONS = [
    ["Which element has the chemical symbol Fe?", "What is its atomic number?", "Name one everyday object made mostly of it."],
    ["Write two lines about a lighthouse.", "Now two lines about a windmill.", "Which of the two did you find easier to write, and why?"],
    ["What is 144 divided by 12?", "Multiply that result by 7.", "Is the product a prime number?"],
]
if SMOKE:
    CONVERSATIONS = [CONVERSATIONS[0][:2]]
SUITES_LOCAL = ["tests/test_qgdn.py", "tests/test_qtower_recurrence.py", "tests/test_qtower_adoption.py", "tests/test_qdefaults.py",
                "tests/test_qtemplate.py", "tests/test_qprefill.py", "tests/test_qbattery.py"]
SUITES_LIVE = ["tests/test_qserve.py", "tests/test_qrelease.py"]
THINK_CLOSE_ID = 248069


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_ids(ids):
    return hashlib.sha256(json.dumps(list(map(int, ids))).encode()).hexdigest()


def mem():
    from llmopt.lab import qprefill
    return qprefill.memory_report()


def model_pin(vendor_dir, art_dir, info):
    """The exact model artifact and configuration this run served."""
    from mlx_lm.models import gated_delta, qwen3_5
    cfg = json.load(open(os.path.join(vendor_dir, "config.json")))
    tc = cfg["text_config"]
    idx = json.load(open(os.path.join(vendor_dir, "model.safetensors.index.json")))
    if not os.path.exists(TEACHER_MANIFEST):
        raise SystemExit(f"REFUSING: {TEACHER_MANIFEST} absent; the source-weights revision is read from it")
    tm = json.load(open(TEACHER_MANIFEST))
    revision = tm.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise SystemExit(f"REFUSING: the teacher manifest carries no 40-hex revision ({revision!r})")
    chain = info["chain"]
    pin = {"source_weights": {"hf_repo": HF_REPO, "hf_revision": revision, "revision_source": TEACHER_MANIFEST, "teacher_manifest_sha256": fsha(TEACHER_MANIFEST),
                              "vendor_index_total_size_bytes": idx.get("metadata", {}).get("total_size"), "vendor_index_tensors": len(idx.get("weight_map", {}))},
           "vendor_dir_files_sha256": {fn: fsha(os.path.join(vendor_dir, fn)) for fn in ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
                                       if os.path.exists(os.path.join(vendor_dir, fn))},
           "config": {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"), "text_model_type": tc.get("model_type"), "hidden_size": tc.get("hidden_size"),
                      "num_hidden_layers": tc.get("num_hidden_layers"), "layer_types": {k: tc["layer_types"].count(k) for k in sorted(set(tc["layer_types"]))},
                      "linear_num_key_heads": tc.get("linear_num_key_heads"), "linear_num_value_heads": tc.get("linear_num_value_heads"),
                      "linear_key_head_dim": tc.get("linear_key_head_dim"), "linear_value_head_dim": tc.get("linear_value_head_dim"), "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim"),
                      "num_attention_heads": tc.get("num_attention_heads"), "num_key_value_heads": tc.get("num_key_value_heads"), "head_dim": tc.get("head_dim"),
                      "vocab_size": tc.get("vocab_size"), "mamba_ssm_dtype": tc.get("mamba_ssm_dtype"), "dtype": tc.get("dtype"), "transformers_version": cfg.get("transformers_version")},
           "artifact": {"art_dir_basename": os.path.basename(os.path.expanduser(art_dir).rstrip("/")), "chain": chain, "chain_sha256": info["chain_sha256"],
                        "manifest_sha256": open(chain).readline().split()[0], "chain_report": info.get("chain_report"), "families": info.get("families"),
                        "fused": info.get("fused"), "raw": info.get("raw"), "payload_resident_bytes": info.get("payload_resident_bytes")},
           "architecture_module": {"name": "mlx_lm.models.qwen3_5 (the shared Qwen3.5-family architecture implementation; the weights are Qwen3.8-27B)",
                                   "qwen3_5_sha256": fsha(qwen3_5.__file__), "gated_delta_sha256": fsha(gated_delta.__file__)}}
    return pin


def engine_block(eng):
    from llmopt.lab import qtower
    return {"recurrence": eng.info.get("recurrence"), "recurrence_layers": eng.info.get("recurrence_layers"), "recurrence_of": qtower.recurrence_of(eng.model),
            "model_training_flag": bool(eng.model.training), "kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"),
            "cache_limit_bytes": eng.cfg.cache_limit_bytes, "chunk": eng.cfg.chunk, "matmul": eng.cfg.matmul, "context_cap": eng.context_cap,
            "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {}), "stop_ids": sorted(eng.stop_ids), "cold_load_s": eng.load_s, "v_live": len(eng.tokenizer)}


def load_engine():
    from llmopt.lab import qserve
    eng = qserve.Engine()                       # the promoted default: no recurrence argument
    if eng.info.get("recurrence") != "kernel" or eng.info.get("recurrence_layers") != eng.info["families"]["linear_attn"]:
        raise SystemExit(f"REFUSING: the default engine did not select the kernel on every DeltaNet layer: {eng.info.get('recurrence')} on {eng.info.get('recurrence_layers')}")
    if eng.model.training is not True:
        raise SystemExit("REFUSING: the tower is expected in its default (training) mode; the selection never flips it")
    return eng


class arm:
    """``with arm(eng, "LOOP", receipt, label) as sel:`` selects the arm's path
    on the loaded engine, counts executions through the instrument's
    counting mode, records the dispatch under `label`, asserts exclusivity,
    and restores the promoted default afterwards."""

    def __init__(self, eng, name, receipt, label):
        from llmopt.lab import qgdn
        self.eng, self.name, self.receipt, self.label = eng, name, receipt, label
        self.sel = qgdn.recurrence_path("vendor")

    def __enter__(self):
        from llmopt.lab import qtower
        self.eng.select_recurrence(ARMS[self.name])
        self.before = qtower.recurrence_of(self.eng.model)
        self.sel.__enter__()
        return self.sel

    def __exit__(self, *exc):
        from llmopt.lab import qtower
        self.sel.__exit__(*exc)
        after = qtower.recurrence_of(self.eng.model)
        want = "kernel" if ARMS[self.name] == "kernel" else "ops"
        other = "ops" if want == "kernel" else "kernel"
        ex = self.sel.count.executed
        ok = ex[other] == 0 and ex[want] > 0
        self.receipt.setdefault("dispatch", []).append({"label": self.label, "arm": self.name, "path": ARMS[self.name], "selection_before": self.before, "selection_after": after,
                                                        "exclusive": ok, **self.sel.receipt()})
        self.eng.select_recurrence("kernel")        # the promoted default is restored after every arm block
        if not ok:
            print(f"[promote] DISPATCH NOT EXCLUSIVE for {self.name} ({self.label}): executed {ex} (recorded; bar 1 reads it)", flush=True)
        return False


# ---------------------------------------------------------------- tests

def stage_tests(eng, receipt):
    """eng is None here: the suites load their own engines."""
    suites = SUITES_LOCAL + ([] if SMOKE else SUITES_LIVE)
    receipt["suites"] = []
    live_env = "0" if SMOKE else "1"
    env = dict(os.environ, LLMOPT_QWEN_LIVE=live_env)
    receipt["live_preconditions"] = {"LLMOPT_QWEN_LIVE": live_env, "artifact_manifest_present": os.path.exists(os.path.expanduser("~/qwen_whole0t/C/manifest.json")),
                                     "vendor_tokenizer_present": os.path.exists(os.path.expanduser("~/qwen_vendor/tokenizer_config.json"))}
    for s in suites:
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "pytest", s, "-q", "-p", "no:cacheprovider"], capture_output=True, text=True, env=env, timeout=3600)
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors|xfailed|deselected)", tail)}
        receipt["suites"].append({"suite": s, "returncode": p.returncode, "summary": tail, "counts": counts, "wall_s": time.perf_counter() - t0,
                                  "stderr_tail": p.stderr[-800:] if p.returncode else ""})
        print(f"[promote] tests {s}: rc {p.returncode} {tail}", flush=True)
    receipt["summary"] = {"suites": len(receipt["suites"]), "nonzero_returncodes": sum(1 for s in receipt["suites"] if s["returncode"] != 0),
                          "failed": sum(s["counts"].get("failed", 0) + s["counts"].get("error", 0) + s["counts"].get("errors", 0) for s in receipt["suites"]),
                          "passed": sum(s["counts"].get("passed", 0) for s in receipt["suites"]), "skipped": sum(s["counts"].get("skipped", 0) for s in receipt["suites"]),
                          "live_suites": {s["suite"]: {"passed": s["counts"].get("passed", 0), "skipped": s["counts"].get("skipped", 0), "rc": s["returncode"]} for s in receipt["suites"] if s["suite"] in SUITES_LIVE},
                          "live": live_env == "1" and all(receipt["live_preconditions"].values())}


# ---------------------------------------------------------------- heldout

def topk(row, k=16):
    import mlx.core as mx
    row = row.astype(mx.float32)
    top = mx.argpartition(-row, k)[:k]
    vals = row[top]
    order = mx.argsort(-vals)
    return [int(i) for i in top[order].tolist()], [float(v) for v in vals[order].tolist()]


def run_prefill(eng, ids, name, receipt, label, decode=0):
    import mlx.core as mx
    from llmopt.lab import qprefill
    mx.clear_cache()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    with arm(eng, name, receipt, label):
        cache = eng.model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(eng.model, ids, cache, chunk=eng.cfg.chunk, head="final", matmul=eng.cfg.matmul)
        mx.eval(lg)
        wall = time.perf_counter() - t0
        ids_top, vals_top = topk(lg[0, -1])
        m = mem()
        m["active_before_bytes"], m["transient_bytes"] = active_before, m["peak_bytes"] - active_before
        row = {"tokens": len(ids), "arm": name, "path": ARMS[name], "ttft_s": wall, "prefill_tok_per_s": len(ids) / wall, "argmax": ids_top[0], "top16_ids": ids_top, "top16_logits": vals_top, "memory": m}
        if decode:
            cur, out, walls = ids_top[0], [], []
            t1 = time.perf_counter()
            for _ in range(decode):
                out.append(cur)
                ts = time.perf_counter()
                lz = eng.model(mx.array([[cur]]), cache=cache)[0, -1]
                mx.eval(lz)
                walls.append(time.perf_counter() - ts)
                cur = int(lz.argmax())
            row["decode_ids"], row["decode_s"] = out, time.perf_counter() - t1
            row["end_to_end_s"] = wall + row["decode_s"]
            row["decode_step_median_s"] = statistics.median(walls[4:]) if len(walls) > 8 else statistics.median(walls)
            row["peak_after_decode_bytes"] = int(mx.get_peak_memory())
        del cache
    mx.clear_cache()
    return row


def top16_diff(x, ref):
    a, b = dict(zip(x["top16_ids"], x["top16_logits"])), dict(zip(ref["top16_ids"], ref["top16_logits"]))
    common = set(a) & set(b)
    return max(abs(a[i] - b[i]) for i in common) if common else 99.0


def fixed_prompt(eng, seed, target):
    import random
    r = random.Random(seed)
    words = ["river", "stone", "lantern", "quiet", "harbour", "copper", "meadow", "signal", "winter", "orbit", "ledger", "canvas", "timber", "glacier", "velvet", "circuit"]
    text = "Summarize the following notes in one paragraph.\n\n" + " ".join(r.choice(words) for _ in range(target * 2))
    ids = eng.render_chat([{"role": "user", "content": text}], False)
    while len(ids) > target:
        text = text.rsplit(" ", 8)[0]
        ids = eng.render_chat([{"role": "user", "content": text}], False)
    return ids


def stage_heldout(eng, receipt):
    if SMOKE:
        prompts = {256: fixed_prompt(eng, "gdn-promote-smoke-256", 256)}
        frozen_rows = {}
    else:
        frozen = json.load(open(HELDOUT_PROMPTS))["prompts"]
        prompts = {t: frozen[str(t)]["ids"] for t in TARGETS}
        for t in TARGETS:
            assert sha_ids(prompts[t]) == frozen[str(t)]["sha256"], f"held-out prompt {t} does not match its frozen sha"
        assert fsha(FROZEN_HELDOUT) == FROZEN_HELDOUT_SHA, "the frozen QWEN-GDN-KERNEL-0 held-out receipt changed"
        fz = json.load(open(FROZEN_HELDOUT))
        frozen_rows = {(r["arm"], r["target"]): r for r in fz["runs"] if r["arm"] == "LOOP"}       # the first LOOP row per target (its arm was self-consistent)
        frozen_rows = {k: v for k, v in sorted(frozen_rows.items())}
        receipt["frozen_reference"] = {"path": FROZEN_HELDOUT, "sha256": FROZEN_HELDOUT_SHA, "commit": fz.get("commit"),
                                       "loop_rows": {str(t): {"argmax": frozen_rows[("LOOP", t)]["argmax"], "decode_ids": frozen_rows[("LOOP", t)]["decode_ids"]} for t in TARGETS}}
    receipt["prompts"] = {str(t): {"tokens": len(p), "sha256": sha_ids(p)} for t, p in prompts.items()}
    receipt["runs"], receipt["warmups"] = [], []
    for t in TARGETS:
        for name in ARMS:
            w = run_prefill(eng, prompts[t], name, receipt, f"warmup {t} {name}")
            receipt["warmups"].append({"tokens": t, "arm": name, "ttft_s": w["ttft_s"]})
        for _ in range(REPS):
            for name in ("LOOP", "DEFAULT", "DEFAULT", "LOOP"):
                r = run_prefill(eng, prompts[t], name, receipt, f"heldout {t} {name}", decode=DECODE_STEPS)
                r["target"] = t
                fr = frozen_rows.get(("LOOP", t))
                r["equals_frozen_loop"] = (r["argmax"] == fr["argmax"] and r["decode_ids"] == fr["decode_ids"]) if fr else None
                receipt["runs"].append(r)
                print(f"[promote] heldout {t} {name:7s} ttft {r['ttft_s']:.2f}s e2e {r['end_to_end_s']:.2f}s peak {r['memory']['peak_bytes'] / 1e9:.2f} GB step {r['decode_step_median_s'] * 1e3:.0f} ms frozen-loop {r['equals_frozen_loop']}", flush=True)
    summ = {}
    med = statistics.median
    for t in TARGETS:
        rows = [r for r in receipt["runs"] if r["target"] == t and r["arm"] == "DEFAULT"]
        base = [r for r in receipt["runs"] if r["target"] == t and r["arm"] == "LOOP"]
        ref = base[0]
        spread = lambda xs, k: 100 * (max(x[k] for x in xs) - min(x[k] for x in xs)) / med(x[k] for x in xs)
        summ[str(t)] = {"n": len(rows), "ttft_median_s": {"LOOP": med(x["ttft_s"] for x in base), "DEFAULT": med(x["ttft_s"] for x in rows)},
                        "ttft_ratio": med(x["ttft_s"] for x in rows) / med(x["ttft_s"] for x in base),
                        "loop_spread_pct": spread(base, "ttft_s"), "default_spread_pct": spread(rows, "ttft_s"),
                        "loop_decode16_spread_pct": spread(base, "decode_s"), "default_decode16_spread_pct": spread(rows, "decode_s"),
                        "decode16_s_ratio": med(x["decode_s"] for x in rows) / med(x["decode_s"] for x in base),
                        "decode_step_ratio": med(x["decode_step_median_s"] for x in rows) / med(x["decode_step_median_s"] for x in base),
                        "e2e16_ratio": med(x["end_to_end_s"] for x in rows) / med(x["end_to_end_s"] for x in base),
                        "parity_mismatches_vs_loop": sum(1 for x in rows if x["argmax"] != ref["argmax"] or x["decode_ids"] != ref["decode_ids"]),
                        "loop_self_mismatches": sum(1 for x in base if x["argmax"] != ref["argmax"] or x["decode_ids"] != ref["decode_ids"]),
                        "default_runs_not_equal_frozen_loop": sum(1 for x in rows if x["equals_frozen_loop"] is False),
                        "loop_runs_not_equal_frozen_loop": sum(1 for x in base if x["equals_frozen_loop"] is False),
                        "top16_max_abs_diff_vs_loop": max(top16_diff(x, ref) for x in rows),
                        "peak_ratio": max(x["memory"]["peak_bytes"] for x in rows) / max(x["memory"]["peak_bytes"] for x in base),
                        "peak_after_decode_ratio": max(x["peak_after_decode_bytes"] for x in rows) / max(x["peak_after_decode_bytes"] for x in base),
                        "footprint_ratio": max(x["memory"]["footprint_bytes"] for x in rows) / max(x["memory"]["footprint_bytes"] for x in base),
                        "transient_ratio": max(x["memory"]["transient_bytes"] for x in rows) / max(1, max(x["memory"]["transient_bytes"] for x in base))}
    receipt["summary"] = summ
    receipt["engine_after"] = engine_block(eng)


# ---------------------------------------------------------------- divergence classification

def classify_divergence(eng, receipt, prompt_ids, prefix_ids, kernel_tok, loop_tok, label):
    """Both arms' logits at the divergent position, re-read along the SAME path
    the generation took: the prompt through the chunked prefill, then the
    loop's generated prefix fed one token at a time through the decode step
    (the final logits are the position's). Top-2 ids, the margin, the logits
    of the two chosen tokens. A divergence is EXPLAINED (a near tie) when the
    loop's margin is under 0.02 logits, the kernel's choice is the loop's
    second token, each arm reproduces its own choice, and the two arms' gap
    between the two tokens differs by at most ARM_DIFF_BOUND (a defect that
    moves logits by more than the qualified arm difference is not a tie)."""
    import mlx.core as mx
    from llmopt.lab import qprefill
    out = {"label": label, "prompt_tokens": len(prompt_ids), "prefix_steps": len(prefix_ids), "kernel_tok": kernel_tok, "loop_tok": loop_tok}
    for name in ("LOOP", "DEFAULT"):
        with arm(eng, name, receipt, f"classify {label} {name}"):
            cache = eng.model.make_cache()
            lg = qprefill.prefill_chunked(eng.model, list(prompt_ids), cache, chunk=eng.cfg.chunk, head="final", matmul=eng.cfg.matmul)
            mx.eval(lg)
            row = lg[0, -1]
            for tok in prefix_ids:
                row = eng.model(mx.array([[int(tok)]]), cache=cache)[0, -1]
            row = row.astype(mx.float32)
            mx.eval(row)
            ids_top, vals_top = topk(row, 2)
            out[name] = {"top1": ids_top[0], "top2": ids_top[1], "margin": vals_top[0] - vals_top[1], "logit_kernel_tok": float(row[kernel_tok].item()), "logit_loop_tok": float(row[loop_tok].item())}
            del cache
    lp, kp = out["LOOP"], out["DEFAULT"]
    out["loop_reproduces"] = lp["top1"] == loop_tok
    out["kernel_reproduces"] = kp["top1"] == kernel_tok
    out["near_tie"] = lp["margin"] < NEAR_TIE_MARGIN
    out["flip_between_the_two"] = kernel_tok == lp["top2"] and loop_tok == lp["top1"]
    out["arm_gap_difference"] = abs((kp["logit_kernel_tok"] - kp["logit_loop_tok"]) - (lp["logit_kernel_tok"] - lp["logit_loop_tok"]))
    out["arm_difference_within_bound"] = out["arm_gap_difference"] <= ARM_DIFF_BOUND
    out["explained"] = bool(out["near_tie"] and out["flip_between_the_two"] and out["loop_reproduces"] and out["kernel_reproduces"] and out["arm_difference_within_bound"])
    return out


def first_divergence(a, b):
    n = min(len(a), len(b))
    for j in range(n):
        if a[j] != b[j]:
            return j
    return None if len(a) == len(b) else n


def compare_streams(eng, receipt, label, prompt_ids, kernel_ids, loop_ids):
    """Classify the first divergence of two generated streams from one prompt
    (None when equal; a length-only difference past a common prefix is
    unclassifiable and counts as unexplained)."""
    j = first_divergence(kernel_ids, loop_ids)
    if j is None:
        return None
    if j >= len(kernel_ids) or j >= len(loop_ids):
        return {"label": label, "position": j, "explained": False, "reason": "one stream ended where the other continued"}
    c = classify_divergence(eng, receipt, list(prompt_ids), list(loop_ids[:j]), int(kernel_ids[j]), int(loop_ids[j]), label)
    c["position"] = j
    return c


# ---------------------------------------------------------------- cache

def stage_cache(eng, receipt):
    from llmopt.lab import qserve
    receipt["conversations"], receipt["max_tokens"] = CONVERSATIONS, CHAT_BUDGET
    turns = {}
    for name in ARMS:
        for mode in ("incremental", "replay"):
            with arm(eng, name, receipt, f"cache {name} {mode}"):
                for ci, conv in enumerate(CONVERSATIONS):
                    s = qserve.Session(eng, mode=mode)             # the chat default: thinking low, incremental reuse
                    for ti, user in enumerate(conv):
                        res = s.ask(user, max_tokens=CHAT_BUDGET)
                        turns[(name, mode, ci, ti)] = {"prompt_ids": list(res.prompt_ids), "generated_ids": list(res.generated_ids), "mode": res.mode, "note": res.note, "stop_reason": res.stop_reason,
                                                       "n_prompt": res.n_prompt, "cache_tokens": res.cache_tokens, "cache_offset_measured": res.cache_offset_measured,
                                                       "first_token_s": res.timings["first_token_s"], "decode_s": res.timings["decode_s"], "total_s": res.timings["total_s"], "text": res.text[:160]}
                        print(f"[promote] cache {name} {mode} c{ci} t{ti}: {res.n_generated} tokens, mode {res.mode}, first token {res.timings['first_token_s']:.2f}s", flush=True)
                    del s
    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        for ti in range(len(conv)):
            d_inc, d_rep, l_inc, l_rep = (turns[("DEFAULT", "incremental", ci, ti)], turns[("DEFAULT", "replay", ci, ti)], turns[("LOOP", "incremental", ci, ti)], turns[("LOOP", "replay", ci, ti)])
            rows.append({"conversation": ci, "turn": ti, "default_incremental_equals_default_replay": d_inc["generated_ids"] == d_rep["generated_ids"],
                         "default_replay_equals_loop_replay": d_rep["generated_ids"] == l_rep["generated_ids"],
                         "default_incremental_equals_loop_incremental": d_inc["generated_ids"] == l_inc["generated_ids"],
                         "loop_incremental_equals_loop_replay": l_inc["generated_ids"] == l_rep["generated_ids"],
                         "modes": {"DEFAULT": d_inc["mode"], "LOOP": l_inc["mode"]}, "notes": {"DEFAULT": d_inc["note"], "LOOP": l_inc["note"]},
                         "offsets_ok": all(x["cache_offset_measured"] in (None, x["cache_tokens"]) for x in (d_inc, d_rep, l_inc, l_rep)),
                         "n_generated": {"DEFAULT": len(d_inc["generated_ids"]), "LOOP": len(l_inc["generated_ids"])}, "stop": {"DEFAULT": d_inc["stop_reason"], "LOOP": l_inc["stop_reason"]},
                         "first_token_s": {"DEFAULT_inc": d_inc["first_token_s"], "LOOP_inc": l_inc["first_token_s"], "DEFAULT_replay": d_rep["first_token_s"], "LOOP_replay": l_rep["first_token_s"]},
                         "total_s": {"DEFAULT_inc": d_inc["total_s"], "LOOP_inc": l_inc["total_s"]}, "default_text": d_inc["text"], "loop_text": l_inc["text"]})
    receipt["turns"] = rows
    # classify the first DEFAULT-against-LOOP divergence per conversation on the REPLAY sessions (identical prompts up to that turn)
    receipt["divergences"] = []
    for ci, conv in enumerate(CONVERSATIONS):
        for ti in range(len(conv)):
            d_rep, l_rep = turns[("DEFAULT", "replay", ci, ti)], turns[("LOOP", "replay", ci, ti)]
            if d_rep["generated_ids"] != l_rep["generated_ids"]:
                if d_rep["prompt_ids"] != l_rep["prompt_ids"]:
                    receipt["divergences"].append({"label": f"cache c{ci} t{ti}", "explained": False, "reason": "the prompts differ at the first divergent turn (every earlier turn was equal): a rendering difference, unexplained"})
                else:
                    receipt["divergences"].append(compare_streams(eng, receipt, f"cache c{ci} t{ti}", d_rep["prompt_ids"], d_rep["generated_ids"], l_rep["generated_ids"]))
                break
    receipt["summary"] = {"n_turns": len(rows), "default_inc_vs_replay_unequal": sum(1 for r in rows if not r["default_incremental_equals_default_replay"]),
                          "default_vs_loop_inc_unequal": sum(1 for r in rows if not r["default_incremental_equals_loop_incremental"]),
                          "default_vs_loop_replay_unequal": sum(1 for r in rows if not r["default_replay_equals_loop_replay"]),
                          "loop_inc_vs_replay_unequal": sum(1 for r in rows if not r["loop_incremental_equals_loop_replay"]),
                          "offset_mismatches": sum(1 for r in rows if not r["offsets_ok"]),
                          "incremental_reused_turns": {a: sum(1 for r in rows if r["modes"][a] == "incremental") for a in ARMS},
                          "fallback_turns": {a: sum(1 for r in rows if r["modes"][a] == "replay-fallback") for a in ARMS},
                          "eligible_turns": sum(len(c) - 1 for c in CONVERSATIONS),
                          "divergences_first_per_conversation": len([d for d in receipt["divergences"] if d]),
                          "unexplained_divergences": sum(1 for d in receipt["divergences"] if d and not d.get("explained")),
                          "generated_tokens": {a: sum(len(turns[(a, "incremental", ci, ti)]["generated_ids"]) for ci, c in enumerate(CONVERSATIONS) for ti in range(len(c))) for a in ARMS}}
    receipt["engine_after"] = engine_block(eng)
    print(f"[promote] cache: inc-vs-replay unequal {receipt['summary']['default_inc_vs_replay_unequal']}, default-vs-loop unequal {receipt['summary']['default_vs_loop_inc_unequal']}, unexplained {receipt['summary']['unexplained_divergences']}", flush=True)


# ---------------------------------------------------------------- battery

def run_collection(eng, receipt, label, coll, thinking, effort, budget, system, name="DEFAULT"):
    """Every task as a fresh replay session under the arm; rows like
    scratch/qwen_battery_run.py's; scored by the committed validators."""
    import collections
    from llmopt.lab import qbattery, qserve
    assert eng.tokenizer.convert_tokens_to_ids("</think>") == THINK_CLOSE_ID
    rows = []
    with arm(eng, name, receipt, label):
        for task in coll["tasks"]:
            sess = qserve.Session(eng, system=system, mode="replay", enable_thinking=thinking, reasoning_effort=effort)
            replies, results, err = [], [], None
            try:
                for turn in task["turns"]:
                    r = sess.ask(turn, max_tokens=budget)
                    replies.append(r.text)
                    close = (len(r.generated_ids) - r.generated_ids[::-1].index(THINK_CLOSE_ID)) if THINK_CLOSE_ID in r.generated_ids else None
                    results.append({"prompt_ids": r.prompt_ids, "generated_ids": r.generated_ids, "text": r.text, "stop_reason": r.stop_reason, "stop_token_id": r.stop_token_id,
                                    "n_prompt": r.n_prompt, "n_generated": r.n_generated, "cache_tokens": r.cache_tokens, "cache_offset_measured": r.cache_offset_measured,
                                    "timings": r.timings, "memory": r.memory, "reasoning_tokens": close, "answer_tokens": (r.n_generated - close) if close is not None else None})
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            if err is None:
                try:
                    verdict = (qbattery.score_thinking if thinking else qbattery.score)(task, replies, results)
                    final = (verdict.get("final_answers") or [None])[-1] if thinking else replies[-1]
                    verdict["format_ok"] = qbattery.format_ok(task, final) if final else None
                except Exception as e:  # noqa: BLE001
                    verdict = {"status": "error", "why": f"validator {type(e).__name__}: {e}"}
            else:
                verdict = {"status": "error", "why": err}
            rows.append({"id": task["id"], "kind": task["kind"], "turns": task["turns"], "expected": task.get("expected"), "verdict": verdict, "results": results})
            print(f"[promote] {label} {task['id']:14s} {verdict['status']:5s} {verdict['why'][:80]!r}", flush=True)
    by_kind = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        by_kind[r["kind"]][r["verdict"]["status"]] += 1
    summary = {"pass": sum(1 for r in rows if r["verdict"]["status"] == "pass"), "scored": sum(1 for r in rows if r["verdict"]["status"] in ("pass", "fail")), "tasks": len(rows),
               "by_kind": {k: dict(v) for k, v in by_kind.items()}, "errors": [r["id"] for r in rows if r["verdict"]["status"] == "error"], "holds": [r["id"] for r in rows if r["verdict"]["status"] == "hold"],
               "generated_tokens_total": sum(res["n_generated"] for r in rows for res in r["results"]),
               "unterminated_thinking_turns": sum(1 for r in rows for res in r["results"] if thinking and res["reasoning_tokens"] is None),
               "first_token_s_median": statistics.median(res["timings"]["first_token_s"] for r in rows for res in r["results"]) if rows else None,
               "task_total_s_sum": sum(res["timings"]["total_s"] for r in rows for res in r["results"])}
    return rows, summary


def compare_tasks(eng, receipt, rows, ref_rows, label):
    """Per task: the ids of every turn against the reference rows (by id); the
    first divergent task-turn classified where the prompts agree."""
    ref = {r["id"]: r for r in ref_rows}
    out, divs = [], []
    for r in rows:
        f = ref.get(r["id"])
        if f is None or not r["results"] or not f["results"]:
            out.append({"id": r["id"], "comparable": False})
            continue
        eq = [a["generated_ids"] == b["generated_ids"] for a, b in zip(r["results"], f["results"])]
        row = {"id": r["id"], "comparable": True, "turns_equal": eq, "all_equal": all(eq) and len(r["results"]) == len(f["results"]),
               "verdicts": {"this": r["verdict"]["status"], "reference": f["verdict"]["status"]}}
        if not row["all_equal"]:
            ti = next((i for i, e in enumerate(eq) if not e), None)
            if ti is None:
                d = {"label": f"{label} {r['id']}", "explained": False, "reason": f"turn counts differ ({len(r['results'])} against {len(f['results'])}) with every shared turn equal"}
            else:
                a, b = r["results"][ti], f["results"][ti]
                if a["prompt_ids"] != b["prompt_ids"]:
                    d = {"label": f"{label} {r['id']} t{ti}", "explained": False, "reason": "the prompts differ at the first divergent turn (every earlier turn was equal): a rendering difference, unexplained"}
                else:
                    d = compare_streams(eng, receipt, f"{label} {r['id']} t{ti}", a["prompt_ids"], a["generated_ids"], b["generated_ids"])
            row["divergence"] = d
            divs.append(d)
        out.append(row)
    return out, divs


def stage_battery(eng, receipt):
    assert fsha(FROZEN_V3) == FROZEN_V3_SHA, "the frozen v3 receipt changed"
    frozen = json.load(open(FROZEN_V3))
    coll = json.load(open(V3))
    assert hashlib.sha256(open(V3, "rb").read()).hexdigest() == frozen["collection_sha256"], "the v3 collection is not the one the frozen receipt scored"
    cfg = frozen["execution_config"]
    assert cfg["enable_thinking"] is True and cfg["reasoning_effort"] == "low" and cfg["max_tokens_per_turn"] == 768
    if SMOKE:
        seen, keep = set(), []
        for t in coll["tasks"]:
            if t["kind"] not in seen:
                seen.add(t["kind"])
                keep.append(t)
        coll = dict(coll, tasks=keep)
    receipt["collection"] = {"path": V3, "sha256": frozen["collection_sha256"], "tasks": len(coll["tasks"]), "execution_config": cfg, "rules": coll["rules"]}
    receipt["frozen_reference"] = {"path": FROZEN_V3, "sha256": FROZEN_V3_SHA, "commit": frozen["commit"], "summary": frozen["summary"]}
    rows, summary = run_collection(eng, receipt, "battery DEFAULT", coll, True, "low", 768, coll["rules"]["system"])
    receipt["tasks"] = rows
    comp, divs = compare_tasks(eng, receipt, rows, frozen["tasks"], "battery")
    receipt["comparison"], receipt["divergences"] = comp, divs
    summary.update({"tasks_all_equal_to_frozen": sum(1 for c in comp if c.get("all_equal")), "tasks_compared": sum(1 for c in comp if c["comparable"]),
                    "divergent_tasks": sum(1 for c in comp if c["comparable"] and not c["all_equal"]),
                    "unexplained_divergences": sum(1 for d in divs if not d.get("explained")),
                    "explained_near_tie_divergences": sum(1 for d in divs if d.get("explained")),
                    "verdict_changes_vs_frozen": [c["id"] for c in comp if c["comparable"] and c["verdicts"]["this"] != c["verdicts"]["reference"]],
                    "frozen_pass": frozen["summary"]["pass"], "frozen_scored": frozen["summary"]["scored"]})
    receipt["summary"] = summary
    receipt["engine_after"] = engine_block(eng)
    print(f"[promote] battery: pass {summary['pass']}/{summary['scored']} (frozen {summary['frozen_pass']}), equal to frozen {summary['tasks_all_equal_to_frozen']}/{summary['tasks_compared']}, unexplained {summary['unexplained_divergences']}", flush=True)


# ---------------------------------------------------------------- cli

def cli_task(task, name, outdir):
    """One scripts/qwen_chat.py process for one task and arm; returns the
    CLI's own JSON receipt plus the process record."""
    out = os.path.join(outdir, f"{name}_{task['id']}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    cmd = [sys.executable, "scripts/qwen_chat.py", "--json", out] + (["--recurrence", "loop"] if name == "LOOP" else [])
    if len(task["turns"]) == 1:
        cmd += ["--prompt", task["turns"][0]]
        stdin = None
    else:
        cmd += ["--interactive"]
        stdin = "\n".join(task["turns"]) + "\n/quit\n"
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=1800)
    rec = {"task": task["id"], "arm": name, "argv": cmd[1:], "returncode": p.returncode, "wall_s": time.perf_counter() - t0,
           "stderr_tail": "\n".join(ln for ln in p.stderr.splitlines() if ln.startswith("[qwen_chat]"))[-1500:]}
    if p.returncode == 0 and os.path.exists(out):
        rec["receipt"] = json.load(open(out))
    else:
        rec["error"] = p.stderr[-1200:]
    return rec


def stage_cli(eng, receipt):
    """eng is None at entry: the subprocesses hold the tower; an engine is
    loaded for classification only after they have all exited."""
    from llmopt.lab import qbattery
    coll = json.load(open(V4))
    tasks = coll["tasks"][:2] if SMOKE else coll["tasks"]
    receipt["collection"] = {"path": V4, "sha256": fsha(V4), "tasks": len(tasks), "independence_note": coll["independence_note"], "excluded_by_prior_prompt": coll["excluded_by_prior_prompt"]}
    outdir = os.path.join(OUTDIR, "cli")
    os.makedirs(outdir, exist_ok=True)
    receipt["processes"] = []
    for task in tasks:
        for name in ("DEFAULT", "LOOP"):
            rec = cli_task(task, name, outdir)
            receipt["processes"].append(rec)
            r = rec.get("receipt", {})
            print(f"[promote] cli {task['id']:14s} {name:7s} rc {rec['returncode']} recurrence {r.get('recurrence')} on {r.get('recurrence_layers')} layers, turns {len(r.get('turns', []))}, {rec['wall_s']:.0f}s", flush=True)
    # score and compare from the CLI receipts
    by = {(p["task"], p["arm"]): p for p in receipt["processes"]}
    rows, divergences = [], []
    for task in tasks:
        d, l = by[(task["id"], "DEFAULT")], by[(task["id"], "LOOP")]
        row = {"id": task["id"], "kind": task["kind"], "returncodes": {"DEFAULT": d["returncode"], "LOOP": l["returncode"]}}
        for name, p in (("DEFAULT", d), ("LOOP", l)):
            r = p.get("receipt")
            if r is None:
                row[name] = {"status": "error", "why": "the CLI process failed"}
                continue
            replies = [t["text"] for t in r["turns"]]
            results = [{"generated_ids": t["generated_ids"], "stop_reason": t["stop_reason"], "n_generated": len(t["generated_ids"])} for t in r["turns"]]
            try:
                v = qbattery.score_thinking(task, replies, results)
            except Exception as e:  # noqa: BLE001
                v = {"status": "error", "why": f"validator {type(e).__name__}: {e}"}
            vend = r.get("recurrence_vendor") or {}
            row[name] = {"status": v["status"], "why": v["why"], "recurrence": r.get("recurrence"), "recurrence_layers": r.get("recurrence_layers"), "recurrence_of": r.get("recurrence_of"),
                         "default_device": vend.get("default_device"), "metal_available": vend.get("metal_available"), "vendor_shas": {k: vend.get(k) for k in ("qwen3_5_sha256", "gated_delta_sha256")},
                         "thinking": r.get("thinking"), "effort": r.get("reasoning_effort"),
                         "max_tokens": r.get("max_tokens"), "mode": r.get("mode"), "turns": len(r["turns"]), "generated": [len(t["generated_ids"]) for t in r["turns"]],
                         "first_token_s": [t["timings"]["first_token_s"] for t in r["turns"]], "total_s": sum(t["timings"]["total_s"] for t in r["turns"]), "load_s": r.get("load_s"),
                         "modes": [t["mode"] for t in r["turns"]]}
        if "receipt" in d and "receipt" in l:
            dt, lt = d["receipt"]["turns"], l["receipt"]["turns"]
            eq = [a["generated_ids"] == b["generated_ids"] for a, b in zip(dt, lt)]
            row["turns_equal"], row["all_equal"] = eq, all(eq) and len(dt) == len(lt)
            row["prompts_equal"] = [a["prompt_ids"] == b["prompt_ids"] for a, b in zip(dt, lt)]
        rows.append(row)
    receipt["tasks"] = rows
    receipt["receipt_recurrence_ok"] = all(r.get("DEFAULT", {}).get("recurrence") == "kernel" and r.get("LOOP", {}).get("recurrence") == "loop"
                                           and r.get("DEFAULT", {}).get("recurrence_of") == {"loop": 0, "kernel": 48, "unselected": 0} and r.get("LOOP", {}).get("recurrence_of") == {"loop": 48, "kernel": 0, "unselected": 0}
                                           and all("gpu" in str(r.get(a, {}).get("default_device")) and r.get(a, {}).get("metal_available") is True for a in ARMS) for r in rows)
    need = [r for r in rows if r.get("all_equal") is False]
    receipt["divergences"] = []
    if need:
        eng = load_engine()
        receipt["engine"] = engine_block(eng)
        receipt["model_pin"] = model_pin(eng.vendor_dir, eng.art_dir, eng.info)
        for r in need:
            d, l = by[(r["id"], "DEFAULT")]["receipt"]["turns"], by[(r["id"], "LOOP")]["receipt"]["turns"]
            ti = next((i for i, e in enumerate(r["turns_equal"]) if not e), None)
            if ti is None:
                dv = {"label": f"cli {r['id']}", "explained": False, "reason": f"turn counts differ ({len(d)} against {len(l)}) with every shared turn equal"}
            elif d[ti]["prompt_ids"] != l[ti]["prompt_ids"]:
                dv = {"label": f"cli {r['id']} t{ti}", "explained": False, "reason": "the prompts differ at the first divergent turn (every earlier turn was equal): a rendering difference, unexplained"}
            else:
                dv = compare_streams(eng, receipt, f"cli {r['id']} t{ti}", d[ti]["prompt_ids"], d[ti]["generated_ids"], l[ti]["generated_ids"])
            r["divergence"] = dv
            receipt["divergences"].append(dv)
        receipt["engine_after"] = engine_block(eng)
    receipt["summary"] = {"tasks": len(rows), "pass": {a: sum(1 for r in rows if r.get(a, {}).get("status") == "pass") for a in ARMS},
                          "scored": {a: sum(1 for r in rows if r.get(a, {}).get("status") in ("pass", "fail")) for a in ARMS},
                          "errors": {a: sum(1 for r in rows if r.get(a, {}).get("status") == "error") for a in ARMS},
                          "process_failures": sum(1 for p in receipt["processes"] if p["returncode"] != 0),
                          "tasks_all_equal": sum(1 for r in rows if r.get("all_equal")), "divergent_tasks": len(need),
                          "unexplained_divergences": sum(1 for dv in receipt["divergences"] if not dv.get("explained")),
                          "explained_near_tie_divergences": sum(1 for dv in receipt["divergences"] if dv.get("explained")),
                          "receipt_recurrence_ok": receipt["receipt_recurrence_ok"],
                          "first_token_s_median": {a: statistics.median(x for r in rows for x in r.get(a, {}).get("first_token_s", [])) for a in ARMS if any(r.get(a, {}).get("first_token_s") for r in rows)},
                          "load_s_median": {a: statistics.median(r[a]["load_s"] for r in rows if r.get(a, {}).get("load_s") is not None) for a in ARMS}}
    print(f"[promote] cli: pass {receipt['summary']['pass']}, equal {receipt['summary']['tasks_all_equal']}/{len(rows)}, unexplained {receipt['summary']['unexplained_divergences']}, receipts ok {receipt['receipt_recurrence_ok']}", flush=True)


# ---------------------------------------------------------------- main

def main():
    from llmopt.lab import qgdn, qserve, qtower
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t_proc = time.perf_counter()
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "stage": STAGE, "driver": os.path.relpath(__file__), "driver_sha256": fsha(__file__),
               "sources_sha256": {"llmopt/lab/qtower.py": fsha(qtower.__file__), "llmopt/lab/qserve.py": fsha(qserve.__file__), "llmopt/lab/qgdn.py": fsha(qgdn.__file__), "scripts/qwen_chat.py": fsha("scripts/qwen_chat.py")},
               "default_recurrence_in_code": qserve.DEFAULT_RECURRENCE, "vendor": qgdn.check_vendor(), "near_tie_margin": NEAR_TIE_MARGIN}
    if STAGE in ("tests", "cli"):
        eng = None
    else:
        eng = load_engine()
        receipt["engine"] = engine_block(eng)
        receipt["model_pin"] = model_pin(eng.vendor_dir, eng.art_dir, eng.info)
    {"tests": stage_tests, "heldout": stage_heldout, "cache": stage_cache, "battery": stage_battery, "cli": stage_cli}[STAGE](eng, receipt)
    receipt["wall_s"] = time.perf_counter() - t_proc
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[promote] {STAGE} wrote {OUT} in {receipt['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

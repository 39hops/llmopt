"""Serving-path guard for the kernel generation 5 default flip (Artin's GO
of 2026-10-22, item 1 and item 3): generation 4 (the qualified control)
against generation 5 (the S16 v5 kernel) through the ACTUAL serving
configuration, on the same Mac, the same arm C artifact and identical
real-token inputs. The earlier A/B (scratch/bench_qmetal_ab_s16.py)
measured greedy decode at contexts 126 and 1,024 without the serving
memory configuration; the qualification census's 2,048-token reading
(130 ms median, 162 p90 over 16 steps after a one-shot fused prefill)
was not a paired serving-path comparison. This driver is.

Serving configuration (llmopt.lab.qprefill.ServingConfig, applied through
configure_serving_memory): chunk 256, GEMM backend, final-position head,
an explicit 2 GiB MLX allocator-cache limit. Prefill runs through
qprefill.prefill_chunked(head="final", matmul="gemm"); decode is the plain
model call token by token outside any scope (the serving decode path).
Decode is teacher-forced on the real tokens that follow the prompt, so
both versions see identical inputs by construction whatever their
outputs; the outputs are compared anyway (final logits, the cache state
after prefill, every decode step's logits), expected bit-identical.

Modes (MODE env):
  ab        (default) paired, order-balanced rounds in ONE process: the
            tower is built once; per context the rounds run in the order
            control, candidate, candidate, control, repeated ABBA_REPEATS
            times, each round with a fresh cache and the same tokens.
            Per round: prefill wall and throughput; STEPS decode steps
            timed one at a time (mx.eval on the logits) with the full
            per-step trace kept; median, p90, mean, min, max of steps
            5..STEPS; MLX peak active memory of the decode phase (reset
            after prefill), the memory report after prefill and after
            decode (active, peak, allocator pool, footprint, process-lifetime peak RSS);
            system memory pressure (memory_pressure's free percentage)
            and whatever pmset -g therm reports (CPU speed and scheduler
            limits when a warning level exists, otherwise its note lines)
            before and after each round; the matrix
            path counts of the prefill and of the decode; the set of
            Metal kernels compiled so far (the dispatch witness: which
            (codec, generation) kernels the process has actually run).
            Cross-version: final logits, cache after prefill and per-step
            decode logits of the first round of each version compared;
            every later round of a version compared with its first.
            Receipt OUT (default logs/qmetal/s16_v5_default/ab.json).
  release   single-prompt serving smoke in the process's DEFAULT kernel
            generation (QMETAL_VERSION unset in the promoted build, or
            set explicitly for the control process): contexts CONTEXTS
            (default 512,1024,2048), one round each, the same records as
            ab plus the effective generation (module default, every
            resident's version, the compiled kernels), final argmax and
            top-16, decode argmaxes, and the sha256 of the final logits,
            the decode logits and the cache arrays; the arrays go to
            DUMP (machine-local) for compare. Receipt
            OUT_DIR/release_<tag>.json where tag = TAG or
            v<generation>_<env|default>.
  compare   reads two release receipts (A and B env, paths) and reports
            per context: final logits (max abs, argmax, margin), cache
            after prefill (bit identity per class), decode logits
            (per-step max abs, argmax agreement), prefill throughput and
            decode medians side by side, memory side by side; writes
            OUT_DIR/compare_<tagA>_vs_<tagB>.json.
Env: ART_DIR, VENDOR_DIR, CONTROL_VERSION (4), CAND_VERSION (5),
CONTEXTS ("1024,2048" for ab; "512,1024,2048" for release), STEPS (48),
ABBA_REPEATS (2), CHUNK (256), CACHE_LIMIT (bytes; default the
ServingConfig 2 GiB), OUT_DIR (logs/qmetal/s16_v5_default), OUT (ab
receipt path), TAG, A, B, SMOKE=1 (OUT_DIR forced to
logs/qmetal/s16_v5_default_smoke, contexts 64 and 96, 6 steps, one
ABBA). Every receipt refuses to overwrite. Timing is the claim; the
numerical claim lives in tests/test_qmetal.py and the qualification
receipts, and the comparisons here are guards on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SMOKE = os.environ.get("SMOKE") == "1"
OUT_DIR = "logs/qmetal/s16_v5_default_smoke" if SMOKE else os.environ.get("OUT_DIR", "logs/qmetal/s16_v5_default")
DUMP = os.path.join(OUT_DIR, "dump")


def refuse(path):
    if os.path.exists(path):
        raise SystemExit(f"REFUSING: {path} exists")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_array(a) -> str:
    return sha_bytes(np.ascontiguousarray(a).tobytes())


def provenance(mode: str) -> dict:
    import mlx.core as mx
    import mlx_lm
    from llmopt.lab import qmetal, qprefill
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    cfg = qprefill.ServingConfig()
    return {"mode": mode, "commit": head, "tree_dirty": bool(dirty), "smoke": SMOKE,
            "device": str(mx.default_device()), "device_name": mx.device_info().get("device_name"),
            "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
            "qmetal_version_env": os.environ.get("QMETAL_VERSION"),
            "qmetal_version_module": qmetal.VERSION,
            "process_default_prefill_gemm": qmetal.PREFILL_GEMM,
            "serving_config_defaults": {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes},
            "entry_point": "llmopt.lab.qprefill.prefill_chunked(head='final', matmul='gemm'); decode = plain model call"}


def system_signals() -> dict:
    """Memory pressure and the thermal limits macOS exposes without root.
    powermetrics (power and temperature) needs root and is not read."""
    out = {"memory_free_percent": None, "cpu_speed_limit": None, "cpu_scheduler_limit": None,
           "power_and_temperature": "not read: powermetrics requires root"}
    try:
        mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"free percentage:\s*(\d+)%", mp)
        if m:
            out["memory_free_percent"] = int(m.group(1))
    except Exception as e:  # noqa: BLE001 - a missing tool is recorded, not fatal
        out["memory_pressure_error"] = repr(e)
    try:
        th = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=20).stdout
        out["pmset_therm_raw"] = [ln.strip() for ln in th.splitlines() if ln.strip()]
        for key, field in (("CPU_Speed_Limit", "cpu_speed_limit"), ("CPU_Scheduler_Limit", "cpu_scheduler_limit")):
            m = re.search(key + r"\s*=\s*(\d+)", th)
            if m:
                out[field] = int(m.group(1))
    except Exception as e:  # noqa: BLE001
        out["pmset_error"] = repr(e)
    return out


def apply_serving(chunk: int):
    from llmopt.lab import qprefill
    lim = os.environ.get("CACHE_LIMIT", "")
    cfg = qprefill.ServingConfig(chunk=chunk) if lim == "" else qprefill.ServingConfig(chunk=chunk, cache_limit_bytes=int(lim))
    prev = qprefill.configure_serving_memory(cfg)
    return cfg, prev


def load_tower():
    from qwen_metal_qual import build_metal_model
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    t0 = time.perf_counter()
    model, info = build_metal_model(art, vdir)
    return model, {k: v for k, v in info.items() if k != "chain_report"} | {"chain_rung0": info["chain_report"]["rung0"]}, \
        time.perf_counter() - t0, os.path.basename(art.rstrip("/"))


def compiled_kernels() -> list:
    from llmopt.lab import qmetal
    return sorted(f"{k[0]}:{k[1]}" for k in qmetal._kernels)


def one_round(model, residents, ids, ctx, steps, cfg, keep_arrays: bool):
    """Fresh cache, serving prefill of ids[:ctx], STEPS teacher-forced
    decode steps on ids[ctx:ctx+steps]. Returns the record and, when
    keep_arrays, the final logits, the cache snapshot after prefill and the
    decode logits for comparison."""
    import mlx.core as mx
    from llmopt.lab import qmetal, qprefill
    from qwen_metal_prefill_head import snapshot_cache
    versions = sorted({int(r.version) for r in residents})
    sig0 = system_signals()
    cache = model.make_cache()
    mx.reset_peak_memory()
    c0 = qmetal.path_counts(residents)
    t0 = time.perf_counter()
    lg = qprefill.prefill_chunked(model, ids[:ctx], cache, chunk=cfg.chunk, head="final", matmul=cfg.matmul)
    mx.eval(lg)
    t_pre = time.perf_counter() - t0
    c1 = qmetal.path_counts(residents)
    mem_pre = qprefill.memory_report()
    # host COPIES: np.asarray on an MLX array is a zero-copy view that keeps
    # the device buffer alive, which would carry each round's logits and
    # cache into the next round's memory readings
    final = np.array(np.asarray(lg[0, -1], dtype=np.float32), copy=True)
    snap = snapshot_cache(cache)
    for s in snap:
        s.state = tuple(np.array(a, copy=True) for a in s.state) if isinstance(s.state, tuple) else np.array(s.state, copy=True)
    mx.reset_peak_memory()
    walls, logits = [], []
    for t in ids[ctx:ctx + steps]:
        t0 = time.perf_counter()
        lz = model(mx.array([[int(t)]]), cache=cache)[0, -1]
        mx.eval(lz)
        walls.append(time.perf_counter() - t0)
        logits.append(np.array(np.asarray(lz, dtype=np.float32), copy=True))
        del lz
    c2 = qmetal.path_counts(residents)
    mem_dec = qprefill.memory_report()
    sig1 = system_signals()
    st = walls[4:] if len(walls) > 4 else walls
    dec = np.stack(logits)
    cache_arrays = []
    for s in snap:
        stt = s.state if isinstance(s.state, tuple) else (s.state,)
        cache_arrays.extend(np.ascontiguousarray(a) for a in stt)
    h = hashlib.sha256()
    for a in cache_arrays:
        h.update(a.tobytes())
    rec = {"context": ctx, "steps": len(walls), "resident_versions": versions,
           "prefill_s": t_pre, "prefill_tok_per_s": ctx / t_pre,
           "median_step_s": float(np.median(st)), "p90_step_s": float(np.percentile(st, 90)),
           "mean_step_s": float(np.mean(st)), "min_step_s": float(np.min(st)), "max_step_s": float(np.max(st)),
           "tok_per_s_median": 1.0 / float(np.median(st)),
           "step_trace_ms": [round(1e3 * w, 3) for w in walls],
           "decode_peak_active_bytes": int(mx.get_peak_memory()),
           "memory_after_prefill": mem_pre, "memory_after_decode": mem_dec,
           "system_before": sig0, "system_after": sig1,
           "matrix_calls_prefill": {k: c1[k] - c0[k] for k in c0},
           "matrix_calls_decode": {k: c2[k] - c1[k] for k in c0},
           "compiled_kernels_after_round": compiled_kernels(),
           "final_argmax": int(final.argmax()),
           "decode_argmax": [int(a) for a in dec.argmax(-1)],
           "sha256": {"final_logits": sha_array(final), "decode_logits": sha_array(dec),
                      "cache_after_prefill": h.hexdigest(), "cache_arrays": len(cache_arrays)}}
    del cache, lg
    if keep_arrays:
        return rec, (final, snap, dec)
    return rec, None


def cross(a, b) -> dict:
    from qwen_metal_prefill_head import compare_caches, compare_logits
    fa, sa, da = a
    fb, sb, db = b
    per_step = [float(np.abs(x - y).max()) for x, y in zip(da, db)]
    return {"final_logits": compare_logits(fa[None], fb[None]),
            "cache_after_prefill": compare_caches(sa, sb),
            "decode_logits": {"per_step_max_abs": per_step, "max_abs": max(per_step),
                              "argmax_identical_steps": int((da.argmax(-1) == db.argmax(-1)).sum()), "steps": len(per_step),
                              "bit_identical": max(per_step) == 0.0}}


def mode_ab():
    import mlx.core as mx
    from llmopt.lab import qmetal
    from qwen_metal_prefill_head import real_tokens, residents_of
    prov = provenance("ab")
    vc = int(os.environ.get("CONTROL_VERSION", "4"))
    vk = int(os.environ.get("CAND_VERSION", "5"))
    contexts = [64, 96] if SMOKE else [int(c) for c in os.environ.get("CONTEXTS", "1024,2048").split(",") if c]
    steps = 6 if SMOKE else int(os.environ.get("STEPS", "48"))
    abba = 1 if SMOKE else int(os.environ.get("ABBA_REPEATS", "2"))
    chunk = int(os.environ.get("CHUNK", "256"))
    # SMOKE writes to the smoke directory whatever OUT says (path isolation)
    out = os.path.join(OUT_DIR, "ab.json") if SMOKE else os.environ.get("OUT", os.path.join(OUT_DIR, "ab.json"))
    refuse(out)
    cfg, prev_limit = apply_serving(chunk)
    ids = real_tokens(max(contexts) + steps)
    model, info, load_wall, arm = load_tower()
    residents = residents_of(model)
    receipt = {"provenance": prov, "arm": arm, "control_version": vc, "candidate_version": vk,
               "contexts": contexts, "steps": steps, "abba_repeats": abba,
               "serving_applied": {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes,
                                   "previous_cache_limit_bytes": prev_limit, "head": "final"},
               "tokens": {"source": "logs/qwenteacher_v2 rollouts concatenated (qwen_metal_prefill_head.real_tokens)",
                          "n": len(ids), "sha256": sha_array(np.asarray(ids, dtype=np.int64))},
               "tower": info, "load_wall_s": load_wall,
               "residents": {"total": len(residents), "s16": sum(1 for r in residents if r.codec == "s16"),
                             "w4": sum(1 for r in residents if r.codec == "w4")},
               "s16_v5_max_tile": qmetal.S16_V5_MAX_TILE, "v4_max_tile": qmetal.V4_MAX_TILE,
               "note": "timing is the claim; teacher-forced decode on real tokens so both versions see identical inputs; "
                       "outputs compared as a guard (expected bit-identical); parity lives in tests/test_qmetal.py and the "
                       "qualification receipts; power and temperature not read (powermetrics requires root)"}

    def set_version(v):
        for r in residents:
            r.version = v

    rows, crosses = [], {}
    order = [vc, vk, vk, vc] * abba
    for ctx in contexts:
        first = {}
        for rnd, v in enumerate(order):
            set_version(v)
            rec, arrays = one_round(model, residents, ids, ctx, steps, cfg, keep_arrays=True)
            rec = {"round": rnd, "version": v, **rec}
            if v not in first:
                first[v] = arrays
                rec["first_round_of_version"] = True
            else:
                same = cross(first[v], arrays)
                rec["vs_first_round_of_this_version"] = {
                    "final_logits_max_abs": same["final_logits"]["max_abs"],
                    "cache_bit_identical": same["cache_after_prefill"]["bit_identical"],
                    "decode_logits_max_abs": same["decode_logits"]["max_abs"]}
            rows.append(rec)
            print(f"[serve-ab] ctx {ctx} round {rnd} v{v}: median {rec['median_step_s']*1e3:.1f} ms p90 "
                  f"{rec['p90_step_s']*1e3:.1f} max {rec['max_step_s']*1e3:.1f} prefill {rec['prefill_tok_per_s']:.1f} tok/s "
                  f"decode peak {rec['decode_peak_active_bytes']/1e9:.2f} GB footprint "
                  f"{rec['memory_after_decode']['footprint_bytes']/1e9:.2f} GB free {rec['system_after']['memory_free_percent']}% "
                  f"therm {rec['system_after']['cpu_speed_limit']}", flush=True)
        crosses[str(ctx)] = cross(first[vc], first[vk])
        del first
    receipt["rounds"] = rows
    receipt["cross_version_first_rounds"] = crosses
    summary = {}
    for ctx in contexts:
        for v in (vc, vk):
            rs = [r for r in rows if r["version"] == v and r["context"] == ctx]
            meds = [r["median_step_s"] for r in rs]
            summary[f"ctx{ctx}_v{v}_median_step_s"] = float(np.median(meds))
            summary[f"ctx{ctx}_v{v}_median_step_s_by_round"] = meds
            summary[f"ctx{ctx}_v{v}_p90_step_s_by_round"] = [r["p90_step_s"] for r in rs]
            summary[f"ctx{ctx}_v{v}_max_step_s_by_round"] = [r["max_step_s"] for r in rs]
            summary[f"ctx{ctx}_v{v}_spread_s"] = float(max(meds) - min(meds))
            summary[f"ctx{ctx}_v{v}_prefill_tok_per_s_by_round"] = [r["prefill_tok_per_s"] for r in rs]
            summary[f"ctx{ctx}_v{v}_decode_peak_active_bytes_max"] = max(r["decode_peak_active_bytes"] for r in rs)
            summary[f"ctx{ctx}_v{v}_footprint_after_decode_bytes_max"] = max(r["memory_after_decode"]["footprint_bytes"] for r in rs)
            summary[f"ctx{ctx}_v{v}_prefill_peak_active_bytes_max"] = max(r["memory_after_prefill"]["peak_bytes"] for r in rs)
        a, b = summary[f"ctx{ctx}_v{vc}_median_step_s"], summary[f"ctx{ctx}_v{vk}_median_step_s"]
        summary[f"ctx{ctx}_improvement_percent"] = 100.0 * (a - b) / a
        cm, km = summary[f"ctx{ctx}_v{vc}_median_step_s_by_round"], summary[f"ctx{ctx}_v{vk}_median_step_s_by_round"]
        summary[f"ctx{ctx}_improvement_percent_worst_pairing"] = 100.0 * (min(cm) - max(km)) / min(cm)
        pc, pk = summary[f"ctx{ctx}_v{vc}_p90_step_s_by_round"], summary[f"ctx{ctx}_v{vk}_p90_step_s_by_round"]
        summary[f"ctx{ctx}_p90_improvement_percent"] = 100.0 * (float(np.median(pc)) - float(np.median(pk))) / float(np.median(pc))
        summary[f"ctx{ctx}_decode_peak_active_ratio_candidate_over_control"] = (
            summary[f"ctx{ctx}_v{vk}_decode_peak_active_bytes_max"] / summary[f"ctx{ctx}_v{vc}_decode_peak_active_bytes_max"])
        summary[f"ctx{ctx}_footprint_ratio_candidate_over_control"] = (
            summary[f"ctx{ctx}_v{vk}_footprint_after_decode_bytes_max"] / summary[f"ctx{ctx}_v{vc}_footprint_after_decode_bytes_max"])
        fc, fk = summary[f"ctx{ctx}_v{vc}_prefill_tok_per_s_by_round"], summary[f"ctx{ctx}_v{vk}_prefill_tok_per_s_by_round"]
        summary[f"ctx{ctx}_prefill_change_percent_candidate_vs_control"] = 100.0 * (float(np.median(fk)) - float(np.median(fc))) / float(np.median(fc))
    receipt["summary"] = summary
    receipt["compiled_kernels"] = compiled_kernels()
    receipt["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[serve-ab] {json.dumps({k: v for k, v in summary.items() if 'by_round' not in k})}; wrote {out}", flush=True)


def mode_release():
    from llmopt.lab import qmetal
    from qwen_metal_prefill_head import real_tokens, residents_of
    prov = provenance("release")
    tag = os.environ.get("TAG") or f"v{qmetal.VERSION}_{'env' if os.environ.get('QMETAL_VERSION') else 'default'}"
    contexts = [64, 96] if SMOKE else [int(c) for c in os.environ.get("CONTEXTS", "512,1024,2048").split(",") if c]
    steps = 6 if SMOKE else int(os.environ.get("STEPS", "48"))
    chunk = int(os.environ.get("CHUNK", "256"))
    out = os.path.join(OUT_DIR, f"release_{tag}.json")
    refuse(out)
    os.makedirs(DUMP, exist_ok=True)
    cfg, prev_limit = apply_serving(chunk)
    ids = real_tokens(max(contexts) + steps)
    model, info, load_wall, arm = load_tower()
    residents = residents_of(model)
    versions = sorted({int(r.version) for r in residents})
    receipt = {"provenance": prov, "tag": tag, "arm": arm, "contexts": contexts, "steps": steps,
               "effective_generation": {"module_default": qmetal.VERSION, "resident_versions": versions,
                                        "all_residents_at_module_default": versions == [qmetal.VERSION],
                                        "s16_v5_max_tile": qmetal.S16_V5_MAX_TILE, "v4_max_tile": qmetal.V4_MAX_TILE},
               "serving_applied": {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes,
                                   "previous_cache_limit_bytes": prev_limit, "head": "final"},
               "tokens": {"source": "logs/qwenteacher_v2 rollouts concatenated (qwen_metal_prefill_head.real_tokens)",
                          "n": len(ids), "sha256": sha_array(np.asarray(ids, dtype=np.int64))},
               "tower": info, "load_wall_s": load_wall, "rounds": []}
    for ctx in contexts:
        rec, (final, snap, dec) = one_round(model, residents, ids, ctx, steps, cfg, keep_arrays=True)
        top = np.argsort(-final.astype(np.float64))[:16]
        rec["final_top16_ids"] = [int(t) for t in top]
        rec["final_top16_logits"] = [float(final[t]) for t in top]
        names = {"final": f"final_{tag}_ctx{ctx}.npy", "decode": f"decode_{tag}_ctx{ctx}.npy", "cache": f"cache_{tag}_ctx{ctx}.npz"}
        for nm in names.values():
            refuse(os.path.join(DUMP, nm))
        np.save(os.path.join(DUMP, names["final"]), final)
        np.save(os.path.join(DUMP, names["decode"]), dec)
        arrs = {}
        for i, s in enumerate(snap):
            stt = s.state if isinstance(s.state, tuple) else (s.state,)
            for j, a in enumerate(stt):
                arrs[f"c{i}_{j}"] = a
            if hasattr(s, "offset"):
                arrs[f"c{i}_offset"] = np.array(s.offset)
        np.savez(os.path.join(DUMP, names["cache"]), **arrs)
        rec["dumps"] = names
        receipt["rounds"].append(rec)
        print(f"[release {tag}] ctx {ctx}: prefill {rec['prefill_tok_per_s']:.1f} tok/s, decode median {rec['median_step_s']*1e3:.1f} ms "
              f"p90 {rec['p90_step_s']*1e3:.1f}, versions {versions}, kernels {rec['compiled_kernels_after_round']}, "
              f"footprint {rec['memory_after_decode']['footprint_bytes']/1e9:.2f} GB", flush=True)
    receipt["compiled_kernels"] = compiled_kernels()
    receipt["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[release {tag}] wrote {out}", flush=True)


def mode_compare():
    from qwen_metal_prefill_head import compare_logits
    a_path, b_path = os.environ["A"], os.environ["B"]
    ra, rb = json.load(open(a_path)), json.load(open(b_path))
    out = os.path.join(OUT_DIR, f"compare_{ra['tag']}_vs_{rb['tag']}.json")
    refuse(out)
    per = {}
    for xa, xb in zip(ra["rounds"], rb["rounds"]):
        if xa["context"] != xb["context"]:
            raise SystemExit("REFUSING: context lists differ")
        ctx = xa["context"]
        fa = np.load(os.path.join(DUMP, xa["dumps"]["final"]))
        fb = np.load(os.path.join(DUMP, xb["dumps"]["final"]))
        da = np.load(os.path.join(DUMP, xa["dumps"]["decode"]))
        db = np.load(os.path.join(DUMP, xb["dumps"]["decode"]))
        ca = np.load(os.path.join(DUMP, xa["dumps"]["cache"]))
        cb = np.load(os.path.join(DUMP, xb["dumps"]["cache"]))
        cache_cmp = {"keys_equal": sorted(ca.files) == sorted(cb.files),
                     "max_abs_by_key_nonzero": {k: float(np.abs(ca[k].astype(np.float64) - cb[k].astype(np.float64)).max())
                                                for k in ca.files if k in cb.files and ca[k].shape == cb[k].shape
                                                and np.abs(ca[k].astype(np.float64) - cb[k].astype(np.float64)).max() != 0.0}}
        cache_cmp["bit_identical"] = cache_cmp["keys_equal"] and not cache_cmp["max_abs_by_key_nonzero"]
        per_step = [float(np.abs(x - y).max()) for x, y in zip(da, db)]
        per[str(ctx)] = {
            "final_logits": compare_logits(fa[None], fb[None]),
            "cache_after_prefill": cache_cmp,
            "decode_logits": {"per_step_max_abs": per_step, "max_abs": max(per_step), "bit_identical": max(per_step) == 0.0,
                              "argmax_identical_steps": int((da.argmax(-1) == db.argmax(-1)).sum()), "steps": len(per_step)},
            "sha_equal": {k: xa["sha256"][k] == xb["sha256"][k] for k in ("final_logits", "decode_logits", "cache_after_prefill")},
            "prefill_tok_per_s": {ra["tag"]: xa["prefill_tok_per_s"], rb["tag"]: xb["prefill_tok_per_s"],
                                  "change_percent_b_vs_a": 100.0 * (xb["prefill_tok_per_s"] - xa["prefill_tok_per_s"]) / xa["prefill_tok_per_s"]},
            "decode_median_ms": {ra["tag"]: 1e3 * xa["median_step_s"], rb["tag"]: 1e3 * xb["median_step_s"],
                                 "improvement_percent_b_vs_a": 100.0 * (xa["median_step_s"] - xb["median_step_s"]) / xa["median_step_s"]},
            "decode_p90_ms": {ra["tag"]: 1e3 * xa["p90_step_s"], rb["tag"]: 1e3 * xb["p90_step_s"]},
            "decode_peak_active_bytes": {ra["tag"]: xa["decode_peak_active_bytes"], rb["tag"]: xb["decode_peak_active_bytes"]},
            "footprint_after_decode_bytes": {ra["tag"]: xa["memory_after_decode"]["footprint_bytes"],
                                             rb["tag"]: xb["memory_after_decode"]["footprint_bytes"]},
            "resident_versions": {ra["tag"]: xa["resident_versions"], rb["tag"]: xb["resident_versions"]},
            "compiled_kernels": {ra["tag"]: xa["compiled_kernels_after_round"], rb["tag"]: xb["compiled_kernels_after_round"]}}
    rec = {"a": {"path": a_path, "tag": ra["tag"], "commit": ra["provenance"]["commit"], "effective_generation": ra["effective_generation"]},
           "b": {"path": b_path, "tag": rb["tag"], "commit": rb["provenance"]["commit"], "effective_generation": rb["effective_generation"]},
           "per_context": per,
           "all_bit_identical": all(p["final_logits"]["bit_identical"] and p["cache_after_prefill"]["bit_identical"]
                                    and p["decode_logits"]["bit_identical"] for p in per.values())}
    json.dump(rec, open(out, "w"), indent=1)
    for ctx, p in per.items():
        print(f"[compare] ctx {ctx}: final bit {p['final_logits']['bit_identical']} cache bit {p['cache_after_prefill']['bit_identical']} "
              f"decode bit {p['decode_logits']['bit_identical']} decode {p['decode_median_ms']} prefill {p['prefill_tok_per_s']}", flush=True)
    print(f"[compare] all_bit_identical {rec['all_bit_identical']}; wrote {out}", flush=True)


def main():
    mode = os.environ.get("MODE", "ab")
    {"ab": mode_ab, "release": mode_release, "compare": mode_compare}[mode]()


if __name__ == "__main__":
    main()

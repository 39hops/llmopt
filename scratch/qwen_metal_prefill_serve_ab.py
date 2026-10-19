"""Final bounded A/B and E0 recount for the hardened prefill entry point
(llmopt.lab.qprefill.prefill_chunked with matmul="gemm" against the
fused-only control, under the serving memory configuration).

Modes (MODE env):
  ab        ONE arm per process (ARM=fused|gemm): the serving config is
            applied (ServingConfig: chunk 256, 2 GiB allocator-cache limit;
            CACHE_LIMIT=none skips the limit), CTX real tokens are
            prefilled through prefill_chunked(head="final", matmul=ARM),
            then DECODE_STEPS teacher-forced decode steps with the plain
            model call (token by token, outside any scope). Recorded: the
            memory classes after load, after prefill and after decode
            (active, peak since reset, allocator pool, footprint = active +
            pool, peak RSS), walls, matrix-call path counts for the prefill
            and for the decode separately, and the dispatch-isolation
            check: after the call a 64-row batch on the head resident and
            on a body resident, outside any scope, must take the fused
            path. Final logits, cache state after prefill and decode
            logits go to the dump directory (machine-local) for compare.
  compare   reads both arms' receipts and dumps: final logits (max abs,
            relative, argmax, margin), cache state after prefill (conv,
            recurrent, KV, offsets), decode logits (argmax per step, max
            abs), with both arms' walls and memory tables side by side.
  census    the 24 E0 teacher-greedy rollouts (prompt + recorded tokens)
            through prefill_chunked(head="full", matmul="gemm") under the
            serving config; at every live position the argmax is compared
            with the frozen E0 row (logs/qwen_e0/C/positions.jsonl,
            draft_argmax) and with the teacher token, and the draft's top-16
            logits with the row's; the full draft logits (fp16, live
            vocabulary) and the final-norm hidden states (fp32) of every
            live position are written to the dump directory for the
            disagreement census (read-only analysis).
Env: ART_DIR, VENDOR_DIR, CTX (1024), CHUNK (256), DECODE_STEPS (16), ARM,
TAG (suffix for an extra arm's receipt and dumps), CACHE_LIMIT (bytes or
none), OUT (logs/qmetal/prefill_serve), SMOKE=1
(forces OUT to logs/qmetal/prefill_serve_smoke, CTX 64, 2 decode steps,
census on one prompt and 8 steps). Receipts refuse to overwrite.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SMOKE = os.environ.get("SMOKE") == "1"
OUT = "logs/qmetal/prefill_serve_smoke" if SMOKE else os.environ.get("OUT", "logs/qmetal/prefill_serve")
DUMP = os.path.join(OUT, "dump")
V_LIVE = 248077   # tokenizer length recorded in logs/qwen_e0/C/summary.json (v_live)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def refuse(path):
    if os.path.exists(path):
        raise SystemExit(f"REFUSING: {path} exists")


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


def provenance(mode, arm=None):
    import mlx.core as mx
    import mlx_lm
    from llmopt.lab import qmetal, qprefill
    cfg = qprefill.ServingConfig()
    return {"mode": mode, "arm": arm, "code_commit": git_head(), "tree_dirty": git_dirty(), "smoke": SMOKE,
            "device": str(mx.default_device()), "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
            "qmetal_version": qmetal.VERSION, "process_default_prefill_gemm": qmetal.PREFILL_GEMM,
            "resident_prefill_gemm_attr": bool(qmetal._Resident.prefill_gemm), "gemm_min_t_override": qmetal.GEMM_MIN_T,
            "gemm_min_t_by_codec": qmetal.GEMM_MIN_T_BY_CODEC, "gemm_slab_bytes": qmetal.GEMM_SLAB_BYTES,
            "serving_config_defaults": {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes},
            "serving_applied": {"matmul": arm if mode == "ab" else ("gemm" if mode == "census" else None),
                                "cache_limit_bytes": (None if os.environ.get("CACHE_LIMIT", "") == "none" else int(os.environ.get("CACHE_LIMIT") or cfg.cache_limit_bytes)) if mode in ("ab", "census") else None},
            "entry_point": "llmopt.lab.qprefill.prefill_chunked"}


def load_model():
    from qwen_metal_qual import build_metal_model
    art = os.environ.get("ART_DIR", os.path.expanduser("~/qwen_whole0t/C"))
    vdir = os.environ.get("VENDOR_DIR", os.path.expanduser("~/qwen_vendor"))
    t0 = time.time()
    model, info = build_metal_model(art, vdir)
    return model, info, time.time() - t0


def apply_cache_limit():
    from llmopt.lab import qprefill
    lim = os.environ.get("CACHE_LIMIT", "")
    if lim == "none":
        return None
    cfg = qprefill.ServingConfig() if lim == "" else qprefill.ServingConfig(cache_limit_bytes=int(lim))
    qprefill.configure_serving_memory(cfg)
    return cfg.cache_limit_bytes


def mode_ab():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from llmopt.lab import qmetal, qprefill
    from qwen_metal_prefill_head import real_tokens, residents_of, decode_after, snapshot_cache
    arm = os.environ["ARM"]
    if arm not in ("fused", "gemm"):
        raise SystemExit("ARM must be fused or gemm")
    ctx = 64 if SMOKE else int(os.environ.get("CTX", "1024"))
    chunk = int(os.environ.get("CHUNK", "256"))
    steps = 2 if SMOKE else int(os.environ.get("DECODE_STEPS", "16"))
    os.makedirs(DUMP, exist_ok=True)
    tag = os.environ.get("TAG", "")          # distinguishes extra arms (for example the no-cache-limit control)
    commit = git_head()
    rpath = os.path.join(OUT, f"ab_{arm}{tag}_{commit}.json")
    refuse(rpath)
    cache_limit = apply_cache_limit()
    ids = real_tokens(ctx + steps)
    model, info, load_wall = load_model()
    residents = residents_of(model)
    head_res = model.language_model.lm_head.res
    body_res = [r for r in residents if r is not head_res and r.gemm_ok][0]
    mx.reset_peak_memory()
    mem = {"after_load": qprefill.memory_report()}
    cache = make_prompt_cache(model)
    counts0 = qmetal.path_counts(residents)
    t0 = time.time()
    lg = qprefill.prefill_chunked(model, ids[:ctx], cache, chunk=chunk, head="final", matmul=arm)
    mx.eval(lg)
    prefill_wall = time.time() - t0
    counts1 = qmetal.path_counts(residents)
    mem["after_prefill"] = qprefill.memory_report()
    final = np.asarray(lg[0, -1], dtype=np.float32)
    snap = snapshot_cache(cache)
    mx.reset_peak_memory()            # after_decode.peak_bytes is the decode phase's own peak
    t0 = time.time()
    dec = decode_after(model, cache, ids[ctx:ctx + steps])
    decode_wall = time.time() - t0
    counts2 = qmetal.path_counts(residents)
    mem["after_decode"] = qprefill.memory_report()
    # dispatch isolation outside the entry point: 64 rows must stay fused
    iso = {}
    for name, r in (("head", head_res), ("body", body_res)):
        x = mx.random.normal((64, r.shape[1])).astype(mx.float32)
        y = r.gemm(x)
        mx.eval(y)
        iso[name] = {"rows": 64, "codec": r.codec, "gemm_ok": bool(r.gemm_ok), "path": r.last_path,
                     "scope_state": qmetal.gemm_scope_state()}
    names = {k: f"{k}_{arm}{tag}_{commit}.{ext}" for k, ext in (("final", "npy"), ("decode", "npy"), ("cache", "npz"))}
    for nm in names.values():
        refuse(os.path.join(DUMP, nm))
    np.save(os.path.join(DUMP, names["final"]), final)
    np.save(os.path.join(DUMP, names["decode"]), dec)
    arrs = {}
    for i, s in enumerate(snap):
        st = s.state if isinstance(s.state, tuple) else (s.state,)
        for j, a in enumerate(st):
            arrs[f"c{i}_{j}"] = a
        if hasattr(s, "offset"):
            arrs[f"c{i}_offset"] = np.array(s.offset)
    np.savez(os.path.join(DUMP, names["cache"]), **arrs)
    top = np.argsort(-final.astype(np.float64))[:16]
    rec = provenance("ab", arm) | {
        "ctx": ctx, "chunk": chunk, "decode_steps": steps, "cache_limit_bytes_applied": cache_limit,
        "artifact": info, "load_wall_s": load_wall,
        "prefill_wall_s": prefill_wall, "prefill_tok_s": ctx / prefill_wall,
        "decode_wall_s": decode_wall, "decode_ms_per_token": 1000 * decode_wall / steps,
        "matrix_calls_prefill": {k: counts1[k] - counts0[k] for k in counts0},
        "matrix_calls_decode": {k: counts2[k] - counts1[k] for k in counts0},
        "memory": mem, "dispatch_isolation_outside_scope": iso,
        "final_argmax": int(final.argmax()), "final_top16_ids": [int(t) for t in top],
        "final_top16_logits": [float(final[t]) for t in top],
        "decode_argmax": [int(a) for a in dec.argmax(-1)],
        "memory_note": "peak_bytes is MLX peak active memory since the last reset (reset after load and again before decode); footprint is MLX active + allocator pool",
        "dumps": {nm: sha_file(os.path.join(DUMP, nm)) for nm in names.values()}}
    json.dump(rec, open(rpath, "w"), indent=1)
    print(f"[ab {arm}] prefill {prefill_wall:.2f} s ({ctx / prefill_wall:.2f} tok/s), decode {1000 * decode_wall / steps:.1f} ms/token, "
          f"calls prefill {rec['matrix_calls_prefill']} decode {rec['matrix_calls_decode']}, footprint after prefill "
          f"{mem['after_prefill']['footprint_bytes'] / 1e9:.2f} GB, peak {mem['after_prefill']['peak_bytes'] / 1e9:.2f} GB, iso {iso}; wrote {rpath}")


def mode_compare():
    from qwen_metal_prefill_head import compare_logits
    import glob
    recs, files = {}, {}
    commit = git_head()
    for arm in ("fused", "gemm"):
        paths = [p for p in glob.glob(os.path.join(OUT, f"ab_{arm}_{commit}.json"))]
        if len(paths) != 1:
            raise SystemExit(f"REFUSING: expected one ab_{arm}_{commit}.json under {OUT}, found {paths}")
        recs[arm] = json.load(open(paths[0]))
        files[arm] = {}
        for nm, h in recs[arm]["dumps"].items():
            if sha_file(os.path.join(DUMP, nm)) != h:
                raise SystemExit(f"REFUSING: dump {nm} does not match its receipt sha")
            files[arm][nm.split("_")[0]] = nm
    for k in ("code_commit", "ctx", "chunk", "decode_steps", "cache_limit_bytes_applied", "smoke"):
        if recs["fused"][k] != recs["gemm"][k]:
            raise SystemExit(f"REFUSING: arms differ in {k}: {recs['fused'][k]} v {recs['gemm'][k]}")
    rpath = os.path.join(OUT, f"compare_{commit}.json")
    refuse(rpath)
    f = {a: np.load(os.path.join(DUMP, files[a]["final"])) for a in recs}
    d = {a: np.load(os.path.join(DUMP, files[a]["decode"])) for a in recs}
    c = {a: np.load(os.path.join(DUMP, files[a]["cache"])) for a in recs}
    # cache classes: DeltaNet caches hold (conv tail, recurrent state), attention caches (keys, values) with an offset
    cache_cmp = {"offsets_equal": True, "bit_identical": True,
                 "classes": {k: {"max_abs": 0.0, "max_rel_to_array_max": 0.0, "arrays": 0} for k in ("conv", "state", "kv_k", "kv_v")}}
    for k in c["fused"].files:
        if k.endswith("_offset"):
            cache_cmp["offsets_equal"] &= bool(int(c["fused"][k]) == int(c["gemm"][k]))
            continue
        a, b = c["fused"][k], c["gemm"][k]
        ci, j = k[1:].split("_")
        is_attn = f"c{ci}_offset" in c["fused"].files
        cls = ("kv_k" if j == "0" else "kv_v") if is_attn else ("conv" if j == "0" else "state")
        dd = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        den = float(np.abs(a).max()) or 1e-30
        e = cache_cmp["classes"][cls]
        e["max_abs"] = max(e["max_abs"], dd)
        e["max_rel_to_array_max"] = max(e["max_rel_to_array_max"], dd / den)
        e["arrays"] += 1
        cache_cmp["bit_identical"] &= dd == 0.0
    dec_cmp = compare_logits(d["fused"], d["gemm"])
    dec_cmp["per_step_argmax_equal"] = [bool(x) for x in (d["fused"].argmax(-1) == d["gemm"].argmax(-1))]
    rec = provenance("compare") | {
        "arms": {a: {k: recs[a][k] for k in ("code_commit", "ctx", "chunk", "decode_steps", "cache_limit_bytes_applied",
                                              "prefill_wall_s", "prefill_tok_s", "decode_ms_per_token",
                                              "matrix_calls_prefill", "matrix_calls_decode", "memory",
                                              "dispatch_isolation_outside_scope")} for a in recs},
        "final_logits": compare_logits(f["fused"][None], f["gemm"][None]),
        "cache_after_prefill": cache_cmp, "decode_logits": dec_cmp}
    json.dump(rec, open(rpath, "w"), indent=1)
    cl = cache_cmp["classes"]
    print(f"[compare] final {rec['final_logits']}; cache bit_identical {cache_cmp['bit_identical']} state rel {cl['state']['max_rel_to_array_max']:.1e} "
          f"conv abs {cl['conv']['max_abs']:.1e} kv abs {max(cl['kv_k']['max_abs'], cl['kv_v']['max_abs']):.1e} "
          f"offsets {cache_cmp['offsets_equal']}; decode argmax {dec_cmp['argmax_identical']}/{dec_cmp['positions']} max abs {dec_cmp['max_abs']:.2e}; wrote {rpath}")


def mode_census():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from llmopt.lab import qmetal, qprefill
    from qwen_metal_prefill_head import residents_of
    os.makedirs(DUMP, exist_ok=True)
    rpath = os.path.join(OUT, f"census_{git_head()}.json")
    refuse(rpath)
    cache_limit = apply_cache_limit()
    tdir = "logs/qwenteacher_v2"
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    rows = [json.loads(line) for line in open("logs/qwen_e0/C/positions.jsonl")]
    e0_summary = json.load(open("logs/qwen_e0/C/summary.json"))
    if e0_summary["v_live"] != V_LIVE:
        raise SystemExit("REFUSING: v_live differs from the E0 summary")
    input_ids, mask = np.asarray(ro["input_ids"]), np.asarray(ro["attention_mask"])
    gen_lengths = list(ro["gen_lengths"])
    n_prompts = 1 if SMOKE else len(recs)
    chunk = int(os.environ.get("CHUNK", "256"))
    model, info, load_wall = load_model()
    residents = residents_of(model)
    lm = model.language_model.lm_head

    class _RecordingHead:
        """Applies the fused head and keeps the hidden rows it saw (host)."""
        def __init__(self):
            self.hidden = []

        def __call__(self, h):
            self.hidden.append(np.asarray(h[0], dtype=np.float32))
            return lm(h)
    rec_head = _RecordingHead()
    model.language_model.lm_head = rec_head
    n_live = sum(gen_lengths) if not SMOKE else min(8, gen_lengths[0])
    commit = git_head()
    dump_names = {"logits": f"draft_logits_fp16_{commit}.npy", "hidden": f"draft_hidden_fp32_{commit}.npy"}
    for nm in dump_names.values():
        refuse(os.path.join(DUMP, nm))
    logits_out = np.lib.format.open_memmap(os.path.join(DUMP, dump_names["logits"]), mode="w+", dtype=np.float16, shape=(n_live, V_LIVE))
    hidden_out = np.lib.format.open_memmap(os.path.join(DUMP, dump_names["hidden"]), mode="w+", dtype=np.float32, shape=(n_live, 5120))
    pos = 0
    agree_e0 = agree_teacher = 0
    topk_max_abs = 0.0
    mismatches = []
    per_prompt = []
    counts0 = qmetal.path_counts(residents)
    mx.reset_peak_memory()
    t_all = time.time()
    row_i = 0
    for i in range(n_prompts):
        prompt = [int(x) for x in input_ids[i][mask[i] == 1]]
        gen = [int(x) for x in recs[i]["tokens"][:gen_lengths[i]]]
        if SMOKE:
            gen = gen[:8]
        ids = prompt + gen
        p_len = len(prompt)
        chunks = []
        rec_head.hidden = []
        cache = make_prompt_cache(model)
        t0 = time.time()
        qprefill.prefill_chunked(model, ids, cache, chunk=chunk, head="full", matmul="gemm",
                                 on_chunk=lambda s, lg: chunks.append(np.asarray(lg[0], dtype=np.float32)))
        wall = time.time() - t0
        lg_all = np.concatenate(chunks, axis=0)
        h_all = np.concatenate(rec_head.hidden, axis=0)
        if lg_all.shape[0] != len(ids) or h_all.shape[0] != len(ids):
            raise SystemExit("REFUSING: logits or hidden rows != tokens")
        a_e0 = a_t = 0
        for s in range(len(gen)):
            zd = lg_all[p_len - 1 + s, :V_LIVE]
            row = rows[row_i]
            if row["prompt_id"] != recs[i]["id"] or row["step"] != s:
                raise SystemExit(f"REFUSING: E0 row order mismatch at prompt {i} step {s}")
            am = int(zd.argmax())
            a_e0 += am == row["draft_argmax"]
            a_t += am == row["teacher_token"]
            if am != row["draft_argmax"]:
                mismatches.append({"prompt_id": row["prompt_id"], "step": s, "e0": row["draft_argmax"], "metal": am,
                                   "margin_metal": float(np.sort(zd)[-1] - np.sort(zd)[-2])})
            topk_max_abs = max(topk_max_abs, float(np.abs(zd[row["draft_topk_ids"]] - np.asarray(row["draft_topk_logits"])).max()))
            logits_out[pos] = zd.astype(np.float16)
            hidden_out[pos] = h_all[p_len - 1 + s]
            pos += 1
            row_i += 1
        # E0 rows of this prompt beyond the smoke's 8 steps are skipped only in smoke
        if SMOKE:
            row_i = sum(gen_lengths[:i + 1])
        agree_e0 += a_e0
        agree_teacher += a_t
        per_prompt.append({"prompt_id": recs[i]["id"], "prompt_len": p_len, "gen_len": len(gen), "wall_s": wall,
                           "tok_s": len(ids) / wall, "argmax_equal_e0": a_e0, "argmax_equal_teacher": a_t})
        print(f"[census] {recs[i]['id']}: {len(ids)} tokens {wall:.1f} s, e0 {a_e0}/{len(gen)}, teacher {a_t}/{len(gen)}", flush=True)
        del lg_all, h_all, chunks, cache
    wall_all = time.time() - t_all
    counts1 = qmetal.path_counts(residents)
    logits_out.flush()
    hidden_out.flush()
    del logits_out, hidden_out
    mem = qprefill.memory_report()
    rec = provenance("census") | {
        "cache_limit_bytes_applied": cache_limit, "chunk": chunk, "artifact": info, "load_wall_s": load_wall,
        "teacher": {"dir": tdir, "code_commit": man["code_commit"], "revision": man["revision"], "rollout_sha_manifest": ro["sha256"],
                    "rollout_tokens_sha256": sha_file(os.path.join(tdir, "rollout_tokens.jsonl")), "manifest_sha256": sha_file(os.path.join(tdir, "teacher_manifest.json"))},
        "e0_rows": "logs/qwen_e0/C/positions.jsonl", "e0_rows_sha256": sha_file("logs/qwen_e0/C/positions.jsonl"),
        "e0_agree_rate_pooled": e0_summary["agree_rate_pooled"],
        "n_prompts": n_prompts, "n_positions": pos, "argmax_equal_e0": agree_e0, "argmax_equal_teacher": agree_teacher,
        "agree_rate_teacher": agree_teacher / pos, "topk16_logit_max_abs_diff_vs_e0": topk_max_abs,
        "mismatches_vs_e0": mismatches, "per_prompt": per_prompt, "wall_s": wall_all,
        "matrix_calls": {k: counts1[k] - counts0[k] for k in counts0}, "memory_after": mem,
        "v_live": V_LIVE, "dump_dir": DUMP, "dumps": {nm: sha_file(os.path.join(DUMP, nm)) for nm in dump_names.values()}}
    json.dump(rec, open(rpath, "w"), indent=1)
    print(f"[census] {pos} positions: argmax equal to E0 {agree_e0}/{pos}, teacher agreement {agree_teacher}/{pos} "
          f"({agree_teacher / pos:.4f} v E0 {e0_summary['agree_rate_pooled']:.4f}), top-16 max abs {topk_max_abs:.2e}, "
          f"{wall_all:.0f} s, calls {rec['matrix_calls']}; wrote {rpath}")


def main():
    mode = os.environ.get("MODE", "ab")
    {"ab": mode_ab, "compare": mode_compare, "census": mode_census}[mode]()


if __name__ == "__main__":
    main()

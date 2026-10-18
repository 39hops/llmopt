"""Metal-versus-CPU qualification of arm C over the frozen E0 teacher-greedy
contexts. Separate from QWEN-METAL-TOWER-0 (one prompt, 78 positions) and
from E0 itself, whose receipts under logs/qwen_e0/C are read, never
rewritten.

The Metal tower (same construction as scratch/qwen_metal_tower.py: mlx-lm
qwen3_5 architecture, fused W4/S16 linears, raw tensors with the sanitizer
conventions, exact HF l2norm on the DeltaNet q/k) is run teacher-forced in
ONE forward per prompt over prompt + recorded rollout tokens, exactly the
construction of scratch/qwen_e0_spec_census.py. Position s of rollout i is
then compared with
  * the CPU arm C row at that position (logs/qwen_e0/C/positions.jsonl):
    argmax identity, logit margin difference, Metal logits at the CPU
    top-16 ids against the stored CPU values, agreement-with-teacher flip,
  * the teacher record (logs/qwenteacher_v2 rollout_logits, pinned by
    sha): shift-aligned sup error and agreement, which E0 computed for the
    CPU arm with the same functions, so the two instruments are compared on
    the same law.
Replay yield (llmopt.lab.spec_reference.replay_blocks) is recomputed from
the Metal argmax stream, concatenated and per stream, beside the CPU
values read from the E0 receipts.

A cached-decode cross-check runs on DECODE_PROMPTS prompts (default 2):
prompt prefill into a fresh cache, then one recorded token per step; every
step's logits are compared with the same prompt's one-forward logits.

LONG_CTX (default "512,1024,2048,4096") times prefill plus 16 decode steps
at synthetic contexts (the prompt tokens repeated), with peak unified
memory, under a wall budget (LONG_BUDGET_S, default 480 per length).

Receipts under OUT (default logs/qwen_metal_qual/<arm>, refuse-if-exists):
positions.jsonl (one row per teacher-greedy position), summary.json.
SMOKE=1: one prompt, 8 steps, no long-context rung, OUT forced to
logs/qwen_metal_qual_smoke/<arm>.
Env: ART_DIR, VENDOR_DIR, TEACHER_DIR, CPU_DIR (default logs/qwen_e0/C),
OUT, SMOKE, DECODE_PROMPTS, LONG_CTX, LONG_BUDGET_S.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TEACHER_PINS = {"logs/qwenteacher_v2": "0ca4151"}
KS = tuple(range(1, 9))
NORM_SHIFT_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                       "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]


def fsha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=os.path.dirname(HERE)).stdout.strip()


def margin_bin(m: float) -> int:
    for i in range(len(MARGIN_EDGES) - 1):
        if MARGIN_EDGES[i] <= m < MARGIN_EDGES[i + 1]:
            return i
    return len(MARGIN_EDGES) - 2


def quantiles(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "max": float(a.max()), "p99": float(np.percentile(a, 99)),
            "p90": float(np.percentile(a, 90)), "median": float(np.median(a)), "mean": float(a.mean())}


def build_metal_model(art: str, vdir: str):
    """Arm C through the mlx-lm Qwen3.5 architecture with fused linears.
    Same construction as scratch/qwen_metal_tower.py (frozen as the
    QWEN-METAL-TOWER-0 record); returns (model, info)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import dec_raw
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows

    arm = os.path.basename(art.rstrip("/"))
    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(qwen3_5.GatedDeltaNet.__call__))
    a_line = "q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)"
    b_line = "k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)"
    if a_line not in src or b_line not in src:
        raise SystemExit("REFUSING: mlx-lm GatedDeltaNet source changed; re-derive the l2norm override")
    src = src.replace(a_line, "q = inv_scale * (q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6))")
    src = src.replace(b_line, "k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)")
    ns: dict = {}
    exec(compile(src, "qwen3_5_gdn_exact_l2norm", "exec"), vars(qwen3_5), ns)
    qwen3_5.GatedDeltaNet.__call__ = ns["__call__"]

    cfg = json.load(open(os.path.join(vdir, "config.json")))
    args = qwen3_5.ModelArgs.from_dict(cfg)
    model = qwen3_5.Model(args)

    class QLinear(nn.Module):
        def __init__(self, res):
            super().__init__()
            self.res = res
            self.out_features, self.in_features = res.shape

        def __call__(self, x):
            lead = x.shape[:-1]
            y = self.res.gemm(x.reshape(-1, x.shape[-1]).astype(mx.float32))
            return y.reshape(*lead, self.out_features)

    class QEmbedding(nn.Module):
        def __init__(self, rows):
            super().__init__()
            self.rows = rows

        def __call__(self, ids):
            flat = np.asarray(ids).reshape(-1)
            vecs = np.stack([self.rows.rows(int(t), int(t) + 1)[0] for t in flat]).astype(np.float32)
            return mx.array(vecs).reshape(*ids.shape, -1)

    def set_path(root, path, value):
        parts = path.split(".")
        obj = root
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        leaf = parts[-1]
        if leaf.isdigit():
            obj[int(leaf)] = value
        else:
            setattr(obj, leaf, value)

    n_fused = n_raw = 0
    bytes_resident = 0
    for key, e in MAN.items():
        if not isinstance(e, dict) or e["codec"] == "excluded":
            continue
        if key.startswith("model.visual") or ".mtp." in key or key.startswith("mtp."):
            continue
        k = key.replace("model.language_model.", "language_model.model.") if key.startswith("model.language_model.") \
            else "language_model." + key
        if e["codec"] in ("w4", "s16"):
            if k.endswith("embed_tokens.weight"):
                buf = payload(e)
                rows = (S16Rows if e["codec"] == "s16" else W4Rows)(buf, e["shape"])
                set_path(model, k[:-len(".weight")], QEmbedding(rows))
                bytes_resident += len(buf)
            else:
                res = qmetal.resident(payload(e), e)
                bytes_resident += res.payload_bytes()
                set_path(model, k[:-len(".weight")], QLinear(res))
            n_fused += 1
        else:
            arr = dec_raw(payload(e), e["shape"], e["dtype"]).astype(np.float32)
            if "conv1d.weight" in k and arr.shape[-1] != 1:
                arr = np.moveaxis(arr, 2, 1)
            if any(k.endswith(s) for s in NORM_SHIFT_SUFFIXES) and arr.ndim == 1:
                arr = arr + 1.0
            set_path(model, k, mx.array(arr))
            n_raw += 1
    mx.eval(model.parameters())
    tm = model.language_model.model
    fams = ["linear_attn" if hasattr(ly, "linear_attn") else "full_attn" for ly in tm.layers]
    info = {"chain": chain, "chain_sha256": fsha(chain), "chain_report": q["report"],
            "n_layers": len(tm.layers), "families": {"linear_attn": fams.count("linear_attn"),
                                                     "full_attn": fams.count("full_attn")},
            "fused": n_fused, "raw": n_raw, "payload_resident_bytes": bytes_resident,
            "kernel_version": qmetal.VERSION}
    return model, info


def main() -> None:
    import mlx.core as mx
    import mlx_lm
    from llmopt.lab import spec_reference as sr
    from transformers import AutoTokenizer

    smoke = os.environ.get("SMOKE") == "1"
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    tdir = os.environ.get("TEACHER_DIR", "logs/qwenteacher_v2")
    cpu_dir = os.environ.get("CPU_DIR", f"logs/qwen_e0/{arm}")
    out = os.environ.get("OUT", f"logs/qwen_metal_qual/{arm}")
    if smoke:
        out = f"logs/qwen_metal_qual_smoke/{arm}"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists (refuse-if-exists)")
    if tdir not in TEACHER_PINS:
        raise SystemExit(f"REFUSING: no source-frozen pin for teacher {tdir}")
    for f in os.listdir(vdir):
        if f.endswith(".safetensors"):
            raise SystemExit("REFUSING: vendor weight shard present; only config/tokenizer/index are read")

    # ---- CPU reference rows (E0, read-only) ----
    cpu_summary = json.load(open(os.path.join(cpu_dir, "summary.json")))
    if cpu_summary.get("smoke") or cpu_summary["arm"] != arm or cpu_summary["device_actual"] != "cpu":
        raise SystemExit("REFUSING: CPU reference is not the real cpu census of this arm")
    cpu_rows = [json.loads(line) for line in open(os.path.join(cpu_dir, "positions.jsonl"))]
    cpu_pos_sha = fsha(os.path.join(cpu_dir, "positions.jsonl"))
    if len(cpu_rows) != cpu_summary["n_positions"]:
        raise SystemExit("REFUSING: CPU rows != summary n_positions")

    # ---- teacher records: pinned, sha-checked (same as E0) ----
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    if man.get("smoke") or man["code_commit"] != TEACHER_PINS[tdir]:
        raise SystemExit("REFUSING: teacher record pin")
    ro = man["records"]["rollouts"]
    rollouts = np.load(os.path.join(tdir, "rollout_logits.npy"), mmap_mode="r")
    rl_sha = hashlib.sha256(np.ascontiguousarray(rollouts).tobytes()).hexdigest()
    if rl_sha != ro["sha256"] or list(rollouts.shape) != ro["shape"]:
        raise SystemExit("REFUSING: rollout_logits array sha/shape != manifest")
    if rl_sha != cpu_summary["teacher"]["rollout_sha"]:
        raise SystemExit("REFUSING: teacher record differs from the one E0 read")
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    gen_lengths = list(ro["gen_lengths"])
    if ro["padding_side"] != "left":
        raise SystemExit("REFUSING: unexpected padding side")
    tok = AutoTokenizer.from_pretrained(vdir)
    gcfg = json.load(open(os.path.join(vdir, "generation_config.json")))
    pad_id = gcfg.get("pad_token_id", tok.pad_token_id)
    v_live = len(tok)
    if v_live != cpu_summary["v_live"]:
        raise SystemExit("REFUSING: live vocab differs from E0")
    n_prompts = len(recs) if not smoke else 1

    # ---- Metal tower ----
    t0 = time.time()
    model, info = build_metal_model(art, vdir)
    print(f"[qual] built arm {arm} on Metal: {info['fused']} fused, {info['raw']} raw, "
          f"{info['payload_resident_bytes']/1e9:.2f} GB resident, {time.time()-t0:.0f}s; "
          f"families {info['families']}", flush=True)
    if info["families"] != {"linear_attn": 48, "full_attn": 16}:
        raise SystemExit(f"REFUSING: layer families {info['families']}")
    if info["chain_sha256"] != cpu_summary["artifact_chain_sha256"]:
        raise SystemExit("REFUSING: artifact digest chain differs from the one E0 qualified")

    os.makedirs(out, exist_ok=True)
    rows_f = open(os.path.join(out, "positions.jsonl"), "w")
    per_prompt = []
    t_all: list[int] = []
    d_cpu_all: list[int] = []
    d_metal_all: list[int] = []
    top16_abs: list[float] = []
    top16_rel: list[float] = []
    margin_abs: list[float] = []
    sup_shift_diff: list[float] = []
    argmax_mismatch: list[dict] = []
    agree_flip: list[dict] = []
    bins_cpu = np.zeros((len(MARGIN_EDGES) - 1, 2), dtype=int)   # by CPU draft margin: [n, identical]
    prefill_walls = []
    row_i = 0
    mx.reset_peak_memory()
    t_start = time.time()
    for i in range(n_prompts):
        rec = recs[i]
        m_i = mask[i]
        prompt = [int(x) for x in input_ids[i][m_i == 1]]
        if pad_id is not None and pad_id in prompt:
            raise SystemExit(f"REFUSING: pad id inside prompt {i}")
        gen_full = [int(x) for x in rec["tokens"][:gen_lengths[i]]]
        gen = gen_full[:8] if smoke else gen_full
        ids = prompt + gen
        p_len = len(prompt)
        t0 = time.time()
        lg = model(mx.array([ids]))
        mx.eval(lg)
        wall = time.time() - t0
        prefill_walls.append((len(ids), wall))
        lg = np.asarray(lg[0], dtype=np.float32)
        if not np.isfinite(lg).all():
            raise SystemExit("REFUSING: non-finite Metal logits")
        t_stream, dc_stream, dm_stream = [], [], []
        for s, tok_s in enumerate(gen):
            cr = cpu_rows[row_i]
            if cr["prompt_id"] != rec["id"] or cr["step"] != s or cr["teacher_token"] != tok_s:
                raise SystemExit(f"REFUSING: CPU row {row_i} misaligned with prompt {rec['id']} step {s}")
            zm = lg[p_len - 1 + s, :v_live].astype(np.float64)
            zt = np.asarray(rollouts[s, i, :v_live], dtype=np.float64)
            am, mm = sr.top1_and_margin(zm)
            # Metal logits at the CPU top-16 ids against the CPU values (the
            # only CPU logits the frozen record keeps)
            cids = np.asarray(cr["draft_topk_ids"])
            cvals = np.asarray(cr["draft_topk_logits"], dtype=np.float64)
            d16 = np.abs(zm[cids] - cvals)
            scale = max(abs(cvals[0]), 1e-30)
            sup_m = sr.shift_aligned_sup_error(zt, zm)
            agree_m = bool(am == tok_s)
            row = {"prompt_id": rec["id"], "step": s, "teacher_token": tok_s,
                   "cpu_argmax": cr["draft_argmax"], "metal_argmax": am,
                   "argmax_identical": bool(am == cr["draft_argmax"]),
                   "cpu_margin": cr["draft_margin_logit"], "metal_margin": mm,
                   "top16_max_abs_err": float(d16.max()), "top16_max_rel_err": float(d16.max() / scale),
                   "cpu_agree": cr["agree"], "metal_agree": agree_m,
                   "cpu_sup_err_shift": cr["sup_err_shift"], "metal_sup_err_shift": sup_m,
                   "teacher_margin_logit": cr["teacher_margin_logit"], "ambiguous": cr["ambiguous"]}
            rows_f.write(json.dumps(row) + "\n")
            top16_abs.append(float(d16.max()))
            top16_rel.append(float(d16.max() / scale))
            margin_abs.append(abs(mm - cr["draft_margin_logit"]))
            sup_shift_diff.append(abs(sup_m - cr["sup_err_shift"]))
            b = margin_bin(cr["draft_margin_logit"])
            bins_cpu[b, 0] += 1
            bins_cpu[b, 1] += int(row["argmax_identical"])
            if not row["argmax_identical"]:
                argmax_mismatch.append({k: row[k] for k in ("prompt_id", "step", "cpu_argmax", "metal_argmax",
                                                             "cpu_margin", "metal_margin", "teacher_token")})
            if agree_m != cr["agree"]:
                agree_flip.append({"prompt_id": rec["id"], "step": s, "cpu_agree": cr["agree"],
                                   "metal_agree": agree_m, "cpu_margin": cr["draft_margin_logit"]})
            t_stream.append(tok_s)
            dc_stream.append(cr["draft_argmax"])
            dm_stream.append(am)
            row_i += 1
        t_all += t_stream
        d_cpu_all += dc_stream
        d_metal_all += dm_stream
        per_prompt.append({"prompt_id": rec["id"], "category": rec["category"], "tokens": len(ids),
                           "forward_wall_s": wall, "tok_per_s_prefill": len(ids) / wall,
                           "argmax_identical": int(sum(a == b for a, b in zip(dc_stream, dm_stream))),
                           "positions": len(gen),
                           "metal_agree_rate": float(np.mean([t == d for t, d in zip(t_stream, dm_stream)])),
                           "cpu_agree_rate": float(np.mean([t == d for t, d in zip(t_stream, dc_stream)])),
                           "replay_k4_metal": sr.replay_blocks(t_stream, dm_stream, 4)["mean_emitted"],
                           "replay_k4_cpu": sr.replay_blocks(t_stream, dc_stream, 4)["mean_emitted"]})
        pp = per_prompt[-1]
        print(f"[qual] {rec['id']}: {len(ids)} tok {wall:.0f}s; argmax identical {pp['argmax_identical']}/{len(gen)}; "
              f"top16 max abs {max(top16_abs[-len(gen):]):.2e}; agree metal {pp['metal_agree_rate']:.3f} "
              f"cpu {pp['cpu_agree_rate']:.3f}", flush=True)
    rows_f.close()
    peak_prefill = int(mx.get_peak_memory())
    if row_i != len(cpu_rows) and not smoke:
        raise SystemExit(f"REFUSING: consumed {row_i} CPU rows of {len(cpu_rows)}")

    n = len(t_all)
    ident = int(sum(a == b for a, b in zip(d_cpu_all, d_metal_all)))
    replay_metal = {k: sr.replay_blocks(t_all, d_metal_all, k)["mean_emitted"] for k in KS}
    replay_cpu = {k: sr.replay_blocks(t_all, d_cpu_all, k)["mean_emitted"] for k in KS}
    # per-stream replay (blocks never cross a prompt boundary), pooled over blocks
    streams: list[tuple[list, list, list]] = []
    off = 0
    for pp in per_prompt:
        m = pp["positions"]
        streams.append((t_all[off:off + m], d_cpu_all[off:off + m], d_metal_all[off:off + m]))
        off += m
    rps = {}
    for k in KS:
        for name, idx in (("metal", 2), ("cpu", 1)):
            tot_em = tot_bl = 0
            for st in streams:
                r = sr.replay_blocks(st[0], st[idx], k)
                tot_em += r["mean_emitted"] * r["blocks"]
                tot_bl += r["blocks"]
            rps.setdefault(k, {})[name] = tot_em / tot_bl if tot_bl else None
    summary = {
        "arm": arm, "artifact": art, "smoke": smoke, "device_actual": "metal",
        "code_commit": git_commit(), "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
        "tower": info,
        "cpu_reference": {"dir": cpu_dir, "code_commit": cpu_summary["code_commit"],
                          "positions_sha256": cpu_pos_sha, "n_positions": len(cpu_rows),
                          "agree_rate_pooled": cpu_summary["agree_rate_pooled"],
                          "replay_pooled_k4": cpu_summary["replay_pooled"]["4"]["mean_emitted"]},
        "teacher": {"dir": tdir, "code_commit": man["code_commit"], "rollout_sha": rl_sha},
        "v_live": v_live, "n_prompts": n_prompts, "n_positions": n,
        "argmax_identical": ident, "argmax_identical_fraction": ident / n,
        "argmax_mismatches": argmax_mismatch,
        "identity_by_cpu_margin_bin": {"edges": MARGIN_EDGES[:-1] + ["inf"], "n_identical": bins_cpu.tolist()},
        "top16_abs_err": quantiles(top16_abs), "top16_rel_err": quantiles(top16_rel),
        "margin_abs_diff": quantiles(margin_abs),
        "sup_err_shift_vs_teacher_abs_diff": quantiles(sup_shift_diff),
        "metal_agree_rate_pooled": float(np.mean([t == d for t, d in zip(t_all, d_metal_all)])),
        "cpu_agree_rate_pooled_recomputed": float(np.mean([t == d for t, d in zip(t_all, d_cpu_all)])),
        "agree_flips": agree_flip,
        "replay_concatenated": {k: {"metal": replay_metal[k], "cpu": replay_cpu[k]} for k in KS},
        "replay_per_stream": rps,
        "per_prompt": per_prompt,
        "prefill": {"tok_per_s_overall": sum(a for a, _ in prefill_walls) / sum(b for _, b in prefill_walls),
                    "peak_mem_bytes": peak_prefill},
    }

    # ---- cached decode cross-check (teacher-forced, logits v the one-forward logits) ----
    n_dec = int(os.environ.get("DECODE_PROMPTS", "1" if smoke else "2"))
    dec = []
    for i in range(min(n_dec, n_prompts)):
        rec = recs[i]
        prompt = [int(x) for x in input_ids[i][mask[i] == 1]]
        gen = [int(x) for x in rec["tokens"][:gen_lengths[i]]]
        gen = gen[:8] if smoke else gen
        ids = prompt + gen
        p_len = len(prompt)
        full = model(mx.array([ids]))
        mx.eval(full)
        full = np.asarray(full[0], dtype=np.float64)
        cache = model.make_cache()
        mx.reset_peak_memory()
        lp = model(mx.array([ids[:p_len]]), cache=cache)
        mx.eval(lp)
        walls, diffs, same = [], [], 0
        for s in range(len(gen)):
            pos = p_len - 1 + s
            t0 = time.time()
            lz = lp[0, -1] if s == 0 else model(mx.array([[ids[pos]]]), cache=cache)[0, -1]
            mx.eval(lz)
            walls.append(time.time() - t0)
            g = np.asarray(lz, dtype=np.float64)
            diffs.append(float(np.abs(g[:v_live] - full[pos, :v_live]).max()))
            same += int(g[:v_live].argmax() == full[pos, :v_live].argmax())
        steady = walls[1:]
        dec.append({"prompt_id": rec["id"], "prompt_len": p_len, "steps": len(gen),
                    "argmax_same_as_one_forward": same,
                    "max_abs_diff_vs_one_forward": float(max(diffs)),
                    "median_step_s": float(np.median(steady)), "p90_step_s": float(np.percentile(steady, 90)),
                    "tok_per_s_steady": 1.0 / float(np.median(steady)),
                    "context_len_at_end": len(ids), "peak_mem_bytes": int(mx.get_peak_memory())})
        print(f"[qual] decode {rec['id']}: {len(gen)} steps, argmax same {same}/{len(gen)}, "
              f"max |decode - forward| {max(diffs):.2e}, median {np.median(steady)*1e3:.0f} ms", flush=True)
    summary["cached_decode"] = dec

    # ---- long-context timing (synthetic tokens, no oracle) ----
    long_ctx = [] if smoke else [int(x) for x in os.environ.get("LONG_CTX", "512,1024,2048,4096").split(",") if x]
    budget = float(os.environ.get("LONG_BUDGET_S", "480"))
    base = [int(x) for x in input_ids[0][mask[0] == 1]]
    longs = []
    for L in long_ctx:
        seq = (base * (L // len(base) + 1))[:L]
        cache = model.make_cache()
        mx.reset_peak_memory()
        t0 = time.time()
        lp = model(mx.array([seq]), cache=cache)
        mx.eval(lp)
        t_pre = time.time() - t0
        cur = int(lp[0, -1].argmax())
        walls = []
        for s in range(16):
            t0 = time.time()
            lz = model(mx.array([[cur]]), cache=cache)[0, -1]
            cur = int(lz.argmax())
            walls.append(time.time() - t0)
        st = walls[2:]
        longs.append({"context": L, "prefill_s": t_pre, "prefill_tok_per_s": L / t_pre,
                      "decode_median_s": float(np.median(st)), "decode_p90_s": float(np.percentile(st, 90)),
                      "decode_tok_per_s": 1.0 / float(np.median(st)),
                      "peak_mem_bytes": int(mx.get_peak_memory())})
        print(f"[qual] ctx {L}: prefill {t_pre:.0f}s ({L/t_pre:.1f} tok/s), decode median "
              f"{np.median(st)*1e3:.0f} ms, peak {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
        del cache, lp
        if t_pre > budget:
            print(f"[qual] prefill wall {t_pre:.0f}s over budget {budget:.0f}s; longer contexts skipped", flush=True)
            break
    summary["long_context"] = longs
    summary["wall_s"] = time.time() - t_start
    summary["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"[qual] wrote {out}: {n} positions, argmax identical {ident}/{n}, top16 max abs "
          f"{summary['top16_abs_err'].get('max', 0):.2e}, replay k4 metal {replay_metal[4]:.4f} "
          f"cpu {replay_cpu[4]:.4f}, wall {summary['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

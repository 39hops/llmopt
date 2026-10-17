"""Arm C on Metal: the mlx-lm Qwen3.5 architecture with every w4 / s16
matrix served by the fused decode-and-multiply kernels (llmopt.lab.qmetal),
raw tensors decoded once, no decoded weight matrix ever materialized.

Separate from the frozen CPU reference (scratch/qwen_e0_spec_census.py,
transformers streamed decode) which stays the oracle. This driver:
  1. builds the model from ~/qwen_vendor/config.json (text config);
  2. loads arm C (qualified against its frozen digest chain) into
     QLinear modules (resident payload parts) and raw mx arrays, with
     the mlx-lm sanitizer's conventions (conv1d [D,1,K] -> [D,K,1];
     zero-centered norm weights + 1 for the five HF norm families);
  3. PREFILL parity: logits over the oracle dump's token sequence
     against logs/qmetal/oracle_<arm>_p<i>_s<n>.npz (token identity at
     every position, max abs and relative logit error);
  4. DECODE parity and timing: prompt prefill into a fresh cache, then
     one token at a time; each step's logits are compared with the
     prefill logits at that position (cache and DeltaNet/conv state
     correctness) and the greedy token with the oracle's; wall per
     token, peak unified memory.
Receipts under logs/qmetal/tower_<arm>_<commit>.json (refuse-if-exists).
Env: ART_DIR, VENDOR_DIR, ORACLE (npz path), STEPS (decode steps, default
all generated tokens in the oracle), LAYERS (debug: truncate the tower).
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

NORM_SHIFT_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                       "model.norm.weight", ".q_norm.weight", ".k_norm.weight")


def main() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import dec_raw
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows

    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    oracle_path = os.environ.get("ORACLE", f"logs/qmetal/oracle_{arm}_p0_s8.npz")
    out = os.environ.get("OUT", f"logs/qmetal/tower_{arm}_{commit}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    n_layers_env = os.environ.get("LAYERS")

    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    print(f"[tower] qualified {arm}: {q['report']['rung0']}", flush=True)
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    cfg = json.load(open(os.path.join(vdir, "config.json")))
    args = qwen3_5.ModelArgs.from_dict(cfg)
    if n_layers_env:
        args.text_config["num_hidden_layers"] = int(n_layers_env)
        args.text_config["layer_types"] = args.text_config["layer_types"][:int(n_layers_env)]
    model = qwen3_5.Model(args)
    tm = model.language_model.model
    print(f"[tower] built {len(tm.layers)} layers; dtype float32", flush=True)

    class QLinear(nn.Module):
        """y = x W^T through the resident fused kernel; no weight array."""
        def __init__(self, res):
            super().__init__()
            self.res = res
            self.out_features, self.in_features = res.shape

        def __call__(self, x):
            lead = x.shape[:-1]
            y = self.res.gemm(x.reshape(-1, x.shape[-1]).astype(mx.float32))
            return y.reshape(*lead, self.out_features)

    class QEmbedding(nn.Module):
        """Token rows decoded on demand with the canonical row decoder."""
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

    t0 = time.time()
    n_fused = n_raw = 0
    bytes_resident = 0
    for key, e in MAN.items():
        if not isinstance(e, dict) or e["codec"] == "excluded":
            continue
        if key.startswith("model.visual") or ".mtp." in key or key.startswith("mtp."):
            continue
        k = key.replace("model.language_model.", "language_model.model.") if key.startswith("model.language_model.") \
            else "language_model." + key
        if ".layers." in k and n_layers_env:
            li = int(k.split(".layers.")[1].split(".")[0])
            if li >= int(n_layers_env):
                continue
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
    print(f"[tower] loaded: {n_fused} fused matrices, {n_raw} raw tensors, "
          f"{bytes_resident/1e9:.2f} GB payload resident, {time.time()-t0:.0f}s", flush=True)

    orc = np.load(oracle_path)
    ids = [int(t) for t in orc["ids"]]
    plen = int(orc["prompt_len"])
    ref = orc["logits"].astype(np.float64)
    steps = int(os.environ.get("STEPS", str(len(ids) - plen)))
    receipt = {"arm": arm, "commit": commit, "chain_report": q["report"], "oracle": oracle_path,
               "n_layers": len(tm.layers), "fused": n_fused, "raw": n_raw,
               "payload_resident_bytes": bytes_resident, "kernel_version": qmetal.VERSION}

    # ---- optional per-layer capture for localization (DUMP_HIDDEN=path) ----
    dump_hidden = os.environ.get("DUMP_HIDDEN")
    captured: dict = {}
    if dump_hidden:
        if os.path.exists(dump_hidden):
            raise SystemExit(f"REFUSING: {dump_hidden} exists")

        def wrap(i, layer):
            orig = layer.__call__

            def call(x, *a, **k):
                y = orig(x, *a, **k)
                captured[f"layer_{i}"] = y
                return y
            layer.__call__ = call
        for i, ly in enumerate(tm.layers):
            wrap(i, ly)
        orig_norm = tm.norm.__call__
        def norm_call(x, *a, **k):
            y = orig_norm(x, *a, **k)
            captured["final_norm"] = y
            return y
        tm.norm.__call__ = norm_call
        orig_emb = tm.embed_tokens.__call__
        def emb_call(x, *a, **k):
            y = orig_emb(x, *a, **k)
            captured["embed"] = y
            return y
        tm.embed_tokens.__call__ = emb_call

    # ---- prefill parity ----
    mx.reset_peak_memory()
    t0 = time.time()
    lg = model(mx.array([ids]))
    mx.eval(lg)
    t_pre = time.time() - t0
    if dump_hidden:
        mx.eval(*captured.values())
        np.savez(dump_hidden, **{k: np.asarray(v[0], dtype=np.float32) for k, v in captured.items()},
                 logits=np.asarray(lg[0], dtype=np.float32))
        print(f"[tower] wrote hidden states to {dump_hidden}", flush=True)
    got = np.asarray(lg[0], dtype=np.float64)
    if n_layers_env is None:
        tok_ok = (got.argmax(1) == ref.argmax(1))
        den = np.abs(ref).max(axis=1)
        rel = np.abs(got - ref).max(axis=1) / den
        receipt["prefill"] = {"tokens": len(ids), "wall_s": t_pre, "tok_per_s": len(ids) / t_pre,
                              "argmax_identical": int(tok_ok.sum()), "positions": len(ids),
                              "max_abs_err": float(np.abs(got - ref).max()), "max_rel_err": float(rel.max()),
                              "median_rel_err": float(np.median(rel)),
                              "peak_mem_bytes": int(mx.get_peak_memory())}
        print(f"[tower] prefill {len(ids)} tokens {t_pre:.1f}s; argmax identical {tok_ok.sum()}/{len(ids)}; "
              f"max abs err {np.abs(got-ref).max():.3e}, max rel {rel.max():.2e}, peak {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
        mism = np.nonzero(~tok_ok)[0]
        for p in mism[:8]:
            m_ref = np.sort(ref[p])[-1] - np.sort(ref[p])[-2]
            print(f"   mismatch at {p}: oracle argmax {ref[p].argmax()} margin {m_ref:.4f}, metal argmax {got[p].argmax()}", flush=True)
    else:
        receipt["prefill"] = {"tokens": len(ids), "wall_s": t_pre, "truncated_layers": True}
        print(f"[tower] prefill {len(ids)} tokens {t_pre:.1f}s (truncated tower, no parity)", flush=True)

    # ---- decode parity and timing ----
    cache = model.make_cache()
    mx.reset_peak_memory()
    t0 = time.time()
    lp = model(mx.array([ids[:plen]]), cache=cache)
    mx.eval(lp)
    t_prompt = time.time() - t0
    walls = []
    dec_ok = 0
    dec_vs_prefill = []
    cur = ids[plen - 1]
    emitted = []
    for s in range(steps):
        pos = plen - 1 + s
        t0 = time.time()
        if s == 0:
            lz = lp[0, -1]
        else:
            lz = model(mx.array([[cur]]), cache=cache)[0, -1]
        mx.eval(lz)
        walls.append(time.time() - t0)
        g = np.asarray(lz, dtype=np.float64)
        dec_vs_prefill.append(float(np.abs(g - got[pos]).max()))
        tok = int(g.argmax())
        emitted.append(tok)
        dec_ok += int(tok == int(ref[pos].argmax()))
        cur = ids[pos + 1] if pos + 1 < len(ids) else tok     # teacher-forced on the recorded tokens
    walls_steady = walls[1:] if len(walls) > 1 else walls
    receipt["decode"] = {"prompt_tokens": plen, "prompt_prefill_s": t_prompt, "steps": steps,
                         "first_step_s": walls[0], "median_step_s": float(np.median(walls_steady)),
                         "p90_step_s": float(np.percentile(walls_steady, 90)),
                         "tok_per_s_steady": 1.0 / float(np.median(walls_steady)),
                         "argmax_matches_oracle": dec_ok,
                         "max_abs_diff_vs_prefill": float(max(dec_vs_prefill)),
                         "peak_mem_bytes": int(mx.get_peak_memory()),
                         "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    print(f"[tower] decode: prompt {plen} tokens {t_prompt:.1f}s; {steps} steps, median {np.median(walls_steady)*1e3:.0f} ms "
          f"({1/np.median(walls_steady):.2f} tok/s), argmax == oracle {dec_ok}/{steps}, "
          f"max |decode - prefill| {max(dec_vs_prefill):.3e}, peak {mx.get_peak_memory()/1e9:.2f} GB, "
          f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[tower] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

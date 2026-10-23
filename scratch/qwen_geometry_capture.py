"""QWEN-LOSSY-GEOMETRY-0, stage 1: the frozen inputs of the local
representation study. For the selected layer-22 matrices (mlp.gate_proj
[17408, 5120] and mlp.down_proj [5120, 17408], both W4; linear_attn.
in_proj_qkv [10240, 5120], S16; gate_proj and up_proj share an input and
up_proj is reported through gate_proj's captures) this driver:
  1. verifies vendor shard 7's pin (sha256 and size) and reads the
     ORIGINAL BF16 tensors from it (exact uint16 patterns widened to
     float32, which is exact for BF16);
  2. reads the arm C payloads of the same tensors from the artifact under
     the digest chain, checks the manifest entries (codec, shape, length
     formula), the exponent-scale range and the canonical decoder output
     (llmopt.lab.qcodec.decode_entry) against the original (relative
     Frobenius error of arm C, the control's figure);
  3. builds the qualified tower (generation 5 default) and runs the 24
     teacher rollouts (prompt plus recorded continuation, teacher-forced)
     through the serving prefill path (prefill_chunked, chunk 256, GEMM),
     recording the input of each selected module at EVERY position;
     prompts are split by index into a development set (0 to 11) and a
     blocked evaluation set (12 to 23) before any candidate is built;
  4. writes the captures as float32 .npy files under DUMP (machine-local,
     untracked; sha256 in the receipt) and the receipt
     logs/qwen_geometry/capture.json (refuse-if-exists).
Nothing in the artifact, the shards or the kernels is modified. Env:
ART_DIR, VENDOR_DIR, SHARD_DIR, OUT_DIR (logs/qwen_geometry), SMOKE=1
(2 prompts, 32 tokens each, OUT_DIR logs/qwen_geometry/smoke).
"""
from __future__ import annotations

import gc
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
from bf16_entropy_census import SHARDS, safetensors_header, tensor_view  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
OUT_DIR = "logs/qwen_geometry/smoke" if SMOKE else os.environ.get("OUT_DIR", "logs/qwen_geometry")
DUMP = os.path.join(OUT_DIR, "dump")
SHARD_DIR = os.path.expanduser(os.environ.get("SHARD_DIR", "~/qwen_vendor_shards/1d4bf0f2"))
SHARD = "model-00007-of-00018.safetensors"
LAYER = 22
TENSORS = {  # manifest name -> module path in the mlx-lm tower
    f"model.language_model.layers.{LAYER}.mlp.gate_proj.weight": f"language_model.model.layers.{LAYER}.mlp.gate_proj",
    f"model.language_model.layers.{LAYER}.mlp.up_proj.weight": f"language_model.model.layers.{LAYER}.mlp.up_proj",
    f"model.language_model.layers.{LAYER}.mlp.down_proj.weight": f"language_model.model.layers.{LAYER}.mlp.down_proj",
    f"model.language_model.layers.{LAYER}.linear_attn.in_proj_qkv.weight": f"language_model.model.layers.{LAYER}.linear_attn.in_proj_qkv",
}
CAPTURE = [f"language_model.model.layers.{LAYER}.mlp.gate_proj", f"language_model.model.layers.{LAYER}.mlp.down_proj",
           f"language_model.model.layers.{LAYER}.linear_attn.in_proj_qkv"]
DEV_PROMPTS = list(range(0, 12))
EVAL_PROMPTS = list(range(12, 24))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qprefill
    from llmopt.lab.hash import require_pinned_file
    from llmopt.lab.qcodec import decode_entry, expected_len
    from qwen_metal_qual import build_metal_model
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    out = os.path.join(OUT_DIR, "capture.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    os.makedirs(DUMP, exist_ok=True)
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    chain = "logs/qwenwhole/artifact_digest_C.txt"
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    path = os.path.join(SHARD_DIR, SHARD)
    pin = require_pinned_file(path, SHARDS[SHARD]["sha256"], SHARDS[SHARD]["bytes"])
    hdr, base = safetensors_header(path)
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "layer": LAYER, "shard": SHARD, "pin_check": pin,
               "chain_rung0": q["report"]["rung0"], "arm": os.path.basename(art.rstrip("/")),
               "dev_prompts": DEV_PROMPTS, "eval_prompts": EVAL_PROMPTS, "tensors": {}, "captures": {}}
    # ---- originals and arm C payloads ----
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    for name, mod in TENSORS.items():
        e = MAN[name]
        mm = tensor_view(path, hdr, base, name)
        u16 = np.ascontiguousarray(np.asarray(mm))
        del mm
        if list(hdr[name]["shape"]) != list(e["shape"]):
            raise SystemExit(f"REFUSING: shape mismatch {name}: vendor {hdr[name]['shape']} manifest {e['shape']}")
        W = bf16_to_f32(u16).reshape(e["shape"])
        buf = payload(e)
        if len(buf) != expected_len(e["codec"], e["shape"]) or len(buf) != e["len"]:
            raise SystemExit(f"REFUSING: payload length {len(buf)} != formula {expected_len(e['codec'], e['shape'])} / manifest {e['len']}")
        nb = int(np.prod(e["shape"])) // 128
        exps = np.frombuffer(buf, np.uint8, nb, 0)
        t0 = time.perf_counter()
        D = decode_entry(buf, e).astype(np.float32)
        t_dec = time.perf_counter() - t0
        d64 = D.astype(np.float64)
        w64 = W.astype(np.float64)
        rel = float(np.linalg.norm(d64 - w64) / np.linalg.norm(w64))
        orig_path = os.path.join(DUMP, f"orig_{name.split('layers.')[1].replace('.weight', '')}.npy")
        np.save(orig_path, u16.reshape(e["shape"]))
        receipt["tensors"][name] = {
            "module": mod, "codec": e["codec"], "shape": e["shape"], "meta": {k: v for k, v in e.items() if k not in ("codec", "shape", "shard", "off", "len")},
            "payload_bytes": len(buf), "payload_sha256": hashlib.sha256(buf).hexdigest(),
            "orig_bf16_sha256": hashlib.sha256(u16.tobytes()).hexdigest(), "orig_dump": orig_path, "orig_dump_sha256": sha_file(orig_path),
            "exp_range": [int(exps.min()), int(exps.max())], "exp_distinct": int(np.unique(exps).size),
            "bits_per_weight_payload": 8.0 * len(buf) / W.size,
            "orig_frobenius": float(np.linalg.norm(w64)), "orig_max_abs": float(np.abs(W).max()),
            "armC_relative_frobenius_error": rel, "canonical_decode_s": t_dec,
            "decoded_finite": bool(np.isfinite(D).all())}
        print(f"[capture] {name.split('layers.')[1]} {e['codec']} {e['shape']} rel Frobenius (arm C) {rel:.4f} exps {receipt['tensors'][name]['exp_range']} bits/w {receipt['tensors'][name]['bits_per_weight_payload']:.4f}", flush=True)
        del W, D, d64, w64, u16
        gc.collect()
    # ---- activations ----
    tdir = "logs/qwenteacher_v2"
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    model, info = build_metal_model(art, vdir)
    cfg = qprefill.ServingConfig()
    qprefill.configure_serving_memory(cfg)
    mods = {name: mod for name, mod in model.named_modules() if type(mod).__name__ == "QLinear" and name in CAPTURE}
    if set(mods) != set(CAPTURE):
        raise SystemExit(f"REFUSING: modules not found: {set(CAPTURE) - set(mods)}")
    cls = type(next(iter(mods.values())))
    orig_call = cls.__call__
    want = {id(m): n for n, m in mods.items()}
    buffers: dict = {n: [] for n in CAPTURE}

    def rec(self, x):
        if id(self) in want:
            buffers[want[id(self)]].append(np.array(np.asarray(x, dtype=np.float32), copy=True).reshape(-1, x.shape[-1]))
        return orig_call(self, x)
    cls.__call__ = rec
    prompts = list(range(2)) if SMOKE else list(range(24))
    per_prompt = []
    t_all = time.perf_counter()
    for i in prompts:
        prompt = [int(t) for t in input_ids[i][mask[i] == 1]]
        gen = [int(t) for t in recs[i]["tokens"][:ro["gen_lengths"][i]]]
        ids = prompt + gen
        if SMOKE:
            ids = ids[:32]
        for n in CAPTURE:
            buffers[n].clear()
        cache = model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(model, ids, cache, chunk=cfg.chunk, head="final", matmul=cfg.matmul)
        mx.eval(lg)
        wall = time.perf_counter() - t0
        row = {"prompt": i, "record_id": recs[i]["id"], "prompt_len": len(prompt), "n_positions": len(ids),
               "split": "dev" if i in DEV_PROMPTS else "eval", "prefill_s": wall, "files": {}}
        for n in CAPTURE:
            X = np.concatenate(buffers[n], axis=0)
            if X.shape[0] != len(ids):
                raise SystemExit(f"REFUSING: captured {X.shape[0]} positions for {n}, expected {len(ids)}")
            fn = os.path.join(DUMP, f"x_{n.split('layers.')[1]}_p{i:02d}.npy")
            np.save(fn, X)
            row["files"][n] = {"path": fn, "shape": list(X.shape), "sha256": sha_file(fn),
                               "rms": float(np.sqrt((X.astype(np.float64) ** 2).mean())), "max_abs": float(np.abs(X).max())}
        per_prompt.append(row)
        print(f"[capture] prompt {i} ({row['split']}) {len(ids)} positions in {wall:.1f}s", flush=True)
        del cache, lg
    cls.__call__ = orig_call
    receipt["captures"] = {"per_prompt": per_prompt, "wall_s": time.perf_counter() - t_all,
                           "prefill_path": "qprefill.prefill_chunked(chunk 256, head final, matmul gemm) under the 2 GiB cache limit",
                           "tower_kernel_version": info["kernel_version"], "chain_sha256": info["chain_sha256"],
                           "positions_dev": sum(r["n_positions"] for r in per_prompt if r["split"] == "dev"),
                           "positions_eval": sum(r["n_positions"] for r in per_prompt if r["split"] == "eval"),
                           "teacher_rollout_sha256": ro["sha256"]}
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[capture] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

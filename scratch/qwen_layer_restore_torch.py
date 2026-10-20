"""QWEN-LAYER-RESTORE-0, the torch half of the one-layer oracle.

Runs with .venv_teacher (transformers 5.15.0, torch 2.13, CPU float32):
the vendor PyTorch Qwen3_5DecoderLayer for layers 22 and 23, holding the
same weights the mlx side held (WEIGHTS=armc: decoded arm C tensors via
llmopt.lab.qcodec; WEIGHTS=vendor: the BF16 shard widened; smoke labels
the arm C tensors "standin"), on the frozen inputs the capture stage
dumped (the compressed tower's residual stream entering each layer),
one sequence at a time from a zero state (no cache object; the chunked
gated delta rule and eager attention with an additive causal mask). The
torch side never applies the mlx-lm load conventions: vendor layout
(zero-centred norms, conv1d [D, 1, K]) is what this implementation
expects, which is the point of the cross-check.

Compares its outputs to the mlx one-shot outputs of oracle_mlx_<weights>
with llmopt.lab.layer_restore.boundary_stats under the registered
tolerance (relative L2 at most 1e-4, max abs over the reference row RMS
at most 1e-3, per layer, all positions pooled). Receipt
logs/qwen_layer_restore[_smoke]/oracle_torch_<weights>.json
(refuse-if-exists). Env: ART_DIR, VENDOR_DIR, VENDOR_SHARD_DIR, OUT,
SMOKE, WEIGHTS, THREADS.
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

from llmopt.lab import layer_restore as lr  # noqa: E402
from llmopt.lab import logit_stats as ls  # noqa: E402
from llmopt.lab.strict_json import write_strict  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
ART_DIR = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
VENDOR_DIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SHARD = "model-00007-of-00018.safetensors"
SHARD_BYTES = 2108759344
SHARD_SHA256 = "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
VENDOR_SHARD_DIR = os.path.expanduser(os.environ.get("VENDOR_SHARD_DIR", f"~/qwen_vendor_shards/{REVISION[:8]}"))
LAYERS = (22, 23)
OUT = "logs/qwen_layer_restore_smoke" if SMOKE else os.environ.get("OUT", "logs/qwen_layer_restore")
ORACLE_REL_L2 = 1e-4
ORACLE_MAX_ABS_OVER_RMS = 1e-3


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
    from llmopt.lab.qcodec import decode_entry
    torch.set_num_threads(int(os.environ.get("THREADS", max(1, os.cpu_count() or 1))))
    src = os.environ.get("WEIGHTS", "armc")
    label = "standin" if (SMOKE and src == "vendor") else src
    if label == "standin":
        src = "armc"
    rpath = os.path.join(OUT, f"oracle_torch_{label}.json")
    if os.path.exists(rpath):
        raise SystemExit(f"REFUSING: {rpath} exists")
    if not SMOKE and git_dirty():
        raise SystemExit("REFUSING: dirty tree")
    cap = json.load(open(os.path.join(OUT, "capture.json")))
    mlx_rec = json.load(open(os.path.join(OUT, f"oracle_mlx_{label}.json")))
    if not cap.get("ok") or not mlx_rec.get("ok"):
        raise SystemExit("REFUSING: capture or oracle_mlx receipt is not ok")
    rec = {"mode": "oracle_torch", "smoke": SMOKE, "weights": label, "code_commit": git_head(), "tree_dirty": git_dirty(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "driver": "scratch/qwen_layer_restore_torch.py",
           "python": sys.version.split()[0], "torch": torch.__version__,
           "transformers": __import__("transformers").__version__, "threads": torch.get_num_threads(), "layers": list(LAYERS),
           "repo": "Qwen/Qwen3.8-27B", "revision": REVISION, "shard": SHARD, "art_role": "ART_DIR (arm C)",
           "vendor_shard_role": "VENDOR_SHARD_DIR (outside the repository)"}
    man = json.load(open(os.path.join(ART_DIR, "manifest.json")))
    idx = json.load(open(os.path.join(VENDOR_DIR, "model.safetensors.index.json")))["weight_map"]
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(ART_DIR, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    def tensor(name):
        e = man[name]
        if src == "armc":
            return np.ascontiguousarray(decode_entry(payload(e), e)).astype(np.float32)
        p = os.path.join(VENDOR_SHARD_DIR, SHARD)
        return ls.read_bf16_tensor(p, name, expect_shape=tuple(e["shape"]))

    if src == "vendor":
        p = os.path.join(VENDOR_SHARD_DIR, SHARD)
        if not os.path.exists(p) or os.path.getsize(p) != SHARD_BYTES or sha_file(p) != SHARD_SHA256:
            raise SystemExit("REFUSING: vendor shard missing or its sha differs from the pin")
        rec["vendor_shard_check"] = "sha re-verified at stage start"
    cfg = AutoConfig.from_pretrained(VENDOR_DIR)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    rec["config"] = {"layer_types": [tc.layer_types[L] for L in LAYERS], "rope_parameters": dict(tc.rope_parameters),
                     "rms_norm_eps": tc.rms_norm_eps, "head_dim": tc.head_dim, "attn_implementation": "eager"}
    t0 = time.time()
    layers = {}
    for L in LAYERS:
        names = sorted(lr.layer_tensor_map(idx, L))
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(tc, L)
        want = {n for n, _ in layer.named_parameters()}
        got = {n.split(f"layers.{L}.", 1)[1]: n for n in names}
        if set(got) != want:
            raise SystemExit(f"REFUSING: layer {L} parameter set differs: missing {want - set(got)}, extra {set(got) - want}")
        for rel, full in got.items():
            mod = layer.get_submodule(rel.rsplit(".", 1)[0]) if "." in rel else layer
            leaf = rel.rsplit(".", 1)[1]
            arr = tensor(full)
            if tuple(arr.shape) != tuple(mod._parameters[leaf].shape):
                raise SystemExit(f"REFUSING: {full} shape {arr.shape} != module {tuple(mod._parameters[leaf].shape)}")
            mod._parameters[leaf] = torch.nn.Parameter(torch.from_numpy(arr), requires_grad=False)
        layer.eval()
        layers[L] = layer
    rec["build_wall_s"] = time.time() - t0
    rot = Qwen3_5TextRotaryEmbedding(tc)
    inputs = {22: "in22", 23: "out22"}
    per = {}
    pooled = {L: [] for L in LAYERS}
    pooled_fused = {L: [] for L in LAYERS}
    with torch.no_grad():
        for key in cap["dumps"]:
            cname, b = key.split(":")
            if b != "in22":
                continue
            for L in LAYERS:
                xin = np.load(os.path.join(cap["dump_dir"], cap["dumps"][f"{cname}:{inputs[L]}"]["file"]))
                y_mlx = np.load(os.path.join(mlx_rec["dump_dir"], mlx_rec["dumps"][f"{cname}:out{L}"]["file"]))
                fused = np.load(os.path.join(cap["dump_dir"], cap["dumps"][f"{cname}:out{L}"]["file"]))
                n = xin.shape[0]
                h = torch.from_numpy(xin)[None]
                pos = torch.arange(n).view(1, 1, -1).expand(4, 1, -1)
                text_pos, pos3 = pos[0], pos[1:]
                pe = rot(h, pos3)
                mask = None if layers[L].block_type == "linear_attention" else torch.from_numpy(lr.causal_additive_mask(n))[None, None]
                t1 = time.time()
                y = layers[L](h, position_embeddings=pe, attention_mask=mask, position_ids=text_pos, past_key_values=None)
                y = np.asarray(y[0].float().numpy(), dtype=np.float32)
                st = lr.boundary_stats(y_mlx, y)
                per[f"{cname}:out{L}"] = {"n": int(n), "wall_s": time.time() - t1, "torch_vs_mlx": st, "torch_vs_fused_tower": lr.boundary_stats(fused, y)}
                pooled[L].append((y_mlx, y))
                pooled_fused[L].append((fused, y))
                print(f"[lr] oracle_torch {label} {cname} layer {L}: torch-vs-mlx rel {st['rel_l2']:.2e} max/rms {st['max_abs_over_row_rms']:.2e} "
                      f"({time.time() - t1:.1f} s)", flush=True)
    summary = {}
    ok = True
    for L in LAYERS:
        a = np.concatenate([p[0] for p in pooled[L]]); b = np.concatenate([p[1] for p in pooled[L]])
        st = lr.boundary_stats(a, b)
        fa = np.concatenate([p[0] for p in pooled_fused[L]]); fb = np.concatenate([p[1] for p in pooled_fused[L]])
        summary[str(L)] = {"torch_vs_mlx": st, "torch_vs_fused_tower": lr.boundary_stats(fa, fb),
                           "within_tolerance": st["rel_l2"] <= ORACLE_REL_L2 and st["max_abs_over_row_rms"] <= ORACLE_MAX_ABS_OVER_RMS}
        ok = ok and summary[str(L)]["within_tolerance"]
    rec.update({"inputs": {str(L): inputs[L] for L in LAYERS}, "per_context": per, "pooled": summary,
                "tolerance": {"rel_l2": ORACLE_REL_L2, "max_abs_over_row_rms": ORACLE_MAX_ABS_OVER_RMS},
                "mlx_receipt": {"path": os.path.join(OUT, f"oracle_mlx_{label}.json"), "code_commit": mlx_rec["code_commit"]},
                "ok": ok, "wall_s": time.time() - t0})
    write_strict(rec, rpath)
    print(f"[lr] oracle_torch {label}: {json.dumps({L: {'rel_l2': s['torch_vs_mlx']['rel_l2'], 'max_abs_over_row_rms': s['torch_vs_mlx']['max_abs_over_row_rms'], 'ok': s['within_tolerance']} for L, s in summary.items()})}; wrote {rpath}", flush=True)
    if not ok:
        raise SystemExit("oracle_torch: outside the registered tolerance")


if __name__ == "__main__":
    main()

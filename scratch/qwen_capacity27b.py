"""QWEN-CAPACITY-METER-1: the 27B cell — M = span_bits - code_entropy
per family x projection on the PINNED vendor checkpoint, streaming.

Registered corrections carried (PRE-REG QWEN-CAPACITY-METER-1): this
reads the pinned vendor Qwen3.8-27B shards (revision asserted via the
shard-index sha the teacher manifest locked), NOT the 0.5B the old
MODELS=qwen hook loaded; M is an OUTLIER-PRESSURE DIAGNOSTIC, never
an allocator — the deliverable is per-group M for retrodiction
against the measured codec/value pattern. meter() is imported
verbatim from scratch/capacity_meter.py (no fork of frozen math).

    .venv/bin/python scratch/qwen_capacity27b.py

Receipt: logs/qwencapacity/meter27b.json. Desk class, Mac CPU.
"""
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
SHARD_INDEX_SHA = ("77042094076611b69791a610065f28b7013b8c621795"
                   "fa86ddccc8bac7d1b9df")   # teacher manifest lock
MAX_ROWS = int(os.environ.get("MAX_ROWS", "1024"))
OUT = "logs/qwencapacity"

_spec = importlib.util.spec_from_file_location(
    "capacity_meter", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scratch/capacity_meter.py"))
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


GROUPS = [
    ("io:embed", r"language_model\.embed_tokens\.weight$"),
    ("io:lm_head", r"^lm_head\.weight$"),
    ("ffn:gate", r"\.mlp\.gate_proj\.weight$"),
    ("ffn:up", r"\.mlp\.up_proj\.weight$"),
    ("ffn:down", r"\.mlp\.down_proj\.weight$"),
    ("full_attn:q", r"\.self_attn\.q_proj\.weight$"),
    ("full_attn:k", r"\.self_attn\.k_proj\.weight$"),
    ("full_attn:v", r"\.self_attn\.v_proj\.weight$"),
    ("full_attn:o", r"\.self_attn\.o_proj\.weight$"),
    ("linear_attn:qkv", r"\.linear_attn\.in_proj_qkv\.weight$"),
    ("linear_attn:z", r"\.linear_attn\.in_proj_z\.weight$"),
    ("linear_attn:out", r"\.linear_attn\.out_proj\.weight$"),
]
LBAND = {("linear_attn", i): b for b, band in enumerate(
    ([*range(0, 21)], [*range(21, 42)], [*range(42, 63)]))
    for i in band}


def group_of(name):
    for g, pat in GROUPS:
        if re.search(pat, name):
            return g
    return None


def main():
    from safetensors import safe_open
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "meter27b.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    idx_path = os.path.join(VDIR, "model.safetensors.index.json")
    got = hashlib.sha256(open(idx_path, "rb").read()).hexdigest()
    if got != SHARD_INDEX_SHA:
        raise SystemExit(f"REFUSING: shard index sha {got[:12]} != "
                         f"locked {SHARD_INDEX_SHA[:12]}")
    wmap = json.load(open(idx_path))["weight_map"]
    agg = {}
    t0 = time.time()
    handles = {}
    n_done = 0
    for name, sh in sorted(wmap.items()):
        g = group_of(name)
        if g is None:
            continue
        if sh not in handles:
            handles[sh] = safe_open(os.path.join(VDIR, sh),
                                    framework="pt", device="cpu")
        W = handles[sh].get_tensor(name)
        R = W.shape[0]
        rng = random.Random(f"qcap27b-{name}")
        rows = (list(range(R)) if R <= MAX_ROWS
                else sorted(rng.sample(range(R), MAX_ROWS)))
        m, k = cm.meter(W[rows].float())
        n = len(rows) * W.shape[1]
        keys = [g]
        lm = re.search(r"layers\.(\d+)\.", name)
        if g.startswith("linear_attn") and lm:
            band = LBAND[("linear_attn", int(lm.group(1)))]
            keys.append(f"{g}:band{band}")
        for key in keys:
            a = agg.setdefault(key, {"wm": 0.0, "wk": 0.0, "n": 0,
                                     "tensors": 0})
            a["wm"] += m * n
            a["wk"] += k * n
            a["n"] += n
            a["tensors"] += 1
        n_done += 1
        if n_done % 40 == 0:
            print(f"[cap] {n_done} tensors {time.time()-t0:.0f}s",
                  flush=True)
        del W
    groups = {key: {"M_bits": round(a["wm"] / a["n"], 3),
                    "kurtosis": round(a["wk"] / a["n"], 2),
                    "tensors": a["tensors"],
                    "sampled_params_M": round(a["n"] / 1e6, 1)}
              for key, a in sorted(agg.items())}
    fam = {k: v["M_bits"] for k, v in groups.items() if ":band" not in k}
    spread = max(fam.values()) - min(fam.values())
    rcpt = {"gate": "QWEN-CAPACITY-METER-1 (diagnostic, never allocator)",
            "vendor_dir": VDIR, "shard_index_sha256": got,
            "max_rows_sampled": MAX_ROWS,
            "groups": groups,
            "family_M_spread_bits": round(spread, 3),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
            "tree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "-uno"]).decode().strip()),
            "wall_s": round(time.time() - t0, 1)}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    for key, v in groups.items():
        print(f"[cap] {key:24s} M={v['M_bits']:6.3f} bits "
              f"kurt={v['kurtosis']:8.2f} ({v['tensors']} tensors)",
              flush=True)
    print(f"[cap] family spread {spread:.3f} bits -> {rcpt_path} "
          f"wall {rcpt['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

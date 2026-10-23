"""QWEN-MLP-COUPLED-0, stage 1: a NEW, versioned, reproducible evaluation
collection and the frozen inputs of layer 22's MLP captured from the
unchanged arm C model (generation 5 default) on it.

Collection: scratch/evalsets/mlp_coupled_v1.json (hand-written prompts
across chat, math, code, reasoning, knowledge, long_context and
degeneration, plus an eval-only ood category: non-English, structured
data, adversarial repetition), plus ARITH mechanically checkable
arithmetic prompts generated from the collection's seed (the answers go
into the receipt; the study's target is still the BF16 MLP on identical
inputs). The collection's sha256 is recorded. Split: within each
in-domain category, prompts sorted by id take alternate splits (even
index eval, odd index dev); ood is eval-only. The split and the
categories are frozen here, before any candidate is calibrated or
scored.

Two activation distributions per prompt, labelled separately:
  prefill   the input of layer 22's MLP (the post-attention RMSNorm
            output, the shared input of gate_proj and up_proj) at every
            position of the prompt's own tokens through the serving
            prefill path (prefill_chunked, chunk 256, GEMM, head final)
  freerun   the same input at each of FREE (48) greedy decode steps the
            arm C model generates after the prompt with its KV cache (its
            own continuation; the generated ids and decoded text are
            recorded)
Dumps are float32 .npy under DUMP (machine-local, sha256 in the receipt);
receipt logs/qwen_mlp/capture.json (refuse-if-exists). Env: ART_DIR,
VENDOR_DIR, OUT_DIR (logs/qwen_mlp), FREE (48), ARITH (4), SMOKE=1
(first prompt of chat and math, 24 prefill tokens, 4 free tokens, OUT_DIR
logs/qwen_mlp/smoke).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SMOKE = os.environ.get("SMOKE") == "1"
OUT_DIR = "logs/qwen_mlp/smoke" if SMOKE else os.environ.get("OUT_DIR", "logs/qwen_mlp")
DUMP = os.path.join(OUT_DIR, "dump")
COLLECTION = os.path.join(HERE, "evalsets", "mlp_coupled_v1.json")
LAYER = 22
MODULE = f"language_model.model.layers.{LAYER}.mlp.gate_proj"
FREE = 4 if SMOKE else int(os.environ.get("FREE", "48"))
N_ARITH = int(os.environ.get("ARITH", "4"))
IN_DOMAIN = ["chat", "math", "code", "reasoning", "knowledge", "long_context", "degeneration"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def arithmetic_prompts(seed: str, n: int) -> list[dict]:
    r = random.Random(seed)
    out = []
    for i in range(n):
        a, b = r.randint(1000, 9999), r.randint(100, 9999)
        op = r.choice(["+", "-", "*"])
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({"id": f"math-M{i + 1}", "category": "math", "mechanical": True,
                    "text": f"Compute {a} {op} {b}. Give only the number.", "answer": str(ans)})
    return out


def assign_splits(prompts: list[dict]) -> None:
    by_cat: dict = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    for cat, ps in by_cat.items():
        ps.sort(key=lambda p: p["id"])
        for i, p in enumerate(ps):
            if cat == "ood":
                p["split"] = "eval"
            else:
                p["split"] = "eval" if i % 2 == 0 else "dev"


def main() -> None:
    import mlx.core as mx
    from transformers import AutoTokenizer
    from llmopt.lab import qprefill
    from qwen_metal_qual import build_metal_model
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    out = os.path.join(OUT_DIR, "capture.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    os.makedirs(DUMP, exist_ok=True)
    coll = json.load(open(COLLECTION))
    coll_sha = sha_file(COLLECTION)
    prompts = [dict(p) for p in coll["prompts"]] + arithmetic_prompts(coll["rules"]["arith_seed"], N_ARITH)
    assign_splits(prompts)
    if SMOKE:
        prompts = [next(p for p in prompts if p["category"] == "chat"), next(p for p in prompts if p["category"] == "math")]
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    tok = AutoTokenizer.from_pretrained(vdir)
    model, info = build_metal_model(art, vdir)
    cfg = qprefill.ServingConfig()
    qprefill.configure_serving_memory(cfg)
    mod = next((m for n, m in model.named_modules() if n == MODULE), None)
    if mod is None:
        raise SystemExit(f"REFUSING: {MODULE} not found")
    cls = type(mod)
    orig_call = cls.__call__
    buf: list = []
    want = id(mod)

    def rec(self, x):
        if id(self) == want:
            buf.append(np.array(np.asarray(x, dtype=np.float32), copy=True).reshape(-1, x.shape[-1]))
        return orig_call(self, x)
    cls.__call__ = rec
    rows = []
    t_all = time.perf_counter()
    for p in prompts:
        ids = tok(p["text"])["input_ids"]
        if SMOKE:
            ids = ids[:24]
        buf.clear()
        cache = model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(model, ids, cache, chunk=cfg.chunk, head="final", matmul=cfg.matmul)
        mx.eval(lg)
        t_pre = time.perf_counter() - t0
        Xp = np.concatenate(buf, 0)
        if Xp.shape[0] != len(ids):
            raise SystemExit(f"REFUSING: prefill captured {Xp.shape[0]} positions for {len(ids)} tokens")
        buf.clear()
        cur = int(lg[0, -1].argmax())
        gen = []
        t0 = time.perf_counter()
        for _ in range(FREE):
            gen.append(cur)
            lz = model(mx.array([[cur]]), cache=cache)[0, -1]
            cur = int(lz.argmax())
        t_free = time.perf_counter() - t0
        Xf = np.concatenate(buf, 0)
        if Xf.shape[0] != FREE:
            raise SystemExit(f"REFUSING: freerun captured {Xf.shape[0]} positions for {FREE} steps")
        fp = os.path.join(DUMP, f"x_prefill_{p['id']}.npy")
        ff = os.path.join(DUMP, f"x_freerun_{p['id']}.npy")
        np.save(fp, Xp)
        np.save(ff, Xf)
        row = {"id": p["id"], "category": p["category"], "split": p["split"], "ood_kind": p.get("ood_kind"),
               "mechanical": p.get("mechanical", False), "answer": p.get("answer"),
               "text_sha256": hashlib.sha256(p["text"].encode()).hexdigest(), "n_tokens": len(ids),
               "prefill_s": t_pre, "freerun_s": t_free, "generated_ids": gen, "generated_text": tok.decode(gen),
               "files": {"prefill": {"path": fp, "shape": list(Xp.shape), "sha256": sha_file(fp),
                                     "rms": float(np.sqrt((Xp.astype(np.float64) ** 2).mean())), "max_abs": float(np.abs(Xp).max())},
                         "freerun": {"path": ff, "shape": list(Xf.shape), "sha256": sha_file(ff),
                                     "rms": float(np.sqrt((Xf.astype(np.float64) ** 2).mean())), "max_abs": float(np.abs(Xf).max())}}}
        rows.append(row)
        print(f"[mlpcap] {p['id']:10s} {p['category']:13s} {p['split']:4s} {len(ids):4d} tok, free {FREE}: {row['generated_text'][:60]!r}", flush=True)
        del cache, lg
    cls.__call__ = orig_call
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "collection": COLLECTION, "collection_sha256": coll_sha,
               "collection_version": coll["version"], "rules": coll["rules"], "layer": LAYER, "module": MODULE, "free_steps": FREE,
               "tokenizer": type(tok).__name__, "arm": os.path.basename(art.rstrip("/")), "tower_kernel_version": info["kernel_version"],
               "chain_sha256": info["chain_sha256"], "serving": {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes},
               "prompts": rows, "wall_s": time.perf_counter() - t_all,
               "counts": {s: {"prompts": sum(1 for r in rows if r["split"] == s), "prefill_positions": sum(r["n_tokens"] for r in rows if r["split"] == s),
                              "freerun_positions": FREE * sum(1 for r in rows if r["split"] == s)} for s in ("dev", "eval")},
               "categories": sorted({r["category"] for r in rows})}
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[mlpcap] wrote {out}: {receipt['counts']}", flush=True)


if __name__ == "__main__":
    main()

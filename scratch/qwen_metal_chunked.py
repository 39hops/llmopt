"""Chunked prefill for the arm C Metal tower: the prompt is fed to the
qualified model in consecutive chunks through the same cache objects the
one-shot path and cached decode use, every chunk's outputs and cache
state are evaluated before the next chunk is built, and the one-shot
path stays available as the reference.

State ownership (mlx-lm 0.31.3 qwen3_5, inspected 2026-10-18):
  * 48 Gated DeltaNet layers: ArraysCache(size=2); cache[0] holds the
    last conv_kernel_size - 1 (= 3) rows of the layer's conv input and
    is rewritten from the current chunk's tail every call, cache[1] the
    recurrent state [B, H, Dv, Dk] that gated_delta_update reads as
    state_in and returns as state_out; cache.advance(S) only moves the
    padding and length bookkeeping, which is None for a single unpadded
    prompt, so the SSM mask is None.
  * 16 full-attention layers: KVCache with offset; RoPE is applied with
    offset=cache.offset for the chunk, keys and values are appended at
    [offset, offset + S) and the attention runs over all cached keys
    with the mask "causal" (bottom-right aligned in mx.fast.sdpa, so a
    chunk's queries see every earlier key and only their own prefix of
    the chunk). The same objects and offsets serve cached decode.
  * Embedding rows and the fused linears are stateless; the exact HF
    q/k l2norm override is in the GatedDeltaNet __call__ and runs
    identically per chunk.
So a chunk boundary changes nothing in the arithmetic of the DeltaNet
recurrence (sequential per token in the kernel) and changes only the
shapes attention operates on; what can differ is float32 reduction
order inside the attention kernel and inside the fused GEMV tiles (a
chunk of S tokens is multiplied in tiles of 8, the last tile padded; the
tile boundaries move with the chunk start). That is the drift this
driver measures instead of assuming bit identity.

Modes (MODE env):
  boundary  for each chunk size in CHUNKS, lengths chunk - 1, chunk,
            chunk + 1 and 2 * chunk + 1 (real tokens: the E0 teacher
            records concatenated); one-shot against chunked on
            identical ids: all-position logits (max abs, per-position
            relative, argmax identity, margin difference), cache state
            after prefill (conv state, recurrent state, KV up to
            offset, offsets), then DECODE_STEPS teacher-forced steps from
            each cache (logit difference, argmax identity). Plus a fixed
            length (INDEP_LEN) prefilled with every chunk size against
            one-shot, for chunk-size dependence. Plus the CPU oracle
            spot check (logs/qmetal/oracle_C_p0_s8.npz) with chunk 64.
  mem       ONE configuration per process (CTX, CHUNK; CHUNK=0 is
            one-shot): peak unified memory after a reset, active memory,
            process RSS, prefill wall and tokens per second, then
            DECODE_STEPS greedy steps (median, p90). Appends nothing;
            writes logs/qmetal/chunked/mem_ctx<CTX>_chunk<CHUNK>.json
            (refuse-if-exists). A shell loop runs the grid so allocator
            state never carries between configurations.
Env: ART_DIR, VENDOR_DIR, CHUNKS ("64,128,256,512"), INDEP_LEN (1030),
DECODE_STEPS (16), CTX, CHUNK, OUT.
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
sys.path.insert(0, HERE)


def cache_arrays(cache):
    """Every array a cache list holds (for mx.eval and for comparison)."""
    import mlx.core as mx
    out = []
    for c in cache:
        st = c.state
        if isinstance(st, (list, tuple)):
            out += [a for a in st if isinstance(a, mx.array)]
        elif isinstance(st, mx.array):
            out.append(st)
    return out


def prefill_oneshot(model, ids, cache):
    import mlx.core as mx
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, *cache_arrays(cache))
    return lg


def prefill_chunked(model, ids, cache, chunk, keep_logits=False):
    """Feed ids in consecutive chunks; evaluate each chunk's logits and the
    cache state before building the next chunk, so the lazy graph never
    spans two chunks. Returns the last chunk's logits (all positions of
    that chunk) or, with keep_logits, the per-position logits of the
    whole prompt as a host numpy array (moved off the device chunk by
    chunk)."""
    import mlx.core as mx
    host = []
    last = None
    for s in range(0, len(ids), chunk):
        lg = model(mx.array([ids[s:s + chunk]]), cache=cache)
        mx.eval(lg, *cache_arrays(cache))
        if keep_logits:
            host.append(np.asarray(lg[0], dtype=np.float32))
            del lg
        else:
            last = lg
    if keep_logits:
        return np.concatenate(host, axis=0)
    return last


def main() -> None:
    import mlx.core as mx
    from qwen_metal_qual import build_metal_model
    from llmopt.lab import qmetal, spec_reference as sr

    mode = os.environ.get("MODE", "boundary")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    dec_steps = int(os.environ.get("DECODE_STEPS", "16"))
    prov = {"commit": head, "tree_dirty_at_launch": dirty, "mlx_version": mx.__version__,
            "kernel_version": qmetal.VERSION, "mode": mode}

    if mode == "mem":
        ctx = int(os.environ["CTX"])
        chunk = int(os.environ["CHUNK"])
        out = os.environ.get("OUT", f"logs/qmetal/chunked/mem_ctx{ctx}_chunk{chunk}.json")
        if os.path.exists(out):
            raise SystemExit(f"REFUSING: {out} exists")
        model, info = build_metal_model(art, vdir)
        base = [int(t) for t in np.random.default_rng(1).integers(1000, 100000, size=ctx)]
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mx.clear_cache()
        mx.reset_peak_memory()
        cache = model.make_cache()
        t0 = time.perf_counter()
        if chunk == 0:
            lg = prefill_oneshot(model, base, cache)
        else:
            lg = prefill_chunked(model, base, cache, chunk)
        t_pre = time.perf_counter() - t0
        peak = int(mx.get_peak_memory())
        active = int(mx.get_active_memory())
        kv_offset_prefill = int(cache[3].offset)   # layer 3 is the first full-attention layer
        cur = int(lg[0, -1].argmax())
        del lg
        walls = []
        for s in range(dec_steps):
            t0 = time.perf_counter()
            lz = model(mx.array([[cur]]), cache=cache)[0, -1]
            cur = int(lz.argmax())
            walls.append(time.perf_counter() - t0)
        st = walls[2:]
        row = {"provenance": prov, "context": ctx, "chunk": chunk, "resident_payload_bytes": info["payload_resident_bytes"],
               "prefill_s": t_pre, "prefill_tok_per_s": ctx / t_pre,
               "peak_unified_bytes": peak, "active_after_prefill_bytes": active,
               "rss_before_prefill_bytes": int(rss0), "rss_after_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "decode_median_s": float(np.median(st)), "decode_p90_s": float(np.percentile(st, 90)),
               "decode_steps": dec_steps, "kv_offset_after_prefill": kv_offset_prefill,
               "kv_offset_after_decode": int(cache[3].offset)}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(row, open(out, "w"), indent=1)
        print(f"[chunked mem] ctx {ctx} chunk {chunk}: prefill {t_pre:.1f}s ({ctx/t_pre:.2f} tok/s), peak {peak/1e9:.2f} GB, "
              f"active after {active/1e9:.2f} GB, rss {row['rss_after_bytes']/1e9:.2f} GB, decode median {np.median(st)*1e3:.1f} ms", flush=True)
        return

    # ---- boundary mode ----
    chunks = [int(c) for c in os.environ.get("CHUNKS", "64,128,256,512").split(",") if c]
    indep_len = int(os.environ.get("INDEP_LEN", "1030"))
    out = os.environ.get("OUT", f"logs/qmetal/chunked/boundary_{head}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    # real tokens: the E0 teacher records (prompt + rollout), concatenated
    tdir = "logs/qwenteacher_v2"
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    pool: list[int] = []
    for i, rec in enumerate(recs):
        pool += [int(x) for x in input_ids[i][mask[i] == 1]] + [int(x) for x in rec["tokens"][:ro["gen_lengths"][i]]]
    print(f"[chunked] token pool {len(pool)} real tokens from {len(recs)} records", flush=True)
    model, info = build_metal_model(art, vdir)
    v_live = None

    def compare_logits(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = np.abs(a - b)
        den = np.abs(a).max(axis=-1)
        am_a, am_b = a.argmax(-1), b.argmax(-1)
        ma = np.array([sr.top1_and_margin(r)[1] for r in a])
        mb = np.array([sr.top1_and_margin(r)[1] for r in b])
        return {"positions": int(a.shape[0]), "max_abs": float(d.max()), "median_abs": float(np.median(d.max(-1))),
                "max_rel": float((d.max(-1) / den).max()), "argmax_identical": int((am_a == am_b).sum()),
                "margin_abs_diff_max": float(np.abs(ma - mb).max()),
                "min_margin_at_mismatch": (float(ma[am_a != am_b].min()) if (am_a != am_b).any() else None)}

    def compare_caches(c1, c2):
        rows = []
        for li, (a, b) in enumerate(zip(c1, c2)):
            if hasattr(a, "offset"):
                ka, va = a.state
                kb, vb = b.state
                rows.append({"layer": li, "kind": "kv", "offset": [int(a.offset), int(b.offset)],
                             "k_max_abs": float(np.abs(np.asarray(ka, dtype=np.float64) - np.asarray(kb, dtype=np.float64)).max()),
                             "v_max_abs": float(np.abs(np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)).max()),
                             "k_max_abs_ref": float(np.abs(np.asarray(ka)).max())})
            else:
                conv_a, st_a = a.state
                conv_b, st_b = b.state
                rows.append({"layer": li, "kind": "deltanet",
                             "conv_max_abs": float(np.abs(np.asarray(conv_a, dtype=np.float64) - np.asarray(conv_b, dtype=np.float64)).max()),
                             "state_max_abs": float(np.abs(np.asarray(st_a, dtype=np.float64) - np.asarray(st_b, dtype=np.float64)).max()),
                             "state_max_abs_ref": float(np.abs(np.asarray(st_a)).max()),
                             "state_rel": float(np.abs(np.asarray(st_a, dtype=np.float64) - np.asarray(st_b, dtype=np.float64)).max()
                                                / max(float(np.abs(np.asarray(st_a)).max()), 1e-30))})
        kv = [r for r in rows if r["kind"] == "kv"]
        dn = [r for r in rows if r["kind"] == "deltanet"]
        return {"kv_offsets_equal": all(r["offset"][0] == r["offset"][1] for r in kv),
                "kv_max_abs": max(max(r["k_max_abs"], r["v_max_abs"]) for r in kv),
                "conv_max_abs": max(r["conv_max_abs"] for r in dn),
                "state_max_abs": max(r["state_max_abs"] for r in dn),
                "state_max_rel": max(r["state_rel"] for r in dn),
                "per_layer": rows}

    def decode_after(cache, ids_next):
        outs = []
        for t in ids_next:
            lz = model(mx.array([[t]]), cache=cache)[0, -1]
            mx.eval(lz)
            outs.append(np.asarray(lz, dtype=np.float32))
        return np.stack(outs)

    def one_case(length, chunk, label):
        ids = pool[:length]
        nxt = pool[length:length + dec_steps]
        c1 = model.make_cache()
        lg1 = np.asarray(prefill_oneshot(model, ids, c1)[0], dtype=np.float32)
        c2 = model.make_cache()
        lg2 = prefill_chunked(model, ids, c2, chunk, keep_logits=True)
        row = {"label": label, "length": length, "chunk": chunk, "n_chunks": -(-length // chunk),
               "prefill_logits": compare_logits(lg1, lg2),
               "last_position": compare_logits(lg1[-1:], lg2[-1:]),
               "cache": compare_caches(c1, c2)}
        d1 = decode_after(c1, nxt)
        d2 = decode_after(c2, nxt)
        row["decode_after"] = compare_logits(d1, d2)
        row["cache"].pop("per_layer")
        print(f"[chunked] {label}: L={length} chunk={chunk} ({row['n_chunks']} chunks) prefill argmax {row['prefill_logits']['argmax_identical']}/{length} "
              f"max abs {row['prefill_logits']['max_abs']:.2e} rel {row['prefill_logits']['max_rel']:.1e}; state rel {row['cache']['state_max_rel']:.1e} "
              f"kv {row['cache']['kv_max_abs']:.1e} offsets {row['cache']['kv_offsets_equal']}; decode argmax {row['decode_after']['argmax_identical']}/{dec_steps} "
              f"max abs {row['decode_after']['max_abs']:.2e}", flush=True)
        del c1, c2
        mx.clear_cache()
        return row

    results = []
    for ch in chunks:
        for L in (ch - 1, ch, ch + 1, 2 * ch + 1):
            if L > len(pool) - dec_steps:
                continue
            results.append(one_case(L, ch, f"boundary_c{ch}_L{L}"))
    indep = []
    if indep_len <= len(pool) - dec_steps:
        ids = pool[:indep_len]
        nxt = pool[indep_len:indep_len + dec_steps]
        c1 = model.make_cache()
        lg1 = np.asarray(prefill_oneshot(model, ids, c1)[0], dtype=np.float32)
        d1 = decode_after(c1, nxt)
        del c1
        lgs = {}
        for ch in chunks:
            c2 = model.make_cache()
            lg2 = prefill_chunked(model, ids, c2, ch, keep_logits=True)
            d2 = decode_after(c2, nxt)
            lgs[ch] = lg2
            indep.append({"chunk": ch, "n_chunks": -(-indep_len // ch), "vs_oneshot": compare_logits(lg1, lg2),
                          "decode_after_vs_oneshot": compare_logits(d1, d2)})
            print(f"[chunked] indep L={indep_len} chunk={ch}: argmax {indep[-1]['vs_oneshot']['argmax_identical']}/{indep_len} "
                  f"max abs {indep[-1]['vs_oneshot']['max_abs']:.2e}; decode {indep[-1]['decode_after_vs_oneshot']['argmax_identical']}/{dec_steps}", flush=True)
            del c2
            mx.clear_cache()
        pairs = []
        cs = sorted(lgs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pairs.append({"chunks": [cs[i], cs[j]], **compare_logits(lgs[cs[i]], lgs[cs[j]])})
        indep_pairs = pairs
    else:
        indep_pairs = []

    # CPU oracle spot check: prompt 0 (70 tokens + 8) with chunk 64 against the census CPU logits
    oracle = {}
    op = "logs/qmetal/oracle_C_p0_s8.npz"
    if os.path.exists(op):
        orc = np.load(op)
        ids = [int(t) for t in orc["ids"]]
        ref = orc["logits"].astype(np.float64)
        v_live = ref.shape[1]
        c = model.make_cache()
        lg = prefill_chunked(model, ids, c, 64, keep_logits=True)[:, :v_live]
        oracle = {"tokens": len(ids), "chunk": 64, **compare_logits(ref, lg)}
        print(f"[chunked] CPU oracle spot check: argmax {oracle['argmax_identical']}/{len(ids)}, max rel {oracle['max_rel']:.1e}", flush=True)
    receipt = {"provenance": prov, "tower": {k: v for k, v in info.items() if k != "chain_report"},
               "token_pool": len(pool), "decode_steps": dec_steps, "boundary": results,
               "chunk_size_independence": {"length": indep_len, "per_chunk": indep, "pairwise": indep_pairs},
               "cpu_oracle_spot_check": oracle}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[chunked] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

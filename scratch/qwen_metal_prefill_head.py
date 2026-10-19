"""Final-position vocabulary projection for the chunked prefill of the
arm C Metal tower, and the 4,096-token chunk-size comparison.

The qualified path (scratch/qwen_metal_chunked.py) calls the whole model
per chunk, so every prompt position is projected through the 248,320-row
S16 head (an [S, 248320] float32 logits array per chunk, 32 tile-8 head
launches per 256 tokens). Generation needs the logits of the last prompt
position only, while every position must still update the DeltaNet and
attention state. mlx-lm's TextModel is body then head: language_model.model
(embedding, 64 layers with their caches, final norm) returns the normed
hidden states [1, S, 5120] and language_model.lm_head (the fused S16
resident) is applied to them, so the body can run over the whole chunk
and the head over h[:, -1:, :] alone. Nothing upstream of the head
changes: the same body call, the same caches, the same evaluation of
every cache array per chunk.

Arms (HEAD env or per mode):
  full    model(chunk, cache): logits at every position (the qualified path)
  final   body(chunk, cache) then lm_head on the last hidden position
  none    body(chunk, cache) only, hidden states evaluated; the control
          with the head omitted (timing and memory only; decode after it
          uses the full model call)

Modes (MODE env):
  time    CTX tokens (real E0 tokens) in CHUNK chunks, ROUNDS rounds of
          full / final / none in rotating order: prefill wall, tokens per
          second, in-process peak after a reset; the final-position logits
          of full against final (max abs, relative, argmax, margin), the
          cache state of the three arms against each other and DECODE_STEPS
          teacher-forced steps from each cache; then the isolated head
          time (lm_head on [1, 5120] and [8, 5120], synced and pipelined,
          and the 32-launch chunk equivalent).
  mem     ONE configuration per process (CTX, CHUNK, HEAD): peak unified
          memory after a reset, active memory, RSS, prefill wall and
          tokens per second, KV offset after prefill and after
          DECODE_STEPS greedy steps. Writes
          logs/qmetal/prefill_head/mem_ctx<CTX>_chunk<CHUNK>_<HEAD>.json.
  ab      CTX (1,024) real tokens in CHUNK chunks: the fused-only control
          against the decode-once GEMM path (qmetal._Resident.prefill_gemm
          switched per arm; decode steps always on the fused path), with
          the full head (per-position argmax and margins) and the final
          head: final logits, caches after prefill, 16 decode steps, walls.
  ctx4k   CTX (4,096) real tokens: full at chunks 128, 256, 512 and final
          at 256, one process, fresh cache each; per position the argmax
          and the top-1 minus top-2 margin (computed on device per chunk,
          kept on host), the final-position logits, the cache state
          after prefill (conv, recurrent, KV with offsets; snapshotted to
          host before decoding) and again after DECODE_STEPS teacher-forced
          steps, and the decode logits, all compared pairwise. No one-shot
          run at 4,096.
Env: ART_DIR, VENDOR_DIR, CTX, CHUNK, HEAD, ROUNDS (2), DECODE_STEPS (16),
OUT, SMOKE (=1 writes under logs/qmetal/prefill_head_smoke/). Receipts
refuse to overwrite. RSS fields are ru_maxrss, the process peak so far.
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

FA_LAYER = 3   # the first full-attention layer (full_attention_interval 4)


def real_tokens(n: int) -> list[int]:
    """The E0 teacher records (prompt + rollout) concatenated, first n."""
    tdir = "logs/qwenteacher_v2"
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    pool: list[int] = []
    for i, rec in enumerate(recs):
        pool += [int(x) for x in input_ids[i][mask[i] == 1]] + [int(x) for x in rec["tokens"][:ro["gen_lengths"][i]]]
    if len(pool) < n:
        raise SystemExit(f"REFUSING: token pool {len(pool)} < {n}")
    return pool[:n]


def prefill(model, ids, cache, chunk, head="full", stats=None):
    """Chunked prefill with the head applied to every position (full), to the
    last position only (final) or not at all (none). Each chunk's logits (or
    hidden states) and every cache array are evaluated before the next chunk
    is built. With stats (a dict), per-position argmax and top-1 minus top-2
    margin are computed on device per chunk and appended on host. Returns
    the last chunk's logits ([1, S, V] for full, [1, 1, V] for final, None
    for none)."""
    import mlx.core as mx
    from qwen_metal_chunked import cache_arrays
    body = model.language_model.model
    lm = model.language_model.lm_head
    last = None
    for s in range(0, len(ids), chunk):
        h = body(mx.array([ids[s:s + chunk]]), cache)
        if head == "full":
            lg = lm(h)
        elif head == "final":
            lg = lm(h[:, -1:, :])
        elif head == "none":
            lg = None
        else:
            raise ValueError(head)
        mx.eval(lg if lg is not None else h, *cache_arrays(cache))
        if stats is not None and lg is not None:
            m1 = lg.max(-1, keepdims=True)
            m2 = mx.where(lg == m1, -mx.inf, lg).max(-1)
            am = lg.argmax(-1)
            mx.eval(m1, m2, am)
            stats.setdefault("argmax", []).append(np.asarray(am[0]).astype(np.int64))
            stats.setdefault("margin", []).append((np.asarray(m1[0, :, 0], dtype=np.float64) - np.asarray(m2[0], dtype=np.float64)))
        last = lg
        del h
    return last


def compare_logits(a, b):
    from llmopt.lab import spec_reference as sr
    a = np.asarray(a, dtype=np.float64).reshape(-1, np.shape(a)[-1])
    b = np.asarray(b, dtype=np.float64).reshape(-1, np.shape(b)[-1])
    d = np.abs(a - b)
    den = np.abs(a).max(axis=-1)
    am_a, am_b = a.argmax(-1), b.argmax(-1)
    ma = np.array([sr.top1_and_margin(r)[1] for r in a])
    mb = np.array([sr.top1_and_margin(r)[1] for r in b])
    return {"positions": int(a.shape[0]), "max_abs": float(d.max()), "max_rel": float((d.max(-1) / den).max()),
            "bit_identical": bool(d.max() == 0.0), "argmax_identical": int((am_a == am_b).sum()),
            "margin_abs_diff_max": float(np.abs(ma - mb).max()),
            "min_margin_at_mismatch": (float(ma[am_a != am_b].min()) if (am_a != am_b).any() else None)}


def compare_caches(c1, c2):
    kv_k = kv_v = conv = st = st_rel = 0.0
    offsets_equal = True
    offs = []
    for a, b in zip(c1, c2):
        if hasattr(a, "offset"):
            ka, va = a.state
            kb, vb = b.state
            offs.append([int(a.offset), int(b.offset)])
            offsets_equal &= int(a.offset) == int(b.offset)
            kv_k = max(kv_k, float(np.abs(np.asarray(ka, dtype=np.float64) - np.asarray(kb, dtype=np.float64)).max()))
            kv_v = max(kv_v, float(np.abs(np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)).max()))
        else:
            conv_a, st_a = a.state
            conv_b, st_b = b.state
            conv = max(conv, float(np.abs(np.asarray(conv_a, dtype=np.float64) - np.asarray(conv_b, dtype=np.float64)).max()))
            dd = float(np.abs(np.asarray(st_a, dtype=np.float64) - np.asarray(st_b, dtype=np.float64)).max())
            st = max(st, dd)
            st_rel = max(st_rel, dd / max(float(np.abs(np.asarray(st_a)).max()), 1e-30))
    return {"kv_offsets_equal": offsets_equal, "kv_offsets": offs[:1], "kv_k_max_abs": kv_k, "kv_v_max_abs": kv_v,
            "conv_max_abs": conv, "state_max_abs": st, "state_max_rel": st_rel,
            "bit_identical": kv_k == 0.0 and kv_v == 0.0 and conv == 0.0 and st == 0.0}


class _Snap:
    """Host copy of one cache object's state (offset and arrays) so a cache
    can be compared as it was after prefill after decode has moved on."""
    def __init__(self, c):
        st = c.state
        self.state = tuple(np.asarray(a) for a in st) if isinstance(st, (list, tuple)) else np.asarray(st)
        if hasattr(c, "offset"):
            self.offset = int(c.offset)


def snapshot_cache(cache):
    return [_Snap(c) for c in cache]


def decode_after(model, cache, ids_next):
    import mlx.core as mx
    outs = []
    for t in ids_next:
        lz = model(mx.array([[t]]), cache=cache)[0, -1]
        mx.eval(lz)
        outs.append(np.asarray(lz, dtype=np.float32))
    return np.stack(outs)


def main() -> None:
    import mlx.core as mx
    from qwen_metal_qual import build_metal_model
    from llmopt.lab import qmetal

    mode = os.environ.get("MODE", "time")
    head_cmd = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    dec_steps = int(os.environ.get("DECODE_STEPS", "16"))
    import mlx_lm
    prov = {"commit": head_cmd, "tree_dirty_at_launch": dirty, "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__, "device": mx.device_info().get("device_name"),
            "kernel_version": qmetal.VERSION, "mode": mode, "smoke": os.environ.get("SMOKE") == "1",
            "prefill_gemm": bool(qmetal._Resident.prefill_gemm), "gemm_min_t": (qmetal._Resident.gemm_min_t if qmetal._Resident.gemm_min_t is not None else dict(qmetal.GEMM_MIN_T_BY_CODEC)),
            "gemm_slab_bytes": int(qmetal.GEMM_SLAB_BYTES)}
    # SMOKE=1 writes every receipt under its own directory, never on a real path
    rdir = "logs/qmetal/prefill_head_smoke" if prov["smoke"] else "logs/qmetal/prefill_head"
    os.makedirs(rdir, exist_ok=True)

    if mode == "mem":
        ctx = int(os.environ["CTX"])
        chunk = int(os.environ["CHUNK"])
        head = os.environ.get("HEAD", "full")
        out = os.environ.get("OUT", f"{rdir}/mem_ctx{ctx}_chunk{chunk}_{head}.json")
        if os.path.exists(out):
            raise SystemExit(f"REFUSING: {out} exists")
        model, info = build_metal_model(art, vdir)
        if not hasattr(model.language_model, "lm_head"):
            raise SystemExit("REFUSING: tied embeddings; the final-position path assumes a separate lm_head")
        base = [int(t) for t in np.random.default_rng(1).integers(1000, 100000, size=ctx)]
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mx.clear_cache()
        mx.reset_peak_memory()
        cache = model.make_cache()
        t0 = time.perf_counter()
        lg = prefill(model, base, cache, chunk, head)
        t_pre = time.perf_counter() - t0
        peak = int(mx.get_peak_memory())
        active = int(mx.get_active_memory())
        kv_prefill = int(cache[FA_LAYER].offset)
        if lg is None:   # control arm: first token from a full call on the last prompt token is not available; use the
            cur = base[-1]  # last prompt token again as a stand-in so the decode loop measures the same step cost
        else:
            cur = int(lg[0, -1].argmax())
        del lg
        walls = []
        for _ in range(dec_steps):
            t0 = time.perf_counter()
            lz = model(mx.array([[cur]]), cache=cache)[0, -1]
            cur = int(lz.argmax())
            walls.append(time.perf_counter() - t0)
        st = walls[2:]
        row = {"provenance": prov, "context": ctx, "chunk": chunk, "head": head,
               "resident_payload_bytes": info["payload_resident_bytes"],
               "prefill_s": t_pre, "prefill_tok_per_s": ctx / t_pre,
               "peak_unified_bytes": peak, "active_after_prefill_bytes": active,
               "peak_rss_before_prefill_bytes": int(rss0), "peak_rss_after_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "decode_median_s": float(np.median(st)), "decode_p90_s": float(np.percentile(st, 90)),
               "decode_steps": dec_steps, "kv_offset_after_prefill": kv_prefill,
               "kv_offset_after_decode": int(cache[FA_LAYER].offset)}
        json.dump(row, open(out, "w"), indent=1)
        print(f"[head mem] ctx {ctx} chunk {chunk} head {head}: prefill {t_pre:.1f}s ({ctx/t_pre:.2f} tok/s), peak {peak/1e9:.2f} GB, "
              f"active after {active/1e9:.2f} GB, peak rss {row['peak_rss_after_bytes']/1e9:.2f} GB, decode median {np.median(st)*1e3:.1f} ms, "
              f"kv offset {kv_prefill} then {row['kv_offset_after_decode']}", flush=True)
        return

    ctx = int(os.environ.get("CTX", "4096" if mode == "ctx4k" else "1024"))
    chunk = int(os.environ.get("CHUNK", "256"))
    out = os.environ.get("OUT", f"{rdir}/{mode}_{head_cmd}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    ids = real_tokens(ctx + dec_steps)
    nxt = ids[ctx:]
    ids = ids[:ctx]
    model, info = build_metal_model(art, vdir)
    if not hasattr(model.language_model, "lm_head"):
        raise SystemExit("REFUSING: tied embeddings; the final-position path assumes a separate lm_head")
    tower = {k: v for k, v in info.items() if k != "chain_report"}

    if mode == "time":
        rounds = int(os.environ.get("ROUNDS", "2"))
        arms = ["full", "final", "none"]
        walls = {a: [] for a in arms}
        peaks = {a: [] for a in arms}
        kept = {}
        for r in range(rounds):
            order = arms[r % 3:] + arms[:r % 3]
            for a in order:
                mx.clear_cache()
                mx.reset_peak_memory()
                cache = model.make_cache()
                t0 = time.perf_counter()
                lg = prefill(model, ids, cache, chunk, a)
                t = time.perf_counter() - t0
                walls[a].append(t)
                peaks[a].append(int(mx.get_peak_memory()))
                print(f"[head time] round {r} {a}: {t:.1f} s, {ctx/t:.2f} tok/s, peak {peaks[a][-1]/1e9:.2f} GB", flush=True)
                if r == 0:
                    kept[a] = {"cache": cache, "final_logits": (None if lg is None else np.asarray(lg[0, -1], dtype=np.float32))}
                else:
                    del cache
                del lg
        parity = {"final_logits_full_vs_final": compare_logits(kept["full"]["final_logits"], kept["final"]["final_logits"]),
                  "cache_full_vs_final": compare_caches(kept["full"]["cache"], kept["final"]["cache"]),
                  "cache_full_vs_none": compare_caches(kept["full"]["cache"], kept["none"]["cache"])}
        d = {a: decode_after(model, kept[a]["cache"], nxt) for a in arms}
        parity["decode_full_vs_final"] = compare_logits(d["full"], d["final"])
        parity["decode_full_vs_none"] = compare_logits(d["full"], d["none"])
        print(f"[head time] final logits full vs final: {parity['final_logits_full_vs_final']}", flush=True)
        print(f"[head time] decode full vs final: argmax {parity['decode_full_vs_final']['argmax_identical']}/{dec_steps}, "
              f"max abs {parity['decode_full_vs_final']['max_abs']:.2e}; cache bit-identical {parity['cache_full_vs_final']['bit_identical']}", flush=True)
        for a in arms:
            del kept[a]["cache"]
        mx.clear_cache()
        # isolated head time
        lm = model.language_model.lm_head
        rng = np.random.default_rng(0)
        hs = {T: [mx.array(rng.normal(size=(1, T, 5120)).astype(np.float32)) for _ in range(32)] for T in (1, 8)}
        iso = {}
        for T in (1, 8):
            mx.eval(*hs[T])
            for _ in range(4):
                mx.eval(lm(hs[T][0]))
            ts = []
            for _ in range(20):
                t0 = time.perf_counter()
                mx.eval(lm(hs[T][0]))
                ts.append(time.perf_counter() - t0)
            q = 8
            tp = []
            for _ in range(6):
                t0 = time.perf_counter()
                mx.eval(*[lm(hs[T][i]) for i in range(q)])
                tp.append((time.perf_counter() - t0) / q)
            iso[T] = {"synced_median_s": float(np.median(ts)), "pipelined_per_launch_s": float(np.median(tp))}
        iso["chunk_equivalent_s"] = (chunk // 8) * iso[8]["pipelined_per_launch_s"]
        iso["per_prompt_full_head_s_projected"] = (ctx // 8) * iso[8]["pipelined_per_launch_s"]
        iso["per_prompt_final_head_s_projected"] = (ctx // chunk) * iso[1]["pipelined_per_launch_s"]
        med = {a: float(np.median(walls[a])) for a in arms}
        rec = {"provenance": prov, "tower": tower, "context": ctx, "chunk": chunk, "rounds": rounds, "decode_steps": dec_steps,
               "walls_s": walls, "median_s": med, "tok_per_s": {a: ctx / med[a] for a in arms},
               "peak_in_process_bytes": peaks, "head_saving_full_minus_final_s": med["full"] - med["final"],
               "head_share_full_minus_none_s": med["full"] - med["none"], "isolated_head": iso, "parity": parity,
               "note": "in-process peaks share one allocator across arms; the per-process figures are the mem receipts"}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"[head time] medians {med}; full minus final {rec['head_saving_full_minus_final_s']:.2f} s, "
              f"full minus none {rec['head_share_full_minus_none_s']:.2f} s; isolated head T1 {iso[1]['pipelined_per_launch_s']*1e3:.2f} ms, "
              f"T8 {iso[8]['pipelined_per_launch_s']*1e3:.2f} ms per launch; wrote {out}", flush=True)
        return

    if mode == "ctx4k":
        arms = [("full", 128), ("full", 256), ("full", 512), ("final", 256)]
        runs = {}
        for head, ch in arms:
            label = f"{head}_c{ch}"
            mx.clear_cache()
            mx.reset_peak_memory()
            cache = model.make_cache()
            stats: dict = {}
            t0 = time.perf_counter()
            lg = prefill(model, ids, cache, ch, head, stats=stats)
            t = time.perf_counter() - t0
            fin = np.asarray(lg[0, -1], dtype=np.float32)
            del lg
            peak = int(mx.get_peak_memory())
            kv_prefill = int(cache[FA_LAYER].offset)
            # the cache state AFTER PREFILL, snapshotted to host before the decode steps mutate it
            snap = snapshot_cache(cache)
            dec = decode_after(model, cache, nxt)
            runs[label] = {"head": head, "chunk": ch, "n_chunks": -(-ctx // ch), "prefill_s": t, "tok_per_s": ctx / t,
                           "peak_in_process_bytes_after_prefill": peak, "final_logits": fin, "cache": snap,
                           "cache_after_decode": cache,
                           "argmax": (np.concatenate(stats["argmax"]) if "argmax" in stats else None),
                           "margin": (np.concatenate(stats["margin"]) if "margin" in stats else None),
                           "decode": dec, "kv_offset_after_prefill": kv_prefill,
                           "kv_offset_after_decode": int(cache[FA_LAYER].offset)}
            print(f"[ctx4k] {label}: {t:.1f} s ({ctx/t:.2f} tok/s), peak {runs[label]['peak_in_process_bytes_after_prefill']/1e9:.2f} GB, "
                  f"kv offset {kv_prefill} then {runs[label]['kv_offset_after_decode']}", flush=True)
        pairs = []
        labels = list(runs)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = runs[labels[i]], runs[labels[j]]
                p = {"pair": [labels[i], labels[j]],
                     "final_logits": compare_logits(a["final_logits"], b["final_logits"]),
                     "cache_after_prefill": compare_caches(a["cache"], b["cache"]),
                     "cache_after_decode": compare_caches(a["cache_after_decode"], b["cache_after_decode"]),
                     "decode_after": compare_logits(a["decode"], b["decode"])}
                if a["argmax"] is not None and b["argmax"] is not None and len(a["argmax"]) == len(b["argmax"]):
                    p["all_positions"] = {"positions": int(len(a["argmax"])),
                                          "argmax_identical": int((a["argmax"] == b["argmax"]).sum()),
                                          "margin_abs_diff_max": float(np.abs(a["margin"] - b["margin"]).max()),
                                          "margin_abs_diff_median": float(np.median(np.abs(a["margin"] - b["margin"]))),
                                          "min_margin_at_mismatch": (float(np.minimum(a["margin"], b["margin"])[a["argmax"] != b["argmax"]].min())
                                                                     if (a["argmax"] != b["argmax"]).any() else None)}
                pairs.append(p)
                print(f"[ctx4k] {labels[i]} vs {labels[j]}: final logits max abs {p['final_logits']['max_abs']:.2e} rel {p['final_logits']['max_rel']:.1e} "
                      f"argmax {p['final_logits']['argmax_identical']}/1; after prefill: state rel {p['cache_after_prefill']['state_max_rel']:.1e} "
                      f"kv {max(p['cache_after_prefill']['kv_k_max_abs'], p['cache_after_prefill']['kv_v_max_abs']):.1e} "
                      f"offsets {p['cache_after_prefill']['kv_offsets']}; decode {p['decode_after']['argmax_identical']}/{dec_steps} max abs {p['decode_after']['max_abs']:.2e}"
                      + (f"; all positions argmax {p['all_positions']['argmax_identical']}/{p['all_positions']['positions']}" if "all_positions" in p else ""), flush=True)
        rec = {"provenance": prov, "tower": tower, "context": ctx, "decode_steps": dec_steps,
               "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("final_logits", "cache", "cache_after_decode", "argmax", "margin", "decode")}
                        for k, v in runs.items()},
               "pairs": pairs,
               "note": "real E0 tokens; no one-shot reference at 4,096; margins are top-1 minus top-2 logit computed on device per chunk; "
                       "cache_after_prefill compares host snapshots taken before the decode steps, cache_after_decode the live caches "
                       "after them; peak_in_process_bytes_after_prefill shares one allocator with the earlier arms' retained caches "
                       "and is not the memory measurement (the mem receipts are)"}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"[ctx4k] wrote {out}", flush=True)
        return
    if mode == "ab":
        # fused-only control against the decode-once GEMM path, same ids,
        # same process, the path switched on the resident class between arms
        arms = [("fused", False, "full"), ("gemm", True, "full"), ("fused", False, "final"), ("gemm", True, "final")]
        runs = {}
        for name, on, head in arms:
            label = f"{name}_{head}"
            qmetal._Resident.prefill_gemm = on
            mx.clear_cache()
            mx.reset_peak_memory()
            cache = model.make_cache()
            stats: dict = {}
            t0 = time.perf_counter()
            lg = prefill(model, ids, cache, chunk, head, stats=stats if head == "full" else None)
            t = time.perf_counter() - t0
            fin = np.asarray(lg[0, -1], dtype=np.float32)
            del lg
            peak = int(mx.get_peak_memory())
            kv_prefill = int(cache[FA_LAYER].offset)
            snap = snapshot_cache(cache)
            qmetal._Resident.prefill_gemm = False          # decode steps on the qualified fused path in every arm
            dec = decode_after(model, cache, nxt)
            runs[label] = {"path": name, "head": head, "chunk": chunk, "prefill_s": t, "tok_per_s": ctx / t,
                           "peak_in_process_bytes_after_prefill": peak, "final_logits": fin, "cache": snap,
                           "argmax": (np.concatenate(stats["argmax"]) if "argmax" in stats else None),
                           "margin": (np.concatenate(stats["margin"]) if "margin" in stats else None),
                           "decode": dec, "kv_offset_after_prefill": kv_prefill,
                           "kv_offset_after_decode": int(cache[FA_LAYER].offset)}
            print(f"[ab] {label}: {t:.1f} s ({ctx/t:.2f} tok/s), peak {peak/1e9:.2f} GB, kv offset {kv_prefill} then "
                  f"{runs[label]['kv_offset_after_decode']}", flush=True)
            del cache
        qmetal._Resident.prefill_gemm = qmetal.PREFILL_GEMM
        pairs = []
        for a_l, b_l in (("fused_full", "gemm_full"), ("fused_final", "gemm_final"), ("fused_full", "fused_final"), ("gemm_full", "gemm_final")):
            a, b = runs[a_l], runs[b_l]
            p = {"pair": [a_l, b_l], "final_logits": compare_logits(a["final_logits"], b["final_logits"]),
                 "cache_after_prefill": compare_caches(a["cache"], b["cache"]),
                 "decode_after": compare_logits(a["decode"], b["decode"])}
            if a["argmax"] is not None and b["argmax"] is not None and len(a["argmax"]) == len(b["argmax"]):
                p["all_positions"] = {"positions": int(len(a["argmax"])),
                                      "argmax_identical": int((a["argmax"] == b["argmax"]).sum()),
                                      "margin_abs_diff_max": float(np.abs(a["margin"] - b["margin"]).max()),
                                      "margin_abs_diff_median": float(np.median(np.abs(a["margin"] - b["margin"]))),
                                      "min_margin_at_mismatch": (float(np.minimum(a["margin"], b["margin"])[a["argmax"] != b["argmax"]].min())
                                                                 if (a["argmax"] != b["argmax"]).any() else None)}
            pairs.append(p)
            print(f"[ab] {a_l} vs {b_l}: final logits max abs {p['final_logits']['max_abs']:.2e} rel {p['final_logits']['max_rel']:.1e} "
                  f"margin {p['final_logits']['margin_abs_diff_max']:.1e}; state rel {p['cache_after_prefill']['state_max_rel']:.1e} "
                  f"kv {max(p['cache_after_prefill']['kv_k_max_abs'], p['cache_after_prefill']['kv_v_max_abs']):.1e} offsets "
                  f"{p['cache_after_prefill']['kv_offsets_equal']}; decode {p['decode_after']['argmax_identical']}/{dec_steps} max abs "
                  f"{p['decode_after']['max_abs']:.2e}" + (f"; all positions argmax {p['all_positions']['argmax_identical']}/{p['all_positions']['positions']} "
                  f"margin {p['all_positions']['margin_abs_diff_max']:.1e}" if "all_positions" in p else ""), flush=True)
        rec = {"provenance": prov, "tower": tower, "context": ctx, "chunk": chunk, "decode_steps": dec_steps,
               "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("final_logits", "cache", "argmax", "margin", "decode")}
                        for k, v in runs.items()},
               "pairs": pairs,
               "note": "real E0 tokens; the gemm arms prefill through the decode-once path and decode through the fused path; "
                       "cache comparison on host snapshots taken before the decode steps; in-process peaks share one allocator"}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"[ab] wrote {out}", flush=True)
        return
    raise SystemExit(f"unknown MODE {mode}")


if __name__ == "__main__":
    main()

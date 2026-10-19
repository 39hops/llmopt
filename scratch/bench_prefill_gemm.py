"""Fused tiles against the decode-once float32 GEMM path on the real arm C
matrices: the measured crossover per distinct shape, the per-matrix decode
and GEMM times, and the numerics of the new reduction order against the
fused kernel and the float64 oracle. Both paths are the production
resident (llmopt/lab/qmetal.py) with the switch set per call, so the
timings are the times a tower would see.

Sections (one process):
  crossover   per distinct fused (codec, shape), one representative tensor
              each, T in TS (8, 16, 32, 64, 128, 256): pipelined per-launch
              time (QUEUE launches on independent inputs, one eval) of the
              fused path and the GEMM path, per-activation cost, the
              smallest measured T at which the GEMM path wins, and the
              interpolated crossover; the head is measured on the full
              248,320-row matrix in GEMM_SLAB_BYTES slabs (the production
              law) and on the fused path with QUEUE_HEAD launches.
  parts       per representative matrix: the decode launch alone (all
              slabs), the GEMM on resident float32 weights alone, and the
              combined path, at T = 256.
  numerics    per representative matrix at T in (1, 64, 256), real
              activations drawn from the tower's hidden-state statistics
              (normal, unit scale) and a heavy-tailed draw: max abs and
              relative error of the fused output and of the GEMM output
              against the float64 oracle (T = 1 forces the GEMM path below
              its production threshold, a reference point only), the ratio
              err / absdot (the
              cancellation-aware term of the parity bound), and whether the
              GEMM path's error is larger or smaller than the fused
              kernel's on that matrix; also the decode_f32 reconstruction
              against the canonical numpy decoder (max abs difference, sign
              of zero).
Receipt: logs/qmetal/prefill_gemm/bench_<commit>.json (refuse-if-exists;
SMOKE=1 writes under logs/qmetal/prefill_gemm_smoke/). Env: ART_DIR,
VENDOR_DIR, REPEATS (30), QUEUE (32), QUEUE_HEAD (4), OUT.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

HEAD_KEY = "lm_head.weight"
TS = (8, 16, 32, 64, 128, 256)


def provenance() -> dict:
    import mlx.core as mx
    import mlx_lm
    from llmopt.lab import qmetal
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty_at_launch": bool(dirty), "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__, "device": mx.device_info().get("device_name"),
            "kernel_version": qmetal.VERSION, "gemm_min_t_default": (qmetal.GEMM_MIN_T if qmetal.GEMM_MIN_T is not None else dict(qmetal.GEMM_MIN_T_BY_CODEC)),
            "gemm_slab_bytes": qmetal.GEMM_SLAB_BYTES, "smoke": os.environ.get("SMOKE") == "1"}


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import decode_entry
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows

    prov = provenance()
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    repeats = int(os.environ.get("REPEATS", "30"))
    queue = int(os.environ.get("QUEUE", "32"))
    queue_head = int(os.environ.get("QUEUE_HEAD", "4"))
    rdir = "logs/qmetal/prefill_gemm_smoke" if prov["smoke"] else "logs/qmetal/prefill_gemm"
    out = os.environ.get("OUT", f"{rdir}/bench_{prov['commit']}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    os.makedirs(os.path.dirname(out), exist_ok=True)
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

    shapes: dict = {}
    for key, e in MAN.items():
        if not isinstance(e, dict) or e["codec"] not in ("w4", "s16"):
            continue
        if key.startswith("model.visual") or ".mtp." in key or key.startswith("mtp.") or key.endswith("embed_tokens.weight"):
            continue
        k = (e["codec"], tuple(e["shape"]))
        shapes.setdefault(k, {"codec": e["codec"], "shape": list(e["shape"]), "count": 0, "example": key})
        shapes[k]["count"] += 1

    def set_path(on: bool):
        qmetal._Resident.prefill_gemm = on
        qmetal._Resident.gemm_min_t = 1

    def timed_pipelined(fn_i, q_):
        mx.eval(*[fn_i(i) for i in range(min(2, q_))])
        ts = []
        for _ in range(max(5, repeats // 5)):
            t0 = time.perf_counter()
            mx.eval(*[fn_i(i) for i in range(q_)])
            ts.append((time.perf_counter() - t0) / q_)
        return float(np.median(ts))

    rng = np.random.default_rng(0)
    crossover = []
    parts = []
    numerics = []
    for key, info in sorted(shapes.items(), key=lambda kv: -kv[1]["count"]):
        codec, shape = key
        e = MAN[info["example"]]
        buf = payload(e)
        res = qmetal.resident(buf, e)
        R, C = res.shape
        is_head = info["example"] == HEAD_KEY
        q_ = queue_head if is_head else queue
        row = dict(info, payload_bytes=res.payload_bytes(), gemm_ok=bool(res.gemm_ok), by_T={})
        if not res.gemm_ok:
            raise SystemExit(f"REFUSING: {info['example']} is not GEMM-eligible; the arm C scan found no such matrix")
        for T in TS:
            xs = [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(q_)]
            mx.eval(*xs)
            set_path(False)
            tf = timed_pipelined(lambda i: res.gemm(xs[i]), q_)
            assert res.last_path == "fused"
            set_path(True)
            tg = timed_pipelined(lambda i: res.gemm(xs[i]), q_)
            assert res.last_path == "gemm"
            row["by_T"][T] = {"fused_pipelined_per_launch_s": tf, "gemm_pipelined_per_launch_s": tg,
                              "fused_per_activation_s": tf / T, "gemm_per_activation_s": tg / T,
                              "gemm_over_fused": tg / tf}
            del xs
        wins = [T for T in TS if row["by_T"][T]["gemm_over_fused"] < 1.0]
        row["smallest_T_where_gemm_wins"] = wins[0] if wins else None
        # linear interpolation of the crossover between the last losing and the first winning T
        xo = None
        for a, b in zip(TS, TS[1:]):
            ra, rb = row["by_T"][a]["gemm_over_fused"], row["by_T"][b]["gemm_over_fused"]
            if ra >= 1.0 > rb:
                xo = a + (ra - 1.0) / (ra - rb) * (b - a)
                break
        row["crossover_T_interpolated"] = xo
        print(f"[bench] {codec} {R}x{C} (x{info['count']}): " + ", ".join(
            f"T{T} fused {row['by_T'][T]['fused_pipelined_per_launch_s']*1e3:.2f} / gemm {row['by_T'][T]['gemm_pipelined_per_launch_s']*1e3:.2f} ms"
            for T in TS) + f"; gemm wins from T={row['smallest_T_where_gemm_wins']} (interp {xo})", flush=True)
        crossover.append(row)

        # ---- parts at T = 256 ----
        T = 256
        xs = [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(q_)]
        mx.eval(*xs)
        rows = max(1, min(R, qmetal.GEMM_SLAB_BYTES // (C * 4)))
        slabs = [(r0, min(R, r0 + rows)) for r0 in range(0, R, rows)]
        # one slab live at a time (the production shape): decode, evaluate, drop, next slab
        def decode_all_slabs():
            for a, b in slabs:
                mx.eval(res.decode_f32(a, b))
        for _ in range(2):
            decode_all_slabs()
        ts = []
        for _ in range(max(5, repeats // 5)):
            t0 = time.perf_counter()
            decode_all_slabs()
            ts.append(time.perf_counter() - t0)
        t_dec = float(np.median(ts))
        if not is_head:
            Wd = res.decode_f32(0, R)
            mx.eval(Wd)
            t_mm = timed_pipelined(lambda i: mx.matmul(xs[i], Wd.T), q_)
            del Wd
        else:
            t_mm = None
        set_path(True)
        t_both = timed_pipelined(lambda i: res.gemm(xs[i]), q_)
        parts.append({"tensor": info["example"], "codec": codec, "shape": [R, C], "slabs": len(slabs), "T": T,
                      "decode_all_slabs_s": t_dec, "decode_timing": "synced per slab, one slab live at a time",
                      "gemm_on_resident_f32_s": t_mm,
                      "gemm_on_resident_f32_note": None if t_mm is not None else "not measured: a dense float32 head is 5.1 GB and is not materialised",
                      "decode_plus_gemm_s": t_both, "rows_per_slab": rows, "dense_f32_bytes_per_slab": rows * C * 4})
        print(f"[bench] parts {codec} {R}x{C}: decode {t_dec*1e3:.2f} ms ({len(slabs)} slab(s)), gemm "
              f"{(t_mm or 0)*1e3:.2f} ms, path {t_both*1e3:.2f} ms", flush=True)
        del xs

        # ---- numerics ----
        if is_head:
            rows_dec = S16Rows(buf, e["shape"]) if codec == "s16" else W4Rows(buf, e["shape"])
            Rn = 4096
            W = rows_dec.rows(0, Rn).astype(np.float64)
            sub = None
        else:
            Rn = R
            W = decode_entry(buf, e).astype(np.float64)
        Wd = np.asarray(res.decode_f32(0, Rn))
        W32 = W.astype(np.float32)
        rec = {"tensor": info["example"], "codec": codec, "shape": [R, C], "rows_compared": Rn,
               "decode_f32_max_abs_diff_vs_numpy": float(np.abs(Wd - W32).max()),
               "decode_f32_bitwise_equal": bool(np.array_equal(Wd, W32) and np.array_equal(np.signbit(Wd), np.signbit(W32))),
               "by_T": {}}
        del Wd
        for T in (1, 16, 32, 64, 256):
            for draw in ("normal", "heavy"):
                X = rng.normal(size=(T, C)) if draw == "normal" else rng.standard_t(2.0, size=(T, C)) * 3.0
                X = X.astype(np.float32)
                X64 = X.astype(np.float64)
                ref = X64 @ W.T
                absdot = np.abs(X64) @ np.abs(W).T
                set_path(False)
                yf = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)[:, :Rn]
                set_path(True)
                yg = np.asarray(res.gemm(mx.array(X)), dtype=np.float64)[:, :Rn]
                path = res.last_path
                ef, eg = np.abs(yf - ref), np.abs(yg - ref)
                den = np.abs(ref).max()
                r = {"path_taken_for_gemm_arm": path, "fused_max_abs": float(ef.max()), "gemm_max_abs": float(eg.max()),
                     "fused_max_rel_to_max_ref": float(ef.max() / den), "gemm_max_rel_to_max_ref": float(eg.max() / den),
                     "fused_max_err_over_absdot": float((ef / np.maximum(absdot, 1e-300)).max()),
                     "gemm_max_err_over_absdot": float((eg / np.maximum(absdot, 1e-300)).max()),
                     "gemm_vs_fused_max_abs_diff": float(np.abs(yg - yf).max()),
                     "gemm_error_larger_than_fused": bool(eg.max() > ef.max()),
                     "ratio_gemm_over_fused_max_abs": float(eg.max() / max(ef.max(), 1e-300))}
                rec["by_T"][f"T{T}_{draw}"] = r
        print(f"[bench] numerics {codec} {R}x{C}: decode bitwise {rec['decode_f32_bitwise_equal']}; T256 normal fused "
              f"{rec['by_T']['T256_normal']['fused_max_err_over_absdot']:.2e} gemm {rec['by_T']['T256_normal']['gemm_max_err_over_absdot']:.2e} (err/absdot)", flush=True)
        numerics.append(rec)
        del res, W, W32
        mx.clear_cache()
    set_path(qmetal.PREFILL_GEMM)
    qmetal._Resident.gemm_min_t = qmetal.GEMM_MIN_T
    worst = max(r["by_T"][k]["gemm_max_err_over_absdot"] for r in numerics for k in r["by_T"])
    worst_f = max(r["by_T"][k]["fused_max_err_over_absdot"] for r in numerics for k in r["by_T"])
    receipt = {"provenance": prov, "arm": arm, "chain_report": q["report"].get("rung0", q["report"]), "repeats": repeats,
               "queue": queue, "queue_head": queue_head, "Ts": list(TS),
               "note": "timing: pipelined per-launch times of the production resident with the path switched per call; "
                       "numerics: against the float64 oracle on the decoded weights (the head on its first 4,096 rows)",
               "crossover": crossover, "parts": parts, "numerics": numerics,
               "summary": {"gemm_wins_from_T": {f"{r['codec']} {r['shape']}": r["smallest_T_where_gemm_wins"] for r in crossover},
                           "crossover_T_interpolated": {f"{r['codec']} {r['shape']}": r["crossover_T_interpolated"] for r in crossover},
                           "worst_gemm_err_over_absdot": worst, "worst_fused_err_over_absdot": worst_f,
                           "decode_bitwise_equal_all": all(r["decode_f32_bitwise_equal"] for r in numerics)}}
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[bench] summary {json.dumps(receipt['summary'])}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()

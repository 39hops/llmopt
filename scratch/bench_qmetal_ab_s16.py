"""A/B benchmark of the S16 kernel generations: the CONTROL generation
(the shipped kernels, QMETAL_VERSION 4: W4 v4 dispatch, the original S16
kernel) against the CANDIDATE generation (QMETAL_VERSION 5: the same W4
dispatch, the S16 v5 kernel) on the same Mac, the same arm C artifact and
identical inputs, alternating the two so that thermal and run-order
effects fall on both arms. Modelled on scratch/bench_qmetal_ab.py (the
W4 v3 against v4 receipt); the S16 version is switched on EVERY
resident, which changes nothing for the W4 residents (generation 5
dispatches W4 exactly as generation 4).

Part 1, matrices: real payloads of the S16 shapes in_proj_qkv [10240,
5120], in_proj_z [6144, 5120], out_proj [5120, 6144] (layer 0), q_proj
[12288, 5120] (layer 3) and lm_head [248320, 5120]; per round and
version: synced per-call median over REPEATS, pipelined per-launch time
over QUEUE queued launches, tiles T = 1, 2, 4, 8, for the control, the
candidate as it dispatches (v5 up to S16_V5_MAX_TILE, the shipped kernel
above) and the candidate with the cap lifted (label 55: the v5 kernel at
every tile, the measurement behind the cap); outputs of all three
compared bit for bit on every input, every round.

Part 2, full cached decode: the arm C tower is built once; prompt
prefill through qprefill.prefill_chunked (chunk 256, GEMM path, head
final: at these tiles every S16 matrix takes the decode-once GEMM, so the
prefill and the cache state are the same arrays under both versions),
then STEPS single-token greedy steps timed one at a time with mx.eval on
the logits; median, p90 and mean of steps 5..STEPS; peak unified memory
and the allocator footprint (qprefill.memory_report) per round. Rounds
per context: ABBA repeated ABBA_REPEATS times (control, candidate,
candidate, control, ...), so each version is measured 2 x ABBA_REPEATS
times per context and the run-to-run spread of each version is on the
receipt next to the between-version difference. The first round's
greedy tokens and per-step logits of each version are kept; every later
round of the same version must reproduce the tokens, and the two
versions' first-round logits are compared step by step (max abs
difference; expected exactly 0 when the candidate is bit-identical).

Receipt logs/qmetal/ab_s16_<commit>.json (refuse-if-exists). Timing is the
claim; the parity claim lives in tests/test_qmetal.py and the
qualification receipts. Env: ART_DIR, VENDOR_DIR, CONTROL_VERSION (4),
CAND_VERSION (5), ROUNDS (3), REPEATS (30), QUEUE (32), CONTEXTS
("126,1024"), STEPS (48), ABBA_REPEATS (2), HEAD (1), OUT.
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

MATRICES = [("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "in_proj_qkv"),
            ("model.language_model.layers.0.linear_attn.in_proj_z.weight", "in_proj_z"),
            ("model.language_model.layers.0.linear_attn.out_proj.weight", "out_proj"),
            ("model.language_model.layers.3.self_attn.q_proj.weight", "q_proj")]
HEAD = ("lm_head.weight", "lm_head")


def provenance() -> dict:
    import mlx.core as mx
    import mlx_lm
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty": bool(dirty), "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__}


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qmetal, qprefill

    prov = provenance()
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    vc = int(os.environ.get("CONTROL_VERSION", "4"))
    vk = int(os.environ.get("CAND_VERSION", "5"))
    rounds = int(os.environ.get("ROUNDS", "3"))
    repeats = int(os.environ.get("REPEATS", "30"))
    queue = int(os.environ.get("QUEUE", "32"))
    contexts = [int(c) for c in os.environ.get("CONTEXTS", "126,1024").split(",") if c]
    steps = int(os.environ.get("STEPS", "48"))
    abba = int(os.environ.get("ABBA_REPEATS", "2"))
    with_head = os.environ.get("HEAD", "1") == "1"
    out = os.environ.get("OUT", f"logs/qmetal/ab_s16_{prov['commit']}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    receipt = {"provenance": prov, "arm": arm, "control_version": vc, "candidate_version": vk, "rounds": rounds,
               "repeats": repeats, "queue": queue, "contexts": contexts, "steps": steps, "abba_repeats": abba,
               "default_version_at_import": qmetal.VERSION,
               "note": "timing only; control and candidate outputs compared bit for bit as a guard; parity lives in the "
                       "tests and the qualification receipts"}

    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    receipt["chain_report"] = q["report"]["rung0"]
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    # ---- part 1: matrices ----
    rng = np.random.default_rng(0)
    mats = []
    for nm, short in list(MATRICES) + ([HEAD] if with_head else []):
        e = MAN[nm]
        if e["codec"] != "s16":
            raise SystemExit(f"REFUSING: {nm} is {e['codec']}")
        res = qmetal.S16Metal(payload(e), e["shape"])
        R, C = res.shape
        xs = {T: [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(queue)] for T in (1, 2, 4, 8)}
        for v in xs.values():
            mx.eval(*v)
        mats.append({"tensor": nm, "short": short, "res": res, "xs": xs, "rows": [], "bit": {}})

    cap = qmetal.S16_V5_MAX_TILE

    def run(res, version, x):
        # version 55 = generation 5 with the tile cap lifted (the v5 kernel at
        # every tile), measured so the receipt records why the cap exists
        if version == 55:
            qmetal.S16_V5_MAX_TILE = 8
            res.version = 5
        else:
            qmetal.S16_V5_MAX_TILE = cap
            res.version = version
        return res.gemm(x)

    first_call = {}
    for m in mats:
        for v in (vc, vk):
            t0 = time.perf_counter()
            mx.eval(run(m["res"], v, m["xs"][1][0]))
            first_call[f"{m['short']}_v{v}"] = time.perf_counter() - t0
    receipt["first_call_s_in_process"] = first_call
    receipt["s16_v5_max_tile"] = cap
    for rnd in range(rounds):
        order = (vc, vk, 55) if rnd % 2 == 0 else (55, vk, vc)
        for v in order:
            for m in mats:
                res = m["res"]
                for T in (1, 2, 4, 8):
                    xs = m["xs"][T]
                    for _ in range(3):
                        mx.eval(run(res, v, xs[0]))
                    ts = []
                    for _ in range(repeats):
                        t0 = time.perf_counter()
                        mx.eval(run(res, v, xs[0]))
                        ts.append(time.perf_counter() - t0)
                    pipe = []
                    for _ in range(5):
                        t0 = time.perf_counter()
                        mx.eval(*[run(res, v, xs[i]) for i in range(queue)])
                        pipe.append((time.perf_counter() - t0) / queue)
                    m["rows"].append({"round": rnd, "version": v, "T": T,
                                      "synced_median_s": float(np.median(ts)), "synced_p90_s": float(np.percentile(ts, 90)),
                                      "pipelined_per_launch_s": float(np.median(pipe))})
                for T in (1, 2, 8):
                    gc = np.asarray(run(res, vc, m["xs"][T][1]))
                    gk = np.asarray(run(res, vk, m["xs"][T][1]))
                    ga = np.asarray(run(res, 55, m["xs"][T][1]))
                    m["bit"].setdefault(f"T{T}", []).append(bool(np.array_equal(gc, gk) and np.array_equal(gc, ga)))
                t1 = [r for r in m["rows"] if r["round"] == rnd and r["version"] == v and r["T"] == 1][0]
                print(f"[ab] round {rnd} v{v} {m['short']}: T1 pipelined {t1['pipelined_per_launch_s']*1e6:.0f} us "
                      f"synced {t1['synced_median_s']*1e6:.0f} us", flush=True)
    receipt["matrices"] = [{"tensor": m["tensor"], "short": m["short"], "shape": list(m["res"].shape),
                            "payload_bytes": m["res"].payload_bytes(), "rows": m["rows"],
                            "bit_identical_every_round": m["bit"]} for m in mats]
    for m in mats:
        m["res"].version = qmetal.VERSION
    qmetal.S16_V5_MAX_TILE = cap   # the last part-1 call lifts the cap; the tower runs under the production cap
    del mats

    # ---- part 2: full tower cached decode ----
    from qwen_metal_qual import build_metal_model
    model, info = build_metal_model(art, vdir)
    residents = [mod.res for _, mod in model.named_modules() if type(mod).__name__ == "QLinear"]
    receipt["tower"] = {k: v for k, v in info.items() if k != "chain_report"}
    receipt["residents"] = {"total": len(residents), "s16": sum(1 for r in residents if r.codec == "s16"),
                            "w4": sum(1 for r in residents if r.codec == "w4")}
    receipt["prefill_entry"] = "qprefill.prefill_chunked(chunk=256, head='final', matmul='gemm'); untimed; version-independent"

    def set_version(v):
        for r in residents:
            r.version = v

    base = [int(t) for t in np.random.default_rng(1).integers(1000, 100000, size=max(contexts))]
    tower_rows = []
    order = [vc, vk, vk, vc] * abba
    for ctx in contexts:
        prompt = base[:ctx]
        first = {}
        for rnd, v in enumerate(order):
            set_version(v)
            cache = model.make_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            lp = qprefill.prefill_chunked(model, prompt, cache, chunk=256, head="final", matmul="gemm")
            mx.eval(lp)
            t_pre = time.perf_counter() - t0
            cur = int(lp[0, -1].argmax())
            walls, emitted, logits = [], [], []
            for s in range(steps):
                t0 = time.perf_counter()
                lz = model(mx.array([[cur]]), cache=cache)[0, -1]
                mx.eval(lz)
                walls.append(time.perf_counter() - t0)
                cur = int(lz.argmax())
                emitted.append(cur)
                logits.append(np.asarray(lz, dtype=np.float32))
            st = walls[4:]
            mem = qprefill.memory_report()
            row = {"context": ctx, "round": rnd, "version": v, "prefill_s": t_pre,
                   "median_step_s": float(np.median(st)), "p90_step_s": float(np.percentile(st, 90)),
                   "mean_step_s": float(np.mean(st)), "min_step_s": float(np.min(st)), "max_step_s": float(np.max(st)),
                   "tok_per_s_median": 1.0 / float(np.median(st)),
                   "peak_mem_bytes": int(mx.get_peak_memory()), "memory_report": mem}
            if v not in first:
                first[v] = (emitted, logits)
                row["first_round_of_version"] = True
            else:
                row["greedy_tokens_same_as_first_round_of_this_version"] = emitted == first[v][0]
                row["max_abs_logit_diff_vs_first_round_of_this_version"] = float(max(
                    np.abs(a - b).max() for a, b in zip(logits, first[v][1])))
            tower_rows.append(row)
            print(f"[ab] tower ctx {ctx} round {rnd} v{v}: median {row['median_step_s']*1e3:.1f} ms "
                  f"({row['tok_per_s_median']:.2f} tok/s) p90 {row['p90_step_s']*1e3:.1f} prefill {t_pre:.1f}s "
                  f"peak {row['peak_mem_bytes']/1e9:.2f} GB", flush=True)
            del cache, lp
        ec, lc = first[vc]
        ek, lk = first[vk]
        tower_rows.append({"context": ctx, "cross_version": {
            "greedy_tokens_control_equal_candidate": ec == ek,
            "first_divergence_step": next((i for i, (a, b) in enumerate(zip(ec, ek)) if a != b), None),
            "max_abs_logit_diff_control_vs_candidate": float(max(np.abs(a - b).max() for a, b in zip(lc, lk))),
            "per_step_max_abs_logit_diff": [float(np.abs(a - b).max()) for a, b in zip(lc, lk)]}})
    receipt["tower_decode"] = tower_rows
    summary = {}
    for ctx in contexts:
        for v in (vc, vk):
            meds = [r["median_step_s"] for r in tower_rows if r.get("version") == v and r["context"] == ctx]
            summary[f"ctx{ctx}_v{v}_median_step_s"] = float(np.median(meds))
            summary[f"ctx{ctx}_v{v}_median_step_s_by_round"] = meds
            summary[f"ctx{ctx}_v{v}_spread_s"] = float(max(meds) - min(meds))
        a, b = summary[f"ctx{ctx}_v{vc}_median_step_s"], summary[f"ctx{ctx}_v{vk}_median_step_s"]
        summary[f"ctx{ctx}_speedup_control_over_candidate"] = a / b
        summary[f"ctx{ctx}_improvement_percent"] = 100.0 * (a - b) / a
        # worst case across rounds: slowest candidate round against fastest control round
        summary[f"ctx{ctx}_improvement_percent_worst_pairing"] = 100.0 * (
            min(summary[f"ctx{ctx}_v{vc}_median_step_s_by_round"]) - max(summary[f"ctx{ctx}_v{vk}_median_step_s_by_round"])
        ) / min(summary[f"ctx{ctx}_v{vc}_median_step_s_by_round"])
    receipt["summary"] = summary
    receipt["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[ab] {json.dumps({k: v for k, v in summary.items() if 'by_round' not in k})}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()

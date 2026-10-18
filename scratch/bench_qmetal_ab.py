"""A/B benchmark of the W4 kernel generations v3 (control) and v4 on the
same Mac, the same arm C artifact and identical inputs, alternating the
two versions to spread thermal and run-order effects across both arms.

Part 1, matrices: the three W4 MLP shapes of one layer (gate_proj and
up_proj [17408, 5120], down_proj [5120, 17408]), real payloads. For each
of ROUNDS rounds the order v3, v4 is alternated; per round and version:
synced per-call median over REPEATS (dispatch plus kernel plus drain),
pipelined per-launch time over QUEUE queued launches (the tower's
regime), for tiles T = 1, 2, 4, 8 (gemm on T identical activations),
and the first-call (compilation plus first launch) cost measured once
per version in a fresh process is NOT possible here; instead the first
call of each kernel in this process is recorded as first_call_s and
excluded from the medians. Outputs of v3 and v4 are compared bit for
bit on every input (array_equal) as a guard that the A/B times the same
function.

Part 2, full cached decode: the arm C tower is built once; the W4
residents' version attribute is switched between rounds (kernels are
cached per version, so the switch costs nothing after the first call).
For each context length in CONTEXTS: prompt prefill into a fresh cache
(untimed), then STEPS single-token greedy steps timed one at a time with
mx.eval on the logits; median, p90 and mean of steps 5..STEPS; peak
unified memory of the arm. Rounds alternate v3, v4, v4, v3 (ABBA) so
each version sees both positions. The v3 round's greedy tokens are
compared with the v4 round's on the same prompt (same arithmetic per
row, so they are expected identical; a difference is reported, not
hidden).

Receipt logs/qmetal/ab_v3_v4_<commit>.json (refuse-if-exists). Timing
only; the parity claim lives in tests/test_qmetal.py and the
qualification receipts. Env: ART_DIR, VENDOR_DIR, ROUNDS (3), REPEATS
(30), QUEUE (32), CONTEXTS ("126,1024"), STEPS (48), OUT.
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
sys.path.insert(0, HERE)


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
    from llmopt.lab import qartifact, qmetal

    prov = provenance()
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    rounds = int(os.environ.get("ROUNDS", "3"))
    repeats = int(os.environ.get("REPEATS", "30"))
    queue = int(os.environ.get("QUEUE", "32"))
    contexts = [int(c) for c in os.environ.get("CONTEXTS", "126,1024").split(",") if c]
    steps = int(os.environ.get("STEPS", "48"))
    out = os.environ.get("OUT", f"logs/qmetal/ab_v3_v4_{prov['commit']}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    receipt = {"provenance": prov, "arm": arm, "rounds": rounds, "repeats": repeats, "queue": queue,
               "contexts": contexts, "steps": steps, "default_version_at_import": qmetal.VERSION,
               "note": "timing only; v3 and v4 outputs compared bit for bit as a guard, parity lives in the tests "
                       "and the qualification receipts"}

    # ---- part 1: matrices ----
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

    names = [f"model.language_model.layers.0.mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
    rng = np.random.default_rng(0)
    mats = []
    for nm in names:
        e = MAN[nm]
        assert e["codec"] == "w4"
        res = qmetal.W4Metal(payload(e), e["shape"])
        R, C = res.shape
        xs = {T: [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(queue)] for T in (1, 2, 4, 8)}
        for v in xs.values():
            mx.eval(*v)
        mats.append({"tensor": nm, "res": res, "xs": xs, "rows": []})

    def run(res, version, x):
        res.version = version
        return res.gemm(x)

    first_call = {}
    for m in mats:
        res = m["res"]
        for v in (3, 4):
            t0 = time.perf_counter()
            mx.eval(run(res, v, m["xs"][1][0]))
            first_call[f"{m['tensor'].split('.')[-2]}_v{v}"] = time.perf_counter() - t0
    receipt["first_call_s_in_process"] = first_call

    for rnd in range(rounds):
        order = (3, 4) if rnd % 2 == 0 else (4, 3)
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
                # bit-for-bit guard on the same input, every round
                g3 = np.asarray(run(res, 3, m["xs"][8][0]))
                g4 = np.asarray(run(res, 4, m["xs"][8][0]))
                m.setdefault("bit_identical_T8", []).append(bool(np.array_equal(g3, g4)))
                print(f"[ab] round {rnd} v{v} {m['tensor'].split('.')[-2]}: T1 pipelined "
                      f"{m['rows'][-4]['pipelined_per_launch_s']*1e6:.0f} us synced {m['rows'][-4]['synced_median_s']*1e6:.0f} us", flush=True)
    receipt["matrices"] = [{"tensor": m["tensor"], "shape": list(m["res"].shape), "payload_bytes": m["res"].payload_bytes(),
                            "rows": m["rows"], "bit_identical_T8_every_round": m["bit_identical_T8"]} for m in mats]
    for m in mats:
        res = m["res"]
        res.version = qmetal.VERSION
    del mats

    # ---- part 2: full tower cached decode ----
    from qwen_metal_qual import build_metal_model
    model, info = build_metal_model(art, vdir)
    tm = model.language_model.model
    QLinearT = type(tm.layers[0].mlp.gate_proj)
    w4_res = []
    for ly in tm.layers:
        for nm in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(ly.mlp, nm)
            if isinstance(mod, QLinearT) and mod.res.codec == "w4":
                w4_res.append(mod.res)
    receipt["tower"] = {k: v for k, v in info.items() if k != "chain_report"}
    receipt["w4_residents"] = len(w4_res)

    def set_version(v):
        for r in w4_res:
            r.version = v

    base = [int(t) for t in np.random.default_rng(1).integers(1000, 100000, size=max(contexts))]
    tower_rows = []
    for ctx in contexts:
        prompt = base[:ctx]
        toks = {3: None, 4: None}
        for rnd, v in enumerate((3, 4, 4, 3)):
            set_version(v)
            cache = model.make_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            lp = model(mx.array([prompt]), cache=cache)
            mx.eval(lp)
            t_pre = time.perf_counter() - t0
            cur = int(lp[0, -1].argmax())
            walls, emitted = [], []
            for s in range(steps):
                t0 = time.perf_counter()
                lz = model(mx.array([[cur]]), cache=cache)[0, -1]
                cur = int(lz.argmax())
                walls.append(time.perf_counter() - t0)
                emitted.append(cur)
            st = walls[4:]
            same = None if toks[v] is None else emitted == toks[v]
            if toks[v] is None:
                toks[v] = emitted
            row = {"context": ctx, "round": rnd, "version": v, "prefill_s": t_pre,
                   "median_step_s": float(np.median(st)), "p90_step_s": float(np.percentile(st, 90)),
                   "mean_step_s": float(np.mean(st)), "tok_per_s_median": 1.0 / float(np.median(st)),
                   "peak_mem_bytes": int(mx.get_peak_memory()),
                   "greedy_tokens_same_as_first_round_of_this_version": same}
            tower_rows.append(row)
            print(f"[ab] tower ctx {ctx} round {rnd} v{v}: median {row['median_step_s']*1e3:.1f} ms "
                  f"({row['tok_per_s_median']:.2f} tok/s) p90 {row['p90_step_s']*1e3:.1f} prefill {t_pre:.1f}s "
                  f"peak {row['peak_mem_bytes']/1e9:.2f} GB", flush=True)
            del cache, lp
        tower_rows.append({"context": ctx, "greedy_tokens_v3_equal_v4": toks[3] == toks[4]})
    receipt["tower_decode"] = tower_rows
    summary = {}
    for ctx in contexts:
        for v in (3, 4):
            meds = [r["median_step_s"] for r in tower_rows if r.get("version") == v and r["context"] == ctx]
            summary[f"ctx{ctx}_v{v}_median_step_s"] = float(np.median(meds))
        summary[f"ctx{ctx}_speedup_v3_over_v4"] = summary[f"ctx{ctx}_v3_median_step_s"] / summary[f"ctx{ctx}_v4_median_step_s"]
    receipt["summary"] = summary
    receipt["process_peak_rss_bytes"] = int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[ab] {json.dumps(summary)}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()

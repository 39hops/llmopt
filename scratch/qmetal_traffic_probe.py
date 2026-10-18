"""W4 activation-traffic ceiling probe for the fused Metal kernels, TIMING
ONLY. Controlled variants of the frozen v3 W4 GEMV source (and the S16
source) remove one memory stream at a time; their outputs are wrong by
construction and nothing here makes a numerical claim. The frozen kernels
in llmopt.lab.qmetal are not modified; variants are compiled from edited
copies of their source strings.

Variants (TT = 1, one simdgroup per row unless stated):
  v3        the shipped kernel (baseline; also parity-checked here as a
            sanity guard that the probe harness feeds real inputs)
  nox       activation loads replaced by a constant (no x traffic)
  noidx     index words replaced by a hash of the lane counter (no weight
            stream; codebook gathers kept)
  nocb      codebook gathers replaced by arithmetic on the word (no
            threadgroup gathers; x and index streams kept)
  nostage   codebook read from device memory per gather, no threadgroup
            staging or barrier (the v1 inner loop at TT = 1)
  xonly     index and codebook removed; only x loads and the reduction
  empty     each simdgroup writes one value (launch and drain floor)
  share2/4/8  PROBE of the sharing idea: one simdgroup computes r rows
            and loads each x chunk once for all r (r accumulator triples,
            r index streams). Full arithmetic, outputs real, compared to
            v3 loosely as a sanity check only; not a parity-fixtured
            kernel and not wired into the resident objects.
S16: v3 (= v1 source), nox, nocodes, share2/4.

Two timing modes per variant: SYNCED (one launch, mx.eval, median of
REPEATS after warmup: dispatch plus kernel plus drain) and PIPELINED
(QUEUE launches queued on independent inputs, one mx.eval; per-launch
time = total / QUEUE: the regime the tower runs in). Effective GB/s =
payload bytes / time in both modes. Register pressure and occupancy are
not observable from Python; what is inferred is marked so in the receipt.

Matrices: real arm C payloads, layer 0 mlp.gate_proj [17408, 5120] and
mlp.down_proj [5120, 17408] (w4), layer 0 in_proj_qkv [10240, 5120]
(s16). Receipt logs/qmetal/traffic_probe_<commit>.json (refuse-if-exists).
Env: ART_DIR, VENDOR_DIR, REPEATS (50), QUEUE (32), OUT.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

MATRICES = [("model.language_model.layers.0.mlp.gate_proj.weight", "w4"),
            ("model.language_model.layers.0.mlp.down_proj.weight", "w4"),
            ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "s16")]
ROOFLINE_GBS = 150.0


def provenance() -> dict:
    import mlx.core as mx
    import mlx_lm
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty": bool(dirty), "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__}


# ---- W4 variants from the frozen v3 source ----
def w4_variant(name: str) -> str:
    from llmopt.lab import qmetal
    src = qmetal._W4_V3_SRC
    if name == "v3":
        return src
    if name == "nostage":
        return qmetal._W4_SRC
    if name == "nox":
        return src.replace(
            "            const device float4* x4 = (const device float4*)(x + t * (uint)C + c0);\n"
            "            float d = (metal::dot(x4[0], w0) + metal::dot(x4[1], w1)\n"
            "                     + metal::dot(x4[2], w2) + metal::dot(x4[3], w3)) * s;",
            "            float4 xk = float4(1.0f, 0.5f, 0.25f, 0.125f) + (float)(j & 3u);\n"
            "            float d = (metal::dot(xk, w0) + metal::dot(xk, w1)\n"
            "                     + metal::dot(xk, w2) + metal::dot(xk, w3)) * s;")
    if name == "noidx":
        return src.replace("        uint word = ir[j];", "        uint word = j * 2654435761u + row;")
    if name == "nocb":
        return src.replace(
            "        float4 w0 = (float4)cb4[(word      ) & 0xFFu];\n"
            "        float4 w1 = (float4)cb4[(word >>  8) & 0xFFu];\n"
            "        float4 w2 = (float4)cb4[(word >> 16) & 0xFFu];\n"
            "        float4 w3 = (float4)cb4[(word >> 24) & 0xFFu];",
            "        float4 w0 = float4((float)(word & 0xFFu), 1.0f, 2.0f, 3.0f) * 1e-3f;\n"
            "        float4 w1 = float4((float)((word >> 8) & 0xFFu), 1.0f, 2.0f, 3.0f) * 1e-3f;\n"
            "        float4 w2 = float4((float)((word >> 16) & 0xFFu), 1.0f, 2.0f, 3.0f) * 1e-3f;\n"
            "        float4 w3 = float4((float)((word >> 24) & 0xFFu), 1.0f, 2.0f, 3.0f) * 1e-3f;")
    if name == "xonly":
        return w4_variant("nocb").replace("        uint word = ir[j];", "        uint word = j * 2654435761u + row;")
    if name == "empty":
        return """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    if (lane == 0) out[row] = (float)row;
"""
    m = re.match(r"share(\d+)$", name)
    if m:
        r = int(m.group(1))
        return f"""
    constexpr uint SG = 32;
    constexpr uint NR = {r};
    uint row0 = threadgroup_position_in_grid.x * NR;
    uint lane = thread_position_in_threadgroup.x;
    const uint C4 = (uint)C / 4;
    threadgroup half4 cb4[256];
    for (uint i = lane; i < 256u; i += SG) cb4[i] = ((const device half4*)cb)[i];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const uint nwords = C4 / 4;
    const uint blocks_per_row = (uint)C / 128;
    const device uint* ir[NR];
    for (uint k = 0; k < NR; k++) ir[k] = (const device uint*)(idx + (row0 + k) * C4);
    float acc[NR], lo[NR], hi[NR];
    for (uint k = 0; k < NR; k++) {{ acc[k] = 0.0f; lo[k] = 0.0f; hi[k] = 0.0f; }}
    for (uint j = lane; j < nwords; j += SG) {{
        uint c0 = j * 16;
        const device float4* x4 = (const device float4*)(x + c0);
        float4 xa = x4[0], xb = x4[1], xc = x4[2], xd = x4[3];      // loaded once for NR rows
        for (uint k = 0; k < NR; k++) {{
            uint word = ir[k][j];
            uint e = (uint)exps[(row0 + k) * blocks_per_row + c0 / 128];
            uint cls = q_class(e);
            float s = q_scale(e, cls);
            float4 w0 = (float4)cb4[(word      ) & 0xFFu];
            float4 w1 = (float4)cb4[(word >>  8) & 0xFFu];
            float4 w2 = (float4)cb4[(word >> 16) & 0xFFu];
            float4 w3 = (float4)cb4[(word >> 24) & 0xFFu];
            float d = (metal::dot(xa, w0) + metal::dot(xb, w1) + metal::dot(xc, w2) + metal::dot(xd, w3)) * s;
            if (cls == 0u) acc[k] += d; else if (cls == 1u) lo[k] += d; else hi[k] += d;
        }}
    }}
    for (uint k = 0; k < NR; k++) {{
        float rr = metal::simd_sum(acc[k]);
        float rl = metal::simd_sum(lo[k]);
        float rh = metal::simd_sum(hi[k]);
        if (lane == 0) out[row0 + k] = (rr + rl * LO_REST) + rh * HI_REST;
    }}
"""
    raise ValueError(name)


def s16_variant(name: str) -> str:
    from llmopt.lab import qmetal
    src = qmetal._S16_SRC
    if name == "v3":
        return src
    if name == "nox":
        return src.replace(
            "            const device float4* x4 = (const device float4*)(x + t * (uint)C + c0);\n"
            "            float d = (metal::dot(x4[0], w0) + metal::dot(x4[1], w1)) * s;",
            "            float4 xk = float4(1.0f, 0.5f, 0.25f, 0.125f) + (float)(j & 3u);\n"
            "            float d = (metal::dot(xk, w0) + metal::dot(xk, w1)) * s;")
    if name == "nocodes":
        return src.replace("        uint word = cr[j];", "        uint word = j * 2654435761u + row;")
    if name == "empty":
        return w4_variant("empty")
    m = re.match(r"share(\d+)$", name)
    if m:
        r = int(m.group(1))
        return f"""
    constexpr uint SG = 32;
    constexpr uint NR = {r};
    uint row0 = threadgroup_position_in_grid.x * NR;
    uint lane = thread_position_in_threadgroup.x;
    const uint C2 = (uint)C / 2;
    const uint nwords = C2 / 4;
    const uint blocks_per_row = (uint)C / 128;
    float lvf[16];
    for (uint i = 0; i < 16u; i++) lvf[i] = (float)lv[i];
    const device uint* cr[NR];
    for (uint k = 0; k < NR; k++) cr[k] = (const device uint*)(codes + (row0 + k) * C2);
    float acc[NR], lo[NR], hi[NR];
    for (uint k = 0; k < NR; k++) {{ acc[k] = 0.0f; lo[k] = 0.0f; hi[k] = 0.0f; }}
    for (uint j = lane; j < nwords; j += SG) {{
        uint c0 = j * 8;
        const device float4* x4 = (const device float4*)(x + c0);
        float4 xa = x4[0], xb = x4[1];
        for (uint k = 0; k < NR; k++) {{
            uint word = cr[k][j];
            uint e = (uint)exps[(row0 + k) * blocks_per_row + c0 / 128];
            uint cls = q_class(e);
            float s = q_scale(e, cls);
            uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu;
            uint b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
            float4 w0 = float4(lvf[b0 >> 4], lvf[b0 & 0xFu], lvf[b1 >> 4], lvf[b1 & 0xFu]);
            float4 w1 = float4(lvf[b2 >> 4], lvf[b2 & 0xFu], lvf[b3 >> 4], lvf[b3 & 0xFu]);
            float d = (metal::dot(xa, w0) + metal::dot(xb, w1)) * s;
            if (cls == 0u) acc[k] += d; else if (cls == 1u) lo[k] += d; else hi[k] += d;
        }}
    }}
    for (uint k = 0; k < NR; k++) {{
        float rr = metal::simd_sum(acc[k]);
        float rl = metal::simd_sum(lo[k]);
        float rh = metal::simd_sum(hi[k]);
        if (lane == 0) out[row0 + k] = (rr + rl * LO_REST) + rh * HI_REST;
    }}
"""
    raise ValueError(name)


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import decode_entry

    prov = provenance()
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    repeats = int(os.environ.get("REPEATS", "50"))
    queue = int(os.environ.get("QUEUE", "32"))
    out = os.environ.get("OUT", f"logs/qmetal/traffic_probe_{prov['commit']}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
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

    kernels: dict = {}

    def kern(codec, name):
        key = (codec, name)
        if key not in kernels:
            inputs = ["x", "idx", "cb", "exps"] if codec == "w4" else ["x", "codes", "lv", "exps"]
            src = w4_variant(name) if codec == "w4" else s16_variant(name)
            kernels[key] = mx.fast.metal_kernel(name=f"probe_{codec}_{name}", input_names=inputs,
                                                output_names=["out"], header=qmetal._HEADER, source=src)
        return kernels[key]

    def launch(codec, name, res, x):
        R, C = res.shape
        m = re.match(r"share(\d+)$", name)
        if m:
            r = int(m.group(1))
            if R % r:
                raise ValueError("rows not divisible")
            grid, tg = ((R // r) * 32, 1, 1), (32, 1, 1)
        else:
            grid, tg = (R * 32, 1, 1), (32, 1, 1)
        (o,) = kern(codec, name)(inputs=[x, *res._parts()], template=[("C", C), ("R", R), ("TT", 1)],
                                 grid=grid, threadgroup=tg, output_shapes=[(1, R)], output_dtypes=[mx.float32])
        return o

    def timed_synced(fn):
        for _ in range(5):
            mx.eval(fn())
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.percentile(ts, 90))

    def timed_pipelined(fn_i):
        # QUEUE launches on independent inputs, one eval: per-launch time in
        # the regime where the host does not wait between kernels
        mx.eval(*[fn_i(i) for i in range(4)])
        ts = []
        for _ in range(max(5, repeats // 5)):
            t0 = time.perf_counter()
            mx.eval(*[fn_i(i) for i in range(queue)])
            ts.append((time.perf_counter() - t0) / queue)
        return float(np.median(ts))

    rng = np.random.default_rng(0)
    receipt = {"provenance": prov, "arm": arm, "chain_report": q["report"]["rung0"], "repeats": repeats,
               "queue": queue, "roofline_gbs": ROOFLINE_GBS, "kernel_version_baseline": qmetal.VERSION,
               "note": "TIMING ONLY. Variant outputs are wrong by construction (streams removed); "
                       "no numerical claim. share* are probes of the sharing idea, not parity-fixtured kernels.",
               "matrices": []}
    for name, codec in MATRICES:
        e = MAN[name]
        if e["codec"] != codec:
            raise SystemExit(f"REFUSING: {name} is {e['codec']}")
        buf = payload(e)
        res = qmetal.resident(buf, e)
        R, C = res.shape
        W = decode_entry(buf, e).astype(np.float64)
        xs = [mx.array(rng.normal(size=(1, C)).astype(np.float32)) for _ in range(queue)]
        mx.eval(*xs)
        ref = W @ np.asarray(xs[0][0], dtype=np.float64)
        nbytes = res.payload_bytes()
        xbytes_per_row = C * 4
        variants = (["v3", "nostage", "nox", "noidx", "nocb", "xonly", "empty", "share2", "share4", "share8"]
                    if codec == "w4" else ["v3", "nox", "nocodes", "empty", "share2", "share4"])
        rows = []
        for v in variants:
            try:
                o = launch(codec, v, res, xs[0])
                mx.eval(o)
            except Exception as ex:  # compile or launch failure is itself a finding
                rows.append({"variant": v, "error": str(ex)[:300]})
                print(f"[probe] {codec} {name.split('.')[-2]} {v}: ERROR {str(ex)[:120]}", flush=True)
                continue
            got = np.asarray(o[0], dtype=np.float64)
            sanity = None
            if v == "v3" or v.startswith("share"):
                err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
                sanity = {"max_rel_err_vs_oracle": float(err),
                          "note": "sanity only for the probe harness; the fixtured parity suite is tests/test_qmetal.py"}
            med, p90 = timed_synced(lambda v=v: launch(codec, v, res, xs[0]))
            pipe = timed_pipelined(lambda i, v=v: launch(codec, v, res, xs[i]))
            row = {"variant": v, "synced_median_s": med, "synced_p90_s": p90, "pipelined_per_launch_s": pipe,
                   "payload_gbs_synced": nbytes / med / 1e9, "payload_gbs_pipelined": nbytes / pipe / 1e9,
                   "sanity": sanity}
            rows.append(row)
            print(f"[probe] {codec} {name.split('.')[-2]} {v:8s}: synced {med*1e6:7.0f} us  pipelined {pipe*1e6:7.0f} us  "
                  f"({nbytes/pipe/1e9:5.1f} GB/s pipelined)" + (f"  sanity rel {sanity['max_rel_err_vs_oracle']:.1e}" if sanity else ""),
                  flush=True)
        receipt["matrices"].append({"tensor": name, "codec": codec, "shape": [R, C], "payload_bytes": nbytes,
                                    "x_bytes_per_row_pass": xbytes_per_row,
                                    "x_traffic_bytes_if_reread_per_row": xbytes_per_row * R,
                                    "variants": rows})
    # launch floor: the empty kernel on the largest grid, synced and pipelined
    receipt["inferred"] = {
        "occupancy_and_registers": "not observable from Python/MLX; no compiler statistics are exposed. "
                                   "Inferred only from the timing deltas between variants.",
        "cache_behaviour": "not measured; the x-reread cost is inferred from the nox delta, "
                           "the codebook staging cost from the nostage and nocb deltas.",
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

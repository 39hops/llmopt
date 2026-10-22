"""S16 lookup probe for the fused Metal kernels, TIMING plus a bit-identity
guard. The shipped S16 kernel (llmopt.lab.qmetal._S16_SRC, the kernel every
generation 1 to 4 runs for S16) expands the 16-entry level table into a
per-lane float array and indexes it at run time with each nibble; the
decode breakdown (OBSERVATION QWEN-METAL-DECODE-BREAKDOWN-0) left that
family unexplained: unmoved by removing its activation loads, 9 percent
from its code stream. This probe compiles controlled variants of the
frozen source and times each on real arm C S16 matrices at a T = 1
decode workload. The frozen kernels are not modified.

Diagnostics (outputs WRONG by construction, timing only):
  v3        the shipped kernel (baseline; parity-checked here as a guard
            that the harness feeds real inputs)
  nox       activation loads replaced by a constant
  nocodes   code words replaced by a hash of the lane counter
  notable   table lookups replaced by arithmetic on the nibbles (the
            unpacking stays, the run-time-indexed table goes)
  noexp     exponent byte replaced by a constant (no exps stream, no class)
  xonly     nocodes plus notable: activation loads and the reduction only
  empty     each simdgroup writes one value (launch and drain floor)
Candidates (outputs REAL; each lookup returns the same float32 level
value as the shipped kernel, and the per-row sequence of float32
operations is kept, so bit identity with v3 is expected and checked):
  cmem      no per-lane table: each nibble reads the device level table
  tg16      the 16 levels staged once per threadgroup, looked up there
  tg256     a 256-entry byte table (level of the high nibble, level of the
            low nibble) staged per threadgroup: one gather per code byte,
            the W4 v3 codebook pattern
  shuf      lane i holds level i & 15; each nibble is a simd_shuffle (no
            memory lookup at all; the word loop is made uniform so the
            shuffles stay in convergent control flow)
  tg256_u4  tg256 with 128-bit code loads (a lane owns four consecutive
            words): a DIFFERENT lane-to-word assignment, so the per-row
            summation order changes and bit identity is NOT expected
            (parity-bound only; reported, never promoted on timing alone)
  tg256_share2 / tg256_share4 / shuf_share4
            the W4 v4 pattern on top of a candidate: one simdgroup owns r
            consecutive rows and loads each activation chunk once for all
            r; per-row operation order unchanged (bit identity expected)
Matrices: real arm C payloads: layer 0 in_proj_qkv [10240, 5120],
in_proj_z [6144, 5120], out_proj [5120, 6144]; layer 3 q_proj [12288,
5120], k_proj [1024, 5120]; lm_head [248320, 5120] (HEAD=0 skips it).
Per variant: SYNCED median and p90 of REPEATS single launches (dispatch
plus kernel plus drain) and PIPELINED per-launch time over QUEUE launches
queued on independent inputs before one eval (the tower's regime), both
as payload GB/s; candidates also carry max relative error against the
float64 canonical decode and bit identity with v3 on the same input.
Register pressure, occupancy and cache hit rates are not observable from
Python; everything inferred from deltas is marked so in the receipt.
Receipt logs/qmetal/s16_probe_<commit>.json (refuse-if-exists). Env:
ART_DIR, VENDOR_DIR, REPEATS (50), QUEUE (32), HEAD (1), OUT.
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

MATRICES = [("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "in_proj_qkv"),
            ("model.language_model.layers.0.linear_attn.in_proj_z.weight", "in_proj_z"),
            ("model.language_model.layers.0.linear_attn.out_proj.weight", "out_proj"),
            ("model.language_model.layers.3.self_attn.q_proj.weight", "q_proj"),
            ("model.language_model.layers.3.self_attn.k_proj.weight", "k_proj")]
HEAD = ("lm_head.weight", "lm_head")
DIAGNOSTICS = ["v3", "nox", "nocodes", "notable", "noexp", "xonly", "empty"]
CANDIDATES = ["cmem", "tg16", "tg256", "shuf", "tg256_u4", "tg256_share2", "tg256_share4", "shuf_share4"]
ROOFLINE_GBS = 150.0


def provenance() -> dict:
    import mlx.core as mx
    import mlx_lm
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty": bool(dirty), "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__}


# ---- source pieces -------------------------------------------------------
# The shipped kernel at TT = 1, written out as the common skeleton with
# three holes: TABLE (per-threadgroup or per-lane setup), LOOKUP (two
# float4 of level values from the four code bytes) and the x load. The
# skeleton reproduces _S16_SRC's per-row operation sequence exactly:
# word j of the row is handled by lane j % 32 in increasing j, the word's
# two dots are summed then scaled, and the row is a simd_sum per class.

LOOKUP = {
    "lvf": ("    float lvf[16];\n    for (uint i = 0; i < 16u; i++) lvf[i] = (float)lv[i];\n",
            "float4(lvf[b0 >> 4], lvf[b0 & 0xFu], lvf[b1 >> 4], lvf[b1 & 0xFu])",
            "float4(lvf[b2 >> 4], lvf[b2 & 0xFu], lvf[b3 >> 4], lvf[b3 & 0xFu])"),
    "cmem": ("",
             "float4((float)lv[b0 >> 4], (float)lv[b0 & 0xFu], (float)lv[b1 >> 4], (float)lv[b1 & 0xFu])",
             "float4((float)lv[b2 >> 4], (float)lv[b2 & 0xFu], (float)lv[b3 >> 4], (float)lv[b3 & 0xFu])"),
    "tg16": ("    threadgroup float lvt[16];\n    if (lane < 16u) lvt[lane] = (float)lv[lane];\n"
             "    threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n",
             "float4(lvt[b0 >> 4], lvt[b0 & 0xFu], lvt[b1 >> 4], lvt[b1 & 0xFu])",
             "float4(lvt[b2 >> 4], lvt[b2 & 0xFu], lvt[b3 >> 4], lvt[b3 & 0xFu])"),
    "tg256": ("    threadgroup float2 tb[256];\n"
              "    for (uint i = lane; i < 256u; i += SG) tb[i] = float2((float)lv[i >> 4], (float)lv[i & 0xFu]);\n"
              "    threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n",
              "float4(tb[b0], tb[b1])",
              "float4(tb[b2], tb[b3])"),
    "shuf": ("    float myl = (float)lv[lane & 0xFu];\n",
             "float4(metal::simd_shuffle(myl, (ushort)(b0 >> 4)), metal::simd_shuffle(myl, (ushort)(b0 & 0xFu)), "
             "metal::simd_shuffle(myl, (ushort)(b1 >> 4)), metal::simd_shuffle(myl, (ushort)(b1 & 0xFu)))",
             "float4(metal::simd_shuffle(myl, (ushort)(b2 >> 4)), metal::simd_shuffle(myl, (ushort)(b2 & 0xFu)), "
             "metal::simd_shuffle(myl, (ushort)(b3 >> 4)), metal::simd_shuffle(myl, (ushort)(b3 & 0xFu)))"),
    "notable": ("",
                "float4((float)(b0 >> 4), (float)(b0 & 0xFu), (float)(b1 >> 4), (float)(b1 & 0xFu)) * 0.01f",
                "float4((float)(b2 >> 4), (float)(b2 & 0xFu), (float)(b3 >> 4), (float)(b3 & 0xFu)) * 0.01f"),
}
# tg256 staging is done by the WHOLE threadgroup; in the share variants
# the threadgroup is still one simdgroup of 32 lanes, so the same loop works.


def single_row(table: str, *, word_expr="cr[j]", exp_expr="(uint)exps[row * blocks_per_row + c0 / 128]",
               x_load=("const device float4* x4 = (const device float4*)(x + c0);\n"
                       "        float4 xa = x4[0], xb = x4[1];"),
               uniform=False) -> str:
    setup, l0, l1 = LOOKUP[table]
    if uniform:
        loop_head = ("    for (uint jj = 0; jj < nwords; jj += SG) {\n"
                     "        uint j = jj + lane;\n"
                     "        bool ok = j < nwords;\n"
                     "        uint c0 = (ok ? j : 0u) * 8;\n"
                     f"        uint word = ok ? ({word_expr}) : 0u;\n")
        tail = "        if (ok) { if (cls == 0u) acc += d; else if (cls == 1u) lo += d; else hi += d; }\n"
    else:
        loop_head = ("    for (uint j = lane; j < nwords; j += SG) {\n"
                     f"        uint word = {word_expr};\n"
                     "        uint c0 = j * 8;\n")
        tail = "        if (cls == 0u) acc += d; else if (cls == 1u) lo += d; else hi += d;\n"
    return f"""
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const uint C2 = (uint)C / 2;
    const device uint* cr = (const device uint*)(codes + row * C2);
    const uint nwords = C2 / 4;
    const uint blocks_per_row = (uint)C / 128;
{setup}    float acc = 0.0f, lo = 0.0f, hi = 0.0f;
{loop_head}        uint e = {exp_expr};
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu;
        uint b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
        float4 w0 = {l0};
        float4 w1 = {l1};
        {x_load}
        float d = (metal::dot(xa, w0) + metal::dot(xb, w1)) * s;
{tail}    }}
    float r = metal::simd_sum(acc);
    float rl = metal::simd_sum(lo);
    float rh = metal::simd_sum(hi);
    if (lane == 0) out[row] = (r + rl * LO_REST) + rh * HI_REST;
"""


def share_rows(table: str, nr: int, uniform=False) -> str:
    setup, l0, l1 = LOOKUP[table]
    if uniform:
        loop_head = ("    for (uint jj = 0; jj < nwords; jj += SG) {\n"
                     "        uint j = jj + lane;\n"
                     "        bool ok = j < nwords;\n"
                     "        uint c0 = (ok ? j : 0u) * 8;\n")
        guard_w = "ok ? cr[k][j] : 0u"
        tail = "            if (ok) { if (cls == 0u) acc[k] += d; else if (cls == 1u) lo[k] += d; else hi[k] += d; }\n"
    else:
        loop_head = ("    for (uint j = lane; j < nwords; j += SG) {\n"
                     "        uint c0 = j * 8;\n")
        guard_w = "cr[k][j]"
        tail = "            if (cls == 0u) acc[k] += d; else if (cls == 1u) lo[k] += d; else hi[k] += d;\n"
    return f"""
    constexpr uint SG = 32;
    constexpr uint NR = {nr};
    uint row0 = threadgroup_position_in_grid.x * NR;
    uint lane = thread_position_in_threadgroup.x;
    const uint C2 = (uint)C / 2;
    const uint nwords = C2 / 4;
    const uint blocks_per_row = (uint)C / 128;
{setup}    const device uint* cr[NR];
    for (uint k = 0; k < NR; k++) cr[k] = (const device uint*)(codes + (row0 + k) * C2);
    float acc[NR], lo[NR], hi[NR];
    for (uint k = 0; k < NR; k++) {{ acc[k] = 0.0f; lo[k] = 0.0f; hi[k] = 0.0f; }}
{loop_head}        const device float4* x4 = (const device float4*)(x + c0);
        float4 xa = x4[0], xb = x4[1];
        for (uint k = 0; k < NR; k++) {{
            uint word = {guard_w};
            uint e = (uint)exps[(row0 + k) * blocks_per_row + c0 / 128];
            uint cls = q_class(e);
            float s = q_scale(e, cls);
            uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu;
            uint b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
            float4 w0 = {l0};
            float4 w1 = {l1};
            float d = (metal::dot(xa, w0) + metal::dot(xb, w1)) * s;
{tail}        }}
    }}
    for (uint k = 0; k < NR; k++) {{
        float rr = metal::simd_sum(acc[k]);
        float rl = metal::simd_sum(lo[k]);
        float rh = metal::simd_sum(hi[k]);
        if (lane == 0) out[row0 + k] = (rr + rl * LO_REST) + rh * HI_REST;
    }}
"""


def u4_rows(table: str) -> str:
    """128-bit code loads: lane owns four consecutive words (32 weights) per
    iteration. Different per-row summation order from the shipped kernel."""
    setup, l0, l1 = LOOKUP[table]
    return f"""
    constexpr uint SG = 32;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const uint C2 = (uint)C / 2;
    const device uint4* cr = (const device uint4*)(codes + row * C2);
    const uint nq = C2 / 16;
    const uint blocks_per_row = (uint)C / 128;
{setup}    float acc = 0.0f, lo = 0.0f, hi = 0.0f;
    for (uint q = lane; q < nq; q += SG) {{
        uint4 qq = cr[q];
        uint words[4] = {{qq.x, qq.y, qq.z, qq.w}};
        uint cbase = q * 32;
        uint e = (uint)exps[row * blocks_per_row + cbase / 128];
        uint cls = q_class(e);
        float s = q_scale(e, cls);
        for (uint w = 0; w < 4u; w++) {{
            uint word = words[w];
            uint c0 = cbase + w * 8;
            uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu;
            uint b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
            float4 w0 = {l0};
            float4 w1 = {l1};
            const device float4* x4 = (const device float4*)(x + c0);
            float d = (metal::dot(x4[0], w0) + metal::dot(x4[1], w1)) * s;
            if (cls == 0u) acc += d; else if (cls == 1u) lo += d; else hi += d;
        }}
    }}
    float r = metal::simd_sum(acc);
    float rl = metal::simd_sum(lo);
    float rh = metal::simd_sum(hi);
    if (lane == 0) out[row] = (r + rl * LO_REST) + rh * HI_REST;
"""


EMPTY = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    if (lane == 0) out[row] = (float)row;
"""
NOX = ("float4 xa = float4(1.0f, 0.5f, 0.25f, 0.125f) + (float)(j & 3u);\n"
       "        float4 xb = xa * 0.5f;")


def variant_source(name: str) -> str:
    if name == "v3":
        return single_row("lvf")
    if name == "nox":
        return single_row("lvf", x_load=NOX)
    if name == "nocodes":
        return single_row("lvf", word_expr="j * 2654435761u + row")
    if name == "notable":
        return single_row("notable")
    if name == "noexp":
        return single_row("lvf", exp_expr="120u")
    if name == "xonly":
        return single_row("notable", word_expr="j * 2654435761u + row")
    if name == "empty":
        return EMPTY
    if name in ("cmem", "tg16", "tg256"):
        return single_row(name)
    if name == "shuf":
        return single_row("shuf", uniform=True)
    if name == "tg256_u4":
        return u4_rows("tg256")
    m = re.match(r"(tg256|shuf)_share(\d+)$", name)
    if m:
        return share_rows(m.group(1), int(m.group(2)), uniform=(m.group(1) == "shuf"))
    raise ValueError(name)


def share_of(name: str) -> int:
    m = re.match(r".*_share(\d+)$", name)
    return int(m.group(1)) if m else 1


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
    with_head = os.environ.get("HEAD", "1") == "1"
    out = os.environ.get("OUT", f"logs/qmetal/s16_probe_{prov['commit']}.json")
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

    # the shipped S16 source must still be the one this probe's skeleton reproduces
    if "float lvf[16];" not in qmetal._S16_SRC or "lvf[b0 >> 4]" not in qmetal._S16_SRC:
        raise SystemExit("REFUSING: qmetal._S16_SRC changed; re-derive the probe skeleton")

    kernels: dict = {}

    def kern(name):
        if name not in kernels:
            kernels[name] = mx.fast.metal_kernel(name=f"s16probe_{name}", input_names=["x", "codes", "lv", "exps"],
                                                 output_names=["out"], header=qmetal._HEADER, source=variant_source(name))
        return kernels[name]

    def launch(name, res, x):
        R, C = res.shape
        r = share_of(name)
        if R % r:
            raise ValueError("rows not divisible by the share factor")
        grid, tg = ((R // r) * 32, 1, 1), (32, 1, 1)
        (o,) = kern(name)(inputs=[x, *res._parts()], template=[("C", C), ("R", R)],
                          grid=grid, threadgroup=tg, output_shapes=[(R,)], output_dtypes=[mx.float32])
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
        mx.eval(*[fn_i(i) for i in range(4)])
        ts = []
        for _ in range(max(5, repeats // 5)):
            t0 = time.perf_counter()
            mx.eval(*[fn_i(i) for i in range(queue)])
            ts.append((time.perf_counter() - t0) / queue)
        return float(np.median(ts)), float(np.percentile(ts, 90))

    rng = np.random.default_rng(0)
    mats = list(MATRICES) + ([HEAD] if with_head else [])
    receipt = {"provenance": prov, "arm": arm, "chain_report": q["report"]["rung0"], "repeats": repeats,
               "queue": queue, "roofline_gbs": ROOFLINE_GBS, "kernel_version_baseline": qmetal.VERSION,
               "workload": "T = 1 decode GEMV, float32 activations, one launch per matrix",
               "diagnostics": DIAGNOSTICS, "candidates": CANDIDATES,
               "note": "Diagnostic variants' outputs are wrong by construction (a stream removed); candidates' "
                       "outputs are real and compared with the shipped kernel bit for bit and with the float64 "
                       "canonical decode. Timing is the claim here; the parity suite is tests/test_qmetal.py.",
               "matrices": []}
    for name, short in mats:
        e = MAN[name]
        if e["codec"] != "s16":
            raise SystemExit(f"REFUSING: {name} is {e['codec']}")
        buf = payload(e)
        res = qmetal.resident(buf, e)
        R, C = res.shape
        nbytes = res.payload_bytes()
        xs = [mx.array(rng.normal(size=(C,)).astype(np.float32)) for _ in range(queue)]
        mx.eval(*xs)
        t0 = time.time()
        W = decode_entry(buf, e).astype(np.float64)
        ref = W @ np.asarray(xs[0], dtype=np.float64)
        absdot = np.abs(W) @ np.abs(np.asarray(xs[0], dtype=np.float64))
        del W
        t_dec = time.time() - t0
        # the shipped kernel through the resident (the production call), the bit reference
        y_ship = np.asarray(res.gemv(xs[0]), dtype=np.float32)
        rows = []
        for v in DIAGNOSTICS + CANDIDATES:
            try:
                o = launch(v, res, xs[0])
                mx.eval(o)
            except Exception as ex:
                rows.append({"variant": v, "error": str(ex)[:400]})
                print(f"[s16] {short} {v}: ERROR {str(ex)[:160]}", flush=True)
                continue
            got = np.asarray(o, dtype=np.float32)
            check = None
            if v == "v3" or v in CANDIDATES:
                g64 = got.astype(np.float64)
                err = np.abs(g64 - ref)
                bound = 1e-5 * np.abs(ref) + 2e-6 * absdot
                check = {"bit_identical_to_shipped": bool(np.array_equal(got, y_ship)),
                         "max_rel_err_vs_oracle": float(err.max() / max(np.abs(ref).max(), 1e-30)),
                         "within_parity_bound": bool((err <= bound).all()),
                         "n_rows_differing_from_shipped": int((got != y_ship).sum())}
            med, p90 = timed_synced(lambda v=v: launch(v, res, xs[0]))
            pipe, pipe_p90 = timed_pipelined(lambda i, v=v: launch(v, res, xs[i]))
            row = {"variant": v, "class": "candidate" if v in CANDIDATES else "diagnostic",
                   "synced_median_s": med, "synced_p90_s": p90,
                   "pipelined_per_launch_s": pipe, "pipelined_per_launch_p90_s": pipe_p90,
                   "payload_gbs_synced": nbytes / med / 1e9, "payload_gbs_pipelined": nbytes / pipe / 1e9,
                   "check": check}
            rows.append(row)
            flag = ""
            if check is not None:
                flag = f"  bit={'Y' if check['bit_identical_to_shipped'] else 'n'} rel {check['max_rel_err_vs_oracle']:.1e}"
            print(f"[s16] {short:12s} {v:13s}: synced {med*1e6:8.0f} us  pipelined {pipe*1e6:8.0f} us "
                  f"({nbytes/pipe/1e9:5.1f} GB/s){flag}", flush=True)
        base = next((r for r in rows if r["variant"] == "v3" and "error" not in r), None)
        deltas = {}
        if base:
            b = base["pipelined_per_launch_s"]
            for r in rows:
                if "error" not in r:
                    deltas[r["variant"]] = {"pipelined_ratio_to_v3": r["pipelined_per_launch_s"] / b,
                                            "pipelined_delta_us": (b - r["pipelined_per_launch_s"]) * 1e6}
        receipt["matrices"].append({"tensor": name, "short": short, "shape": [R, C], "payload_bytes": nbytes,
                                    "x_bytes": C * 4, "x_traffic_bytes_if_reread_per_row": C * 4 * R,
                                    "cpu_decode_s": t_dec, "variants": rows, "relative_to_v3": deltas})
        del res, xs
    receipt["inferred"] = {
        "occupancy_registers_cache": "not observable from Python/MLX; inferred only from the deltas between variants "
                                     "(nox: activation traffic; nocodes: code stream; notable: run-time table lookup; "
                                     "noexp: exponent stream and class; xonly: loads plus reduction floor; empty: launch)"}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[s16] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

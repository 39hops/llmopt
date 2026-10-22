"""W4 T = 1 bottleneck probe for the fused Metal kernels, TIMING plus a
bit-identity guard, in the shape of the S16 lookup probe
(scratch/qmetal_s16_probe.py). The shipped W4 kernel under generations 4
and 5 at tiles 1 and 2 is v4 (llmopt.lab.qmetal._W4_V4_SRC): one
simdgroup owns ROWS4 = 4 consecutive rows, stages the 2 KB codebook as
half4 in threadgroup memory once per row group, loads each 16-wide
activation chunk once for the four rows, and per index word gathers four
half4 codebook entries, scales the word's partial dot by the block's
power-of-two scale and accumulates per exponent class; simd_sum per row.
The generation 3 probe (OBSERVATION QWEN-METAL-DECODE-BREAKDOWN-0)
measured v3 (one row per simdgroup) and the share4 variant that became
v4; nothing has measured v4's own composition. This probe compiles
controlled variants of a skeleton that reproduces v4's per-row operation
sequence exactly and times each on real arm C W4 matrices at a T = 1
decode workload with REAL one-token activations captured from the
qualified tower (the input to each probed matrix at one decode step of
a real prompt). The frozen kernels are not modified.

Diagnostics (outputs WRONG by construction, timing only):
  v4        the skeleton (baseline; bit-identical to the shipped resident
            call, checked as a guard that the harness feeds real inputs)
  nox       activation loads replaced by a constant
  noidx     index words replaced by a hash of the lane counter
  nocb      codebook gathers replaced by arithmetic on the index bytes (the
            unpacking stays, the staged table and its gathers go)
  noexp     exponent byte replaced by a constant (no exps stream, no class)
  xonly     noidx plus nocb: activation loads and the reduction only
  empty     each simdgroup writes one value per row (launch and drain floor)
Candidates (outputs REAL; each gather returns the same float32 values as
the shipped kernel and the per-row float32 operation sequence is kept,
so bit identity with v4 is expected and checked unless stated):
  cmem      no threadgroup staging: each index byte reads its half4 entry
            from the device codebook (the S16 v5 pattern, here on a 2 KB
            per-byte table; a hypothesis, not an assumption)
  tgf32     the codebook staged as float4 (4 KB) so the gather returns
            float4 without a half-to-float conversion per lookup
  tgexp     the four rows' exponent bytes staged in threadgroup memory once
            per row group (no per-word device read of the exponent)
  tgf32_tgexp  both
  cmem_tgexp   device codebook reads with the staged exponents (the first
            smoke read cmem and tgexp as the two largest single savings)
  cmem_tgexp_share8  the same with 8 rows per simdgroup
  share2 / share8  the v4 pattern with 2 or 8 rows per simdgroup (the
            share curve around 4; share8 read slower than share4 in the
            generation 3 probe)
  u2        64-bit index loads, a lane owns two consecutive words: a
            DIFFERENT lane-to-word assignment, so the per-row summation
            order changes and bit identity is NOT expected (parity-bound
            only; reported, never promoted on timing alone)
Matrices: real arm C W4 payloads, layers LAYERS (default 0 and 32):
mlp.gate_proj [17408, 5120], mlp.up_proj [17408, 5120], mlp.down_proj
[5120, 17408]. Activations: the tower is built once, a real prompt
(the E0 teacher records) is prefilled and one decode step run with the
probed modules recording their input; the captured float32 vectors are
the x of every timing and parity row (QUEUE independent inputs for the
pipelined measurement are the captured x with a per-launch permutation
of a 16-element block, so each launch reads a different vector of the
same statistics). The tower is released before the timing.
Per variant: SYNCED median and p90 of REPEATS single launches and
PIPELINED per-launch time over QUEUE launches queued before one eval
(the tower's regime), both as payload GB/s; candidates also carry max
relative error against the float64 canonical decode and bit identity
with the shipped resident call on the same input. Register pressure,
occupancy and cache hit rates are not observable from Python; everything
inferred from deltas is marked so in the receipt. Receipt
logs/qmetal/w4_probe_<commit>.json (refuse-if-exists). Env: ART_DIR,
VENDOR_DIR, LAYERS ("0,32"), REPEATS (50), QUEUE (32), OUT, SMOKE=1
(layer 0 only, REPEATS 5, QUEUE 8, receipt under logs/qmetal/smoke/).
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SMOKE = os.environ.get("SMOKE") == "1"
MATS = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
DIAGNOSTICS = ["v4", "nox", "noidx", "nocb", "noexp", "xonly", "empty"]
CANDIDATES = ["cmem", "tgf32", "tgexp", "tgf32_tgexp", "cmem_tgexp", "cmem_tgexp_share8", "share2", "share8", "u2"]
ROOFLINE_GBS = 150.0


def provenance() -> dict:
    import mlx.core as mx
    import mlx_lm
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty": bool(dirty), "smoke": SMOKE, "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__}


# ---- source pieces -------------------------------------------------------
# The shipped v4 kernel at TT = 1 as a skeleton with holes: CB (codebook
# staging and gather), EXP (exponent access), the x load and the word
# load. Word j of a row is handled by lane j % 32 in increasing j; the
# word's four dots are summed then scaled; per-class accumulation; one
# simd_sum per class per row; the same float32 sequence as _W4_V4_SRC.

CB = {
    "tgh": ("    threadgroup half4 cb4[256];\n"
            "    for (uint i = lane; i < 256u; i += SG) cb4[i] = ((const device half4*)cb)[i];\n"
            "    threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n",
            "(float4)cb4[{b}]"),
    "cmem": ("    const device half4* cbd = (const device half4*)cb;\n", "(float4)cbd[{b}]"),
    "tgf32": ("    threadgroup float4 cbf[256];\n"
              "    for (uint i = lane; i < 256u; i += SG) cbf[i] = (float4)((const device half4*)cb)[i];\n"
              "    threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n",
              "cbf[{b}]"),
    "nocb": ("", "(float4((float)({b}), (float)(({b}) >> 2), (float)(({b}) >> 4), (float)(({b}) >> 6)) * 0.001f)"),
}
EXP_DEV = "(uint)exps[(row0 + k) * blocks_per_row + c0 / 128]"
EXP_TG_SETUP = ("    threadgroup uchar et[NR * MAXB];\n"
                "    for (uint i = lane; i < NR * blocks_per_row; i += SG)\n"
                "        et[i] = exps[(row0 + (i / blocks_per_row)) * blocks_per_row + (i % blocks_per_row)];\n"
                "    threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n")
EXP_TG = "(uint)et[k * blocks_per_row + c0 / 128]"
X_DEV = ("        const device float4* x4 = (const device float4*)(x + c0);\n"
         "        float4 xa = x4[0], xb = x4[1], xc = x4[2], xd = x4[3];\n")
X_CONST = ("        float4 xa = float4(1.0f, 0.5f, 0.25f, 0.125f) + (float)(j & 3u);\n"
           "        float4 xb = xa * 0.5f, xc = xa * 0.25f, xd = xa * 0.125f;\n")


def rows_source(cb: str, nr: int, *, exp="dev", x_load=X_DEV, word_expr="ir[k][j]") -> str:
    setup, gather = CB[cb]
    exp_setup = EXP_TG_SETUP if exp == "tg" else ""
    exp_expr = EXP_TG if exp == "tg" else EXP_DEV
    g = [gather.format(b=f"(word >> {s}) & 0xFFu" if s else "word & 0xFFu") for s in (0, 8, 16, 24)]
    return f"""
    constexpr uint SG = 32;
    constexpr uint NR = {nr};
    constexpr uint MAXB = 256;
    uint row0 = threadgroup_position_in_grid.x * NR;
    uint lane = thread_position_in_threadgroup.x;
    const uint C4 = (uint)C / 4;
    const uint nwords = C4 / 4;
    const uint blocks_per_row = (uint)C / 128;
{setup}{exp_setup}    const device uint* ir[NR];
    for (uint k = 0; k < NR; k++) ir[k] = (const device uint*)(idx + (row0 + k) * C4);
    float acc[NR], lo[NR], hi[NR];
    for (uint k = 0; k < NR; k++) {{ acc[k] = 0.0f; lo[k] = 0.0f; hi[k] = 0.0f; }}
    for (uint j = lane; j < nwords; j += SG) {{
        uint c0 = j * 16;
{x_load}        for (uint k = 0; k < NR; k++) {{
            uint word = {word_expr};
            uint e = {exp_expr};
            uint cls = q_class(e);
            float s = q_scale(e, cls);
            float4 w0 = {g[0]};
            float4 w1 = {g[1]};
            float4 w2 = {g[2]};
            float4 w3 = {g[3]};
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


def u2_source(nr: int = 4) -> str:
    """64-bit index loads: a lane owns two consecutive words (32 weights) per
    iteration. Different per-row summation order from the shipped kernel."""
    setup, gather = CB["tgh"]
    g = [gather.format(b=f"(word >> {s}) & 0xFFu" if s else "word & 0xFFu") for s in (0, 8, 16, 24)]
    return f"""
    constexpr uint SG = 32;
    constexpr uint NR = {nr};
    uint row0 = threadgroup_position_in_grid.x * NR;
    uint lane = thread_position_in_threadgroup.x;
    const uint C4 = (uint)C / 4;
    const uint npairs = C4 / 8;
    const uint blocks_per_row = (uint)C / 128;
{setup}    const device uint2* ir[NR];
    for (uint k = 0; k < NR; k++) ir[k] = (const device uint2*)(idx + (row0 + k) * C4);
    float acc[NR], lo[NR], hi[NR];
    for (uint k = 0; k < NR; k++) {{ acc[k] = 0.0f; lo[k] = 0.0f; hi[k] = 0.0f; }}
    for (uint p = lane; p < npairs; p += SG) {{
        uint cbase = p * 32;
        const device float4* x4 = (const device float4*)(x + cbase);
        float4 xs[8];
        for (uint q = 0; q < 8u; q++) xs[q] = x4[q];
        for (uint k = 0; k < NR; k++) {{
            uint2 pair = ir[k][p];
            uint e = (uint)exps[(row0 + k) * blocks_per_row + cbase / 128];
            uint cls = q_class(e);
            float s = q_scale(e, cls);
            float dsum = 0.0f;
            for (uint w = 0; w < 2u; w++) {{
                uint word = w == 0u ? pair.x : pair.y;
                float4 w0 = {g[0]};
                float4 w1 = {g[1]};
                float4 w2 = {g[2]};
                float4 w3 = {g[3]};
                float d = (metal::dot(xs[w * 4 + 0], w0) + metal::dot(xs[w * 4 + 1], w1)
                         + metal::dot(xs[w * 4 + 2], w2) + metal::dot(xs[w * 4 + 3], w3)) * s;
                if (cls == 0u) acc[k] += d; else if (cls == 1u) lo[k] += d; else hi[k] += d;
            }}
            (void)dsum;
        }}
    }}
    for (uint k = 0; k < NR; k++) {{
        float rr = metal::simd_sum(acc[k]);
        float rl = metal::simd_sum(lo[k]);
        float rh = metal::simd_sum(hi[k]);
        if (lane == 0) out[row0 + k] = (rr + rl * LO_REST) + rh * HI_REST;
    }}
"""


EMPTY = """
    constexpr uint NR = 4;
    uint row0 = threadgroup_position_in_grid.x * NR;
    uint lane = thread_position_in_threadgroup.x;
    if (lane < NR) out[row0 + lane] = (float)(row0 + lane);
"""


def variant_source(name: str) -> str:
    if name == "v4":
        return rows_source("tgh", 4)
    if name == "nox":
        return rows_source("tgh", 4, x_load=X_CONST)
    if name == "noidx":
        return rows_source("tgh", 4, word_expr="j * 2654435761u + (row0 + k)")
    if name == "nocb":
        return rows_source("nocb", 4)
    if name == "noexp":
        src = rows_source("tgh", 4)
        return src.replace(f"uint e = {EXP_DEV};", "uint e = 120u;")
    if name == "xonly":
        return rows_source("nocb", 4, word_expr="j * 2654435761u + (row0 + k)")
    if name == "empty":
        return EMPTY
    if name == "cmem":
        return rows_source("cmem", 4)
    if name == "tgf32":
        return rows_source("tgf32", 4)
    if name == "tgexp":
        return rows_source("tgh", 4, exp="tg")
    if name == "tgf32_tgexp":
        return rows_source("tgf32", 4, exp="tg")
    if name == "cmem_tgexp":
        return rows_source("cmem", 4, exp="tg")
    if name == "cmem_tgexp_share8":
        return rows_source("cmem", 8, exp="tg")
    m = re.match(r"share(\d+)$", name)
    if m:
        return rows_source("tgh", int(m.group(1)))
    if name == "u2":
        return u2_source(4)
    raise ValueError(name)


def share_of(name: str) -> int:
    m = re.search(r"share(\d+)$", name)
    return int(m.group(1)) if m else 4


def capture_activations(art: str, vdir: str, paths: list[str]) -> dict:
    """Build the qualified tower, run a real prompt (prefill then one decode
    step) and return the float32 input vector each named module saw at the
    decode step. The tower is deleted before return."""
    import mlx.core as mx
    from qwen_metal_prefill_head import real_tokens
    from qwen_metal_qual import build_metal_model
    model, info = build_metal_model(art, vdir)
    mods = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "QLinear" and name in paths:
            mods[name] = mod
    missing = [p for p in paths if p not in mods]
    if missing:
        raise SystemExit(f"REFUSING: modules not found: {missing}")
    cls = type(next(iter(mods.values())))
    orig = cls.__call__
    captured: dict = {}
    want = {id(m): n for n, m in mods.items()}

    def rec(self, x):
        if id(self) in want:
            captured[want[id(self)]] = np.array(np.asarray(x, dtype=np.float32), copy=True)
        return orig(self, x)
    cls.__call__ = rec
    n_prompt = 32 if SMOKE else 96
    ids = real_tokens(n_prompt + 1)
    cache = model.make_cache()
    lp = model(mx.array([ids[:n_prompt]]), cache=cache)
    mx.eval(lp)
    captured.clear()                      # keep the decode-step inputs only
    lz = model(mx.array([[ids[n_prompt]]]), cache=cache)
    mx.eval(lz)
    cls.__call__ = orig
    out = {k: v.reshape(-1) for k, v in captured.items()}
    bad = [k for k, v in out.items() if v.ndim != 1 or not np.isfinite(v).all()]
    if bad or set(out) != set(paths):
        raise SystemExit(f"REFUSING: capture incomplete or non-finite: {bad} / {sorted(out)}")
    meta = {"prompt_tokens": n_prompt, "decode_token": int(ids[n_prompt]),
            "tower_kernel_version": info["kernel_version"], "chain_sha256": info["chain_sha256"],
            "x_stats": {k: {"rms": float(np.sqrt((v.astype(np.float64) ** 2).mean())), "max_abs": float(np.abs(v).max()),
                            "zero_fraction": float((v == 0).mean())} for k, v in out.items()}}
    del model, cache, lp, lz, mods
    gc.collect()
    mx.clear_cache()
    return {"x": out, "meta": meta}


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import decode_entry

    prov = provenance()
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    layers = [0] if SMOKE else [int(v) for v in os.environ.get("LAYERS", "0,32").split(",") if v]
    repeats = 5 if SMOKE else int(os.environ.get("REPEATS", "50"))
    queue = 8 if SMOKE else int(os.environ.get("QUEUE", "32"))
    out = (f"logs/qmetal/smoke/w4_probe_{prov['commit']}.json" if SMOKE
           else os.environ.get("OUT", f"logs/qmetal/w4_probe_{prov['commit']}.json"))
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    if qmetal.ROWS_V4 != 4 or "threadgroup half4 cb4[256];" not in qmetal._W4_V4_SRC or "xa[t] = x4[0];" not in qmetal._W4_V4_SRC:
        raise SystemExit("REFUSING: qmetal._W4_V4_SRC changed; re-derive the probe skeleton")
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

    tensors = [(f"model.language_model.layers.{L}.{m}.weight", f"L{L}.{m.split('.')[-1]}",
                f"language_model.model.layers.{L}.{m}") for L in layers for m in MATS]
    cap = capture_activations(art, vdir, [p for _, _, p in tensors])

    kernels: dict = {}

    def kern(name):
        if name not in kernels:
            kernels[name] = mx.fast.metal_kernel(name=f"w4probe_{name}", input_names=["x", "idx", "cb", "exps"],
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

    receipt = {"provenance": prov, "arm": arm, "chain_report": q["report"]["rung0"], "repeats": repeats,
               "queue": queue, "roofline_gbs": ROOFLINE_GBS, "kernel_version_baseline": qmetal.VERSION,
               "shipped_w4_dispatch_at_tile_1": "v4 (ROWS4 = 4) under generations 4 and 5",
               "workload": "T = 1 decode GEMV, captured float32 activations of one real decode step, one launch per matrix",
               "activations": cap["meta"], "layers": layers,
               "diagnostics": DIAGNOSTICS, "candidates": CANDIDATES,
               "note": "Diagnostic variants' outputs are wrong by construction (a stream removed); candidates' "
                       "outputs are real and compared with the shipped resident call bit for bit and with the "
                       "float64 canonical decode. Timing is the claim here; the parity suite is tests/test_qmetal.py.",
               "matrices": []}
    rng = np.random.default_rng(0)
    for name, short, path in tensors:
        e = MAN[name]
        if e["codec"] != "w4":
            raise SystemExit(f"REFUSING: {name} is {e['codec']}")
        buf = payload(e)
        res = qmetal.resident(buf, e)
        R, C = res.shape
        nbytes = res.payload_bytes()
        x0 = cap["x"][path]
        if x0.shape[0] != C:
            raise SystemExit(f"REFUSING: captured x length {x0.shape[0]} != C {C} for {name}")
        # independent inputs of the same statistics: a per-launch permutation of 16-element blocks
        xs = [mx.array(x0)]
        for _ in range(queue - 1):
            perm = rng.permutation(C // 16)
            xs.append(mx.array(x0.reshape(-1, 16)[perm].reshape(-1)))
        mx.eval(*xs)
        t0 = time.time()
        W = decode_entry(buf, e).astype(np.float64)
        ref = W @ x0.astype(np.float64)
        absdot = np.abs(W) @ np.abs(x0.astype(np.float64))
        del W
        t_dec = time.time() - t0
        y_ship = np.asarray(res.gemv(xs[0]), dtype=np.float32)
        rows = []
        for v in DIAGNOSTICS + CANDIDATES:
            try:
                o = launch(v, res, xs[0])
                mx.eval(o)
            except Exception as ex:
                rows.append({"variant": v, "error": str(ex)[:400]})
                print(f"[w4] {short} {v}: ERROR {str(ex)[:160]}", flush=True)
                continue
            got = np.asarray(o, dtype=np.float32)
            check = None
            if v == "v4" or v in CANDIDATES:
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
            print(f"[w4] {short:14s} {v:12s}: synced {med*1e6:8.0f} us  pipelined {pipe*1e6:8.0f} us "
                  f"({nbytes/pipe/1e9:5.1f} GB/s){flag}", flush=True)
        base = next((r for r in rows if r["variant"] == "v4" and "error" not in r), None)
        deltas = {}
        if base:
            b = base["pipelined_per_launch_s"]
            for r in rows:
                if "error" not in r:
                    deltas[r["variant"]] = {"pipelined_ratio_to_v4": r["pipelined_per_launch_s"] / b,
                                            "pipelined_delta_us": (b - r["pipelined_per_launch_s"]) * 1e6}
        receipt["matrices"].append({"tensor": name, "short": short, "shape": [R, C], "payload_bytes": nbytes,
                                    "x_bytes": C * 4, "cpu_decode_s": t_dec, "variants": rows, "relative_to_v4": deltas})
        del res, xs
    receipt["inferred"] = {
        "occupancy_registers_cache": "not observable from Python/MLX; inferred only from the deltas between variants "
                                     "(nox: activation traffic; noidx: index stream; nocb: staged codebook gathers; "
                                     "noexp: exponent stream and class; xonly: loads plus reduction floor; empty: launch)"}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[w4] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

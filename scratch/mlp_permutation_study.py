"""SwiGLU intermediate-neuron permutation: mechanical companion to
scratch/leancheck/MlpPermutation.lean (Phase C of the 2026-10-19 round).

The theorem: for a permutation P of the intermediate index, the MLP with
Wg' = P Wg, Wu' = P Wu, Wd' = Wd P^T equals the original for every input
(exact arithmetic, any gate function). The study checks the identity
mechanically and then asks the only question the theorem leaves open: a
block-scaled codec groups weights along a matrix's input dimension, so
the permutation reorders rows of Wg and Wu (block contents unchanged) but
regroups the columns of Wd into new 128-blocks; does a chosen permutation
lower the quantization error of Wd, and does any output improvement come
from that lower error rather than from a changed evaluation order?

Sections:
  exact       Fractions arithmetic on small matrices: permuted equals
              original exactly (bit-for-bit as rationals)
  float       float32 and float64 dense evaluation of original against
              permuted weights on Gaussian inputs: the difference is at the
              reduction-order level (reported in ulps of the output) and is
              zero in exact arithmetic
  grouping    two block-128 e8m0-scaled proxy codecs applied to Wg, Wu,
              Wd: "s16_proxy" (16 scalar levels per block; the arm C S16
              table of a named S16 entry when the artifact is present,
              else 16 uniform levels) and "w4_proxy" (a 256-entry
              codebook over 4-weight vectors fitted by k-means on the
              block-normalised matrix, the shape of the artifact's W4
              codec without its trained codebook or seeds). Arm C's MLP
              matrices are W4 and its attention and IO matrices S16; the
              proxies share the artifact's block scale law, which is the
              part the permutation regroups, and nothing else is claimed
              under identity, random permutations (controls) and
              magnitude-grouping permutations (columns of Wd sorted by
              max-abs or L2 norm); per-matrix relative Frobenius error, the
              fact that Wg/Wu errors are permutation-invariant, and the
              MLP output error against the unquantized float64 reference
              evaluated in float64 (quantization only) and in float32
              (quantization plus evaluation order), with the same quantized
              weights evaluated in both orders to isolate the order term
  sources     synthetic iid Gaussian, synthetic with log-normal per-neuron
              scales (heavy-tailed column magnitudes), and, when ART_DIR
              exists, the decoded arm C layer-0 down_proj as a realistic
              value distribution (requantizing a decoded matrix measures
              the codec's grouping sensitivity on that distribution; it
              says nothing about the information the original compression
              discarded, which is booked as the limitation)
Env: ART_DIR (~/qwen_whole0t/C), OUT (logs/qmetal/mlp_perm/study_<commit>.json;
ignored under SMOKE), SMOKE=1 (small sizes, logs/qmetal/mlp_perm_smoke/).
Refuses to overwrite. Limitation booked with the receipt: the decoded
arm C window is a matrix already quantised on the original block grid,
so the identity permutation keeps that grid and is favoured; on that
source the comparison that carries information is sorted against
random (both leave the grid), never sorted against identity.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SMOKE = os.environ.get("SMOKE") == "1"
BLOCK = 128


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


def fit_codebook(Wn, rng, k=256, iters=12, samples=1 << 18):
    """k-means codebook over 4-weight vectors of the block-normalised matrix."""
    v = Wn.reshape(-1, 4)
    idx = rng.choice(len(v), min(samples, len(v)), replace=False)
    S = v[idx]
    C = S[rng.choice(len(S), k, replace=False)].copy()
    for _ in range(iters):
        d = ((S[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = S[m].mean(0)
    return C


def quantize_w4_proxy(W, rng):
    """Block-128 e8m0 scale, then nearest codebook vector per 4 weights (codebook fitted on this matrix)."""
    R, C = W.shape
    Wb = W.reshape(R, C // BLOCK, BLOCK).astype(np.float64)
    sc = e8m0(Wb)
    Wn = (Wb / sc).reshape(R, C)
    cb = fit_codebook(Wn, rng)
    v = Wn.reshape(-1, 4)
    out = np.empty_like(v)
    for lo in range(0, len(v), 1 << 20):
        d = ((v[lo:lo + (1 << 20), None, :] - cb[None, :, :]) ** 2).sum(-1)
        out[lo:lo + (1 << 20)] = cb[d.argmin(1)]
    return (out.reshape(R, C // BLOCK, BLOCK) * sc).reshape(R, C)


def silu(t):
    return t / (1.0 + np.exp(-t))


def mlp(Wg, Wu, Wd, X, dtype):
    """X[N, d] -> [N, d] with gate silu, all in dtype (numpy matmul order)."""
    Wg, Wu, Wd, X = (np.asarray(a, dtype=dtype) for a in (Wg, Wu, Wd, X))
    g = X @ Wg.T
    u = X @ Wu.T
    return (silu(g) * u) @ Wd.T


def permute(Wg, Wu, Wd, perm):
    return Wg[perm], Wu[perm], Wd[:, perm]


# ------------------------------------------------------------ exact
def exact_section(rng):
    d, m, N = (4, 6, 3) if SMOKE else (6, 10, 4)
    def frac_mat(shape):
        return [[Fraction(int(rng.integers(-5, 6)), int(rng.integers(1, 4))) for _ in range(shape[1])] for _ in range(shape[0])]
    Wg, Wu, Wd = frac_mat((m, d)), frac_mat((m, d)), frac_mat((d, m))
    X = frac_mat((N, d))
    act = lambda t: t * t / (1 + abs(t))          # any function of the gate; rational-closed
    def run(Wg, Wu, Wd):
        out = []
        for x in X:
            g = [sum(Wg[j][k] * x[k] for k in range(d)) for j in range(m)]
            u = [sum(Wu[j][k] * x[k] for k in range(d)) for j in range(m)]
            hmid = [act(g[j]) * u[j] for j in range(m)]
            out.append([sum(Wd[i][j] * hmid[j] for j in range(m)) for i in range(d)])
        return out
    ref = run(Wg, Wu, Wd)
    results = []
    for s in range(3):
        perm = [int(p) for p in rng.permutation(m)]
        Wg2 = [Wg[p] for p in perm]
        Wu2 = [Wu[p] for p in perm]
        Wd2 = [[Wd[i][p] for p in perm] for i in range(d)]
        results.append(run(Wg2, Wu2, Wd2) == ref)
    return {"d": d, "m": m, "inputs": N, "permutations": 3, "all_exactly_equal": all(results), "per_perm": results}


# ------------------------------------------------------------ float
def ulps(a, b):
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    sp = np.spacing(np.abs(np.asarray(a, np.float32))).astype(np.float64)
    return np.abs(a64 - b64) / sp


def float_section(rng):
    d, m, N = (64, 256, 16) if SMOKE else (512, 2048, 64)
    Wg, Wu = rng.standard_normal((m, d)) / np.sqrt(d), rng.standard_normal((m, d)) / np.sqrt(d)
    Wd = rng.standard_normal((d, m)) / np.sqrt(m)
    X = rng.standard_normal((N, d))
    out = {"d": d, "m": m, "inputs": N}
    for dtype in (np.float32, np.float64):
        ref = mlp(Wg, Wu, Wd, X, dtype)
        diffs, ul = [], []
        for s in range(3):
            perm = rng.permutation(m)
            y = mlp(*permute(Wg, Wu, Wd, perm), X, dtype)
            diffs.append(float(np.abs(y.astype(np.float64) - ref.astype(np.float64)).max()))
            ul.append(float(ulps(y, ref).max()) if dtype is np.float32 else None)
        scale = float(np.abs(ref).max())
        out[np.dtype(dtype).name] = {"max_abs_diff_vs_original_order": diffs, "max_abs_diff_over_output_scale": [x / scale for x in diffs],
                                     "max_ulps_float32_note": "ulps of the nearest output element; dominated by near-zero outputs, kept for the record",
                                     "max_ulps_float32": ul, "output_scale": scale, "bit_identical": all(x == 0.0 for x in diffs)}
    return out


# ------------------------------------------------------------ codec
def e8m0(blocks):
    """Per-block power-of-two scale: smallest 2^e with max|block| <= 2^e (zero block -> 2^-127)."""
    mx = np.abs(blocks).max(axis=-1, keepdims=True)
    e = np.ceil(np.log2(np.where(mx > 0, mx, 2.0 ** -127)))
    return np.exp2(e)


def quantize_block16(W, levels):
    """Block-128 along the input dimension, e8m0 scale, nearest of 16 levels (levels in [-1, 1])."""
    R, C = W.shape
    if C % BLOCK:
        raise ValueError("input dimension must be a multiple of 128")
    Wb = W.reshape(R, C // BLOCK, BLOCK).astype(np.float64)
    sc = e8m0(Wb)
    Wn = Wb / sc
    idx = np.abs(Wn[..., None] - levels[None, None, None, :]).argmin(-1)
    return (levels[idx] * sc).reshape(R, C)


def rel_fro(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def arm_c_levels(art):
    """The 16 half levels of an arm C S16 payload (the first S16 entry in
    manifest order, named in the receipt), else 16 uniform levels."""
    try:
        man = json.load(open(os.path.join(art, "manifest.json")))
        name = next(k for k, e in man.items() if e.get("codec") == "s16")
        e = man[name]
        with open(os.path.join(art, e["shard"] + ".bin"), "rb") as f:
            f.seek(e["off"])
            n = e["shape"][0] * e["shape"][1]
            f.seek(n // BLOCK, 1)
            lv = np.frombuffer(f.read(32), dtype=np.float16).astype(np.float64)
        return np.sort(lv), f"arm C S16 table of {name} (an attention or IO matrix; the MLP matrices are W4)"
    except (FileNotFoundError, StopIteration, OSError):
        return np.linspace(-1, 1, 16), "uniform 16 levels"


def grouping_perms(Wd, rng, n_random=3):
    out = {"identity": np.arange(Wd.shape[1])}
    for s in range(n_random):
        out[f"random_{s}"] = rng.permutation(Wd.shape[1])
    out["sort_col_maxabs"] = np.argsort(np.abs(Wd).max(axis=0), kind="stable")
    out["sort_col_l2"] = np.argsort(np.linalg.norm(Wd, axis=0), kind="stable")
    return out


def grouping_section(name, Wg, Wu, Wd, X, levels, rng, codec):
    ref64 = mlp(Wg, Wu, Wd, X, np.float64)
    res = {"source": name, "codec": codec, "shapes": {"Wg": list(Wg.shape), "Wd": list(Wd.shape)}, "inputs": int(X.shape[0]),
           "input_scale_std": float(X.std()), "arms": {}}
    base_q = None
    for pname, perm in grouping_perms(Wd, rng).items():
        Wg2, Wu2, Wd2 = permute(Wg, Wu, Wd, perm)
        t0 = time.time()
        if codec == "s16_proxy":
            Qg, Qu, Qd = quantize_block16(Wg2, levels), quantize_block16(Wu2, levels), quantize_block16(Wd2, levels)
        else:
            qrng = np.random.default_rng(7)      # the same codebook seed for every permutation of a source
            Qg, Qu, Qd = quantize_w4_proxy(Wg2, qrng), quantize_w4_proxy(Wu2, qrng), quantize_w4_proxy(Wd2, qrng)
        tq = time.time() - t0
        inv = np.argsort(perm)
        # the same quantized weights brought back to the original order (identical values, original evaluation order)
        Qg_o, Qu_o, Qd_o = Qg[inv], Qu[inv], Qd[:, inv]
        y64 = mlp(Qg, Qu, Qd, X, np.float64)
        y32 = mlp(Qg, Qu, Qd, X, np.float32)
        y32_o = mlp(Qg_o, Qu_o, Qd_o, X, np.float32)
        y64_o = mlp(Qg_o, Qu_o, Qd_o, X, np.float64)
        y64_wd_only = mlp(Wg2, Wu2, Qd, X, np.float64)        # Wd quantized alone: the regrouped matrix's own contribution
        arm = {"quantize_wall_s": tq,
               "output_rel_err_wd_quantized_only_f64": rel_fro(y64_wd_only, ref64),
               "rel_fro_err": {"Wg": rel_fro(Qg, Wg2), "Wu": rel_fro(Qu, Wu2), "Wd": rel_fro(Qd, Wd2)},
               "wd_blocks_regrouped": bool(not np.array_equal(perm // BLOCK, np.arange(len(perm)) // BLOCK)),
               "output_rel_err_vs_dense_f64": {"eval_f64": rel_fro(y64, ref64), "eval_f32": rel_fro(y32, ref64)},
               "same_quantized_weights_original_order": {"eval_f64": rel_fro(y64_o, ref64), "eval_f32": rel_fro(y32_o, ref64),
                                                          "f64_permuted_vs_original_order_max_abs": float(np.abs(y64 - y64_o).max()),
                                                          "f32_permuted_vs_original_order_max_abs": float(np.abs(y32 - y32_o).max())}}
        if base_q is None:
            base_q = arm["rel_fro_err"]
        else:
            # row permutation keeps every row's blocks intact: exact for the scalar codec; the W4 proxy refits its
            # codebook from a sample of the permuted matrix, so its Wg/Wu errors agree to the sampling, not bitwise
            tol = 1e-12 if codec == "s16_proxy" else 2e-2
            arm["wg_wu_err_equal_to_identity"] = bool(abs(arm["rel_fro_err"]["Wg"] - base_q["Wg"]) < tol and abs(arm["rel_fro_err"]["Wu"] - base_q["Wu"]) < tol)
        res["arms"][pname] = arm
    ident = res["arms"]["identity"]
    for pname, arm in res["arms"].items():
        arm["wd_err_ratio_vs_identity"] = arm["rel_fro_err"]["Wd"] / ident["rel_fro_err"]["Wd"]
        arm["output_err_f64_ratio_vs_identity"] = arm["output_rel_err_vs_dense_f64"]["eval_f64"] / ident["output_rel_err_vs_dense_f64"]["eval_f64"]
        arm["output_err_wd_only_ratio_vs_identity"] = arm["output_rel_err_wd_quantized_only_f64"] / ident["output_rel_err_wd_quantized_only_f64"]
    return res


def sources(rng, art):
    d, m, N = (256, 1024, 64) if SMOKE else (1024, 4096, 128)
    out = []
    Wg, Wu = rng.standard_normal((m, d)) / np.sqrt(d), rng.standard_normal((m, d)) / np.sqrt(d)
    Wd = rng.standard_normal((d, m)) / np.sqrt(m)
    out.append(("gaussian_iid", Wg, Wu, Wd))
    scale = np.exp(rng.standard_normal(m) * 1.0)       # log-normal per-neuron scale, sigma 1
    out.append(("lognormal_neuron_scales_sigma1", Wg * scale[:, None] ** 0.5, Wu * scale[:, None] ** 0.5, Wd * scale[None, :]))
    if os.path.isdir(art) and not SMOKE:
        try:
            from llmopt.lab import qcodec
            man = json.load(open(os.path.join(art, "manifest.json")))
            names = {"gate": "model.language_model.layers.0.mlp.gate_proj.weight", "up": "model.language_model.layers.0.mlp.up_proj.weight",
                     "down": "model.language_model.layers.0.mlp.down_proj.weight"}
            mats = {}
            for k, nm in names.items():
                e = man[nm]
                with open(os.path.join(art, e["shard"] + ".bin"), "rb") as f:
                    f.seek(e["off"])
                    buf = f.read(e["len"])
                mats[k] = qcodec.decode_entry(buf, e).astype(np.float64)
            # a 1,024 x 4,096 window keeps the study bounded: 1,024 hidden rows, 4,096 intermediate neurons
            sel_m = np.arange(4096)
            sel_d = np.arange(1024)
            import hashlib
            shas = {k: hashlib.sha256(mats[k].tobytes()).hexdigest()[:16] for k in mats}
            out.append((f"arm_c_layer0_decoded_window(decoded_sha16 {shas})", mats["gate"][np.ix_(sel_m, sel_d)], mats["up"][np.ix_(sel_m, sel_d)], mats["down"][np.ix_(sel_d, sel_m)]))
        except Exception as ex:  # noqa: BLE001 - the real-matrix source is optional, its absence is booked
            out.append(("arm_c_layer0_decoded_window_unavailable", None, None, str(ex)))
    return out, d, N


def main():
    rng = np.random.default_rng(20261019)
    art = os.environ.get("ART_DIR", os.path.expanduser("~/qwen_whole0t/C"))
    out_dir = "logs/qmetal/mlp_perm_smoke" if SMOKE else "logs/qmetal/mlp_perm"
    out = os.path.join(out_dir, f"study_{git_head()}.json") if SMOKE else os.environ.get("OUT", os.path.join(out_dir, f"study_{git_head()}.json"))
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    levels, lv_src = arm_c_levels(art)
    import hashlib
    chain = "logs/qwenwhole/artifact_digest_C.txt"
    rec = {"code_commit": git_head(), "tree_dirty": git_dirty(), "smoke": SMOKE, "art_dir_present": os.path.isdir(art),
           "art_dir": art.replace(os.path.expanduser("~"), "~"),
           "artifact_manifest_sha256": hashlib.sha256(open(os.path.join(art, "manifest.json"), "rb").read()).hexdigest() if os.path.isdir(art) else None,
           "artifact_chain_sha256": hashlib.sha256(open(chain, "rb").read()).hexdigest() if os.path.exists(chain) else None,
           "levels": [float(v) for v in levels], "levels_source": lv_src, "block": BLOCK,
           "codecs": {"s16_proxy": "block-128 e8m0 scale, nearest of 16 scalar levels",
                      "w4_proxy": "block-128 e8m0 scale, nearest of 256 k-means 4-vectors fitted per matrix (12 iterations, 262,144 samples)"}}
    rec["exact"] = exact_section(rng)
    print("[perm] exact", rec["exact"]["all_exactly_equal"], flush=True)
    rec["float"] = float_section(rng)
    print("[perm] float32 max abs over scale", rec["float"]["float32"]["max_abs_diff_over_output_scale"], "float64", rec["float"]["float64"]["max_abs_diff_over_output_scale"], flush=True)
    srcs, d, N = sources(rng, art)
    rec["grouping"] = []
    for name, Wg, Wu, Wd in srcs:
        if Wg is None:
            rec["grouping"].append({"source": name, "error": Wd})
            continue
        X = rng.standard_normal((N, Wg.shape[1]))
        if name.startswith("arm_c"):
            X = X * 0.05      # a hidden-state scale stand-in, recorded in the receipt (the gate is nonlinear, so results depend on it)
        for codec in ("s16_proxy", "w4_proxy"):
          g = grouping_section(name, Wg, Wu, Wd, X, levels, rng, codec)
          rec["grouping"].append(g)
          a = g["arms"]
          print(f"[perm] {name} {codec}: Wd rel err identity {a['identity']['rel_fro_err']['Wd']:.4f}, random {a['random_0']['rel_fro_err']['Wd']:.4f}, "
              f"sort maxabs {a['sort_col_maxabs']['rel_fro_err']['Wd']:.4f} (x{a['sort_col_maxabs']['wd_err_ratio_vs_identity']:.3f}), "
              f"sort l2 {a['sort_col_l2']['rel_fro_err']['Wd']:.4f}; output f64 ratio maxabs {a['sort_col_maxabs']['output_err_f64_ratio_vs_identity']:.3f} "
              f"l2 {a['sort_col_l2']['output_err_f64_ratio_vs_identity']:.3f} (Wd-only {a['sort_col_l2']['output_err_wd_only_ratio_vs_identity']:.3f}); order-only f32 diff {a['sort_col_l2']['same_quantized_weights_original_order']['f32_permuted_vs_original_order_max_abs']:.2e}", flush=True)
    json.dump(rec, open(out, "w"), indent=1)
    print("[perm] wrote", out)


if __name__ == "__main__":
    main()

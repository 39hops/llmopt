"""STREAM-WDISTILL-0 PASS 1 + PASS 2 (pre-reg RESULTS L30921; amendments
-BUDGET L31031, -CONTRACT L31106, -ARITH L31182, -READING).

Sibling of the FROZEN scratch/stream_wdistill0.py (results-cited: it is
the PASS-0 evidence record and is not edited). Everything registered is
read from the amendments and hardcoded here as CONTRACT constants; no
value below may be tuned after seeing an error.

PASS 1  range-fetch one expert projection -> dequantize -> update
        sufficient statistics -> DISCARD. Nothing accumulates but the
        Gram matrices and a seeded VQ training sample.
SOLVE   exact symmetric eigendecomposition (residual + hidden Grams);
        per-projection residual-VQ codebook stacks under a pinned wall.
PASS 2  range-fetch again -> encode per arm -> serialize -> score.

Zero teacher forward passes. Zero calibration data. Disk residence is
O(one expert).

    SMOKE=1 .venv/bin/python -u scratch/stream_wdistill1.py
    .venv/bin/python -u scratch/stream_wdistill1.py
"""
import io
import json
import os
import struct
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

from llmopt.lab.shards import dequant  # noqa: E402  (frozen exact MXFP4)

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"  # PASS 0
REPO = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
SHARD = "model-00024-of-00048.safetensors"
LAYER = 22

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EXPERTS = int(os.environ.get("N_EXPERTS", "4" if SMOKE else "256"))
BUDGET_NAME = os.environ.get("BUDGET", "B1")

# ---- CONTRACT CONSTANTS (frozen in the amendments; do not tune) ----
VENDOR_BYTES = 3_422_552_064
BUDGET = {"B1": VENDOR_BYTES // 2, "B2": VENDOR_BYTES // 4}[BUDGET_NAME]
SEED = 20260816
D_MODEL, D_FF = 4096, 2048
SCALAR_BITS, SCALAR_BLOCK = 2, 128           # arm A: 2 + 16/128 = 2.125 bpw
VQ_WIDTH, VQ_K = 32, 256                     # arm B
VQ_STAGES = {"B1": 8, "B2": 4}[BUDGET_NAME]
VQ_SAMPLE = 1 << 20
VQ_KMEANSPP_SUB = 1 << 16
VQ_LLOYD_ITERS = 15
VQ_WALL_S = {"B1": 2700.0, "B2": 1350.0}[BUDGET_NAME]
PROBE_N, PROBE_ITERS = 64, 30                # operator probes; power iters
RSVD_VERIFY_N = 8                            # frozen subset for the rSVD check
PROJS = ("w1", "w3", "w2")
RUN_TAG = os.environ.get("RUN_TAG", "")   # a re-run MUST tag itself:
# writing a second execution into a path a booked entry cites is the
# frozen-receipt violation, and a manual rename defeats the guard below.
OUT = (f"logs/streamwd/pass12_{BUDGET_NAME}"
       f"{'_smoke' if SMOKE else ''}{RUN_TAG}.jsonl")

# captured at IMPORT so the field names the commit the run STARTED at,
# not whatever HEAD drifted to while it ran (receipt-auditor S3).
CODE_COMMIT = __import__("subprocess").check_output(
    ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
rng_global = np.random.default_rng(SEED)


# ------------------------------------------------------------------ io
def _get(url, lo=None, hi=None, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "llmopt-streamwd/1 (research)"})
            if lo is not None:
                req.add_header("Range", f"bytes={lo}-{hi - 1}")
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            if lo is not None and len(raw) != hi - lo:
                raise AssertionError(f"truncated {len(raw)} != {hi - lo}")
            return raw
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(2 * (a + 1))


_MAN = {}


def manifest():
    if _MAN:
        return _MAN
    hlen = struct.unpack("<Q", _get(f"{REPO}/{SHARD}", 0, 8))[0]
    hdr = json.loads(_get(f"{REPO}/{SHARD}", 8, 8 + hlen))
    hdr.pop("__metadata__", None)
    for k, v in hdr.items():
        _MAN[k] = (8 + hlen, v["data_offsets"], v["dtype"], v["shape"])
    return _MAN


FETCHED = [0]


def stream_expert(e, proj):
    """Range-fetch + exact dequant -> fp32 [out, in]. Nothing cached."""
    man = manifest()
    base = f"layers.{LAYER}.ffn.experts.{e}.{proj}"
    bo, (wlo, whi), _, wsh = man[f"{base}.weight"]
    _, (slo, shi), _, ssh = man[f"{base}.scale"]
    wraw = _get(f"{REPO}/{SHARD}", bo + wlo, bo + whi)
    sraw = _get(f"{REPO}/{SHARD}", bo + slo, bo + shi)
    FETCHED[0] += len(wraw) + len(sraw)
    packed = np.frombuffer(wraw, np.uint8).reshape(wsh)
    scale = np.frombuffer(sraw, np.uint8).reshape(ssh)
    _, _, w = dequant(packed, scale)
    return w                                   # fp32, [out, in*2]


# ------------------------------------------------------- serialization
def ser_bytes(parts, meta):
    """Exact artifact size: JSON manifest (counted) + raw LE buffers."""
    buf = io.BytesIO()
    mj = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode()
    buf.write(struct.pack("<Q", len(mj)))
    buf.write(mj)
    for a in parts:
        buf.write(np.ascontiguousarray(a).tobytes())
    return buf.getbuffer().nbytes


def pick_rank(bytes_of_r, B, hi):
    """Largest r whose MEASURED serialized size is <= B (dry-run)."""
    lo, best = 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if bytes_of_r(mid) <= B:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# ------------------------------------------------------------ metrics
def q_int8_rows(M):
    """int8 + per-row fp16 scale (the frozen coefficient dtype)."""
    s = np.abs(M).max(axis=1, keepdims=True) / 127.0
    s = np.maximum(s, 1e-12).astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(M / s), -127, 127).astype(np.int8)
    return q, s.astype(np.float16)


def deq_int8_rows(q, s):
    return q.astype(np.float32) * s.astype(np.float32)


def spectral_ratio(D, W, seed):
    """||D||_2 / ||W||_2 by seeded power iteration (fixed iters)."""
    r = np.random.default_rng(seed)

    def top(M):
        v = r.standard_normal(M.shape[1]).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-30
        for _ in range(PROBE_ITERS):
            u = M @ v
            nu = np.linalg.norm(u)
            if nu < 1e-30:
                return 0.0
            u /= nu
            v = M.T @ u
            nv = np.linalg.norm(v)
            if nv < 1e-30:
                return 0.0
            v /= nv
        return float(np.linalg.norm(M @ v))
    return top(D) / max(top(W), 1e-30)


def op_parts(D, W, seed):
    """Seeded isotropic-probe operator error, returned as (num2, den2)
    so the LAYER figure pools sums rather than averaging ratios. The
    same seeded X serves every arm within a tensor."""
    r = np.random.default_rng(seed)
    X = r.standard_normal((PROBE_N, W.shape[1])).astype(np.float32)
    return (float(np.linalg.norm(X @ D.T) ** 2),
            float(np.linalg.norm(X @ W.T) ** 2))


# ------------------------------------------------------------------ vq
def kmeans(V, K, seed, iters):
    r = np.random.default_rng(seed)
    sub = V[r.choice(len(V), min(VQ_KMEANSPP_SUB, len(V)), replace=False)]
    C = [sub[r.integers(len(sub))]]
    d2 = ((sub - C[0]) ** 2).sum(1)
    for _ in range(K - 1):
        p = d2 / max(d2.sum(), 1e-30)
        C.append(sub[r.choice(len(sub), p=p)])
        d2 = np.minimum(d2, ((sub - C[-1]) ** 2).sum(1))
    C = np.stack(C).astype(np.float32)
    for _ in range(iters):
        a = assign(V, C)
        # scatter-add centroid update (bincount, not a K-loop): same
        # Lloyd step, ~100x faster, and the wall is a registered gate
        cnt = np.bincount(a, minlength=K).astype(np.float32)
        S = np.zeros((K, V.shape[1]), np.float32)
        np.add.at(S, a, V)
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
    # codebooks are charged at fp16; decode from the stored values
    return C.astype(np.float16).astype(np.float32)


def assign(V, C, chunk=1 << 16):
    out = np.empty(len(V), np.int32)
    cn = (C ** 2).sum(1)
    for i in range(0, len(V), chunk):
        v = V[i:i + chunk]
        d = cn[None, :] - 2.0 * (v @ C.T)
        out[i:i + chunk] = d.argmin(1)
    return out


def main():
    os.makedirs("logs/streamwd", exist_ok=True)
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t0 = time.time()
    print(f"[wd1] budget {BUDGET_NAME} = {BUDGET:,} B | experts "
          f"{N_EXPERTS} | smoke {SMOKE}", flush=True)

    # ---------------------------------------------------------- PASS 1
    Cin = np.zeros((D_MODEL, D_MODEL), np.float64)
    Cout = np.zeros((D_MODEL, D_MODEL), np.float64)
    Hin = np.zeros((D_FF, D_FF), np.float64)      # arm E, hidden axis
    Hout = np.zeros((D_FF, D_FF), np.float64)
    samp = {p: [] for p in PROJS}
    per_exp = max(1, VQ_SAMPLE // max(N_EXPERTS, 1))
    t_p1 = time.time()
    for e in range(N_EXPERTS):
        for p in PROJS:
            W = stream_expert(e, p)
            if p in ("w1", "w3"):
                Cin += (W.T @ W).astype(np.float64)      # [4096,4096]
                Hin += (W @ W.T).astype(np.float64)      # [2048,2048]
            else:
                Cout += (W @ W.T).astype(np.float64)     # [4096,4096]
                Hout += (W.T @ W).astype(np.float64)     # [2048,2048]
            v = W.reshape(-1, VQ_WIDTH)
            idx = np.random.default_rng(SEED + 7919 * e).choice(
                len(v), min(per_exp, len(v)), replace=False)
            samp[p].append(v[idx].copy())
            del W
        if (e + 1) % 16 == 0 or e + 1 == N_EXPERTS:
            print(f"  [pass1] {e+1}/{N_EXPERTS} experts "
                  f"({FETCHED[0]/2**20:.0f} MiB, {time.time()-t_p1:.0f}s)",
                  flush=True)
    p1_wall = time.time() - t_p1
    samp = {p: np.concatenate(v)[:VQ_SAMPLE] for p, v in samp.items()}
    print(f"[wd1] PASS1 done {p1_wall:.0f}s, fetched "
          f"{FETCHED[0]/2**20:.0f} MiB", flush=True)

    # ----------------------------------------------------------- SOLVE
    def topvec(G, r):
        """Exact: full symmetric eigendecomposition, top-r."""
        w, V = np.linalg.eigh(G)
        V = np.ascontiguousarray(V[:, ::-1][:, :r])
        # fp16 is the CHARGED dtype: round the basis to what the
        # artifact actually stores, then compute in fp32.
        return V.astype(np.float16).astype(np.float32)

    def topvec_rand(G, r, seed, oversample=64, power=4):
        """Randomized top-r for a PSD Gram. Used only for arm D's 512
        per-expert decompositions (exact eigh there is hours); verified
        against topvec() on the frozen 8-expert subset below, per the
        -CONTRACT clause. Solver strength (oversample/power) was raised
        from 32/2 to 64/4 BEFORE the real run on the smoke's SOLVER-
        ACCURACY reading (0.0062 at the degenerate rank 2048-of-4096),
        never on any arm's error; disclosed in the verdict."""
        rg = np.random.default_rng(seed)
        n = G.shape[0]
        Om = rg.standard_normal((n, min(r + oversample, n))).astype(np.float64)
        Y = G @ Om
        for _ in range(power):
            Y = G @ Y
        Q, _ = np.linalg.qr(Y)
        T = Q.T @ G @ Q
        w, U = np.linalg.eigh(T)
        V = np.ascontiguousarray(Q @ U[:, ::-1][:, :r])
        return V.astype(np.float16).astype(np.float32)

    def subspace_dev(Ve, Va, G):
        """Relative captured-energy deviation between two r-subspaces."""
        ea = float(np.trace(Va.T @ (G @ Va)))
        ee = float(np.trace(Ve.T @ (G @ Ve)))
        return abs(ee - ea) / max(abs(ee), 1e-30)

    # arm C rank by MEASURED serialized size
    def bytes_C(r):
        parts = [np.zeros((D_MODEL, r), np.float16),
                 np.zeros((D_MODEL, r), np.float16)]
        for _ in range(N_EXPERTS):
            parts += [np.zeros((D_FF, r), np.int8),
                      np.zeros((D_FF, r), np.int8),
                      np.zeros((r, D_FF), np.int8),
                      np.zeros((D_FF, 1), np.float16),
                      np.zeros((D_FF, 1), np.float16),
                      np.zeros((r, 1), np.float16)]
        return ser_bytes(parts, {"arm": "C", "r": r})

    def bytes_D(r):
        parts = []
        for _ in range(N_EXPERTS):
            parts += [np.zeros((D_MODEL, r), np.float16),
                      np.zeros((D_MODEL, r), np.float16),
                      np.zeros((D_FF, r), np.int8),
                      np.zeros((D_FF, r), np.int8),
                      np.zeros((r, D_FF), np.int8),
                      np.zeros((D_FF, 1), np.float16),
                      np.zeros((D_FF, 1), np.float16),
                      np.zeros((r, 1), np.float16)]
        return ser_bytes(parts, {"arm": "D", "r": r})

    def bytes_E(r):
        parts = [np.zeros((D_FF, r), np.float16),
                 np.zeros((D_FF, r), np.float16)]
        for _ in range(N_EXPERTS):
            parts += [np.zeros((r, D_MODEL), np.int8),
                      np.zeros((r, D_MODEL), np.int8),
                      np.zeros((D_MODEL, r), np.int8),
                      np.zeros((r, 1), np.float16),
                      np.zeros((r, 1), np.float16),
                      np.zeros((D_MODEL, 1), np.float16)]
        return ser_bytes(parts, {"arm": "E", "r": r})

    def bytes_A():
        parts = []
        for _ in range(N_EXPERTS):
            for rows, cols in ((D_FF, D_MODEL), (D_FF, D_MODEL),
                               (D_MODEL, D_FF)):
                parts.append(np.zeros(rows * cols * SCALAR_BITS // 8,
                                      np.uint8))
                parts.append(np.zeros((rows, cols // SCALAR_BLOCK),
                                      np.float16))
        return ser_bytes(parts, {"arm": "A"})

    def bytes_B(stages):
        parts = [np.zeros((stages, VQ_K, VQ_WIDTH), np.float16)
                 for _ in PROJS]
        n_el = N_EXPERTS * 3 * D_FF * D_MODEL
        parts.append(np.zeros(n_el // VQ_WIDTH * stages, np.uint8))
        return ser_bytes(parts, {"arm": "B", "stages": stages})

    rC = pick_rank(bytes_C, BUDGET, D_MODEL)
    rD = pick_rank(bytes_D, BUDGET, D_FF)
    rE = pick_rank(bytes_E, BUDGET, D_FF)
    print(f"[wd1] measured ranks  C {rC} ({bytes_C(rC):,} B)  "
          f"D {rD} ({bytes_D(rD):,} B)  E {rE} ({bytes_E(rE):,} B)",
          flush=True)

    Vin, Vout = topvec(Cin, rC), topvec(Cout, rC)
    Uh, Vh = topvec(Hin, rE), topvec(Hout, rE)

    # arm B codebooks, per projection, under the pinned wall
    t_vq = time.time()
    books, vq_ok = {}, True
    for p in PROJS:
        V = samp[p].astype(np.float32)
        res, stack = V.copy(), []
        for s in range(VQ_STAGES):
            if time.time() - t_vq > VQ_WALL_S:
                vq_ok = False
                print(f"  [vq] WALL HIT at {p} stage {s}", flush=True)
                break
            C = kmeans(res, VQ_K, SEED + 104729 * s, VQ_LLOYD_ITERS)
            res = res - C[assign(res, C)]
            stack.append(C)
            print(f"  [vq] {p} stage {s+1}/{VQ_STAGES} "
                  f"({time.time()-t_vq:.0f}s)", flush=True)
        books[p] = stack
        if not vq_ok:
            break
    stages_done = min(len(v) for v in books.values()) if books else 0
    print(f"[wd1] VQ stages complete {stages_done}/{VQ_STAGES} "
          f"({time.time()-t_vq:.0f}s, wall {VQ_WALL_S:.0f}s) "
          f"-> BAR2-eligible {stages_done == VQ_STAGES}", flush=True)

    # ---------------------------------------------------------- PASS 2
    ARMS = "ABCDE"
    rsvd_dev = []
    acc = {a: {"se": 0.0, "n2": 0.0} for a in ARMS}
    spec, opr = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    t_p2 = time.time()

    def scalarA(W):
        Wb = W.reshape(W.shape[0], -1, SCALAR_BLOCK)
        lim = 2 ** (SCALAR_BITS - 1) - 1
        s = np.abs(Wb).max(-1, keepdims=True) / lim
        s = np.maximum(s, 1e-12).astype(np.float16).astype(np.float32)
        return (np.clip(np.rint(Wb / s), -lim, lim) * s).reshape(W.shape)

    def vqB(W, p):
        res = W.reshape(-1, VQ_WIDTH).copy()
        out = np.zeros_like(res)
        for C in books[p][:stages_done]:
            a = assign(res, C)
            out += C[a]
            res -= C[a]
        return out.reshape(W.shape)

    def proj_rows(W, V):      # W [rows, d] , V [d, r]  -> W V V^T
        q, s = q_int8_rows(W @ V)
        return deq_int8_rows(q, s) @ V.T

    def proj_cols(W, V):      # W [d, cols] , V [d, r]  -> V V^T W
        q, s = q_int8_rows(V.T @ W)
        return V @ deq_int8_rows(q, s)

    for e in range(N_EXPERTS):
        # all three projections together: arm D's private input basis is
        # JOINT over W1 and W3 (the faithful analogue of C's pooled Cin).
        Ws = {p: stream_expert(e, p) for p in PROJS}
        # --- arm D: per-expert bases, EXACT eigh of the expert's own
        # Grams (the randomized path failed its 1e-3 verification at
        # 0.0243 on the B1 run: a near-isotropic spectrum has no gap
        # for power iteration to exploit, which is the finding itself)
        gin = (Ws["w1"].T @ Ws["w1"] + Ws["w3"].T @ Ws["w3"]).astype(np.float64)
        gout = (Ws["w2"] @ Ws["w2"].T).astype(np.float64)
        Vin_e, Vout_e = topvec(gin, rD), topvec(gout, rD)
        if e < RSVD_VERIFY_N:   # kept: exact-v-randomized delta, descriptive
            rsvd_dev.append(subspace_dev(
                Vin_e, topvec_rand(gin, rD, SEED + 31 * e), gin))
        for p, W in Ws.items():
            recon = {"A": scalarA(W)}
            if stages_done:
                recon["B"] = vqB(W, p)
            if p in ("w1", "w3"):
                recon["C"] = proj_rows(W, Vin)
                recon["D"] = proj_rows(W, Vin_e)
                recon["E"] = proj_cols(W, Uh)
            else:
                recon["C"] = proj_cols(W, Vout)
                recon["D"] = proj_cols(W, Vout_e)
                recon["E"] = proj_rows(W, Vh)
            for a, R in recon.items():
                Dm = R - W
                acc[a]["se"] += float((Dm ** 2).sum())
                acc[a]["n2"] += float((W ** 2).sum())
                spec[a].setdefault(p, []).append(
                    spectral_ratio(Dm, W, SEED + 13))
                n2, d2 = op_parts(Dm, W, SEED + 17)
                q = opr[a].setdefault(p, [0.0, 0.0])
                q[0] += n2
                q[1] += d2
            del recon
        del Ws, gin, gout, Vin_e, Vout_e
        if (e + 1) % 8 == 0 or e + 1 == N_EXPERTS:
            print(f"  [pass2] {e+1}/{N_EXPERTS} "
                  f"({time.time()-t_p2:.0f}s)", flush=True)

    bA, bB = bytes_A(), bytes_B(stages_done or VQ_STAGES)
    ab = {"A": bA, "B": bB, "C": bytes_C(rC), "D": bytes_D(rD),
          "E": bytes_E(rE)}
    for a, v in ab.items():
        print(f"[wd1] arm {a} bytes {v:,} <= {BUDGET:,} ? {v <= BUDGET}",
              flush=True)
    import subprocess
    row = {"budget": BUDGET_NAME, "budget_bytes": BUDGET, "smoke": SMOKE,
           "n_experts": N_EXPERTS, "revision": REVISION,
           "ranks": {"C": rC, "D": rD, "E": rE},
           "arm_bytes": ab,
           "arm_within_budget": {a: bool(v <= BUDGET) for a, v in ab.items()},
           "bits_per_weight": {a: 8 * v / (N_EXPERTS * 3 * D_FF * D_MODEL)
                               for a, v in ab.items()},
           "config": {"model": MODEL, "shard": SHARD, "layer": LAYER,
                      "seed": SEED, "scalar_bits": SCALAR_BITS,
                      "scalar_block": SCALAR_BLOCK, "vq_width": VQ_WIDTH,
                      "vq_K": VQ_K, "vq_sample": VQ_SAMPLE,
                      "vq_lloyd_iters": VQ_LLOYD_ITERS,
                      "armD_solver": "exact_eigh"},
           "code_commit": CODE_COMMIT,
           "rsvd_max_dev": (max(rsvd_dev) if rsvd_dev else None),
           "rsvd_verify_n": RSVD_VERIFY_N,
           "rsvd_role": "descriptive_exact_vs_randomized",
           "rsvd_gating": False,
           "rsvd_experts": list(range(min(RSVD_VERIFY_N, N_EXPERTS))),
           "vq_stages_done": stages_done, "vq_target": VQ_STAGES,
           "vq_bar2_eligible": stages_done == VQ_STAGES,
           "frob": {a: (acc[a]["se"] / acc[a]["n2"]) ** 0.5
                    for a in acc if acc[a]["n2"]},
           "spectral_pooled": {a: {p: float(np.mean(x)) for p, x in v.items()}
                               for a, v in spec.items() if v},
           "operator_pooled": {
               a: {p: (v[p][0] / max(v[p][1], 1e-30)) ** 0.5 for p in v}
               for a, v in opr.items() if v},
           "operator_layer": {
               a: (sum(x[0] for x in v.values())
                   / max(sum(x[1] for x in v.values()), 1e-30)) ** 0.5
               for a, v in opr.items() if v},
           "metrics_n_experts": N_EXPERTS,
           "fetched_MiB": round(FETCHED[0] / 2 ** 20, 1),
           "pass1_s": round(p1_wall, 1),
           "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[wd1] FROBENIUS {json.dumps(row['frob'], indent=None)}",
          flush=True)
    print(f"[wd1] -> {OUT}  wall {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

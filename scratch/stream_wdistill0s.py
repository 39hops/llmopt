"""STREAM-WDISTILL-0S: fair scalar baseline + width decomposition
(pre-reg RESULTS L31503; amendments -0S-DESIGN, -0S-CONTROL, -0S-SPEC,
-0S-METRIC; executable projection docs/preregs/stream-wdistill-0s.json).

Sibling of FROZEN scratch/stream_wdistill1.py, which it IMPORTS for the
identical streaming/scoring harness (range-fetch + exact MXFP4 dequant,
seeded operator probes, pooled-by-sums accounting, measured
serialization). Nothing in the frozen file is edited.

THE LADDER, every arm at 2 + 8/128 = 2.0625 bpw index+scale payload,
per-block E8M0 (exponent-only, ROUND UP) normalization at block 128:
  S1-T    ternary max-anchored {-1,0,+1} in a uint2 field (the true
          minimal repair of EXEC1's arm A: scale dtype only)
  S1-U4   4-level uniform mid-rise {-1,-1/3,+1/3,+1}, nearest,
          ties away from zero
  S2      globally optimal 4-level scalar on the frozen 4096-bin
          empirical discretization (exact DP over (count,sum,sumsq)
          bin primitives; interval cost at the nearest-representable
          fp16 level — optimal FOR THE STORED DECODER)
  W4      residual VQ w=4  K=256 s=1   (2.000 index bpw)
  W8      residual VQ w=8  K=256 s=2   (2.000 index bpw)
  W32     residual VQ w=32 K=256 s=8   (2.000 index bpw)
  W*-shuf shuffled twins: independent per-block permutation
          (seeded matrix argsort, rng key (shuffle_seed, proj, expert)),
          OWN codebook trained on the shuffled sample; W32 carries
          three frozen seeds (20260816/17/18, BAR 3), W4/W8 one
          descriptive seed (20260816).

BARS (all pooled operator error over 256 experts; -0S-METRIC: these
are Frobenius bars carrying Monte-Carlo variance, kept as registered):
  1 VQ-SURVIVES            W32 beats min(S1-T, S1-U4, S2) by >= 10%
  2 VECTOR-STRUCTURE-PAYS  W4 beats S2 by >= 10%
  3 LOCALITY-IS-REAL       natural W32 beats the MEAN of its three
                           shuffled twins by >= 5% AND all three
REFUTED-IF best scalar within 5% of W32.

    SMOKE=1 .venv/bin/python -u scratch/stream_wdistill0s.py
    .venv/bin/python -u scratch/stream_wdistill0s.py
"""
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

import stream_wdistill1 as wd1  # noqa: E402  (frozen harness, imported)

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EXPERTS = int(os.environ.get("N_EXPERTS", "4" if SMOKE else "256"))
RUN_TAG = os.environ.get("RUN_TAG", "")
OUT = (f"logs/streamwd/pass0s_B1{'_smoke' if SMOKE else ''}{RUN_TAG}.jsonl")

# ---- CONTRACT CONSTANTS (frozen in pre-reg + amendments; no tuning) --
SEED = 20260816
BLOCK = 128
BITS_PAYLOAD = 2 + 8 / BLOCK                    # 2.0625 bpw, every arm
S2_BINS = 4096
VQ_K = 256
VQ_SAMPLE = 1 << 20
VQ_KMEANSPP_SUB = 1 << 16
VQ_LLOYD_ITERS = 15
WIDTHS = {"W4": (4, 1), "W8": (8, 2), "W32": (32, 8)}   # width, stages
SHUF_SEEDS = {"W4": (20260816,), "W8": (20260816,),
              "W32": (20260816, 20260817, 20260818)}
CODEBOOK_WALL_S = float(os.environ.get("CODEBOOK_WALL_S", "2700"))
BUDGET = wd1.BUDGET                              # B1 = 1,711,276,032
PROJS = wd1.PROJS
D_MODEL, D_FF = wd1.D_MODEL, wd1.D_FF
CODE_COMMIT = wd1.CODE_COMMIT


# ------------------------------------------------------ E8M0 + scalars
def e8m0_scale(Wb):
    """Per-block exponent-only scale, ROUND UP (pinned -0S-DESIGN (4)):
    exponent = ceil(log2(blockmax)), so the block max is always
    representable and max-anchored arms never overload."""
    mx = np.abs(Wb).max(-1, keepdims=True)
    e = np.ceil(np.log2(np.maximum(mx, 2.0 ** -126)))
    e = np.clip(e, -127, 127)
    return np.exp2(e).astype(np.float32)


def blocks(W):
    return W.reshape(W.shape[0], -1, BLOCK)


LEVELS_T = np.array([-1.0, 0.0, 1.0], np.float32)
LEVELS_U4 = np.array([-1.0, -1 / 3, 1 / 3, 1.0], np.float32)


def nearest_level(Wn, levels):
    """Nearest level, ties AWAY FROM ZERO (pinned -0S-SPEC (4)):
    argmin on |w - l| with a tiny bias toward larger |l| on exact ties
    is realized by preferring the later index among equal distances,
    with levels sorted by |l| ascending within each sign pair."""
    d = np.abs(Wn[..., None] - levels)          # [..., L]
    # bias: subtract epsilon * |level| so exact ties resolve to the
    # larger-magnitude level; epsilon far below any real gap
    d -= (np.abs(levels) * 1e-7)
    return levels[d.argmin(-1)]


def s2_levels_dp(count, ssum, ssq):
    """Exact optimal 4-level scalar quantizer on the frozen binning.

    DP over contiguous bin intervals (optimal 1-D cells are contiguous).
    Interval cost uses (count, sum, sumsq) primitives and the NEAREST
    REPRESENTABLE fp16 level (-0S-SPEC (1)): for interval [i, j) with
    count c, sum s, sumsq q and L = fp16(s/c),
    cost = q - 2*L*s + L^2*c. Inner minimization vectorized over i.
    Returns 4 fp16-representable levels."""
    n = len(count)
    cc = np.concatenate([[0.0], np.cumsum(count)])
    cs = np.concatenate([[0.0], np.cumsum(ssum)])
    cq = np.concatenate([[0.0], np.cumsum(ssq)])
    K = 4
    INF = float("inf")
    cost = np.full((K + 1, n + 1), INF)
    lev = np.zeros((K + 1, n + 1))
    cut = np.zeros((K + 1, n + 1), np.int64)
    cost[0, 0] = 0.0
    for k in range(1, K + 1):
        prev = cost[k - 1]
        for j in range(k, n + 1):
            i = np.arange(k - 1, j)
            c = cc[j] - cc[i]
            sm = cs[j] - cs[i]
            q = cq[j] - cq[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                L = np.where(c > 0, sm / np.maximum(c, 1e-300), 0.0)
            L = L.astype(np.float16).astype(np.float64)
            ic = np.where(c > 0, q - 2 * L * sm + L * L * c, 0.0)
            tot = prev[i] + ic
            b = int(np.argmin(tot))
            cost[k, j], cut[k, j], lev[k, j] = tot[b], i[b], L[b]
    levels, j = [], n
    for k in range(K, 0, -1):
        levels.append(lev[k, j])
        j = cut[k, j]
    return np.array(sorted(levels), np.float32)


# ---------------------------------------------------------- shuffling
def block_perms(shuf_seed, proj_idx, expert, n_blocks, inverse=False):
    """Independent permutation per block (-0S-SPEC (2)): one seeded
    matrix per (shuffle_seed, projection, expert), row b argsorted =
    block b's permutation. Fully reproducible; no reuse across blocks."""
    r = np.random.default_rng([shuf_seed, proj_idx, expert])
    P = np.argsort(r.random((n_blocks, BLOCK)), axis=1)
    return np.argsort(P, axis=1) if inverse else P


def apply_perm(Wn_blocks, P):
    return np.take_along_axis(Wn_blocks, P[None] if P.ndim == 2 and
                              Wn_blocks.ndim == 3 else P, axis=-1)


# --------------------------------------------------------------- main
def main():
    os.makedirs("logs/streamwd", exist_ok=True)
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t0 = time.time()
    print(f"[wd0s] experts {N_EXPERTS} | smoke {SMOKE} | "
          f"budget {BUDGET:,} B", flush=True)

    # -------------------------------------------------- byte accounting
    n_el = N_EXPERTS * 3 * D_FF * D_MODEL
    n_blocks_total = n_el // BLOCK

    def ser(parts, meta):
        return wd1.ser_bytes(parts, meta)

    def bytes_scalar(name, n_levels_book):
        parts = [np.zeros(n_el // 4, np.uint8),          # 2-bit codes
                 np.zeros(n_blocks_total, np.uint8)]     # E8M0 exponents
        if n_levels_book:                                # S2 codebook
            parts.append(np.zeros((3, n_levels_book), np.float16))
        return ser(parts, {"arm": name, "block": BLOCK})

    def bytes_vq(name, width, stages, shuf_seed=None):
        parts = [np.zeros(n_el // 4, np.uint8),          # 2.000 bpw idx
                 np.zeros(n_blocks_total, np.uint8),     # E8M0 exponents
                 np.zeros((3, stages, VQ_K, width), np.float16)]
        meta = {"arm": name, "block": BLOCK, "K": VQ_K, "stages": stages}
        if shuf_seed is not None:
            meta["shuffle_seed"] = shuf_seed             # perms derive
        return ser(parts, meta)

    arm_bytes = {"S1-T": bytes_scalar("S1-T", 0),
                 "S1-U4": bytes_scalar("S1-U4", 0),
                 "S2": bytes_scalar("S2", 4)}
    for nm, (w, s) in WIDTHS.items():
        arm_bytes[nm] = bytes_vq(nm, w, s)
        for ss in SHUF_SEEDS[nm]:
            arm_bytes[f"{nm}-shuf{ss}"] = bytes_vq(nm, w, s, ss)
    for a, v in sorted(arm_bytes.items()):
        ok = v <= BUDGET
        print(f"[wd0s] arm {a:12s} {v:,} B <= {BUDGET:,} ? {ok} "
              f"({8 * v / n_el:.4f} bpw)", flush=True)
        if not ok:
            raise SystemExit(f"REFUSING: arm {a} over budget at dry-run")

    # ------------------------------------------------------------ PASS 1
    # samples of NORMALIZED values: per width natural + shuffled twins,
    # plus the S2 histogram primitives over the normalized scalars.
    # codebooks and S2 levels are PER-PROJECTION (frozen recipe)
    per_exp = max(1, VQ_SAMPLE // max(N_EXPERTS, 1))
    nat = {(nm, p): [] for nm in WIDTHS for p in PROJS}
    shuf = {(nm, ss, p): [] for nm in WIDTHS for ss in SHUF_SEEDS[nm]
            for p in PROJS}
    hist_c = {p: np.zeros(S2_BINS, np.float64) for p in PROJS}
    hist_s = {p: np.zeros(S2_BINS, np.float64) for p in PROJS}
    hist_q = {p: np.zeros(S2_BINS, np.float64) for p in PROJS}
    edges = np.linspace(-1.0, 1.0, S2_BINS + 1)
    t1 = time.time()
    for e in range(N_EXPERTS):
        for pi, p in enumerate(PROJS):
            W = wd1.stream_expert(e, p)
            Wb = blocks(W)
            sc = e8m0_scale(Wb)
            Wn = Wb / sc                          # in [-1, 1] by round-up
            flat = Wn.reshape(-1)
            bi = np.clip(np.searchsorted(edges, flat, "right") - 1,
                         0, S2_BINS - 1)
            hist_c[p] += np.bincount(bi, minlength=S2_BINS)
            hist_s[p] += np.bincount(bi, weights=flat, minlength=S2_BINS)
            hist_q[p] += np.bincount(bi,
                                     weights=flat.astype(np.float64) ** 2,
                                     minlength=S2_BINS)
            nb = Wn.shape[0] * Wn.shape[1]
            Wn2 = Wn.reshape(nb, BLOCK)
            rs = np.random.default_rng(SEED + 7919 * e + pi)
            for nm, (w, _) in WIDTHS.items():
                v = Wn2.reshape(-1, w)
                idx = rs.choice(len(v), min(per_exp, len(v)),
                                replace=False)
                nat[(nm, p)].append(v[idx].copy())
                for ss in SHUF_SEEDS[nm]:
                    P = block_perms(ss, pi, e, nb)
                    vs = np.take_along_axis(Wn2, P, axis=1).reshape(-1, w)
                    shuf[(nm, ss, p)].append(vs[idx].copy())
            del W, Wb, Wn, Wn2
        if (e + 1) % 16 == 0 or e + 1 == N_EXPERTS:
            print(f"  [pass1] {e+1}/{N_EXPERTS} "
                  f"({wd1.FETCHED[0]/2**20:.0f} MiB, "
                  f"{time.time()-t1:.0f}s)", flush=True)
    nat = {k: np.concatenate(v)[:VQ_SAMPLE] for k, v in nat.items()}
    shuf = {k: np.concatenate(v)[:VQ_SAMPLE] for k, v in shuf.items()}
    print(f"[wd0s] PASS1 {time.time()-t1:.0f}s "
          f"fetched {wd1.FETCHED[0]/2**20:.0f} MiB", flush=True)

    # ------------------------------------------------------------- SOLVE
    s2lv = {p: s2_levels_dp(hist_c[p], hist_s[p], hist_q[p])
            for p in PROJS}
    for p in PROJS:
        print(f"[wd0s] S2 DP levels {p}: {s2lv[p].tolist()}", flush=True)

    def train_stack(V, stages, tag):
        res, stack = V.astype(np.float32).copy(), []
        for s in range(stages):
            if time.time() - t_vq > CODEBOOK_WALL_S:
                print(f"  [vq] WALL HIT {tag} stage {s}", flush=True)
                return stack, False
            C = wd1.kmeans(res, VQ_K, SEED + 104729 * s + zlib.crc32(tag.encode()) % 9973,
                           VQ_LLOYD_ITERS)
            res = res - C[wd1.assign(res, C)]
            stack.append(C)
            print(f"  [vq] {tag} stage {s+1}/{stages} "
                  f"({time.time()-t_vq:.0f}s)", flush=True)
        return stack, True

    t_vq = time.time()
    books, walled = {}, []
    for nm, (w, s) in WIDTHS.items():
        for p in PROJS:
            books[(nm, p)], ok = train_stack(nat[(nm, p)], s, f"{nm}/{p}")
            if not ok:
                walled.append(f"{nm}/{p}")
        for ss in SHUF_SEEDS[nm]:
            key = f"{nm}-shuf{ss}"
            for p in PROJS:
                books[(key, p)], ok = train_stack(
                    shuf[(nm, ss, p)], s, f"{key}/{p}")
                if not ok:
                    walled.append(f"{key}/{p}")
    vq_wall = time.time() - t_vq
    print(f"[wd0s] codebook phase {vq_wall:.0f}s "
          f"(wall {CODEBOOK_WALL_S:.0f}s) walled={walled}", flush=True)
    del nat, shuf

    # ------------------------------------------------------------ PASS 2
    ARM_ORDER = (["S1-T", "S1-U4", "S2"] + list(WIDTHS)
                 + [f"{nm}-shuf{ss}" for nm in WIDTHS
                    for ss in SHUF_SEEDS[nm]])
    acc = {a: {"se": 0.0, "n2": 0.0} for a in ARM_ORDER}
    spec = {a: {} for a in ARM_ORDER}
    opr = {a: {} for a in ARM_ORDER}

    def vq_recon(Vv, stack):
        res, out = Vv.copy(), np.zeros_like(Vv)
        for C in stack:
            a = wd1.assign(res, C)
            out += C[a]
            res -= C[a]
        return out

    t2 = time.time()
    for e in range(N_EXPERTS):
        for pi, p in enumerate(PROJS):
            W = wd1.stream_expert(e, p)
            Wb = blocks(W)
            sc = e8m0_scale(Wb)
            Wn = Wb / sc
            nb = Wn.shape[0] * Wn.shape[1]
            Wn2 = Wn.reshape(nb, BLOCK)
            recon = {
                "S1-T": nearest_level(Wn, LEVELS_T),
                "S1-U4": nearest_level(Wn, LEVELS_U4),
                "S2": nearest_level(Wn, s2lv[p]),
            }
            for nm, (w, s) in WIDTHS.items():
                if books[(nm, p)]:
                    recon[nm] = vq_recon(
                        Wn2.reshape(-1, w),
                        books[(nm, p)]).reshape(Wn.shape)
                for ss in SHUF_SEEDS[nm]:
                    key = f"{nm}-shuf{ss}"
                    if not books[(key, p)]:
                        continue
                    P = block_perms(ss, pi, e, nb)
                    Pi = block_perms(ss, pi, e, nb, inverse=True)
                    xs = np.take_along_axis(Wn2, P, axis=1)
                    ys = vq_recon(xs.reshape(-1, w),
                                  books[(key, p)]).reshape(nb, BLOCK)
                    recon[key] = np.take_along_axis(
                        ys, Pi, axis=1).reshape(Wn.shape)
            for a, Rn in recon.items():
                R = (Rn * sc).reshape(W.shape)
                Dm = R - W
                acc[a]["se"] += float((Dm ** 2).sum())
                acc[a]["n2"] += float((W ** 2).sum())
                spec[a].setdefault(p, []).append(
                    wd1.spectral_ratio(Dm, W, SEED + 13))
                n2, d2 = wd1.op_parts(Dm, W, SEED + 17)
                q = opr[a].setdefault(p, [0.0, 0.0])
                q[0] += n2
                q[1] += d2
            del W, Wb, Wn, Wn2, recon
        if (e + 1) % 8 == 0 or e + 1 == N_EXPERTS:
            print(f"  [pass2] {e+1}/{N_EXPERTS} ({time.time()-t2:.0f}s)",
                  flush=True)

    op_layer = {a: (sum(x[0] for x in v.values())
                    / max(sum(x[1] for x in v.values()), 1e-30)) ** 0.5
                for a, v in opr.items() if v}
    row = {"rung": "STREAM-WDISTILL-0S", "smoke": SMOKE,
           "n_experts": N_EXPERTS, "revision": wd1.REVISION,
           "code_commit": CODE_COMMIT,
           "arm_bytes": arm_bytes,
           "arm_within_budget": {a: bool(v <= BUDGET)
                                 for a, v in arm_bytes.items()},
           "bits_per_weight": {a: 8 * v / n_el
                               for a, v in arm_bytes.items()},
           "s2_levels": {p: v.tolist() for p, v in s2lv.items()},
           "s2_bins": S2_BINS,
           "codebook_wall_s": CODEBOOK_WALL_S,
           "codebook_phase_s": round(vq_wall, 1),
           "codebook_walled_arms": walled,
           "config": {"seed": SEED, "block": BLOCK, "vq_K": VQ_K,
                      "widths": {k: v for k, v in WIDTHS.items()},
                      "shuf_seeds": {k: list(v)
                                     for k, v in SHUF_SEEDS.items()},
                      "vq_sample": VQ_SAMPLE,
                      "vq_lloyd_iters": VQ_LLOYD_ITERS,
                      "scale": "E8M0_round_up"},
           "frob": {a: (acc[a]["se"] / acc[a]["n2"]) ** 0.5
                    for a in acc if acc[a]["n2"]},
           "spectral_mean_of_ratios_DESCRIPTIVE": {
               a: {p: float(np.mean(x)) for p, x in v.items()}
               for a, v in spec.items() if v},
           "operator_layer": op_layer,
           "metrics_n_experts": N_EXPERTS,
           "fetched_MiB": round(wd1.FETCHED[0] / 2 ** 20, 1),
           "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[wd0s] operator_layer "
          f"{json.dumps({k: round(v, 6) for k, v in op_layer.items()})}",
          flush=True)
    print(f"[wd0s] -> {OUT} wall {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

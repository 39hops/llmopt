"""STREAM-WDISTILL harness v2 — SPEED PROTOTYPE (shard cache + GPU).

NOT a registered instrument. The v1 harness (frozen
scratch/stream_wdistill1.py + scratch/stream_wdistill0s.py) is the
evidence path for the 0S rung running on the Mac. This file explores
how fast the same computation gets with two levers v1 cannot take
mid-rung:

  1. SHARD CACHE. Download model-00024-of-00048.safetensors ONCE to
     local disk (HF_TOKEN honored if set), sha-verifiable, then every
     pass reads expert ranges by mmap — fetch cost ~0 after the first
     run instead of ~40 min of sequential HTTPS ranges per pass.
  2. GPU COMPUTE. k-means, residual-VQ assign, nearest-level scalars,
     probes and spectral power iteration in torch (CUDA on the 3080,
     falls back to CPU).

CORRECTNESS GATE, stated up front: v2 is compared against the Mac v1
SMOKE receipt (logs/streamwd/pass0s_B1_smoke.jsonl, same 4 experts,
same seeds) within TOLERANCE — different device, different BLAS,
different reduction orders, so bit-identity is not expected and no
cross-device number from this file may enter a registered bar. If v2
ever becomes the instrument, it re-runs v1's receipt ON THE SAME
DEVICE first (the eigh-replaces-rSVD precedent).

RNG parity: numpy Generators with v1's exact seeds drive every random
draw (sampling, k-means init, probes); torch is used only for
deterministic linear algebra on arrays those generators produce. So
v1-v2 deltas isolate floating-point/order effects, not stream drift.

    SHARD_CACHE=~/shards SMOKE=1 python -u scratch/streamwd_v2.py
    (WSL: TORCH_DISABLE_NATIVE_JIT=1, see CLAUDE.md)
"""
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

try:
    import torch
    DEV = ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:                                    # numpy-only fallback
    torch = None
    DEV = "numpy"

from llmopt.lab.shards import dequant  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
REPO = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
LAYER = int(os.environ.get("LAYER", "22"))
SHARD = None                     # resolved from the official index

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_EXPERTS = int(os.environ.get("N_EXPERTS", "4" if SMOKE else "256"))
CACHE_DIR = os.path.expanduser(os.environ.get("SHARD_CACHE", "~/shards"))
OUT = (f"logs/streamwd/v2census_L{LAYER}"
       f"{'_smoke' if SMOKE else ''}.jsonl"
       if "LAYER" in os.environ else
       f"logs/streamwd/v2proto{'_smoke' if SMOKE else ''}.jsonl")

# v1 contract constants (verbatim from stream_wdistill0s.py)
SEED = 20260816
BLOCK = 128
S2_BINS = 4096
VQ_K = 256
VQ_SAMPLE = 1 << 20
VQ_KMEANSPP_SUB = 1 << 16
VQ_LLOYD_ITERS = 15
WIDTHS = {"W4": (4, 1), "W8": (8, 2), "W32": (32, 8)}
SHUF_SEEDS = {"W4": (20260816,), "W8": (20260816,),
              "W32": (20260816, 20260817, 20260818)}
PROBE_N, PROBE_ITERS = 64, 30
D_MODEL, D_FF = 4096, 2048
PROJS = ("w1", "w3", "w2")


# ------------------------------------------------------- shard cache
def resolve_shard() -> str:
    """The shard holding LAYER's routed experts, from the official
    safetensors index (never guessed from filename arithmetic).
    Refuses if the layer's expert tensors span multiple shards."""
    global SHARD
    if SHARD:
        return SHARD
    os.makedirs(CACHE_DIR, exist_ok=True)
    ip = os.path.join(CACHE_DIR, f"{REVISION[:12]}_index.json")
    if not os.path.exists(ip):
        req = urllib.request.Request(
            f"{REPO}/model.safetensors.index.json",
            headers={"User-Agent": "llmopt-streamwd-v2 (research)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            open(ip, "wb").write(r.read())
    wm = json.load(open(ip))["weight_map"]
    shards = {v for k, v in wm.items()
              if f"layers.{LAYER}.ffn.experts." in k}
    if len(shards) != 1:
        raise SystemExit(f"REFUSING: layer {LAYER} experts span "
                         f"{sorted(shards)}")
    SHARD = shards.pop()
    return SHARD


def shard_path() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{REVISION[:12]}_{resolve_shard()}")


def ensure_shard() -> str:
    p = shard_path()
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    url = f"{REPO}/{resolve_shard()}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "llmopt-streamwd-v2 (research)"})
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:                     # standard hf CLI token file
        tf = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(tf):
            tok = open(tf).read().strip()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    print(f"[v2] downloading {resolve_shard()} -> {p} "
          f"(auth={'yes' if tok else 'no'})", flush=True)
    t = time.time()
    tmp = p + ".part"
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 24)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got % (1 << 30) < (1 << 24):
                print(f"  [dl] {got/2**30:.1f}/{total/2**30:.1f} GiB "
                      f"({got/max(time.time()-t,1e-9)/2**20:.0f} MiB/s)",
                      flush=True)
    try:
        os.replace(tmp, p)
    except FileNotFoundError:
        # concurrent downloader won the replace; accept its file
        if not os.path.exists(p):
            raise
    print(f"[v2] shard cached in {time.time()-t:.0f}s "
          f"({os.path.getsize(p)/2**30:.2f} GiB)", flush=True)
    return p


_MAN, _MM = {}, [None]


def manifest():
    if _MAN:
        return _MAN
    p = ensure_shard()
    _MM[0] = np.memmap(p, dtype=np.uint8, mode="r")
    hlen = struct.unpack("<Q", bytes(_MM[0][:8]))[0]
    hdr = json.loads(bytes(_MM[0][8:8 + hlen]))
    hdr.pop("__metadata__", None)
    for k, v in hdr.items():
        _MAN[k] = (8 + hlen, v["data_offsets"], v["dtype"], v["shape"])
    return _MAN


def stream_expert(e, proj):
    man = manifest()
    base = f"layers.{LAYER}.ffn.experts.{e}.{proj}"
    bo, (wlo, whi), _, wsh = man[f"{base}.weight"]
    _, (slo, shi), _, ssh = man[f"{base}.scale"]
    packed = np.array(_MM[0][bo + wlo:bo + whi]).reshape(wsh)
    scale = np.array(_MM[0][bo + slo:bo + shi]).reshape(ssh)
    _, _, w = dequant(packed, scale)
    return w


# ----------------------------------------------------- torch helpers
def T(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV)


def e8m0_scale_t(Wb):
    mx = Wb.abs().amax(-1, keepdim=True)
    e = torch.clamp(torch.ceil(torch.log2(torch.clamp(mx, min=2.0 ** -126))),
                    -127, 127)
    return torch.exp2(e)


LEVELS_T = np.array([-1.0, 0.0, 1.0], np.float32)
LEVELS_U4 = np.array([-1.0, -1 / 3, 1 / 3, 1.0], np.float32)


def nearest_level_t(Wn, levels):
    d = (Wn.unsqueeze(-1) - levels).abs() - levels.abs() * 1e-7
    return levels[d.argmin(-1)]


def assign_t(V, C, chunk=1 << 18):
    cn = (C * C).sum(1)
    out = torch.empty(len(V), dtype=torch.long, device=V.device)
    for i in range(0, len(V), chunk):
        v = V[i:i + chunk]
        out[i:i + chunk] = (cn[None] - 2.0 * (v @ C.T)).argmin(1)
    return out


def kmeans_t(Vn, K, seed, iters):
    """v1's kmeans (kmeans++ subsample init + Lloyd + fp16 rounding),
    numpy rng for every random draw, torch for the linear algebra."""
    r = np.random.default_rng(seed)
    sub_idx = r.choice(len(Vn), min(VQ_KMEANSPP_SUB, len(Vn)), replace=False)
    sub = Vn[T(sub_idx)] if isinstance(sub_idx, np.ndarray) else Vn[sub_idx]
    C = [sub[int(r.integers(len(sub)))]]
    d2 = ((sub - C[0]) ** 2).sum(1)
    for _ in range(K - 1):
        p = (d2 / torch.clamp(d2.sum(), min=1e-30)).cpu().numpy().astype(
            np.float64)
        p /= p.sum()
        C.append(sub[int(r.choice(len(sub), p=p))])
        d2 = torch.minimum(d2, ((sub - C[-1]) ** 2).sum(1))
    C = torch.stack(C)
    for _ in range(iters):
        a = assign_t(Vn, C)
        cnt = torch.bincount(a, minlength=K).float()
        S = torch.zeros_like(C)
        S.index_add_(0, a, Vn)
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
    return C.half().float()


def vq_recon_t(Vv, stack):
    res, out = Vv.clone(), torch.zeros_like(Vv)
    for C in stack:
        a = assign_t(res, C)
        out += C[a]
        res -= C[a]
    return out


def block_perms(shuf_seed, proj_idx, expert, n_blocks, inverse=False):
    r = np.random.default_rng([shuf_seed, proj_idx, expert])
    P = np.argsort(r.random((n_blocks, BLOCK)), axis=1)
    return np.argsort(P, axis=1) if inverse else P


def spectral_ratio_t(D, W, seed):
    r = np.random.default_rng(seed)

    def top(M):
        v = T(r.standard_normal(M.shape[1]).astype(np.float32))
        v = v / (v.norm() + 1e-30)
        for _ in range(PROBE_ITERS):
            u = M @ v
            nu = u.norm()
            if nu < 1e-30:
                return 0.0
            u = u / nu
            v = M.T @ u
            nv = v.norm()
            if nv < 1e-30:
                return 0.0
            v = v / nv
        return float((M @ v).norm())
    return top(D) / max(top(W), 1e-30)


def op_parts_t(D, W, seed):
    r = np.random.default_rng(seed)
    X = T(r.standard_normal((PROBE_N, W.shape[1])).astype(np.float32))
    return (float(((X @ D.T) ** 2).sum()), float(((X @ W.T) ** 2).sum()))


# ------------------------------------------------------ S2 (unchanged)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "wd0s_ref", "scratch/stream_wdistill0s.py")
_wd0s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wd0s)
s2_levels_dp = _wd0s.s2_levels_dp                # identical DP, CPU


def main():
    os.makedirs("logs/streamwd", exist_ok=True)
    t0 = time.time()
    print(f"[v2] device {DEV} | experts {N_EXPERTS} | smoke {SMOKE}",
          flush=True)
    ensure_shard()
    t_fetch0 = time.time()

    # ---------------- PASS 1: samples + histograms (v1 rng, GPU math)
    per_exp = max(1, VQ_SAMPLE // max(N_EXPERTS, 1))
    nat = {(nm, p): [] for nm in WIDTHS for p in PROJS}
    shuf = {(nm, ss, p): [] for nm in WIDTHS for ss in SHUF_SEEDS[nm]
            for p in PROJS}
    hist = {p: [np.zeros(S2_BINS, np.float64) for _ in range(3)]
            for p in PROJS}
    edges = T(np.linspace(-1.0, 1.0, S2_BINS + 1).astype(np.float32))
    t1 = time.time()
    for e in range(N_EXPERTS):
        for pi, p in enumerate(PROJS):
            W = T(stream_expert(e, p))
            Wb = W.reshape(W.shape[0], -1, BLOCK)
            sc = e8m0_scale_t(Wb)
            Wn = Wb / sc
            flat = Wn.reshape(-1)
            bi = torch.clamp(torch.bucketize(flat, edges, right=True) - 1,
                             0, S2_BINS - 1)
            hist[p][0] += torch.bincount(
                bi, minlength=S2_BINS).double().cpu().numpy()
            hist[p][1] += torch.bincount(
                bi, weights=flat.double(),
                minlength=S2_BINS).cpu().numpy()
            hist[p][2] += torch.bincount(
                bi, weights=(flat.double() ** 2),
                minlength=S2_BINS).cpu().numpy()
            nb = Wn.shape[0] * Wn.shape[1]
            Wn2 = Wn.reshape(nb, BLOCK)
            rs = np.random.default_rng(SEED + 7919 * e + pi)
            for nm, (w, _) in WIDTHS.items():
                v = Wn2.reshape(-1, w)
                idx = rs.choice(len(v), min(per_exp, len(v)), replace=False)
                ti = T(idx)
                nat[(nm, p)].append(v[ti].cpu().numpy())
                for ss in SHUF_SEEDS[nm]:
                    P = T(block_perms(ss, pi, e, nb))
                    vs = torch.gather(Wn2, 1, P).reshape(-1, w)
                    shuf[(nm, ss, p)].append(vs[ti].cpu().numpy())
    p1 = time.time() - t1
    nat = {k: np.concatenate(v)[:VQ_SAMPLE] for k, v in nat.items()}
    shuf = {k: np.concatenate(v)[:VQ_SAMPLE] for k, v in shuf.items()}
    print(f"[v2] PASS1 {p1:.0f}s", flush=True)

    # ------------------------------------------------------ SOLVE
    s2lv = {p: s2_levels_dp(*hist[p]) for p in PROJS}
    t_vq = time.time()
    books = {}

    def train_stack(Vn, stages, tag):
        res, stack = Vn.clone(), []
        for st in range(stages):
            C = kmeans_t(res, VQ_K,
                         SEED + 104729 * st + zlib.crc32(tag.encode()) % 9973,
                         VQ_LLOYD_ITERS)
            res = res - C[assign_t(res, C)]
            stack.append(C)
        return stack
    for nm, (w, s) in WIDTHS.items():
        for p in PROJS:
            books[(nm, p)] = train_stack(T(nat[(nm, p)]), s, f"{nm}/{p}")
        for ss in SHUF_SEEDS[nm]:
            key = f"{nm}-shuf{ss}"
            for p in PROJS:
                books[(key, p)] = train_stack(
                    T(shuf[(nm, ss, p)]), s, f"{key}/{p}")
    vq_s = time.time() - t_vq
    print(f"[v2] codebooks {vq_s:.0f}s", flush=True)
    del nat, shuf

    # ------------------------------------------------------ PASS 2
    ARMS = (["S1-T", "S1-U4", "S2"] + list(WIDTHS)
            + [f"{nm}-shuf{ss}" for nm in WIDTHS for ss in SHUF_SEEDS[nm]])
    acc = {a: [0.0, 0.0] for a in ARMS}
    opr = {a: {} for a in ARMS}
    spec = {a: {} for a in ARMS}
    lt = T(LEVELS_T)
    lu4 = T(LEVELS_U4)
    s2t = {p: T(s2lv[p]) for p in PROJS}
    t2 = time.time()
    for e in range(N_EXPERTS):
        for pi, p in enumerate(PROJS):
            Wnp = stream_expert(e, p)
            W = T(Wnp)
            Wb = W.reshape(W.shape[0], -1, BLOCK)
            sc = e8m0_scale_t(Wb)
            Wn = Wb / sc
            nb = Wn.shape[0] * Wn.shape[1]
            Wn2 = Wn.reshape(nb, BLOCK)
            recon = {"S1-T": nearest_level_t(Wn, lt),
                     "S1-U4": nearest_level_t(Wn, lu4),
                     "S2": nearest_level_t(Wn, s2t[p])}
            for nm, (w, s) in WIDTHS.items():
                recon[nm] = vq_recon_t(
                    Wn2.reshape(-1, w), books[(nm, p)]).reshape(Wn.shape)
                for ss in SHUF_SEEDS[nm]:
                    key = f"{nm}-shuf{ss}"
                    P = T(block_perms(ss, pi, e, nb))
                    Pi = T(block_perms(ss, pi, e, nb, inverse=True))
                    xs = torch.gather(Wn2, 1, P)
                    ys = vq_recon_t(xs.reshape(-1, w),
                                    books[(key, p)]).reshape(nb, BLOCK)
                    recon[key] = torch.gather(ys, 1, Pi).reshape(Wn.shape)
            for a, Rn in recon.items():
                R = (Rn * sc).reshape(W.shape)
                Dm = R - W
                acc[a][0] += float((Dm ** 2).sum())
                acc[a][1] += float((W ** 2).sum())
                spec[a].setdefault(p, []).append(
                    spectral_ratio_t(Dm, W, SEED + 13))
                n2, d2 = op_parts_t(Dm, W, SEED + 17)
                q = opr[a].setdefault(p, [0.0, 0.0])
                q[0] += n2
                q[1] += d2
            del recon, W, Wb, Wn, Wn2
        if (e + 1) % 8 == 0 or e + 1 == N_EXPERTS:
            print(f"  [pass2] {e+1}/{N_EXPERTS} ({time.time()-t2:.0f}s)",
                  flush=True)
    p2 = time.time() - t2

    code_commit = __import__("subprocess").check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    row = {"proto": "streamwd_v2", "device": DEV, "smoke": SMOKE,
           "layer": LAYER, "shard": resolve_shard(),
           "code_commit": code_commit,
           "n_experts": N_EXPERTS, "revision": REVISION,
           "s2_levels": {p: v.tolist() for p, v in s2lv.items()},
           "frob": {a: (acc[a][0] / acc[a][1]) ** 0.5 for a in ARMS},
           "operator_layer": {
               a: (sum(x[0] for x in v.values())
                   / max(sum(x[1] for x in v.values()), 1e-30)) ** 0.5
               for a, v in opr.items() if v},
           "spectral_mean_of_ratios_DESCRIPTIVE": {
               a: {p: float(np.mean(x)) for p, x in v.items()}
               for a, v in spec.items() if v},
           "pass1_s": round(p1, 1), "codebook_s": round(vq_s, 1),
           "pass2_s": round(p2, 1),
           "wall_s": round(time.time() - t0, 1),
           "shard_cached": True,
           "note": "SPEED PROTOTYPE — tolerance-compare vs v1 smoke "
                   "receipt only; never a registered number"}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[v2] operator_layer "
          f"{json.dumps({k: round(v, 6) for k, v in row['operator_layer'].items()})}",
          flush=True)
    print(f"[v2] -> {OUT} wall {row['wall_s']}s "
          f"(p1 {p1:.0f} vq {vq_s:.0f} p2 {p2:.0f})", flush=True)


if __name__ == "__main__":
    main()

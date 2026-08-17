"""QWEN-STREAM-PROBE-0: does the 0S codec ranking transport to a
DENSE model's FFN? Descriptive, single layer, 3080.

Subject: Qwen/Qwen3.8-27B (released 2026-08-14), revision pinned
below. Resolves the banked weights-availability fence (the repo is
public, safetensors, Apache-2.0) and takes the lab's FIRST
measurement on this model. Dense FFN: gate/up/down at 5120x17408,
NO expert axis — so this is exactly the transport test the Qwen
RIFF bank named (STREAM-WDISTILL arms A/B-class transport; C/D/E
do not exist here).

Arms, verbatim from the 0S registered recipe where applicable:
same seeds, per-block-128 E8M0 round-up normalization, S1-T /
S1-U4 / S2(DP-4096) scalars, W4/W8/W32 residual VQ at 2.000 index
bpw (K=256, 2^20 sample, 15 Lloyd iters), one shuffled twin for
W4 and W32 (seed 20260816). Per-projection codebooks
(gate/up/down). EVIDENCE CLASS: descriptive throughout —
different model, different device than any registered bar; numbers
never compare against V4 receipts numerically, only the RANKING
is read.

    LAYER=32 SMOKE=1 python -u scratch/qwen_stream_probe.py
    LAYER=32 python -u scratch/qwen_stream_probe.py
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

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REPO = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
LAYER = int(os.environ.get("LAYER", "32"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CACHE = os.path.expanduser(os.environ.get("SHARD_CACHE", "~/shards"))
OUT = f"logs/qwenprobe/L{LAYER}{'_smoke' if SMOKE else ''}.jsonl"

SEED = 20260816
BLOCK = 128
S2_BINS = 4096
VQ_K = 256
VQ_SAMPLE = 1 << 20
VQ_LLOYD_ITERS = 15
WIDTHS = {"W4": (4, 1), "W8": (8, 2), "W32": (32, 8)}
SHUF = ("W4", "W32")
PROBE_N, PROBE_ITERS = 64, 30
PROJS = ("gate_proj", "up_proj", "down_proj")

import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("wd0s", "scratch/stream_wdistill0s.py")
_wd0s = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_wd0s)
s2_levels_dp = _wd0s.s2_levels_dp


def _tok():
    tf = os.path.expanduser("~/.cache/huggingface/token")
    return open(tf).read().strip() if os.path.exists(tf) else ""


def _get(url, out=None):
    req = urllib.request.Request(url, headers={
        "User-Agent": "llmopt-qwenprobe/0 (research)"})
    t = _tok()
    if t:
        req.add_header("Authorization", f"Bearer {t}")
    with urllib.request.urlopen(req, timeout=300) as r:
        if out is None:
            return r.read()
        with open(out, "wb") as f:
            while True:
                c = r.read(1 << 24)
                if not c:
                    break
                f.write(c)


def ensure(fname):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"qwen38_{REVISION[:8]}_{os.path.basename(fname)}")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        print(f"[qp] downloading {fname}", flush=True)
        t = time.time()
        _get(f"{REPO}/{fname}", p + ".part")
        os.replace(p + ".part", p)
        print(f"[qp] cached {os.path.getsize(p)/2**30:.2f} GiB "
              f"in {time.time()-t:.0f}s", flush=True)
    return p


def load_tensor(name):
    idx = json.loads(open(ensure("model.safetensors.index.json")).read())
    shard = idx["weight_map"][name]
    p = ensure(shard)
    mm = np.memmap(p, dtype=np.uint8, mode="r")
    hlen = struct.unpack("<Q", bytes(mm[:8]))[0]
    hdr = json.loads(bytes(mm[8:8 + hlen]))
    e = hdr[name]
    lo, hi = e["data_offsets"]
    raw = np.array(mm[8 + hlen + lo:8 + hlen + hi])
    assert e["dtype"] in ("BF16", "F16"), e["dtype"]
    u16 = raw.view(np.uint16).reshape(e["shape"])
    if e["dtype"] == "BF16":
        w = (u16.astype(np.uint32) << 16).view(np.float32)
    else:
        w = u16.view(np.float16).astype(np.float32)
    return w


def T(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV)


def e8m0(Wb):
    mx = Wb.abs().amax(-1, keepdim=True)
    e = torch.clamp(torch.ceil(torch.log2(torch.clamp(mx, min=2.0 ** -126))),
                    -127, 127)
    return torch.exp2(e)


LT = np.array([-1.0, 0.0, 1.0], np.float32)
LU4 = np.array([-1.0, -1 / 3, 1 / 3, 1.0], np.float32)


def nearest(Wn, lv):
    d = (Wn.unsqueeze(-1) - lv).abs() - lv.abs() * 1e-7
    return lv[d.argmin(-1)]


def assign(V, C, chunk=1 << 18):
    cn = (C * C).sum(1)
    out = torch.empty(len(V), dtype=torch.long, device=V.device)
    for i in range(0, len(V), chunk):
        out[i:i + chunk] = (cn[None] - 2.0 * (V[i:i + chunk] @ C.T)).argmin(1)
    return out


def kmeans(Vn, seed):
    r = np.random.default_rng(seed)
    sub = Vn[T(r.choice(len(Vn), min(1 << 16, len(Vn)), replace=False))]
    C = [sub[int(r.integers(len(sub)))]]
    d2 = ((sub - C[0]) ** 2).sum(1)
    for _ in range(VQ_K - 1):
        p = (d2 / torch.clamp(d2.sum(), min=1e-30)).cpu().numpy().astype(np.float64)
        p /= p.sum()
        C.append(sub[int(r.choice(len(sub), p=p))])
        d2 = torch.minimum(d2, ((sub - C[-1]) ** 2).sum(1))
    C = torch.stack(C)
    for _ in range(VQ_LLOYD_ITERS):
        a = assign(Vn, C)
        cnt = torch.bincount(a, minlength=VQ_K).float()
        S = torch.zeros_like(C)
        S.index_add_(0, a, Vn)
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
    return C.half().float()


def stack_train(Vn, stages, tag):
    res, st = Vn.clone(), []
    for s in range(stages):
        C = kmeans(res, SEED + 104729 * s + zlib.crc32(tag.encode()) % 9973)
        res = res - C[assign(res, C)]
        st.append(C)
    return st


def vq(Vv, st):
    res, out = Vv.clone(), torch.zeros_like(Vv)
    for C in st:
        a = assign(res, C)
        out += C[a]
        res -= C[a]
    return out


def perms(seed, pi, nb, inverse=False):
    r = np.random.default_rng([seed, pi, 0])
    P = np.argsort(r.random((nb, BLOCK)), axis=1)
    return np.argsort(P, axis=1) if inverse else P


def op_parts(D, W, seed):
    r = np.random.default_rng(seed)
    X = T(r.standard_normal((PROBE_N, W.shape[1])).astype(np.float32))
    return float(((X @ D.T) ** 2).sum()), float(((X @ W.T) ** 2).sum())


def main():
    os.makedirs("logs/qwenprobe", exist_ok=True)
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t0 = time.time()
    print(f"[qp] {MODEL}@{REVISION[:8]} layer {LAYER} dev {DEV} "
          f"smoke {SMOKE}", flush=True)
    names = [f"model.layers.{LAYER}.mlp.{p}.weight" for p in PROJS]
    Ws = {}
    for p, n in zip(PROJS, names):
        W = load_tensor(n)
        if SMOKE:                       # smoke: a slice, own receipt path
            W = W[:W.shape[0] // 8]
        Ws[p] = T(W)
        print(f"[qp] {p} {tuple(Ws[p].shape)}", flush=True)

    # normalized samples + histograms, per projection
    books, s2lv, norm = {}, {}, {}
    edges = T(np.linspace(-1, 1, S2_BINS + 1).astype(np.float32))
    t_cb = time.time()
    for pi, p in enumerate(PROJS):
        Wb = Ws[p].reshape(Ws[p].shape[0], -1, BLOCK)
        sc = e8m0(Wb)
        Wn = (Wb / sc)
        norm[p] = (Wn, sc)
        flat = Wn.reshape(-1)
        bi = torch.clamp(torch.bucketize(flat, edges, right=True) - 1,
                         0, S2_BINS - 1)
        h = [torch.bincount(bi, minlength=S2_BINS).double().cpu().numpy(),
             torch.bincount(bi, weights=flat.double(),
                            minlength=S2_BINS).cpu().numpy(),
             torch.bincount(bi, weights=flat.double() ** 2,
                            minlength=S2_BINS).cpu().numpy()]
        s2lv[p] = T(s2_levels_dp(*h).astype(np.float32))
        nb = Wn.shape[0] * Wn.shape[1]
        Wn2 = Wn.reshape(nb, BLOCK)
        r = np.random.default_rng(SEED + pi)
        for nm, (w, s) in WIDTHS.items():
            v = Wn2.reshape(-1, w)
            idx = T(r.choice(len(v), min(VQ_SAMPLE, len(v)), replace=False))
            books[(nm, p)] = stack_train(v[idx], s, f"{nm}/{p}")
            if nm in SHUF:
                P = T(perms(SEED, pi, nb))
                vs = torch.gather(Wn2, 1, P).reshape(-1, w)
                books[(f"{nm}-shuf", p)] = stack_train(
                    vs[idx], s, f"{nm}-shuf/{p}")
    print(f"[qp] codebooks {time.time()-t_cb:.0f}s", flush=True)

    ARMS = ["S1-T", "S1-U4", "S2"] + list(WIDTHS) + [f"{n}-shuf" for n in SHUF]
    acc = {a: [0.0, 0.0] for a in ARMS}
    opr = {a: [0.0, 0.0] for a in ARMS}
    lt, lu = T(LT), T(LU4)
    for pi, p in enumerate(PROJS):
        Wn, sc = norm[p]
        W = Ws[p]
        nb = Wn.shape[0] * Wn.shape[1]
        Wn2 = Wn.reshape(nb, BLOCK)
        recon = {"S1-T": nearest(Wn, lt), "S1-U4": nearest(Wn, lu),
                 "S2": nearest(Wn, s2lv[p])}
        for nm, (w, s) in WIDTHS.items():
            recon[nm] = vq(Wn2.reshape(-1, w),
                           books[(nm, p)]).reshape(Wn.shape)
        for nm in SHUF:
            w = WIDTHS[nm][0]
            P = T(perms(SEED, pi, nb))
            Pi = T(perms(SEED, pi, nb, inverse=True))
            xs = torch.gather(Wn2, 1, P)
            ys = vq(xs.reshape(-1, w),
                    books[(f"{nm}-shuf", p)]).reshape(nb, BLOCK)
            recon[f"{nm}-shuf"] = torch.gather(ys, 1, Pi).reshape(Wn.shape)
        for a, Rn in recon.items():
            R = (Rn * sc).reshape(W.shape)
            Dm = R - W
            acc[a][0] += float((Dm ** 2).sum())
            acc[a][1] += float((W ** 2).sum())
            n2, d2 = op_parts(Dm, W, SEED + 17)
            opr[a][0] += n2
            opr[a][1] += d2
        del recon
        print(f"[qp] scored {p}", flush=True)

    row = {"probe": "qwen-stream-probe-0", "model": MODEL,
           "revision": REVISION, "layer": LAYER, "smoke": SMOKE,
           "device": DEV,
           "code_commit": __import__("subprocess").check_output(
               ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
           "config": {"seed": SEED, "block": BLOCK, "vq_K": VQ_K,
                      "widths": WIDTHS, "scale": "E8M0_round_up",
                      "shuf_seed": SEED, "projs": list(PROJS)},
           "frob": {a: (acc[a][0] / acc[a][1]) ** 0.5 for a in ARMS},
           "operator_layer": {a: (opr[a][0] / max(opr[a][1], 1e-30)) ** 0.5
                              for a in ARMS},
           "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[qp] op {json.dumps({k: round(v, 5) for k, v in row['operator_layer'].items()})}",
          flush=True)
    print(f"[qp] -> {OUT} wall {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

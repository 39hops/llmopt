"""QWEN-WHOLE-0T compiler (PRE-REG RESULTS L32776 + -0T-ARMS).

Streams the 18 vendor shards of Qwen/Qwen3.8-27B@1d4bf0f2 under
bounded residency and emits THREE complete compressed text-language
artifacts (rate tables A/B/C) in one pass. Per-source-shard
transaction:

  ensure source shard -> parse full key set -> classify every
  tensor by ROLE -> encode (at most two codecs per tensor, shared
  across arms) or passthrough -> write per-arm output shard .part
  -> conservation check -> sha256 -> atomic rename -> receipt row
  -> delete source shard (KEEP_SOURCE=1 = debug mode, never the
  registered evidence path).

Restartable at shard boundaries: a completed receipt row + all
three output shards present and sha-matching => skip.

Artifact layout (per arm, mirrors vendor sharding):
  <ART>/A/shard-000NN.bin   sequential tensor records
  <ART>/A/manifest.json     name -> {codec, shape, offsets, meta}
Record encodings:
  w4      E8M0 exponents u8[blocks] + codebook fp16[256,4]
          + indices u8[n/4]
  s16     E8M0 exponents u8[blocks] + levels fp16[16]
          + codes u4-packed[n/2]
  raw     original bf16 bytes verbatim
Excluded tensors (vision tower, mtp) are RECORDED in the manifest
with codec "excluded" and zero payload — conservation counts them.

    SMOKE=1  python -u scratch/qwen_whole0t.py   (first shard only,
             smoke-suffixed outputs and receipt)
    python -u scratch/qwen_whole0t.py
"""
import hashlib
import json
import os
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

import torch  # noqa: E402
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("qp", "scratch/qwen_stream_probe.py")
qp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(qp)
_f = importlib.util.spec_from_file_location("fp", "scratch/qwen_family_probe.py")
fp = importlib.util.module_from_spec(_f)
_f.loader.exec_module(fp)          # dp_levels(count,ssum,ssq,K)

DEV = qp.DEV
SMOKE = os.environ.get("SMOKE", "0") == "1"
KEEP_SOURCE = os.environ.get("KEEP_SOURCE", "0") == "1"
SEED = 20260816
BLOCK = 128
CHUNK_ROWS = 8192
ARMS = ("A", "B", "C")
ART = os.path.expanduser(os.environ.get(
    "ART_DIR", f"~/qwen_whole0t{'_smoke' if SMOKE else ''}"))
SUF = "_smoke" if SMOKE else ""
RCPT = f"logs/qwenwhole/compile{SUF}.jsonl"

IO_NAMES = ("model.language_model.embed_tokens.weight", "lm_head.weight")
ATTN_MARKS = (".linear_attn.in_proj_qkv.", ".linear_attn.in_proj_z.",
              ".linear_attn.out_proj.", ".self_attn.q_proj.",
              ".self_attn.k_proj.", ".self_attn.v_proj.",
              ".self_attn.o_proj.")


def classify(name, shape):
    """-> (family, {arm: codec}) per the frozen role/rate tables."""
    n = 1
    for d in shape:
        n *= d
    if name.startswith("model.visual") or ".visual." in name:
        return "vision", {a: "excluded" for a in ARMS}
    if "mtp" in name:
        return "mtp", {a: "excluded" for a in ARMS}
    if name in IO_NAMES:
        fam = "embeddings" if "embed" in name else "lm_head"
        return fam, {"A": "w4", "B": "s16", "C": "s16"}
    if n < (1 << 20) or len(shape) != 2:
        return "small", {a: "raw" for a in ARMS}
    if ".mlp." in name:
        return "ffn", {a: "w4" for a in ARMS}
    if any(m in name for m in ATTN_MARKS):
        fam = "linear_attn" if "linear_attn" in name else "full_attn"
        return fam, {"A": "w4", "B": "w4", "C": "s16"}
    if ".linear_attn." in name:                 # in_proj_a/b 48x5120 etc.
        return "linear_attn_small", {a: "raw" for a in ARMS}
    return "other", {a: "raw" for a in ARMS}


# ------------------------------------------------------------ encoders
def enc_w4(W, name):
    """Per-tensor W4: returns (payload bytes, recon fn parts, meta)."""
    rows = W.shape[0]
    r = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 99991)
    samples = []
    per_chunk = max(1, (1 << 20) // max(rows // CHUNK_ROWS + 1, 1))
    for lo in range(0, rows, CHUNK_ROWS):
        Wc = qp.T(W[lo:lo + CHUNK_ROWS])
        Wn = (Wc.reshape(Wc.shape[0], -1, BLOCK)
              / qp.e8m0(Wc.reshape(Wc.shape[0], -1, BLOCK)))
        v = Wn.reshape(-1, 4)
        samples.append(v[qp.T(r.choice(len(v), min(per_chunk, len(v)),
                                       replace=False))].cpu().numpy())
        del Wc, Wn
    samp = qp.T(np.concatenate(samples)[:1 << 20])
    C = qp.stack_train(samp, 1, f"{name}/w4")[0]
    del samp
    exps, idxs = [], []
    se = n2 = 0.0
    for lo in range(0, rows, CHUNK_ROWS):
        Wc = qp.T(W[lo:lo + CHUNK_ROWS])
        Wb = Wc.reshape(Wc.shape[0], -1, BLOCK)
        sc = qp.e8m0(Wb)
        Wn = Wb / sc
        a = qp.assign(Wn.reshape(-1, 4), C)
        R = (C[a].reshape(Wn.shape) * sc).reshape(Wc.shape)
        Dm = R - Wc
        se += float((Dm ** 2).sum())
        nn, _ = qp.op_parts(Dm, Wc, SEED + 17)
        n2 += nn
        exps.append((torch.log2(sc).reshape(-1) + 127)
                    .to(torch.uint8).cpu().numpy())
        idxs.append(a.to(torch.uint8).cpu().numpy())
        del Wc, Wb, Wn, R, Dm
    payload = (np.concatenate(exps).tobytes()
               + C.half().cpu().numpy().tobytes()
               + np.concatenate(idxs).tobytes())
    return payload, se, n2, {"codec": "w4", "K": 256, "width": 4}


def enc_s16(W, name):
    rows = W.shape[0]
    hist = [np.zeros(4096, np.float64) for _ in range(3)]
    edges = qp.T(np.linspace(-1, 1, 4097).astype(np.float32))
    for lo in range(0, rows, CHUNK_ROWS):
        Wc = qp.T(W[lo:lo + CHUNK_ROWS])
        Wb = Wc.reshape(Wc.shape[0], -1, BLOCK)
        Wn = Wb / qp.e8m0(Wb)
        flat = Wn.reshape(-1)
        bi = torch.clamp(torch.bucketize(flat, edges, right=True) - 1,
                         0, 4095)
        hist[0] += torch.bincount(bi, minlength=4096).double().cpu().numpy()
        hist[1] += torch.bincount(bi, weights=flat.double(),
                                  minlength=4096).cpu().numpy()
        hist[2] += torch.bincount(bi, weights=flat.double() ** 2,
                                  minlength=4096).cpu().numpy()
        del Wc, Wb, Wn, flat
    lv = qp.T(fp.dp_levels(*hist, 16))
    exps, codes = [], []
    se = n2 = 0.0
    for lo in range(0, rows, CHUNK_ROWS):
        Wc = qp.T(W[lo:lo + CHUNK_ROWS])
        Wb = Wc.reshape(Wc.shape[0], -1, BLOCK)
        sc = qp.e8m0(Wb)
        Wn = Wb / sc
        d = (Wn.unsqueeze(-1) - lv).abs()
        code = d.argmin(-1).to(torch.uint8)
        R = (lv[code.long()] * sc).reshape(Wc.shape)
        Dm = R - Wc
        se += float((Dm ** 2).sum())
        nn, _ = qp.op_parts(Dm, Wc, SEED + 17)
        n2 += nn
        exps.append((torch.log2(sc).reshape(-1) + 127)
                    .to(torch.uint8).cpu().numpy())
        c = code.reshape(-1).cpu().numpy()
        codes.append((c[0::2] << 4 | c[1::2]).astype(np.uint8))
        del Wc, Wb, Wn, d, code, R, Dm
    payload = (np.concatenate(exps).tobytes()
               + lv.half().cpu().numpy().tobytes()
               + np.concatenate(codes).tobytes())
    return payload, se, n2, {"codec": "s16", "levels": 16}


# ------------------------------------------------------------ compiler
def main():
    os.makedirs("logs/qwenwhole", exist_ok=True)
    for a in ARMS:
        os.makedirs(os.path.join(ART, a), exist_ok=True)
    done = set()
    if os.path.exists(RCPT):
        for ln in open(RCPT):
            row = json.loads(ln)
            if row.get("kind") == "shard":
                done.add(row["shard"])
    idx = json.loads(open(qp.ensure("model.safetensors.index.json")).read())
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    if SMOKE:
        shards = shards[:1]
    t0 = time.time()
    commit = __import__("subprocess").check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"[w0t] {len(shards)} shards | arms {ARMS} | smoke {SMOKE} | "
          f"resume-skip {len(done)}", flush=True)

    fam_err = {}          # (family, codec) -> [se, wsum, n2, d2]
    counts = {"compressed": 0, "passthrough": 0, "excluded": 0}
    manifests = {a: {} for a in ARMS}
    arm_bytes = {a: 0 for a in ARMS}

    for sh in shards:
        if sh in done:
            print(f"[w0t] skip {sh} (receipted)", flush=True)
            continue
        t_sh = time.time()
        p = qp.ensure(sh)
        mm = np.memmap(p, dtype=np.uint8, mode="r")
        hlen = struct.unpack("<Q", bytes(mm[:8]))[0]
        hdr = json.loads(bytes(mm[8:8 + hlen]))
        hdr.pop("__metadata__", None)
        outs = {a: open(os.path.join(ART, a, f"{sh}.part"), "wb")
                for a in ARMS}
        seen = set()
        for name, e in hdr.items():
            assert name not in seen, f"duplicate {name}"
            seen.add(name)
            fam, codecs = classify(name, e["shape"])
            lo, hi = e["data_offsets"]
            raw = bytes(mm[8 + hlen + lo:8 + hlen + hi])
            enc_cache = {}
            for a in ARMS:
                codec = codecs[a]
                off = outs[a].tell()
                if codec == "excluded":
                    manifests[a][name] = {"codec": "excluded",
                                          "shape": e["shape"]}
                    continue
                if codec == "raw":
                    outs[a].write(raw)
                    manifests[a][name] = {"codec": "raw",
                                          "dtype": e["dtype"],
                                          "shape": e["shape"],
                                          "shard": sh, "off": off,
                                          "len": len(raw)}
                    continue
                if codec not in enc_cache:
                    u16 = np.frombuffer(raw, np.uint16).reshape(e["shape"])
                    W = ((u16.astype(np.uint32) << 16).view(np.float32)
                         if e["dtype"] == "BF16"
                         else u16.view(np.float16).astype(np.float32))
                    fn = enc_w4 if codec == "w4" else enc_s16
                    payload, se, n2, meta = fn(W, name)
                    wsum = float((W.astype(np.float64) ** 2).sum())
                    r = np.random.default_rng(SEED + 17)
                    X = r.standard_normal((qp.PROBE_N, W.shape[1]))
                    d2 = float(((X @ W.T.astype(np.float64)) ** 2).sum())
                    k = (fam, codec)
                    acc = fam_err.setdefault(k, [0.0, 0.0, 0.0, 0.0])
                    acc[0] += se
                    acc[1] += wsum
                    acc[2] += n2
                    acc[3] += d2
                    enc_cache[codec] = (payload, meta)
                    del W
                payload, meta = enc_cache[codec]
                outs[a].write(payload)
                manifests[a][name] = dict(meta, shape=e["shape"],
                                          shard=sh, off=off,
                                          len=len(payload))
            if codecs["A"] == "excluded":
                counts["excluded"] += 1
            elif codecs["A"] == "raw":
                counts["passthrough"] += 1
            else:
                counts["compressed"] += 1
        shas = {}
        for a in ARMS:
            outs[a].close()
            fpth = os.path.join(ART, a, f"{sh}.part")
            h = hashlib.sha256(open(fpth, "rb").read()).hexdigest()
            os.replace(fpth, os.path.join(ART, a, sh + ".bin"))
            shas[a] = h
            arm_bytes[a] += os.path.getsize(os.path.join(ART, a, sh + ".bin"))
        # conservation, per shard
        for a in ARMS:
            miss = set(hdr) - {k for k in manifests[a] if
                               manifests[a][k].get("shard") == sh
                               or manifests[a][k]["codec"] == "excluded"}
            assert not miss, f"CONSERVATION: {a} dropped {sorted(miss)[:3]}"
        with open(RCPT, "a") as f:
            f.write(json.dumps({
                "kind": "shard", "shard": sh, "n_tensors": len(hdr),
                "shas": shas, "wall_s": round(time.time() - t_sh, 1),
                "code_commit": commit}) + "\n")
        del mm
        if not KEEP_SOURCE and not SMOKE:
            os.remove(p)
            print(f"[w0t] {sh} done {time.time()-t_sh:.0f}s "
                  f"(source deleted)", flush=True)
        else:
            print(f"[w0t] {sh} done {time.time()-t_sh:.0f}s", flush=True)

    # manifests + summary
    total_keys = len(wm) if not SMOKE else sum(
        1 for k, v in wm.items() if v in shards)
    viol = 0
    for a in ARMS:
        have = {k for k, v in wm.items()
                if (not SMOKE or v in shards)} - set(manifests[a])
        viol = max(viol, len(have))
        mj = json.dumps(manifests[a], sort_keys=True)
        open(os.path.join(ART, a, "manifest.json"), "w").write(mj)
        arm_bytes[a] += len(mj)
    fam_out = {f"{fam}:{codec}": {
        "frob": (v[0] / max(v[1], 1e-30)) ** 0.5,
        "op": (v[2] / max(v[3], 1e-30)) ** 0.5}
        for (fam, codec), v in fam_err.items()}
    with open(RCPT, "a") as f:
        f.write(json.dumps({
            "kind": "summary", "smoke": SMOKE, "revision": qp.REVISION,
            "code_commit": commit, "n_source_keys": total_keys,
            "counts": counts, "conservation_violations": viol,
            "arm_bytes": arm_bytes,
            "arm_gib": {a: round(b / 2 ** 30, 3)
                        for a, b in arm_bytes.items()},
            "family_errors": fam_out,
            "rate_tables": {"A": "uniform w4@2.0625",
                            "B": "A + io s16@4.0625",
                            "C": "B + attn s16@4.0625"},
            "wall_s": round(time.time() - t0, 1)}) + "\n")
    print(f"[w0t] DONE viol={viol} "
          f"bytes={ {a: round(b/2**30, 2) for a, b in arm_bytes.items()} } "
          f"wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

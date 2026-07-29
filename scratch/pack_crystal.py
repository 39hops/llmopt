"""THE PACKED CRYSTAL C0+C1 (pre-reg 2026-07-29 eve): real bytes
for the sigma-law. C0: per-tensor denominator q_t = ceil(2/sigma_t)
(grid step <= sigma/2, below the knee), codes = round(W*q_t) packed
to ceil(log2(span)) bits, one (q_t, offset) per tensor -> .npz +
reader. Norms/emb/head stay fp32 (tiny, never snapped). C1: full
gates on packed v fresh fp control, same device — bar: within
sigma (~3.5). Reports bits/wt, Shannon entropy of the code stream
(Gaussian-capacity check), artifact bytes v fp32/fp16.
__main__-guarded.
"""
import ast
import io
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CRYSTALS = [
    ("L4d56", "checkpoints/sym_birth_dense_mps_L4_ema.pt",
     dict(d=56, layers=4, heads=4, ffn=224)),
    ("d64h8", "checkpoints/sym_birth_dense_mps_h8_ema.pt",
     dict(d=64, layers=8, heads=8, ffn=256)),
]


def pack_tensor(w):
    """-> (packed bytes, q, minc, bits, shape, entropy_bits_total)"""
    wf = w.float()
    sigma = float(wf.std())
    q = math.ceil(2.0 / sigma)
    codes = torch.round(wf * q).to(torch.int64).numpy()
    minc, maxc = int(codes.min()), int(codes.max())
    span = maxc - minc + 1
    bits = max(1, math.ceil(math.log2(span)))
    u = (codes - minc).astype(np.uint32).ravel()
    # bit-plane pack: n x bits boolean matrix -> packbits
    planes = ((u[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.uint8)
    packed = np.packbits(planes.ravel())
    _, cnt = np.unique(u, return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log2(p)).sum()) * u.size
    return packed, q, minc, bits, tuple(w.shape), ent


def unpack_tensor(packed, q, minc, bits, shape):
    n = int(np.prod(shape))
    planes = np.unpackbits(packed, count=n * bits).reshape(n, bits)
    u = (planes.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(1)
    codes = u.astype(np.int64) + minc
    return torch.from_numpy(codes).float().reshape(shape) / q


def pack_crystal(sd, path):
    """Pack all 2-D block weights; passthrough the rest. Returns stats."""
    blob, meta = {}, {}
    packed_params = raw_bits = ent_bits = 0
    for k, v in sd.items():
        if v.ndim == 2 and k.startswith("blocks."):
            packed, q, minc, bits, shape, ent = pack_tensor(v)
            blob[k + ".codes"] = packed
            meta[k] = (q, minc, bits, shape)
            packed_params += v.numel()
            raw_bits += v.numel() * bits
            ent_bits += ent
        else:
            blob[k + ".fp"] = v.float().numpy()
    blob["__meta__"] = np.frombuffer(
        repr(meta).encode(), dtype=np.uint8).copy()
    buf = io.BytesIO()
    np.savez_compressed(buf, **blob)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return packed_params, raw_bits, ent_bits, len(buf.getvalue())


def load_crystal(path):
    z = np.load(path)
    meta = ast.literal_eval(bytes(z["__meta__"]).decode())
    sd = {}
    for name in z.files:
        if name == "__meta__":
            continue
        if name.endswith(".fp"):
            sd[name[:-3]] = torch.from_numpy(z[name])
        else:
            k = name[:-6]
            q, minc, bits, shape = meta[k]
            sd[k] = unpack_tensor(z[name], q, minc, bits, shape)
    return sd


def main():
    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    def gate(sd, cfg, label):
        m = build_model(len(tok.vocab), **cfg).to(dev)
        m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        m.eval()
        with torch.no_grad():
            solves, valid = G.gate_eval(m, tok, dev)
        print(f"PACK {label}: {sum(solves.values())}/120 "
              f"@ {valid:.2f}%", flush=True)
        del m

    for tag, ckpt, cfg in CRYSTALS:
        base = torch.load(ckpt, map_location="cpu", weights_only=True)
        out = f"checkpoints/packed_{tag}.npz"
        nparam, raw_bits, ent_bits, nbytes = pack_crystal(base, out)
        step_bits = []  # Gaussian capacity per tensor at its step
        for k, v in base.items():
            if v.ndim == 2 and k.startswith("blocks."):
                s = float(v.float().std())
                q = math.ceil(2.0 / s)
                step_bits.append(
                    (v.numel(),
                     0.5 * math.log2(2 * math.pi * math.e)
                     - math.log2((1.0 / q) / s)))
        cap = sum(n * b for n, b in step_bits) / nparam
        tot = sum(v.numel() for v in base.values())
        print(f"PACK {tag}: {nparam}/{tot} params packed | "
              f"raw {raw_bits / nparam:.2f} bits/wt | "
              f"entropy {ent_bits / nparam:.2f} | "
              f"gauss-cap {cap:.2f} | "
              f"artifact {nbytes} B v fp32 {tot * 4} B "
              f"({tot * 4 / nbytes:.2f}x) v fp16 {tot * 2} B",
              flush=True)
        gate(base, cfg, f"{tag} fp control")
        gate(load_crystal(out), cfg, f"{tag} PACKED")


if __name__ == "__main__":
    main()

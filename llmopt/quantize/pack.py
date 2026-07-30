"""The packed-crystal format: sigma-law allocation, bit-packing,
and optional rANS entropy coding (promoted from scratch/, C-series
2026-07-29/30; verdicts in docs/RESULTS.md).

Format (per 2-D tensor): denominator q = ceil(k_step / sigma)
(default k_step=2.0 -> grid step <= sigma/2, the free zone of the
sigma-priced snap law on at-capacity weights), integer codes
round(W * q) packed to ceil(log2(span)) bits, one (q, offset) per
tensor. Non-2-D tensors pass through as fp32.

Measured basis: born crystals gate at parity at ~5 bits/wt with
code-stream entropy within 1% of Gaussian capacity; GPTQ/AWQ/HQQ
tie the closed form at matched bits ON AT-CAPACITY WEIGHTS ONLY —
check `llmopt.quantize.meter` first (M <~ 2 is the sigma-law
domain; web-dense LLMs are not in it).
"""
from __future__ import annotations

import ast
import io
import math


def pack_tensor(w, k_step: float = 2.0):
    """fp tensor [out, in] -> (packed bytes, q, minc, bits, shape,
    entropy_bits_total). Bit-plane packing via numpy packbits."""
    import numpy as np
    import torch

    wf = w.float()
    sigma = float(wf.std())
    q = math.ceil(k_step / max(sigma, 1e-8))
    codes = torch.round(wf * q).to(torch.int64).numpy()
    minc, maxc = int(codes.min()), int(codes.max())
    span = maxc - minc + 1
    bits = max(1, math.ceil(math.log2(span)))
    u = (codes - minc).astype(np.uint32).ravel()
    planes = ((u[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.uint8)
    packed = np.packbits(planes.ravel())
    _, cnt = np.unique(u, return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log2(p)).sum()) * u.size
    return packed, q, minc, bits, tuple(w.shape), ent


def unpack_tensor(packed, q, minc, bits, shape):
    import numpy as np
    import torch

    n = int(np.prod(shape))
    planes = np.unpackbits(packed, count=n * bits).reshape(n, bits)
    u = (planes.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(1)
    codes = u.astype(np.int64) + minc
    return torch.from_numpy(codes).float().reshape(shape) / q


def pack_state_dict(sd, path, select=None, k_step: float = 2.0):
    """Pack a state dict to an .npz container. `select(name, tensor)
    -> bool` chooses which 2-D tensors get packed (default: all);
    everything else passes through fp32.

    Returns (packed_params, raw_bits, entropy_bits, container_bytes).
    """
    import numpy as np

    blob, meta = {}, {}
    packed_params = raw_bits = ent_bits = 0
    for k, v in sd.items():
        take = v.ndim == 2 and (select is None or select(k, v))
        if take:
            packed, q, minc, bits, shape, ent = pack_tensor(v, k_step)
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


def load_state_dict(path):
    """Read a packed .npz container back to an fp state dict."""
    import numpy as np
    import torch

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


def rans_size(codes, verify: bool = True):
    """Entropy-code an integer code array with rANS (constriction).
    Returns (bytes_including_table, entropy_bits_per_symbol). The
    stream is roundtrip-verified when verify=True. Import-guarded:
    raises ImportError with guidance if constriction is missing."""
    import numpy as np

    try:
        import constriction
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rans_size needs `pip install constriction`") from e
    sym = np.asarray(codes).ravel()
    sym0 = (sym - int(sym.min())).astype(np.int32)
    counts = np.bincount(sym0).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    model = constriction.stream.model.Categorical(probs, perfect=False)
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(sym0, model)
    comp = enc.get_compressed()
    if verify:
        dec = constriction.stream.stack.AnsCoder(comp)
        out = dec.decode(model, len(sym0))
        assert (np.asarray(out) == sym0).all(), "rANS roundtrip failed"
    return 4 * len(comp) + 4 * len(counts) + 8, ent

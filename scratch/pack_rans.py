"""P6-v2 (pre-reg 2026-07-30): rANS the packed artifacts — the
entropy bound as real bytes. constriction static-Categorical rANS
per tensor (frequency table stored alongside, overhead counted).
Cells: house packed_*.npz crystals; Qwen3 blackhole parts.
Every stream verified by exact roundtrip. __main__-guarded.
"""
import glob
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import constriction  # noqa: E402


def rans_bytes(codes, verify=True):
    """codes int array -> (compressed bytes incl. table, entropy
    bits/sym). Exact-roundtrip asserted."""
    sym = codes.ravel()
    lo = int(sym.min())
    sym0 = (sym - lo).astype(np.int32)
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
        assert (np.asarray(out) == sym0).all(), "roundtrip FAILED"
    table_b = 4 * len(counts) + 8  # uint32 counts + lo/len header
    return 4 * len(comp) + table_b, ent


def house():
    import torch
    for p in sorted(glob.glob("checkpoints/packed_*.npz")):
        z = np.load(p)
        tot_b = 0
        n_codes = fp_b = 0
        t0 = time.time()
        import ast
        meta = ast.literal_eval(bytes(z["__meta__"]).decode())
        for name in z.files:
            if name == "__meta__":
                continue
            if name.endswith(".fp"):
                fp_b += z[name].nbytes
                continue
            k = name[:-6]
            q, minc, bits, shape = meta[k]
            n = int(np.prod(shape))
            planes = np.unpackbits(z[name], count=n * bits).reshape(n, bits)
            codes = (planes.astype(np.uint32)
                     << np.arange(bits, dtype=np.uint32)).sum(1)
            b, ent = rans_bytes(codes.astype(np.int64))
            tot_b += b
            n_codes += n
        total = tot_b + fp_b
        print(f"P6v2 {p.split('/')[-1]}: rANS {tot_b} B "
              f"({8 * tot_b / n_codes:.3f} bits/wt) + fp {fp_b} B "
              f"= {total} B total | {time.time() - t0:.1f}s",
              flush=True)


def qwen():
    tot_b = tot_n = 0
    scale_b = 0
    t0 = time.time()
    for p in sorted(glob.glob("checkpoints/blackhole_q3_parts/part-*.npz")):
        z = np.load(p)
        for name in z.files:
            if name.endswith(".scale"):
                scale_b += z[name].nbytes
                continue
            if not name.endswith(".codes"):
                continue
            c = z[name].astype(np.int64)
            # verify unconditionally: the old `tot_n < 2e9` gate turned
            # round-trip checking OFF for the tail of large models, so a
            # coder bug past 2B symbols would book a size for a stream
            # that does not decode (opus-5 audit catch, 2026-08-03).
            b, ent = rans_bytes(c, verify=True)
            tot_b += b
            tot_n += c.size
        print(f"P6v2 qwen {p.split('/')[-1]}: cum "
              f"{8 * tot_b / tot_n:.3f} bits/wt "
              f"({tot_n / 1e9:.1f}B params, {time.time() - t0:.0f}s)",
              flush=True)
    total = tot_b + scale_b
    print(f"P6v2 QWEN TOTAL: {total / 1e9:.2f} GB "
          f"({8 * tot_b / tot_n:.3f} bits/wt codes + "
          f"{scale_b / 1e9:.2f} GB scales) v bf16 "
          f"{2 * tot_n / 1e9:.1f} GB = {2 * tot_n / total:.2f}x",
          flush=True)


def main():
    house()
    qwen()


if __name__ == "__main__":
    main()

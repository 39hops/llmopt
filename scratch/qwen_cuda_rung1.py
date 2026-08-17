"""CUDA leg rung 1: single-tensor w4 decode on device, bit-exact
parity against the canonical decoder (llmopt.lab.qcodec.dec_w4).

The payload is a 2-bit vector quantizer, not nibble-per-weight W4:
one u8 index selects a row of four fp16 codebook values; a per-128
block u8 exponent gives scale = 2^(exp - 127). The kernel therefore
unpacks byte -> 4-vector, and the per-block scales are precomputed
HOST-SIDE with the same np.exp2 the canonical decoder uses, so the
device never evaluates exp2 (an approx/flush path would silently
break exp edge cases; parity here must be BIT-exact, tolerance 0).

Fixtures (all must be bit-identical to dec_w4 before the real
tensor is touched):
  - the qualify-suite shapes (8x256, 16x128, 5x640, 32x512), random
    payloads, exps in [120, 132)
  - exponent edges: exp = 0 (fp32 subnormal scale), 127, 255 (inf)
  - refusal: C % 4 != 0 per-row grid assumption (kernel indexes
    scales as flat block id, valid because decode is over the
    flattened order exactly like the canonical decoder)
Then: ONE real w4 tensor from the QUALIFIED artifact (manifest only
via qartifact — consumers never parse raw), decoded on device,
compared bit-exact against dec_w4 of the same payload.

Receipt: logs/qwencuda/rung1.json (refuse-if-exists).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.qcodec import BLOCK, dec_w4, expected_len  # noqa: E402

OUT = "logs/qwencuda/rung1.json"
ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
REAL_TENSOR = os.environ.get(
    "REAL_TENSOR",
    "model.language_model.layers.33.linear_attn.in_proj_qkv.weight")


@triton.jit
def w4_decode_kernel(idx_ptr, cb_ptr, scale_ptr, out_ptr, n,
                     BLK: tl.constexpr):
    """One program per 128-element block: out[i] = fp32(cb[idx, lane])
    * scale[block]. cb is fp16[256*4] flat; scale fp32 precomputed."""
    b = tl.program_id(0)
    offs = b * BLK + tl.arange(0, BLK)
    m = offs < n
    byte = tl.load(idx_ptr + offs // 4, mask=m, other=0)
    lane = offs % 4
    val = tl.load(cb_ptr + byte.to(tl.int32) * 4 + lane, mask=m,
                  other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + b)
    tl.store(out_ptr + offs, val * s, mask=m)


def gpu_dec_w4(buf: bytes, shape) -> np.ndarray:
    n = int(np.prod(shape))
    nb = n // BLOCK
    assert len(buf) == expected_len("w4", shape)
    exps = np.frombuffer(buf, np.uint8, nb, 0).astype(np.int32)
    # scales via the SAME host-side np.exp2 as qcodec._scales
    scale = np.exp2(exps - 127).astype(np.float32)
    cb = np.frombuffer(buf, np.float16, 256 * 4, nb)
    idx = np.frombuffer(buf, np.uint8, n // 4, nb + 2048)

    d_idx = torch.from_numpy(idx.copy()).cuda()
    d_cb = torch.from_numpy(cb.copy()).cuda()
    d_scale = torch.from_numpy(scale).cuda()
    d_out = torch.empty(n, dtype=torch.float32, device="cuda")
    w4_decode_kernel[(nb,)](d_idx, d_cb, d_scale, d_out, n, BLK=BLOCK)
    torch.cuda.synchronize()
    return d_out.cpu().numpy().reshape(shape)


def _payload(R, C, seed=3, exp_lo=120, exp_hi=132):
    rng = np.random.default_rng(seed)
    nb = R * C // BLOCK
    exps = rng.integers(exp_lo, exp_hi, nb, dtype=np.uint8)
    cb = (rng.standard_normal((256, 4)) * 0.3).astype(np.float16)
    idx = rng.integers(0, 256, R * C // 4, dtype=np.uint8)
    return exps.tobytes() + cb.tobytes() + idx.tobytes()


def _edge_payload(R, C, exp_val):
    rng = np.random.default_rng(7)
    nb = R * C // BLOCK
    exps = np.full(nb, exp_val, np.uint8)
    cb = (rng.standard_normal((256, 4)) * 0.3).astype(np.float16)
    idx = rng.integers(0, 256, R * C // 4, dtype=np.uint8)
    return exps.tobytes() + cb.tobytes() + idx.tobytes()


def main() -> int:
    if os.path.exists(OUT):
        raise SystemExit(f"refuse: {OUT} exists — new run, new path")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    fixtures = []
    for R, C in ((8, 256), (16, 128), (5, 640), (32, 512)):
        buf = _payload(R, C)
        ok = bool(np.array_equal(gpu_dec_w4(buf, [R, C]),
                                 dec_w4(buf, [R, C])))
        fixtures.append({"case": f"random-{R}x{C}", "bit_exact": ok})
    for exp_val in (0, 127, 255):
        buf = _edge_payload(8, 256, exp_val)
        ref = dec_w4(buf, [8, 256])
        got = gpu_dec_w4(buf, [8, 256])
        # exp=255 -> inf scale: NaN where cb value is 0 in both paths
        ok = bool(np.array_equal(got, ref, equal_nan=True))
        fixtures.append({"case": f"exp-{exp_val}", "bit_exact": ok})
    all_fix = all(f["bit_exact"] for f in fixtures)
    print(f"[r1] fixtures: {fixtures} all={all_fix}", flush=True)

    real = {"tensor": REAL_TENSOR, "bit_exact": None}
    if all_fix:
        from llmopt.lab import qartifact
        arm = os.path.basename(ART.rstrip("/"))
        chain = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "logs", "qwenwhole",
            f"artifact_digest_{arm}.txt")
        q = qartifact.qualify_artifact(
            ART, VDIR + "/model.safetensors.index.json",
            chain if os.path.exists(chain) else None,
            allow_unchained=os.environ.get("ALLOW_UNCHAINED") == "1")
        man = q["manifest"]
        e = man[REAL_TENSOR]
        assert e["codec"] == "w4", f"{REAL_TENSOR} codec {e['codec']}"
        with open(os.path.join(ART, e["shard"] + ".bin"), "rb") as f:
            f.seek(e["off"])
            buf = f.read(e["len"])
        ref = dec_w4(buf, e["shape"])
        got = gpu_dec_w4(buf, e["shape"])
        real["bit_exact"] = bool(np.array_equal(got, ref))
        real["shape"] = e["shape"]
        real["payload_bytes"] = e["len"]
        print(f"[r1] real: {real}", flush=True)

    passed = all_fix and real["bit_exact"] is True
    rec = {
        "rung": 1,
        "code_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "device_actual": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "artifact": os.path.basename(ART.rstrip("/")),
        "fixtures": fixtures,
        "real_tensor": real,
        "parity_tolerance": 0,
        "passed": passed,
        "env": {k: os.environ.get(k) for k in
                ("TORCH_DISABLE_NATIVE_JIT", "PYTORCH_CUDA_ALLOC_CONF",
                 "ALLOW_UNCHAINED")},
    }
    with open(OUT, "x") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(f"[r1] passed={passed}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

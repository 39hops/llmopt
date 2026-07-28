"""E2: export a MicroLM crystal to AXNN v1.1 (proposed extension:
cfg ffn="swiglu" + fused-qkv + rmsnorm-no-bias + rope + tied head
— every convention DECLARED per the AXNN doctrine).

Container (little-endian, per include/ax/nn/model.hpp v1):
  "AXNN" | u32 version=1 | u32 cfg_len | cfg JSON | tensors
  tensor: u32 name_len | name | u32 ndim | u64 dims[] | f32 data
v1.1 delta (the relay ask): cfg gains ffn:"swiglu",
attn_fused:"qkv"; tensor names ship in HOUSE layout:
  emb.weight [V,D] (tied head), blocks.{i}.n1.g/.n2.g [D],
  blocks.{i}.qkv.weight [3D,D], blocks.{i}.o.weight [D,D],
  blocks.{i}.gate/.up.weight [F,D], blocks.{i}.down.weight [D,F],
  nf.g [D]
Usage: export_axnn.py <ckpt.pt> <out.axnn>
"""
import hashlib
import json
import struct
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

ckpt, out = sys.argv[1], sys.argv[2]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)
D = sd["emb.weight"].shape[1]
V = sd["emb.weight"].shape[0]
L = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
F = sd["blocks.0.gate.weight"].shape[0]
cfg = {"d_model": D, "n_layers": L, "n_heads": 4, "d_ff": F,
       "vocab": V, "max_seq": 512, "norm": "rmsnorm",
       "act": "silu", "pos": "rope", "rope_style": "half",
       "eps": 1e-6, "rope_theta": 10000.0,
       "ffn": "swiglu", "attn_fused": "qkv", "head": "tied",
       "axnn_minor": 1}
cfg_b = json.dumps(cfg).encode()

with open(out, "wb") as f:
    f.write(b"AXNN")
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<I", len(cfg_b)))
    f.write(cfg_b)
    for name in sorted(sd):
        t = sd[name].float().contiguous()
        nb = name.encode()
        f.write(struct.pack("<I", len(nb)))
        f.write(nb)
        f.write(struct.pack("<I", t.dim()))
        for d_ in t.shape:
            f.write(struct.pack("<Q", d_))
        f.write(t.numpy().tobytes())

sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"{out}: {len(sd)} tensors, cfg {cfg}", flush=True)
print(f"sha256 {sha}", flush=True)

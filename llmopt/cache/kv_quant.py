"""KV cache quantization: int8/int4 with per-token-per-head scales.

KV activations have outlier channels but per-(token, head) max-abs
scaling keeps error contained enough for int8 to be near-lossless in
attention; int4 is the aggressive setting. Storage is a symmetric
integer code + one fp scale per (token, head) row.

QuantizedPagedStore mirrors PagedTensorStore's API (bind/write/gather)
over the paged block manager, so the block table code is unchanged —
gather dequantizes on the way out.
"""

from __future__ import annotations

from llmopt.cache.paged import BlockTable


def quantize(x, bits: int = 8):
    """Symmetric per-row quantization over the last dim.

    x: [..., dim] -> (codes int8 [..., dim], scale [..., 1]).
    int4 codes are stored in int8 range [-7, 7] (packing is a deploy
    concern; see quantize/ design notes).
    """
    import torch

    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    codes = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return codes, scale


def dequantize(codes, scale):
    return codes.float() * scale


class QuantizedPagedStore:
    """int8/int4 K/V backing store over paged blocks.

    Same interface as PagedTensorStore; gather returns dequantized
    tensors [length, heads, dim].
    """

    def __init__(
        self, num_blocks: int, block_size: int, heads: int, dim: int,
        *, bits: int = 8,
    ):
        import torch

        self.bits = bits
        self.block_size = block_size
        shape = (num_blocks, block_size, heads, dim)
        self.k = torch.zeros(shape, dtype=torch.int8)
        self.v = torch.zeros(shape, dtype=torch.int8)
        self.k_scale = torch.zeros(num_blocks, block_size, heads, 1)
        self.v_scale = torch.zeros(num_blocks, block_size, heads, 1)

    def bind(self, table: BlockTable) -> BlockTable:
        def hook(src: int, dst: int, used: int) -> None:
            for t in (self.k, self.v, self.k_scale, self.v_scale):
                t[dst, :used] = t[src, :used]

        table.cow_hook = hook
        return table

    def write(self, table: BlockTable, k, v) -> None:
        b, off = table.append()
        self.k[b, off], self.k_scale[b, off] = quantize(k, self.bits)
        self.v[b, off], self.v_scale[b, off] = quantize(v, self.bits)

    def gather(self, table: BlockTable):
        import torch

        ks, vs = [], []
        for pos in range(table.length):
            b, off = table.slot(pos)
            ks.append(dequantize(self.k[b, off], self.k_scale[b, off]))
            vs.append(dequantize(self.v[b, off], self.v_scale[b, off]))
        return torch.stack(ks), torch.stack(vs)

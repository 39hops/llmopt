"""Paged KV cache blocks (vLLM-style, pure-Python manager).

KV memory is a fixed pool of fixed-size blocks; each sequence maps
logical positions to physical blocks through a block table. Sharing is
by reference count: ``fork`` copies the table and bumps refcounts (e.g.
parallel sampling from a shared prompt), and writes to a shared block
trigger copy-on-write. This kills fragmentation: waste is bounded by
one partial block per sequence instead of a contiguous max-length slab.

The manager is storage-agnostic. PagedTensorStore is a torch backing
store keyed by (block, offset) proving the layout round-trips; a real
deployment would point a paged-attention kernel at the same table.
"""

from __future__ import annotations


class BlockAllocator:
    """Fixed pool of ref-counted blocks."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks - 1, -1, -1))
        self.refcount = [0] * num_blocks

    def alloc(self) -> int:
        if not self.free_list:
            raise MemoryError("out of KV blocks")
        b = self.free_list.pop()
        self.refcount[b] = 1
        return b

    def incref(self, block: int) -> None:
        self.refcount[block] += 1

    def decref(self, block: int) -> None:
        self.refcount[block] -= 1
        assert self.refcount[block] >= 0
        if self.refcount[block] == 0:
            self.free_list.append(block)

    @property
    def num_free(self) -> int:
        return len(self.free_list)


class BlockTable:
    """One sequence's logical->physical block mapping.

    append() grows by one token, allocating a block at each boundary.
    fork() shares all blocks with a new table (copy-on-write on the
    writable tail). free() releases every reference.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int):
        self.allocator = allocator
        self.block_size = block_size
        self.blocks: list[int] = []
        self.length = 0
        # storage backends replace this to copy already-written slots on
        # COW: hook(src_block, dst_block, used_offsets)
        self.cow_hook = lambda src, dst, used: None

    def slot(self, pos: int) -> tuple[int, int]:
        """(physical block, offset) for logical position pos."""
        assert 0 <= pos < self.length
        return self.blocks[pos // self.block_size], pos % self.block_size

    def append(self) -> tuple[int, int]:
        """Reserve the next logical slot; returns its (block, offset).
        Copy-on-writes the tail block if it is shared."""
        off = self.length % self.block_size
        if off == 0:
            self.blocks.append(self.allocator.alloc())
        else:
            tail = self.blocks[-1]
            if self.allocator.refcount[tail] > 1:  # shared: copy before write
                self.blocks[-1] = self.allocator.alloc()
                self.allocator.decref(tail)
                self.cow_hook(tail, self.blocks[-1], off)
        self.length += 1
        return self.blocks[-1], off

    def fork(self) -> "BlockTable":
        child = BlockTable(self.allocator, self.block_size)
        child.blocks = list(self.blocks)
        child.length = self.length
        child.cow_hook = self.cow_hook
        for b in self.blocks:
            self.allocator.incref(b)
        return child

    def free(self) -> None:
        for b in self.blocks:
            self.allocator.decref(b)
        self.blocks, self.length = [], 0


class PagedTensorStore:
    """Torch K/V backing store: [num_blocks, block_size, heads, dim].

    write() puts one position's K/V at its (block, offset); gather()
    reconstructs the contiguous [length, heads, dim] view a dense cache
    would hold — byte-identical if the table is consistent.
    """

    def __init__(self, num_blocks: int, block_size: int, heads: int, dim: int):
        import torch

        self.block_size = block_size
        shape = (num_blocks, block_size, heads, dim)
        self.k = torch.zeros(shape)
        self.v = torch.zeros(shape)

    def bind(self, table: BlockTable) -> BlockTable:
        """Install this store's COW copy hook on a table."""
        def hook(src: int, dst: int, used: int) -> None:
            self.k[dst, :used] = self.k[src, :used]
            self.v[dst, :used] = self.v[src, :used]

        table.cow_hook = hook
        return table

    def write(self, table: BlockTable, k, v) -> None:
        b, off = table.append()
        self.k[b, off] = k
        self.v[b, off] = v

    def gather(self, table: BlockTable):
        import torch

        ks, vs = [], []
        for pos in range(table.length):
            b, off = table.slot(pos)
            ks.append(self.k[b, off])
            vs.append(self.v[b, off])
        return torch.stack(ks), torch.stack(vs)

"""Paged KV block manager: allocation, fork/COW, torch round-trip."""

import pytest

torch = pytest.importorskip("torch")

from llmopt.cache.paged import BlockAllocator, BlockTable, PagedTensorStore

BS = 4  # block size


def _table(num_blocks=8):
    return BlockTable(BlockAllocator(num_blocks), BS)


def test_append_allocates_at_boundaries():
    t = _table()
    slots = [t.append() for _ in range(BS + 1)]
    assert len(t.blocks) == 2
    assert [off for _, off in slots] == [0, 1, 2, 3, 0]
    assert t.allocator.num_free == 6


def test_oom_raises():
    t = _table(num_blocks=1)
    for _ in range(BS):
        t.append()
    with pytest.raises(MemoryError):
        t.append()


def test_fork_shares_and_free_releases():
    t = _table()
    for _ in range(BS + 2):
        t.append()
    child = t.fork()
    assert child.blocks == t.blocks
    assert all(t.allocator.refcount[b] == 2 for b in t.blocks)
    child.free()
    assert all(t.allocator.refcount[b] == 1 for b in t.blocks)
    t.free()
    assert t.allocator.num_free == 8


def test_write_after_fork_copy_on_writes_tail_only():
    t = _table()
    for _ in range(BS + 2):  # one full block + partial tail
        t.append()
    child = t.fork()
    full, tail = t.blocks
    child.append()  # writes into shared partial tail -> COW
    assert child.blocks[0] == full  # full block still shared
    assert child.blocks[1] != tail  # tail copied
    assert t.allocator.refcount[tail] == 1  # parent's again exclusively
    t.append()  # parent tail no longer shared: no COW
    assert t.blocks[1] == tail


def test_torch_store_round_trip_and_cow_preserves_prefix():
    heads, dim = 2, 3
    alloc = BlockAllocator(8)
    store = PagedTensorStore(8, BS, heads, dim)
    t = store.bind(BlockTable(alloc, BS))

    ref_k, ref_v = [], []
    for _ in range(BS + 2):
        k, v = torch.randn(heads, dim), torch.randn(heads, dim)
        store.write(t, k, v)
        ref_k.append(k)
        ref_v.append(v)

    child = t.fork()
    ck, cv = torch.randn(heads, dim), torch.randn(heads, dim)
    store.write(child, ck, cv)  # triggers COW; must copy written prefix
    pk, pv = torch.randn(heads, dim), torch.randn(heads, dim)
    store.write(t, pk, pv)  # parent diverges independently

    gk, gv = store.gather(t)
    assert torch.equal(gk, torch.stack(ref_k + [pk]))
    assert torch.equal(gv, torch.stack(ref_v + [pv]))
    gk, gv = store.gather(child)
    assert torch.equal(gk, torch.stack(ref_k + [ck]))
    assert torch.equal(gv, torch.stack(ref_v + [cv]))

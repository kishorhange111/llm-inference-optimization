"""Tests for PagedAttention KV-cache block allocator and fragmentation estimator."""

import pytest

from infopt.kv_cache import KNOWN
from infopt.kv_paging import KVBlockAllocator, OutOfBlocksError, compare_contiguous_vs_paged_waste


def test_allocate_append_and_free_blocks():
    alloc = KVBlockAllocator(num_blocks=8, block_size=16)
    b1 = alloc.allocate_sequence("s1", prompt_tokens=20)  # needs 2 blocks (32 slots)
    assert len(b1) == 2
    assert alloc.stats().used_blocks == 2
    assert alloc.stats().internal_fragmentation_ratio == pytest.approx(12 / 32)

    # Append 12 tokens -> fills block 2 exactly (32 tokens, 0 fragmentation)
    b2 = alloc.append_tokens("s1", 12)
    assert len(b2) == 2
    assert alloc.stats().internal_fragmentation_ratio == 0.0

    # Append 1 more token -> allocates 3rd block
    b3 = alloc.append_tokens("s1", 1)
    assert len(b3) == 3
    assert alloc.stats().used_blocks == 3

    alloc.free_sequence("s1")
    assert alloc.stats().used_blocks == 0
    assert alloc.stats().free_blocks == 8


def test_copy_on_write_fork_and_split():
    alloc = KVBlockAllocator(num_blocks=6, block_size=16)
    src_blocks = alloc.allocate_sequence("parent", prompt_tokens=20)  # 2 blocks, tail partially filled
    child_blocks = alloc.fork_sequence("parent", "child")
    assert child_blocks == src_blocks
    assert alloc.stats().used_blocks == 2  # zero extra physical blocks

    # Appending to child must trigger CoW on the shared tail block
    updated = alloc.append_tokens("child", 1)
    assert updated[0] == src_blocks[0]  # full prefix block still shared
    assert updated[1] != src_blocks[1]  # tail block copied
    assert alloc.stats().cow_copies == 1
    assert alloc.stats().used_blocks == 3

    # Freeing parent keeps shared prefix block alive for child
    alloc.free_sequence("parent")
    assert alloc.stats().used_blocks == 2
    alloc.free_sequence("child")
    assert alloc.stats().used_blocks == 0


def test_swap_out_swap_in_and_oom():
    alloc = KVBlockAllocator(num_blocks=4, block_size=16)
    alloc.allocate_sequence("a", 32)  # 2 blocks
    alloc.allocate_sequence("b", 32)  # 2 blocks

    with pytest.raises(OutOfBlocksError):
        alloc.allocate_sequence("c", 16)
    assert alloc.stats().oom_events == 1

    freed = alloc.swap_out("b")
    assert freed == 2
    assert alloc.stats().swapped_sequences == 1
    alloc.allocate_sequence("c", 16)  # succeeds now
    assert alloc.stats().used_blocks == 3

    alloc.free_sequence("a")
    restored = alloc.swap_in("b")
    assert len(restored) == 2
    assert alloc.stats().swap_outs == 1
    assert alloc.stats().swap_ins == 1


def test_from_gpu_budget_and_fragmentation_comparison():
    alloc = KVBlockAllocator.from_gpu_budget(KNOWN["Qwen2.5-7B"], gpu_gb=40.0, block_size=16)
    # ~21.8 GiB free / (57344 * 16 bytes) -> ~25k blocks
    assert alloc.num_blocks > 20_000

    with pytest.raises(ValueError):
        KVBlockAllocator.from_gpu_budget(KNOWN["Qwen2.5-7B"], gpu_gb=10.0)

    cmp = compare_contiguous_vs_paged_waste([120, 350, 800, 64], max_model_len=4096, block_size=16)
    assert cmp["paged_waste_ratio"] < 0.05
    assert cmp["contiguous_waste_ratio"] > 0.90
    assert cmp["memory_savings_ratio"] > 0.90

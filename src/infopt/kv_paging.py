"""PagedAttention-style KV-cache block allocator and memory fragmentation simulator.

Models fixed-size physical KV blocks, logical-to-physical block tables,
copy-on-write (CoW) sequence forking for beam search / shared system prompts,
CPU swap-out/swap-in under memory pressure, and internal vs contiguous
fragmentation analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from infopt.kv_cache import ModelShape, kv_bytes_per_token, weights_gb


class OutOfBlocksError(MemoryError):
    """Raised when the physical KV block pool cannot satisfy an allocation."""


@dataclass(frozen=True)
class BlockAllocatorStats:
    """Point-in-time telemetry for :class:`KVBlockAllocator`."""

    total_blocks: int
    used_blocks: int
    free_blocks: int
    swapped_sequences: int
    active_sequences: int
    logical_tokens: int
    allocated_token_slots: int
    utilization: float
    internal_fragmentation_ratio: float
    cow_copies: int
    swap_outs: int
    swap_ins: int
    oom_events: int


@dataclass
class _SequenceState:
    seq_id: str
    num_tokens: int
    blocks: list[int]  # physical block IDs when resident on GPU
    swapped: bool = False


class KVBlockAllocator:
    """Simulates a vLLM-style paged KV-cache block manager with CoW and swapping."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self._ref_counts: dict[int, int] = {}
        self._sequences: dict[str, _SequenceState] = {}
        self._cow_copies = 0
        self._swap_outs = 0
        self._swap_ins = 0
        self._oom_events = 0

    @classmethod
    def from_gpu_budget(
        cls,
        shape: ModelShape,
        gpu_gb: float,
        block_size: int = 16,
        weight_bits: float = 16.0,
        kv_bytes: float = 2.0,
        utilization: float = 0.9,
    ) -> KVBlockAllocator:
        """Size the block pool from GPU VRAM remaining after loading model weights."""
        free_gb = gpu_gb * utilization - weights_gb(shape, weight_bits)
        if free_gb <= 0:
            raise ValueError(f"Model {shape.name} weights exceed usable GPU memory ({gpu_gb * utilization:.2f} GiB)")
        bytes_per_block = kv_bytes_per_token(shape, kv_bytes) * block_size
        num_blocks = int((free_gb * (2**30)) // bytes_per_block)
        if num_blocks <= 0:
            raise ValueError("Not enough free memory for even one KV block")
        return cls(num_blocks=num_blocks, block_size=block_size)

    def _alloc_block(self) -> int:
        if not self._free_blocks:
            self._oom_events += 1
            raise OutOfBlocksError("No free physical KV blocks available")
        bid = self._free_blocks.pop()
        self._ref_counts[bid] = 1
        return bid

    def _release_block(self, bid: int) -> None:
        rc = self._ref_counts.get(bid, 0) - 1
        if rc <= 0:
            self._ref_counts.pop(bid, None)
            self._free_blocks.append(bid)
        else:
            self._ref_counts[bid] = rc

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size) if num_tokens > 0 else 0

    def allocate_sequence(self, seq_id: str, prompt_tokens: int) -> list[int]:
        """Allocate initial physical blocks for a new sequence."""
        if seq_id in self._sequences:
            raise ValueError(f"Sequence {seq_id!r} already exists")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        need = self.blocks_needed(prompt_tokens)
        if need > len(self._free_blocks):
            self._oom_events += 1
            raise OutOfBlocksError(f"Need {need} blocks for {seq_id!r}, only {len(self._free_blocks)} free")
        blocks = [self._alloc_block() for _ in range(need)]
        self._sequences[seq_id] = _SequenceState(seq_id=seq_id, num_tokens=prompt_tokens, blocks=blocks)
        return list(blocks)

    def fork_sequence(self, src_seq_id: str, dst_seq_id: str) -> list[int]:
        """Copy-on-write fork: share all physical blocks without copying memory."""
        if src_seq_id not in self._sequences:
            raise KeyError(f"Unknown source sequence {src_seq_id!r}")
        if dst_seq_id in self._sequences:
            raise ValueError(f"Destination sequence {dst_seq_id!r} already exists")
        src = self._sequences[src_seq_id]
        if src.swapped:
            raise RuntimeError(f"Cannot fork swapped-out sequence {src_seq_id!r}")
        for bid in src.blocks:
            self._ref_counts[bid] += 1
        self._sequences[dst_seq_id] = _SequenceState(seq_id=dst_seq_id, num_tokens=src.num_tokens, blocks=list(src.blocks))
        return list(src.blocks)

    def append_tokens(self, seq_id: str, new_tokens: int = 1) -> list[int]:
        """Append tokens to a resident sequence, triggering CoW on shared tail blocks."""
        if seq_id not in self._sequences:
            raise KeyError(f"Unknown sequence {seq_id!r}")
        if new_tokens < 0:
            raise ValueError("new_tokens must be >= 0")
        seq = self._sequences[seq_id]
        if seq.swapped:
            raise RuntimeError(f"Sequence {seq_id!r} is swapped out; swap_in first")
        if new_tokens == 0:
            return list(seq.blocks)

        old_tokens = seq.num_tokens
        new_total = old_tokens + new_tokens
        need_blocks = self.blocks_needed(new_total) - len(seq.blocks)

        # If writing into a partially filled tail block that is shared, copy-on-write first.
        needs_cow = len(seq.blocks) > 0 and (old_tokens % self.block_size != 0) and self._ref_counts.get(seq.blocks[-1], 0) > 1
        total_new_phys = need_blocks + (1 if needs_cow else 0)
        if total_new_phys > len(self._free_blocks):
            self._oom_events += 1
            raise OutOfBlocksError(f"Need {total_new_phys} blocks to append to {seq_id!r}, only {len(self._free_blocks)} free")

        if needs_cow:
            old_tail = seq.blocks[-1]
            new_tail = self._alloc_block()
            self._release_block(old_tail)
            seq.blocks[-1] = new_tail
            self._cow_copies += 1

        for _ in range(need_blocks):
            seq.blocks.append(self._alloc_block())

        seq.num_tokens = new_total
        return list(seq.blocks)

    def swap_out(self, seq_id: str) -> int:
        """Evict a sequence's GPU blocks to simulated CPU RAM and return blocks freed."""
        seq = self._sequences[seq_id]
        if seq.swapped:
            return 0
        before_free = len(self._free_blocks)
        for bid in seq.blocks:
            self._release_block(bid)
        seq.blocks = []
        seq.swapped = True
        self._swap_outs += 1
        return len(self._free_blocks) - before_free

    def swap_in(self, seq_id: str) -> list[int]:
        """Restore a swapped-out sequence back into GPU blocks."""
        seq = self._sequences[seq_id]
        if not seq.swapped:
            return list(seq.blocks)
        need = self.blocks_needed(seq.num_tokens)
        if need > len(self._free_blocks):
            self._oom_events += 1
            raise OutOfBlocksError(f"Need {need} blocks to swap in {seq_id!r}, only {len(self._free_blocks)} free")
        seq.blocks = [self._alloc_block() for _ in range(need)]
        seq.swapped = False
        self._swap_ins += 1
        return list(seq.blocks)

    def free_sequence(self, seq_id: str) -> None:
        """Release all blocks owned by ``seq_id``."""
        seq = self._sequences.pop(seq_id)
        for bid in seq.blocks:
            self._release_block(bid)

    def block_table(self, seq_id: str) -> list[int]:
        return list(self._sequences[seq_id].blocks)

    def stats(self) -> BlockAllocatorStats:
        used = self.num_blocks - len(self._free_blocks)
        active = [s for s in self._sequences.values() if not s.swapped]
        swapped = len(self._sequences) - len(active)
        logical_tokens = sum(s.num_tokens for s in active)
        logical_slots = sum(len(s.blocks) * self.block_size for s in active)
        frag = (logical_slots - logical_tokens) / logical_slots if logical_slots > 0 else 0.0
        return BlockAllocatorStats(
            total_blocks=self.num_blocks,
            used_blocks=used,
            free_blocks=len(self._free_blocks),
            swapped_sequences=swapped,
            active_sequences=len(active),
            logical_tokens=logical_tokens,
            allocated_token_slots=logical_slots,
            utilization=used / self.num_blocks,
            internal_fragmentation_ratio=frag,
            cow_copies=self._cow_copies,
            swap_outs=self._swap_outs,
            swap_ins=self._swap_ins,
            oom_events=self._oom_events,
        )


def compare_contiguous_vs_paged_waste(
    sequence_lengths: list[int],
    max_model_len: int,
    block_size: int = 16,
) -> dict[str, float]:
    """Compare wasted KV slots between naive contiguous pre-allocation and paged blocks."""
    if not sequence_lengths:
        return {"contiguous_waste_ratio": 0.0, "paged_waste_ratio": 0.0, "memory_savings_ratio": 0.0}
    total_tokens = sum(sequence_lengths)
    contiguous_slots = len(sequence_lengths) * max_model_len
    paged_slots = sum(math.ceil(n / block_size) * block_size for n in sequence_lengths)
    contig_waste = (contiguous_slots - total_tokens) / contiguous_slots if contiguous_slots > 0 else 0.0
    paged_waste = (paged_slots - total_tokens) / paged_slots if paged_slots > 0 else 0.0
    savings = 1.0 - (paged_slots / contiguous_slots) if contiguous_slots > 0 else 0.0
    return {
        "contiguous_waste_ratio": contig_waste,
        "paged_waste_ratio": paged_waste,
        "memory_savings_ratio": savings,
    }

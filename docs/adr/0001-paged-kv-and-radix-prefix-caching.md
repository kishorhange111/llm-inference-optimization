# ADR-0001: Paged KV-Cache Allocation & Radix-Tree Prefix Caching

- **Status:** Accepted
- **Date:** 2026-11-24
- **Modules:** `src/infopt/kv_cache.py`, `src/infopt/kv_paging.py`, `src/infopt/prefix_cache.py`

## Context

Autoregressive LLM serving is memory-capacity and memory-bandwidth bound rather than compute bound. Pre-allocating contiguous KV buffers up to `max_model_len` wastes 85–95% of VRAM on typical mixed-length workloads, while recomputing identical system prompts and multi-turn chat histories saturates prefill compute and inflates TTFT p95/p99.

## Decision

1. **PagedAttention-style fixed-size blocks (`KVBlockAllocator`):** Allocate KV memory in fixed-size blocks (default 16 tokens) with logical-to-physical block tables, copy-on-write (CoW) sequence forking for beam search / shared prefixes, and CPU swap-out/swap-in under memory pressure.
2. **Radix-tree KV prefix reuse (`RadixPrefixCache`):** Index cached prompt prefixes in a compressed radix tree with automatic edge splitting, reference-count pinning for active in-flight requests, and LRU leaf eviction bounded by `max_cached_tokens`.
3. **FP8 KV cache quantization:** Combine GQA + 4-bit weights + FP8 KV cache (`kv_bytes=1.0`) to roughly triple concurrent 4k-token sequence capacity on an A100-40GB (99 -> 296 sequences for Qwen2.5-7B).

## Consequences

- **Pros:** Reduces internal KV fragmentation from >85% to <5%, eliminates redundant prefill on shared system prompts (97% token reuse in smoke benchmark), and enables predictable K8s HPA scaling on `infopt_kv_cache_utilization_ratio`.
- **Cons:** Adds block-table indirection and requires careful pin/unpin lifecycle tracking so active sequences are never evicted mid-generation.

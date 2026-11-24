# ADR-0002: Continuous Batching & Lossless Speculative Decoding

- **Status:** Accepted
- **Date:** 2026-11-24
- **Modules:** `src/infopt/scheduler.py`, `src/infopt/speculative.py`

## Context

Static batching forces short responses to wait for the longest sequence in a batch, wasting decode compute on padding tokens and inflating p50/p95 latency. Conversely, at low concurrency (batch size 1–4), decode is strictly memory-bandwidth bound and leaves GPU tensor cores underutilized.

## Decision

1. **Iteration-level continuous batching (`ContinuousBatchingScheduler`):** Evict completed sequences immediately after each decode step and admit waiting requests using a priority queue (`REALTIME`, `INTERACTIVE`, `BATCH`) with starvation-prevention aging and per-step token budgets (`max_tokens_per_step`). Integrate with `KVBlockAllocator` to preempt lowest-priority sequences under VRAM pressure.
2. **Lossless speculative decoding (`SpeculativeDecodingSimulator`, `verify_draft_tokens`):** Use exact Leviathan et al. (2023) rejection sampling so draft-then-verify preserves the exact target token distribution. Select lookahead length `gamma` analytically via `optimal_lookahead_gamma(alpha, cost_ratio)` and disable speculative decoding at high batch sizes where target verification becomes compute-bound.

## Consequences

- **Pros:** Continuous batching delivers >2x output throughput over static batching on mixed-length workloads (`2.03x` in smoke suite); speculative decoding delivers `2.6x` latency reduction at `alpha=0.82, gamma=5` for interactive low-batch traffic with zero quality degradation.
- **Cons:** Speculative decoding hurts throughput when acceptance rate `alpha < 0.4` or when concurrency already saturates GPU FLOPs; hence `deploy/profiles/vllm-qwen7b-awq.yaml` keeps it configurable per workload tier.

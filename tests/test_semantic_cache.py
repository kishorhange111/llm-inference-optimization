"""Tests for semantic response cache and single-flight request coalescing."""

import asyncio

import pytest

from infopt.semantic_cache import RequestCoalescer, SemanticResponseCache


class _Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_semantic_cache_hit_miss_and_model_scoping():
    cache = SemanticResponseCache(similarity_threshold=0.75)
    cache.store(
        "Explain how PagedAttention reduces KV cache fragmentation in vLLM",
        "PagedAttention stores KV cache in fixed-size non-contiguous blocks.",
        model_id="qwen-7b",
    )

    # Near-duplicate phrasing hits for the same model
    hit = cache.lookup(
        "Explain how PagedAttention reduces KV cache fragmentation in vLLM!",
        model_id="qwen-7b",
    )
    assert hit.hit
    assert hit.similarity >= 0.95
    assert "fixed-size" in (hit.response or "")

    # Different model namespace must not cross-contaminate
    other_model = cache.lookup(
        "Explain how PagedAttention reduces KV cache fragmentation in vLLM",
        model_id="llama-8b",
    )
    assert not other_model.hit

    # Unrelated prompt misses
    miss = cache.lookup("What is the weather in Tokyo tomorrow?", model_id="qwen-7b")
    assert not miss.hit
    assert cache.stats().hits == 1
    assert cache.stats().misses == 2


def test_semantic_cache_ttl_and_lru_eviction():
    clock = _Clock()
    cache = SemanticResponseCache(similarity_threshold=0.9, max_entries=2, ttl_seconds=10.0, clock=clock)
    cache.store("alpha prompt", "resp-a")
    cache.store("beta prompt", "resp-b")
    # Refresh alpha so beta is LRU
    assert cache.lookup("alpha prompt").hit
    cache.store("gamma prompt", "resp-c")
    assert cache.stats().evictions == 1
    assert not cache.lookup("beta prompt").hit
    assert cache.lookup("alpha prompt").hit

    # Advance past TTL -> entries expire
    clock.advance(11.0)
    assert not cache.lookup("alpha prompt").hit
    assert cache.stats().expirations >= 1


def test_request_coalescer_deduplicates_concurrent_calls():
    async def _run() -> None:
        coalescer = RequestCoalescer()
        calls = 0

        async def _backend() -> str:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.02)
            return f"generated-{calls}"

        results = await asyncio.gather(*[coalescer.run("same-key", _backend) for _ in range(5)])
        values = [v for v, _ in results]
        flags = [c for _, c in results]
        assert calls == 1
        assert values == ["generated-1"] * 5
        assert flags.count(False) == 1
        assert flags.count(True) == 4
        assert coalescer.unique_executions == 1
        assert coalescer.coalesced_requests == 4

    asyncio.run(_run())


def test_cache_validation():
    with pytest.raises(ValueError):
        SemanticResponseCache(similarity_threshold=0.0)
    with pytest.raises(ValueError):
        SemanticResponseCache(max_entries=0)

"""Tests for radix-tree KV prefix-caching engine."""

import pytest

from infopt.prefix_cache import RadixPrefixCache


def test_prefix_insert_split_and_match():
    cache = RadixPrefixCache(max_cached_tokens=1024)
    sys_prompt = [1, 2, 3, 4, 5]
    q1 = sys_prompt + [10, 11, 12]
    q2 = sys_prompt + [20, 21]

    assert cache.insert(q1) == 8
    # Partial match on shared system prompt before inserting q2
    m = cache.match_prefix(q2)
    assert m.matched_tokens == 5
    assert m.remaining_tokens == [20, 21]
    assert m.hit_ratio == pytest.approx(5 / 7)
    assert len(m.kv_block_ids) == 5

    # Inserting q2 splits the edge after sys_prompt and adds only 2 new tokens
    assert cache.insert(q2) == 2
    assert cache.stats().cached_tokens == 10

    # Full match on both branches now
    assert cache.match_prefix(q1).hit_ratio == 1.0
    assert cache.match_prefix(q2).hit_ratio == 1.0


def test_lru_eviction_respects_recency_and_pins():
    cache = RadixPrefixCache(max_cached_tokens=10)
    cache.insert([1, 2, 3, 4, 5])  # oldest (5 tokens)
    cache.insert([10, 11, 12, 13, 14])  # newer (5 tokens)

    # Touch the first sequence so it becomes most recently used
    assert cache.match_prefix([1, 2, 3, 4, 5]).matched_tokens == 5

    # Insert a 3rd sequence -> should evict [10..14], not [1..5]
    cache.insert([20, 21, 22, 23])
    assert cache.stats().evictions >= 1
    assert cache.match_prefix([1, 2, 3, 4, 5]).matched_tokens == 5
    assert cache.match_prefix([10, 11, 12, 13, 14]).matched_tokens == 0


def test_pinned_prefix_is_protected_from_eviction():
    cache = RadixPrefixCache(max_cached_tokens=10)
    cache.insert([1, 2, 3, 4, 5])
    assert cache.pin([1, 2, 3, 4, 5]) == 5

    cache.insert([10, 11, 12, 13, 14])
    # Trigger eviction: even though [1..5] is older, it is pinned, so [10..14] is evicted
    cache.insert([20, 21, 22])
    assert cache.match_prefix([1, 2, 3, 4, 5]).matched_tokens == 5

    cache.unpin([1, 2, 3, 4, 5])
    freed = cache.evict_lru(5)
    assert freed >= 5


def test_validation_and_empty_queries():
    with pytest.raises(ValueError):
        RadixPrefixCache(max_cached_tokens=0)
    cache = RadixPrefixCache(max_cached_tokens=64)
    assert cache.insert([]) == 0
    assert cache.match_prefix([]).hit_ratio == 0.0
    with pytest.raises(ValueError):
        cache.insert([1, 2, 3], kv_block_ids=[1])

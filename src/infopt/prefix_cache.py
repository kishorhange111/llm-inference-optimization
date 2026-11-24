"""Radix-tree KV prefix-caching engine (SGLang RadixAttention style).

Stores tokenized prompt prefixes in a compressed radix tree so repeated system
prompts, few-shot examples, and multi-turn chat histories reuse already-computed
KV-cache blocks instead of re-running prefill. Supports edge splitting,
reference-count pinning for in-flight sequences, and LRU leaf eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _RadixNode:
    tokens: tuple[int, ...] = ()
    parent: _RadixNode | None = None
    children: dict[int, _RadixNode] = field(default_factory=dict)
    kv_block_ids: list[int] = field(default_factory=list)
    last_access: int = 0
    pin_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass(frozen=True)
class PrefixMatch:
    """Result of looking up a prompt's longest cached prefix."""

    matched_tokens: int
    total_tokens: int
    kv_block_ids: list[int]
    remaining_tokens: list[int]

    @property
    def hit_ratio(self) -> float:
        return self.matched_tokens / self.total_tokens if self.total_tokens > 0 else 0.0


@dataclass(frozen=True)
class PrefixCacheStats:
    """Aggregate reuse and eviction metrics for :class:`RadixPrefixCache`."""

    cached_tokens: int
    max_cached_tokens: int
    queries: int
    total_query_tokens: int
    total_reused_tokens: int
    token_hit_rate: float
    evictions: int
    node_count: int


def _common_prefix_len(a: tuple[int, ...] | list[int], b: tuple[int, ...] | list[int]) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


class RadixPrefixCache:
    """Compressed radix tree for automatic KV-cache prefix reuse with LRU eviction."""

    def __init__(self, max_cached_tokens: int = 65_536) -> None:
        if max_cached_tokens <= 0:
            raise ValueError("max_cached_tokens must be > 0")
        self.max_cached_tokens = max_cached_tokens
        self._root = _RadixNode()
        self._clock = 0
        self._cached_tokens = 0
        self._queries = 0
        self._query_tokens = 0
        self._reused_tokens = 0
        self._evictions = 0
        self._next_block_id = 1

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _alloc_blocks(self, n_tokens: int) -> list[int]:
        ids = list(range(self._next_block_id, self._next_block_id + n_tokens))
        self._next_block_id += n_tokens
        return ids

    def _split_node(self, node: _RadixNode, split_at: int) -> _RadixNode:
        """Split ``node`` at ``split_at`` tokens, returning the new upper parent node."""
        assert node.parent is not None
        upper = _RadixNode(
            tokens=node.tokens[:split_at],
            parent=node.parent,
            kv_block_ids=node.kv_block_ids[:split_at],
            last_access=node.last_access,
            pin_count=node.pin_count,
        )
        node.tokens = node.tokens[split_at:]
        node.kv_block_ids = node.kv_block_ids[split_at:]
        node.parent.children[upper.tokens[0]] = upper
        node.parent = upper
        upper.children[node.tokens[0]] = node
        return upper

    def insert(self, tokens: list[int], kv_block_ids: list[int] | None = None) -> int:
        """Insert a token sequence into the radix tree. Returns number of newly cached tokens."""
        if not tokens:
            return 0
        if kv_block_ids is not None and len(kv_block_ids) != len(tokens):
            raise ValueError("kv_block_ids length must match tokens length")

        ts = self._tick()
        node = self._root
        idx = 0
        newly_cached = 0

        while idx < len(tokens):
            first = tokens[idx]
            child = node.children.get(first)
            if child is None:
                rem = tuple(tokens[idx:])
                blocks = kv_block_ids[idx:] if kv_block_ids is not None else self._alloc_blocks(len(rem))
                leaf = _RadixNode(tokens=rem, parent=node, kv_block_ids=list(blocks), last_access=ts)
                node.children[first] = leaf
                self._cached_tokens += len(rem)
                newly_cached += len(rem)
                break

            cpl = _common_prefix_len(child.tokens, tokens[idx:])
            if cpl < len(child.tokens):
                child = self._split_node(child, cpl)

            child.last_access = ts
            node = child
            idx += cpl

        if self._cached_tokens > self.max_cached_tokens:
            self.evict_lru(self._cached_tokens - self.max_cached_tokens)

        return newly_cached

    def match_prefix(self, tokens: list[int]) -> PrefixMatch:
        """Find the longest cached prefix for ``tokens`` and refresh its LRU timestamps."""
        self._queries += 1
        self._query_tokens += len(tokens)
        if not tokens:
            return PrefixMatch(matched_tokens=0, total_tokens=0, kv_block_ids=[], remaining_tokens=[])

        ts = self._tick()
        node = self._root
        idx = 0
        matched_blocks: list[int] = []

        while idx < len(tokens):
            child = node.children.get(tokens[idx])
            if child is None:
                break
            cpl = _common_prefix_len(child.tokens, tokens[idx:])
            child.last_access = ts
            matched_blocks.extend(child.kv_block_ids[:cpl])
            idx += cpl
            if cpl < len(child.tokens):
                break
            node = child

        self._reused_tokens += idx
        return PrefixMatch(
            matched_tokens=idx,
            total_tokens=len(tokens),
            kv_block_ids=matched_blocks,
            remaining_tokens=list(tokens[idx:]),
        )

    def _walk_path(self, tokens: list[int]) -> list[_RadixNode]:
        path: list[_RadixNode] = []
        node = self._root
        idx = 0
        while idx < len(tokens):
            child = node.children.get(tokens[idx])
            if child is None:
                break
            cpl = _common_prefix_len(child.tokens, tokens[idx:])
            if cpl < len(child.tokens):
                break
            path.append(child)
            node = child
            idx += cpl
        return path

    def pin(self, tokens: list[int]) -> int:
        """Pin all nodes along the matched prefix so LRU eviction cannot remove them."""
        path = self._walk_path(tokens)
        for node in path:
            node.pin_count += 1
        return sum(len(n.tokens) for n in path)

    def unpin(self, tokens: list[int]) -> int:
        """Decrement pin count along the matched prefix."""
        path = self._walk_path(tokens)
        for node in path:
            node.pin_count = max(0, node.pin_count - 1)
        return sum(len(n.tokens) for n in path)

    def _collect_evictable_leaves(self) -> list[_RadixNode]:
        leaves: list[_RadixNode] = []
        stack = list(self._root.children.values())
        while stack:
            n = stack.pop()
            if n.is_leaf:
                if n.pin_count == 0:
                    leaves.append(n)
            else:
                stack.extend(n.children.values())
        return leaves

    def evict_lru(self, tokens_to_free: int) -> int:
        """Evict least-recently-used unpinned leaves until ``tokens_to_free`` tokens are freed."""
        freed = 0
        while freed < tokens_to_free:
            leaves = self._collect_evictable_leaves()
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            assert victim.parent is not None
            victim.parent.children.pop(victim.tokens[0], None)
            freed += len(victim.tokens)
            self._cached_tokens -= len(victim.tokens)
            self._evictions += 1
        return freed

    def _count_nodes(self) -> int:
        count = 0
        stack = list(self._root.children.values())
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children.values())
        return count

    def stats(self) -> PrefixCacheStats:
        hit_rate = self._reused_tokens / self._query_tokens if self._query_tokens > 0 else 0.0
        return PrefixCacheStats(
            cached_tokens=self._cached_tokens,
            max_cached_tokens=self.max_cached_tokens,
            queries=self._queries,
            total_query_tokens=self._query_tokens,
            total_reused_tokens=self._reused_tokens,
            token_hit_rate=hit_rate,
            evictions=self._evictions,
            node_count=self._count_nodes(),
        )

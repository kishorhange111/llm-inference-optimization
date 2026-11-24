"""Semantic response cache and single-flight request coalescing for LLM serving.

Reduces redundant GPU prefill/decode work by:
1. Answering semantically equivalent prompts from an embedding-indexed cache
   with configurable cosine similarity threshold, TTL, model scoping, and LRU eviction.
2. Coalescing concurrent identical in-flight requests (single-flight pattern) so
   bursty duplicate traffic triggers only one backend LLM call.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def hash_text_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words + char-trigram hashing embedding (unit-normalized)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_RE.findall(text.lower())
    features: list[str] = list(tokens)
    for tok in tokens:
        if len(tok) >= 3:
            features.extend(tok[i : i + 3] for i in range(len(tok) - 2))
    if not features:
        return vec
    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little")
        idx = val % dim
        sign = 1.0 if (val & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


@dataclass
class _CacheEntry:
    prompt: str
    model_id: str
    vector: np.ndarray
    response: str
    created_at: float


@dataclass(frozen=True)
class CacheLookupResult:
    hit: bool
    response: str | None = None
    similarity: float = 0.0
    matched_prompt: str | None = None


@dataclass(frozen=True)
class SemanticCacheStats:
    entries: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    expirations: int


class SemanticResponseCache:
    """Embedding-similarity LLM response cache with TTL, model scoping, and LRU eviction."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_entries: int = 1024,
        ttl_seconds: float = 3600.0,
        embedder: Callable[[str], np.ndarray] = hash_text_embedding,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._embedder = embedder
        self._clock = clock
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _prune_expired(self) -> None:
        now = self._clock()
        expired = [k for k, e in self._entries.items() if (now - e.created_at) >= self.ttl_seconds]
        for k in expired:
            self._entries.pop(k, None)
            self._expirations += 1

    def lookup(self, prompt: str, model_id: str = "default") -> CacheLookupResult:
        """Return the best cached response for ``model_id`` above ``similarity_threshold``."""
        self._prune_expired()
        qvec = self._embedder(prompt)
        qnorm = float(np.linalg.norm(qvec))
        if qnorm == 0:
            self._misses += 1
            return CacheLookupResult(hit=False)
        qvec = qvec / qnorm

        best_key: str | None = None
        best_sim = -1.0
        for key, entry in self._entries.items():
            if entry.model_id != model_id:
                continue
            sim = float(np.dot(qvec, entry.vector))
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None and best_sim >= self.similarity_threshold:
            entry = self._entries[best_key]
            self._entries.move_to_end(best_key)
            self._hits += 1
            return CacheLookupResult(
                hit=True,
                response=entry.response,
                similarity=best_sim,
                matched_prompt=entry.prompt,
            )

        self._misses += 1
        return CacheLookupResult(hit=False, similarity=max(0.0, best_sim))

    def store(self, prompt: str, response: str, model_id: str = "default") -> None:
        """Store a normalized prompt embedding and response in the cache."""
        self._prune_expired()
        vec = self._embedder(prompt)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        key = f"{model_id}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]}"
        self._entries[key] = _CacheEntry(
            prompt=prompt,
            model_id=model_id,
            vector=vec,
            response=response,
            created_at=self._clock(),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1

    def stats(self) -> SemanticCacheStats:
        total = self._hits + self._misses
        return SemanticCacheStats(
            entries=len(self._entries),
            hits=self._hits,
            misses=self._misses,
            hit_rate=self._hits / total if total > 0 else 0.0,
            evictions=self._evictions,
            expirations=self._expirations,
        )


class RequestCoalescer:
    """Async single-flight coalescer for identical concurrent inference requests."""

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Future] = {}
        self.unique_executions: int = 0
        self.coalesced_requests: int = 0

    async def run(self, key: str, factory: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Run ``factory`` once per concurrent ``key``; followers receive ``(result, True)``."""
        existing = self._in_flight.get(key)
        if existing is not None:
            self.coalesced_requests += 1
            return await existing, True

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._in_flight[key] = fut
        self.unique_executions += 1
        try:
            result = await factory()
            fut.set_result(result)
            return result, False
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        finally:
            self._in_flight.pop(key, None)

"""Serving resilience: circuit breaker, hedged fallback, and token-budget rate limiter.

Protects LLM inference gateways against cascading backend failures, tail-latency
spikes, and noisy-neighbor token exhaustion.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit breaker is OPEN."""


class RateLimitExceededError(RuntimeError):
    """Raised when a tenant exceeds its token or request budget."""


class CircuitBreaker:
    """Three-state circuit breaker with injectable clock for deterministic testing."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 30.0,
        half_open_success_threshold: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_success_threshold = half_open_success_threshold
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (self._clock() - self._opened_at) >= self.recovery_timeout_s
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_successes = 0
        return self._state

    def record_success(self) -> None:
        current = self.state
        if current == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._opened_at = None
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        current = self.state
        self._failure_count += 1
        if current == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()

    def call(self, fn: Callable[[], T]) -> T:
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker is OPEN")
        try:
            res = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return res


@dataclass
class _TenantBucket:
    tokens: float
    requests: float
    last_refill: float


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining_tokens: float
    retry_after_s: float = 0.0


class TokenBudgetRateLimiter:
    """Per-tenant dual token-bucket limiter (tokens/sec + requests/sec) with reconciliation."""

    def __init__(
        self,
        token_capacity: float,
        token_refill_per_sec: float,
        request_capacity: float = 60.0,
        request_refill_per_sec: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if token_capacity <= 0 or token_refill_per_sec <= 0:
            raise ValueError("Token capacity and refill rate must be > 0")
        self.token_capacity = float(token_capacity)
        self.token_refill_per_sec = float(token_refill_per_sec)
        self.request_capacity = float(request_capacity)
        self.request_refill_per_sec = float(request_refill_per_sec)
        self._clock = clock
        self._buckets: dict[str, _TenantBucket] = {}

    def _get_bucket(self, tenant_id: str) -> _TenantBucket:
        now = self._clock()
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            bucket = _TenantBucket(tokens=self.token_capacity, requests=self.request_capacity, last_refill=now)
            self._buckets[tenant_id] = bucket
            return bucket
        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(self.token_capacity, bucket.tokens + elapsed * self.token_refill_per_sec)
        bucket.requests = min(self.request_capacity, bucket.requests + elapsed * self.request_refill_per_sec)
        bucket.last_refill = now
        return bucket

    def allow(self, tenant_id: str, estimated_tokens: int) -> RateLimitDecision:
        """Attempt to consume ``estimated_tokens`` and 1 request from ``tenant_id``'s bucket."""
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be >= 0")
        bucket = self._get_bucket(tenant_id)
        if bucket.tokens >= estimated_tokens and bucket.requests >= 1.0:
            bucket.tokens -= estimated_tokens
            bucket.requests -= 1.0
            return RateLimitDecision(allowed=True, remaining_tokens=bucket.tokens, retry_after_s=0.0)

        tok_wait = max(0.0, (estimated_tokens - bucket.tokens) / self.token_refill_per_sec)
        req_wait = max(0.0, (1.0 - bucket.requests) / self.request_refill_per_sec)
        return RateLimitDecision(allowed=False, remaining_tokens=bucket.tokens, retry_after_s=max(tok_wait, req_wait))

    def reconcile(self, tenant_id: str, estimated_tokens: int, actual_tokens: int) -> float:
        """Refund unused tokens (or debit overage) once generation completes."""
        bucket = self._get_bucket(tenant_id)
        delta = estimated_tokens - actual_tokens
        bucket.tokens = max(0.0, min(self.token_capacity, bucket.tokens + delta))
        return bucket.tokens


@dataclass(frozen=True)
class ExecutionOutcome(Generic[T]):
    value: T
    backend_name: str
    attempts: int
    fell_back: bool


class HedgedFallbackExecutor(Generic[T]):
    """Executes across ordered backends with per-backend circuit breakers and fallback."""

    def __init__(self, backends: list[tuple[str, Callable[[], T], CircuitBreaker]]) -> None:
        if not backends:
            raise ValueError("At least one backend is required")
        self._backends = backends

    def execute(self) -> ExecutionOutcome[T]:
        errors: list[str] = []
        for idx, (name, fn, breaker) in enumerate(self._backends):
            try:
                val = breaker.call(fn)
                return ExecutionOutcome(value=val, backend_name=name, attempts=idx + 1, fell_back=idx > 0)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError(f"All inference backends failed: {'; '.join(errors)}")

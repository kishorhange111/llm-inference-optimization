"""Tests for circuit breaker, token-budget rate limiter, and hedged fallback."""

import pytest

from infopt.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    HedgedFallbackExecutor,
    TokenBudgetRateLimiter,
)


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_circuit_breaker_lifecycle():
    clock = _FakeClock()
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=10.0, clock=clock)
    assert cb.state == CircuitState.CLOSED

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "ok")

    clock.advance(10.0)
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.call(lambda: "recovered") == "recovered"
    assert cb.state == CircuitState.CLOSED


def test_token_budget_rate_limiter_and_reconcile():
    clock = _FakeClock()
    limiter = TokenBudgetRateLimiter(
        token_capacity=1000,
        token_refill_per_sec=100,
        request_capacity=5,
        request_refill_per_sec=1,
        clock=clock,
    )
    d1 = limiter.allow("tenant-a", 700)
    assert d1.allowed and d1.remaining_tokens == pytest.approx(300.0)

    d2 = limiter.allow("tenant-a", 500)
    assert not d2.allowed and d2.retry_after_s == pytest.approx(2.0)

    # Tenant isolation: tenant-b is unaffected
    assert limiter.allow("tenant-b", 500).allowed

    # Refund unused tokens from d1 (actual was 400 vs estimated 700 -> +300)
    rem = limiter.reconcile("tenant-a", estimated_tokens=700, actual_tokens=400)
    assert rem == pytest.approx(600.0)
    assert limiter.allow("tenant-a", 500).allowed

    # After waiting, tokens refill
    clock.advance(5.0)
    assert limiter.allow("tenant-a", 500).allowed


def test_hedged_fallback_executor():
    clock = _FakeClock()
    cb_primary = CircuitBreaker(failure_threshold=1, recovery_timeout_s=30.0, clock=clock)
    cb_secondary = CircuitBreaker(failure_threshold=2, recovery_timeout_s=30.0, clock=clock)

    executor = HedgedFallbackExecutor(
        [
            ("vllm-primary", lambda: (_ for _ in ()).throw(TimeoutError("gpu busy")), cb_primary),
            ("tgi-fallback", lambda: "fallback-ok", cb_secondary),
        ]
    )
    out = executor.execute()
    assert out.value == "fallback-ok"
    assert out.backend_name == "tgi-fallback"
    assert out.fell_back is True
    assert cb_primary.state == CircuitState.OPEN

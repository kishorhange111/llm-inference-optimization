"""Latency statistics and the standard LLM-serving metrics.

* **TTFT** (time to first token): how long a user stares at a blank screen.
  It is dominated by *prefill*, the one forward pass over the whole prompt, which is compute-bound.
* **TPOT** (time per output token, also called inter-token latency): how
  fast text streams after that. It is dominated by *decode*, one pass per
  token that has to read all the weights, so it is memory-bandwidth-bound.
* **Throughput**: total generated tokens per second across all concurrent
  requests. This is what determines cost per token.

Optimisations trade these off. Larger batches raise throughput but also raise
each request's TPOT. Quantisation lowers TPOT, because there are fewer bytes to read per token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RequestTiming:
    """Wall-clock timestamps (seconds) for one generation request."""

    start: float
    first_token: float
    end: float
    output_tokens: int

    @property
    def ttft(self) -> float:
        return self.first_token - self.start

    @property
    def tpot(self) -> float:
        """Average time between tokens after the first one (undefined for a 1-token output, returned as 0)."""
        return (self.end - self.first_token) / (self.output_tokens - 1) if self.output_tokens > 1 else 0.0

    @property
    def latency(self) -> float:
        return self.end - self.start


def percentiles(values: list[float], ps=(50, 90, 99)) -> dict[str, float]:
    """{'p50': ..., 'p90': ..., 'p99': ...}. Tail latency (p99) is what SLOs are written against."""
    if not values:
        return {f"p{p}": float("nan") for p in ps}
    arr = np.asarray(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


@dataclass
class BenchmarkResult:
    """Aggregated result of one benchmark configuration."""

    config: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    requests: int
    throughput_tok_s: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    latency_s: dict[str, float]
    peak_memory_gb: float | None = None
    extra: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(
    config: str,
    timings: list[RequestTiming],
    wall_seconds: float,
    batch_size: int,
    prompt_tokens: int,
    peak_memory_gb: float | None = None,
    extra: dict | None = None,
) -> BenchmarkResult:
    """Combine per-request timings into one :class:`BenchmarkResult`.

    Throughput uses total wall-clock time, not the sum of request latencies:
    concurrent requests overlap, and that overlap is exactly what batching buys.
    """
    total_out = sum(t.output_tokens for t in timings)
    return BenchmarkResult(
        config=config,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=round(total_out / max(len(timings), 1)),
        requests=len(timings),
        throughput_tok_s=total_out / wall_seconds if wall_seconds > 0 else 0.0,
        ttft_ms={k: v * 1000 for k, v in percentiles([t.ttft for t in timings]).items()},
        tpot_ms={k: v * 1000 for k, v in percentiles([t.tpot for t in timings]).items()},
        latency_s=percentiles([t.latency for t in timings]),
        peak_memory_gb=peak_memory_gb,
        extra=extra,
    )

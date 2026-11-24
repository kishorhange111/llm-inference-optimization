"""Prometheus metrics exporter for LLM inference serving observability.

Exposes histograms and gauges for TTFT, TPOT, token throughput, KV-cache
utilization, active continuous-batch size, and prefix/semantic cache hit rates.
"""

from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class InferenceMetrics:
    """Isolated Prometheus metric registry for LLM serving telemetry."""

    TTFT_BUCKETS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.2, 0.5)

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests_total = Counter(
            "infopt_requests_total",
            "Total LLM inference requests processed",
            labelnames=("model", "status"),
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "infopt_tokens_total",
            "Total prompt and generated output tokens",
            labelnames=("model", "direction"),
            registry=self.registry,
        )
        self.ttft_seconds = Histogram(
            "infopt_ttft_seconds",
            "Time to first token in seconds",
            labelnames=("model",),
            buckets=self.TTFT_BUCKETS,
            registry=self.registry,
        )
        self.tpot_seconds = Histogram(
            "infopt_tpot_seconds",
            "Time per output token (inter-token latency) in seconds",
            labelnames=("model",),
            buckets=self.TPOT_BUCKETS,
            registry=self.registry,
        )
        self.kv_cache_utilization = Gauge(
            "infopt_kv_cache_utilization_ratio",
            "Fraction of physical KV-cache blocks currently occupied",
            labelnames=("model",),
            registry=self.registry,
        )
        self.active_batch_size = Gauge(
            "infopt_active_batch_size",
            "Number of sequences currently in the active decode batch",
            labelnames=("model",),
            registry=self.registry,
        )
        self.cache_events_total = Counter(
            "infopt_cache_events_total",
            "Prefix and semantic cache hit/miss counters",
            labelnames=("cache_type", "outcome"),
            registry=self.registry,
        )

    def observe_request(
        self,
        model: str,
        prompt_tokens: int,
        output_tokens: int,
        ttft_seconds: float,
        tpot_seconds: float,
        status: str = "ok",
    ) -> None:
        """Record latency and token counters for a completed inference request."""
        self.requests_total.labels(model=model, status=status).inc()
        if prompt_tokens > 0:
            self.tokens_total.labels(model=model, direction="prompt").inc(prompt_tokens)
        if output_tokens > 0:
            self.tokens_total.labels(model=model, direction="output").inc(output_tokens)
        if ttft_seconds >= 0:
            self.ttft_seconds.labels(model=model).observe(ttft_seconds)
        if tpot_seconds >= 0:
            self.tpot_seconds.labels(model=model).observe(tpot_seconds)

    def set_kv_utilization(self, model: str, ratio: float) -> None:
        self.kv_cache_utilization.labels(model=model).set(max(0.0, min(1.0, ratio)))

    def set_active_batch(self, model: str, batch_size: int) -> None:
        self.active_batch_size.labels(model=model).set(max(0, batch_size))

    def record_cache_event(self, cache_type: str, hit: bool) -> None:
        self.cache_events_total.labels(cache_type=cache_type, outcome="hit" if hit else "miss").inc()

    def render_prometheus(self) -> str:
        """Serialize all registered metrics in Prometheus text exposition format."""
        return generate_latest(self.registry).decode("utf-8")


def attach_metrics_endpoint(app: FastAPI, metrics: InferenceMetrics, path: str = "/metrics") -> None:
    """Mount a Prometheus ``/metrics`` scrape endpoint onto a FastAPI application."""

    @app.get(path)
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render_prometheus(), media_type=CONTENT_TYPE_LATEST)

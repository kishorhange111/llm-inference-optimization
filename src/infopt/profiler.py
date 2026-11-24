"""Latency, throughput, and unit-economics profiler for LLM inference.

Computes serving SLO metrics (TTFT, TPOT, end-to-end latency at p50/p95/p99),
token throughput (prompt, generation, and total tok/s), GPU cost per 1M tokens,
and SLO compliance checks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from infopt.stats import RequestTiming, percentiles


@dataclass(frozen=True)
class GPUCostModel:
    """Hourly hardware cost and reference specs for $/1M-token estimation."""

    name: str
    hourly_usd: float
    memory_gb: float
    fp16_tflops: float
    bandwidth_gb_s: float

    def cost_per_second(self) -> float:
        return self.hourly_usd / 3600.0


# Representative on-demand cloud GPU rates (USD/hr) for cost-efficiency modeling.
GPU_CATALOG: dict[str, GPUCostModel] = {
    "T4": GPUCostModel("T4", hourly_usd=0.53, memory_gb=16.0, fp16_tflops=65.0, bandwidth_gb_s=300.0),
    "L4": GPUCostModel("L4", hourly_usd=0.80, memory_gb=24.0, fp16_tflops=121.0, bandwidth_gb_s=300.0),
    "A10G": GPUCostModel("A10G", hourly_usd=1.21, memory_gb=24.0, fp16_tflops=125.0, bandwidth_gb_s=600.0),
    "A100-40GB": GPUCostModel("A100-40GB", hourly_usd=2.21, memory_gb=40.0, fp16_tflops=312.0, bandwidth_gb_s=1555.0),
    "A100-80GB": GPUCostModel("A100-80GB", hourly_usd=3.67, memory_gb=80.0, fp16_tflops=312.0, bandwidth_gb_s=2039.0),
    "H100-80GB": GPUCostModel("H100-80GB", hourly_usd=6.98, memory_gb=80.0, fp16_tflops=989.0, bandwidth_gb_s=3350.0),
}


@dataclass(frozen=True)
class ProfileSample:
    """Single observed inference request with prompt/output token counts."""

    request_id: str
    timing: RequestTiming
    prompt_tokens: int
    succeeded: bool = True
    error: str | None = None

    @property
    def output_tokens(self) -> int:
        return self.timing.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.timing.output_tokens


@dataclass(frozen=True)
class SLOTargets:
    """Tail-latency and throughput SLO thresholds."""

    max_ttft_p95_ms: float = 500.0
    max_ttft_p99_ms: float = 1000.0
    max_tpot_p95_ms: float = 50.0
    max_tpot_p99_ms: float = 80.0
    min_output_tok_s: float = 100.0
    max_error_rate: float = 0.01


@dataclass(frozen=True)
class SLOCheckResult:
    """Pass/fail verdict against :class:`SLOTargets`."""

    passed: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class ProfileReport:
    """Aggregated latency, throughput, and cost profile for a serving run."""

    name: str
    requests_total: int
    requests_succeeded: int
    error_rate: float
    wall_seconds: float
    prompt_tokens_total: int
    output_tokens_total: int
    prompt_tok_s: float
    output_tok_s: float
    total_tok_s: float
    requests_per_second: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    e2e_latency_ms: dict[str, float]
    gpu_name: str | None = None
    gpu_count: int = 1
    cost_per_1m_output_tokens_usd: float | None = None
    cost_per_1m_total_tokens_usd: float | None = None

    def check_slo(self, targets: SLOTargets) -> SLOCheckResult:
        violations: list[str] = []
        if self.error_rate > targets.max_error_rate:
            violations.append(f"error_rate {self.error_rate:.4f} > {targets.max_error_rate:.4f}")
        if not math.isnan(self.ttft_ms["p95"]) and self.ttft_ms["p95"] > targets.max_ttft_p95_ms:
            violations.append(f"ttft_p95 {self.ttft_ms['p95']:.1f}ms > {targets.max_ttft_p95_ms:.1f}ms")
        if not math.isnan(self.ttft_ms["p99"]) and self.ttft_ms["p99"] > targets.max_ttft_p99_ms:
            violations.append(f"ttft_p99 {self.ttft_ms['p99']:.1f}ms > {targets.max_ttft_p99_ms:.1f}ms")
        if not math.isnan(self.tpot_ms["p95"]) and self.tpot_ms["p95"] > targets.max_tpot_p95_ms:
            violations.append(f"tpot_p95 {self.tpot_ms['p95']:.1f}ms > {targets.max_tpot_p95_ms:.1f}ms")
        if not math.isnan(self.tpot_ms["p99"]) and self.tpot_ms["p99"] > targets.max_tpot_p99_ms:
            violations.append(f"tpot_p99 {self.tpot_ms['p99']:.1f}ms > {targets.max_tpot_p99_ms:.1f}ms")
        if self.output_tok_s < targets.min_output_tok_s:
            violations.append(f"output_tok_s {self.output_tok_s:.1f} < {targets.min_output_tok_s:.1f}")
        return SLOCheckResult(passed=len(violations) == 0, violations=violations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class LatencyThroughputProfiler:
    """Collects per-request timings and summarizes latency, throughput, and cost."""

    PERCENTILES = (50, 95, 99)

    def __init__(
        self,
        name: str,
        gpu: GPUCostModel | str | None = None,
        gpu_count: int = 1,
    ) -> None:
        if gpu_count < 1:
            raise ValueError("gpu_count must be >= 1")
        if isinstance(gpu, str):
            if gpu not in GPU_CATALOG:
                raise KeyError(f"Unknown GPU model {gpu!r}; available: {sorted(GPU_CATALOG)}")
            self.gpu: GPUCostModel | None = GPU_CATALOG[gpu]
        else:
            self.gpu = gpu
        self.name = name
        self.gpu_count = gpu_count
        self._samples: list[ProfileSample] = []

    @property
    def samples(self) -> list[ProfileSample]:
        return list(self._samples)

    def record(
        self,
        request_id: str,
        start: float,
        first_token: float,
        end: float,
        prompt_tokens: int,
        output_tokens: int,
        succeeded: bool = True,
        error: str | None = None,
    ) -> ProfileSample:
        """Record one completed (or failed) request."""
        if prompt_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        if first_token < start or end < first_token:
            raise ValueError("Timestamps must satisfy start <= first_token <= end")
        sample = ProfileSample(
            request_id=request_id,
            timing=RequestTiming(start=start, first_token=first_token, end=end, output_tokens=output_tokens),
            prompt_tokens=prompt_tokens,
            succeeded=succeeded,
            error=error,
        )
        self._samples.append(sample)
        return sample

    def summarize(self, wall_seconds: float | None = None) -> ProfileReport:
        """Compute p50/p95/p99 latencies, throughput, and $/1M token unit economics."""
        total = len(self._samples)
        ok = [s for s in self._samples if s.succeeded]
        failed = total - len(ok)
        error_rate = failed / total if total > 0 else 0.0

        if wall_seconds is None:
            if self._samples:
                wall_seconds = max(s.timing.end for s in self._samples) - min(s.timing.start for s in self._samples)
            else:
                wall_seconds = 0.0

        prompt_total = sum(s.prompt_tokens for s in ok)
        output_total = sum(s.output_tokens for s in ok)
        all_tokens = prompt_total + output_total

        prompt_tok_s = prompt_total / wall_seconds if wall_seconds > 0 else 0.0
        output_tok_s = output_total / wall_seconds if wall_seconds > 0 else 0.0
        total_tok_s = all_tokens / wall_seconds if wall_seconds > 0 else 0.0
        rps = len(ok) / wall_seconds if wall_seconds > 0 else 0.0

        ttft_ms = {k: v * 1000.0 for k, v in percentiles([s.timing.ttft for s in ok], self.PERCENTILES).items()}
        tpot_ms = {k: v * 1000.0 for k, v in percentiles([s.timing.tpot for s in ok], self.PERCENTILES).items()}
        e2e_ms = {k: v * 1000.0 for k, v in percentiles([s.timing.latency for s in ok], self.PERCENTILES).items()}

        cost_1m_out: float | None = None
        cost_1m_total: float | None = None
        if self.gpu is not None:
            usd_per_sec = self.gpu.cost_per_second() * self.gpu_count
            if output_tok_s > 0:
                cost_1m_out = (1_000_000.0 / output_tok_s) * usd_per_sec
            if total_tok_s > 0:
                cost_1m_total = (1_000_000.0 / total_tok_s) * usd_per_sec

        return ProfileReport(
            name=self.name,
            requests_total=total,
            requests_succeeded=len(ok),
            error_rate=error_rate,
            wall_seconds=wall_seconds,
            prompt_tokens_total=prompt_total,
            output_tokens_total=output_total,
            prompt_tok_s=prompt_tok_s,
            output_tok_s=output_tok_s,
            total_tok_s=total_tok_s,
            requests_per_second=rps,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_latency_ms=e2e_ms,
            gpu_name=self.gpu.name if self.gpu else None,
            gpu_count=self.gpu_count,
            cost_per_1m_output_tokens_usd=cost_1m_out,
            cost_per_1m_total_tokens_usd=cost_1m_total,
        )


def compare_profiles(baseline: ProfileReport, candidate: ProfileReport) -> dict[str, float]:
    """Return relative improvements of ``candidate`` vs ``baseline`` (positive speedup / negative latency delta)."""

    def _pct_change(base: float, cand: float) -> float:
        if base == 0 or math.isnan(base) or math.isnan(cand):
            return 0.0
        return ((cand - base) / base) * 100.0

    deltas = {
        "ttft_p50_delta_pct": _pct_change(baseline.ttft_ms["p50"], candidate.ttft_ms["p50"]),
        "ttft_p95_delta_pct": _pct_change(baseline.ttft_ms["p95"], candidate.ttft_ms["p95"]),
        "ttft_p99_delta_pct": _pct_change(baseline.ttft_ms["p99"], candidate.ttft_ms["p99"]),
        "tpot_p50_delta_pct": _pct_change(baseline.tpot_ms["p50"], candidate.tpot_ms["p50"]),
        "tpot_p95_delta_pct": _pct_change(baseline.tpot_ms["p95"], candidate.tpot_ms["p95"]),
        "tpot_p99_delta_pct": _pct_change(baseline.tpot_ms["p99"], candidate.tpot_ms["p99"]),
        "output_tok_s_speedup": candidate.output_tok_s / baseline.output_tok_s if baseline.output_tok_s > 0 else 0.0,
    }
    if baseline.cost_per_1m_output_tokens_usd and candidate.cost_per_1m_output_tokens_usd:
        deltas["cost_1m_output_delta_pct"] = _pct_change(baseline.cost_per_1m_output_tokens_usd, candidate.cost_per_1m_output_tokens_usd)
    return deltas

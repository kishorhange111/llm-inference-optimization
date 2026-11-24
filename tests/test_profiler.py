"""Tests for latency, throughput, and cost-per-1M-token profiler."""

import json
import math

import pytest

from infopt.profiler import (
    GPU_CATALOG,
    GPUCostModel,
    LatencyThroughputProfiler,
    SLOTargets,
    compare_profiles,
)


def test_profiler_percentiles_throughput_and_cost():
    profiler = LatencyThroughputProfiler("a100-bf16", gpu="A100-40GB", gpu_count=1)
    # 10 overlapping requests across a 2.0s window
    for i in range(10):
        start = i * 0.1
        ttft = 0.05 + i * 0.01  # 50ms .. 140ms
        decode = 0.50  # 500ms for 50 additional tokens -> 10ms TPOT
        profiler.record(
            request_id=f"req-{i}",
            start=start,
            first_token=start + ttft,
            end=start + ttft + decode,
            prompt_tokens=256,
            output_tokens=51,
        )

    report = profiler.summarize(wall_seconds=2.0)
    assert report.requests_total == 10
    assert report.requests_succeeded == 10
    assert report.error_rate == 0.0
    assert report.output_tokens_total == 510
    assert report.prompt_tokens_total == 2560
    assert report.output_tok_s == pytest.approx(255.0)
    assert report.total_tok_s == pytest.approx((2560 + 510) / 2.0)
    assert report.ttft_ms["p50"] == pytest.approx(95.0)
    assert report.ttft_ms["p95"] > report.ttft_ms["p50"]
    assert report.ttft_ms["p99"] >= report.ttft_ms["p95"]
    assert report.tpot_ms["p50"] == pytest.approx(10.0)

    # Cost per 1M output tokens = (1e6 / 255) * (2.21 / 3600)
    expected_cost = (1_000_000.0 / 255.0) * (GPU_CATALOG["A100-40GB"].hourly_usd / 3600.0)
    assert report.cost_per_1m_output_tokens_usd == pytest.approx(expected_cost)
    assert report.cost_per_1m_total_tokens_usd < report.cost_per_1m_output_tokens_usd


def test_profiler_slo_check_and_error_rate():
    profiler = LatencyThroughputProfiler("slo-test", gpu="L4")
    profiler.record("r1", 0.0, 0.2, 1.0, prompt_tokens=100, output_tokens=41)
    profiler.record("r2", 0.0, 0.9, 2.0, prompt_tokens=100, output_tokens=41)
    profiler.record("r3", 0.0, 0.1, 0.1, prompt_tokens=100, output_tokens=0, succeeded=False, error="timeout")

    report = profiler.summarize()
    assert report.error_rate == pytest.approx(1 / 3)
    verdict = report.check_slo(
        SLOTargets(
            max_ttft_p95_ms=500.0,
            max_ttft_p99_ms=800.0,
            max_tpot_p95_ms=25.0,
            min_output_tok_s=100.0,
            max_error_rate=0.05,
        )
    )
    assert not verdict.passed
    assert any("error_rate" in v for v in verdict.violations)
    assert any("ttft_p95" in v for v in verdict.violations)
    assert any("output_tok_s" in v for v in verdict.violations)


def test_profiler_empty_and_validation():
    empty = LatencyThroughputProfiler("empty").summarize()
    assert empty.requests_total == 0
    assert empty.output_tok_s == 0.0
    assert math.isnan(empty.ttft_ms["p50"])
    assert empty.cost_per_1m_output_tokens_usd is None

    with pytest.raises(KeyError):
        LatencyThroughputProfiler("bad-gpu", gpu="UNKNOWN-GPU")
    with pytest.raises(ValueError):
        LatencyThroughputProfiler("bad-count", gpu_count=0)

    p = LatencyThroughputProfiler("val")
    with pytest.raises(ValueError):
        p.record("r1", 1.0, 0.5, 2.0, prompt_tokens=10, output_tokens=5)
    with pytest.raises(ValueError):
        p.record("r2", 0.0, 0.1, 0.2, prompt_tokens=-1, output_tokens=5)


def test_compare_profiles_and_json_roundtrip():
    custom_gpu = GPUCostModel("Custom", hourly_usd=4.0, memory_gb=80.0, fp16_tflops=500.0, bandwidth_gb_s=2000.0)
    base = LatencyThroughputProfiler("fp16", gpu=custom_gpu)
    cand = LatencyThroughputProfiler("awq-int4", gpu=custom_gpu)

    for i in range(5):
        base.record(f"b{i}", 0.0, 0.20, 1.20, prompt_tokens=128, output_tokens=101)
        cand.record(f"c{i}", 0.0, 0.10, 0.60, prompt_tokens=128, output_tokens=101)

    r_base = base.summarize(wall_seconds=1.2)
    r_cand = cand.summarize(wall_seconds=0.6)
    deltas = compare_profiles(r_base, r_cand)

    assert deltas["ttft_p50_delta_pct"] == pytest.approx(-50.0)
    assert deltas["tpot_p50_delta_pct"] == pytest.approx(-50.0)
    assert deltas["output_tok_s_speedup"] == pytest.approx(2.0)
    assert deltas["cost_1m_output_delta_pct"] == pytest.approx(-50.0)

    payload = json.loads(r_cand.to_json())
    assert payload["name"] == "awq-int4"
    assert payload["gpu_name"] == "Custom"

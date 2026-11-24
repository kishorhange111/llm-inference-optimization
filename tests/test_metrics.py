"""Tests for Prometheus metrics exporter and Grafana serving dashboard."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from infopt.metrics import InferenceMetrics, attach_metrics_endpoint
from infopt.router import ModelEndpoint, ModelRegistry, create_inference_app


def test_prometheus_metrics_recording_and_exposition():
    m = InferenceMetrics()
    m.observe_request(
        model="qwen2.5-7b-awq",
        prompt_tokens=256,
        output_tokens=64,
        ttft_seconds=0.08,
        tpot_seconds=0.015,
        status="ok",
    )
    m.set_kv_utilization("qwen2.5-7b-awq", 0.62)
    m.set_active_batch("qwen2.5-7b-awq", 12)
    m.record_cache_event("prefix", hit=True)
    m.record_cache_event("semantic", hit=False)

    text = m.render_prometheus()
    assert 'infopt_requests_total{model="qwen2.5-7b-awq",status="ok"} 1.0' in text
    assert 'infopt_tokens_total{direction="output",model="qwen2.5-7b-awq"} 64.0' in text
    assert 'infopt_kv_cache_utilization_ratio{model="qwen2.5-7b-awq"} 0.62' in text
    assert 'infopt_active_batch_size{model="qwen2.5-7b-awq"} 12.0' in text
    assert 'infopt_cache_events_total{cache_type="prefix",outcome="hit"} 1.0' in text
    assert "infopt_ttft_seconds_bucket" in text
    assert "infopt_tpot_seconds_bucket" in text


def test_metrics_endpoint_mounted_on_fastapi_router():
    reg = ModelRegistry()
    reg.register(ModelEndpoint(model_id="qwen-7b"))
    app = create_inference_app(reg)
    metrics = InferenceMetrics()
    metrics.observe_request("qwen-7b", 10, 20, 0.05, 0.01)
    attach_metrics_endpoint(app, metrics)

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "infopt_requests_total" in resp.text


def test_grafana_dashboard_json_references_real_metrics():
    dash_path = Path(__file__).resolve().parents[1] / "deploy" / "grafana" / "llm-serving-dashboard.json"
    dashboard = json.loads(dash_path.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "infopt-llm-serving"
    assert len(dashboard["panels"]) >= 6
    all_exprs = " ".join(t["expr"] for p in dashboard["panels"] for t in p["targets"])
    for expected in (
        "infopt_ttft_seconds_bucket",
        "infopt_tpot_seconds_bucket",
        "infopt_tokens_total",
        "infopt_kv_cache_utilization_ratio",
        "infopt_active_batch_size",
        "infopt_cache_events_total",
        "infopt_requests_total",
    ):
        assert expected in all_exprs

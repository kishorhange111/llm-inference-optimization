"""Tests for OpenAI-compatible FastAPI inference router and model registry."""

from fastapi.testclient import TestClient

from infopt.router import ModelEndpoint, ModelRegistry, create_inference_app
from infopt.serve_bench import parse_sse_line


def _make_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(
        ModelEndpoint(
            model_id="qwen2.5-7b-awq",
            backend="vllm",
            avg_ttft_ms=65.0,
            cost_per_1k_output_usd=0.0002,
            max_context_tokens=32768,
            handler=lambda prompt, max_tok: f"fast-reply ({max_tok})",
        ),
        aliases=["fast", "default"],
    )
    reg.register(
        ModelEndpoint(
            model_id="llama-3.1-8b-bf16",
            backend="tgi",
            avg_ttft_ms=140.0,
            cost_per_1k_output_usd=0.0005,
            max_context_tokens=8192,
            handler=lambda prompt, max_tok: "quality-reply",
        )
    )
    return reg


def test_health_and_models_listing():
    client = TestClient(create_inference_app(_make_registry()))
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    assert h.json()["healthy_models"] == 2

    m = client.get("/v1/models")
    assert m.status_code == 200
    ids = {item["id"] for item in m.json()["data"]}
    assert ids == {"qwen2.5-7b-awq", "llama-3.1-8b-bf16"}


def test_chat_completions_json_and_streaming_sse():
    client = TestClient(create_inference_app(_make_registry()))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "fast",
            "messages": [{"role": "user", "content": "Hello inference router"}],
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "qwen2.5-7b-awq"
    assert body["choices"][0]["message"]["content"] == "fast-reply (32)"
    assert body["usage"]["total_tokens"] >= 2

    stream_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "routing_strategy": "lowest_latency",
            "messages": [{"role": "user", "content": "Stream this"}],
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    pieces = [parse_sse_line(line) for line in stream_resp.text.splitlines()]
    text = "".join(p for p in pieces if p is not None)
    assert "fast-reply" in text


def test_routing_errors_and_unhealthy_failover():
    reg = _make_registry()
    client = TestClient(create_inference_app(reg))

    # Unknown model -> 404
    missing = client.post("/v1/completions", json={"model": "nonexistent", "prompt": "hi"})
    assert missing.status_code == 404

    # Mark fast model unhealthy -> explicit request returns 503, auto routes to healthy fallback
    reg.set_health("qwen2.5-7b-awq", False)
    unhealthy = client.post("/v1/completions", json={"model": "qwen2.5-7b-awq", "prompt": "hi"})
    assert unhealthy.status_code == 503

    fallback = client.post("/v1/completions", json={"model": "auto", "prompt": "hi"})
    assert fallback.status_code == 200
    assert fallback.json()["model"] == "llama-3.1-8b-bf16"

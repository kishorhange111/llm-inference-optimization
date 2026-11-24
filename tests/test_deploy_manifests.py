"""Validation tests for Docker, vLLM/TGI profiles, and Kubernetes GPU/HPA manifests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_and_compose_present():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "USER infopt" in dockerfile

    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    for token in ("router:", "vllm:", "tgi:", "prometheus:", "grafana:", "--enable-prefix-caching"):
        assert token in compose


def test_vllm_tgi_profiles_and_k8s_manifests():
    vllm_prof = (ROOT / "deploy" / "profiles" / "vllm-qwen7b-awq.yaml").read_text(encoding="utf-8")
    assert "quantization: awq" in vllm_prof
    assert "kv_cache_dtype: fp8" in vllm_prof

    tgi_prof = (ROOT / "deploy" / "profiles" / "tgi-llama8b-fp8.yaml").read_text(encoding="utf-8")
    assert "engine: tgi" in tgi_prof
    assert "quantize: fp8" in tgi_prof

    k8s_dep = (ROOT / "deploy" / "k8s" / "deployment-gpu.yaml").read_text(encoding="utf-8")
    assert "nvidia.com/gpu" in k8s_dep
    assert "prometheus.io/scrape" in k8s_dep
    assert "/dev/shm" in k8s_dep

    k8s_hpa = (ROOT / "deploy" / "k8s" / "hpa.yaml").read_text(encoding="utf-8")
    assert "HorizontalPodAutoscaler" in k8s_hpa
    assert "infopt_kv_cache_utilization_ratio" in k8s_hpa
    assert "infopt_active_batch_size" in k8s_hpa

# ADR-0003: Quantization Quality Gates, Serving Resilience & Autoscaling

- **Status:** Accepted
- **Date:** 2026-11-24
- **Modules:** `src/infopt/quantization.py`, `src/infopt/onnx_export.py`, `src/infopt/router.py`, `src/infopt/resilience.py`, `src/infopt/semantic_cache.py`, `src/infopt/metrics.py`

## Context

Aggressive weight quantization (INT4/AWQ/GPTQ) and edge export (ONNX Runtime) cut memory footprint and TPOT dramatically, but unmonitored compression can silently degrade task accuracy. Meanwhile, production inference endpoints require tenant isolation, overload protection, duplicate-query deduplication, and GPU-aware autoscaling.

## Decision

1. **Quantization & ONNX parity gates:** Pair every quantization preset (`BF16`, `FP8`, `INT8`, `INT4_NF4`, `AWQ`, `GPTQ`) with group-wise calibration (`SQNR`, cosine similarity, relative L2 error) and explicit perplexity delta gates (`evaluate_accuracy_drop`). Validate ONNX Runtime exports via `verify_numerical_parity` (max abs logit diff + top-1 token agreement).
2. **Resilient OpenAI-compatible gateway:** Front vLLM/TGI/ONNX backends with `create_inference_app` (`/v1/chat/completions` JSON & SSE streaming, `/v1/completions`, `/v1/models`, `/healthz`), per-backend `CircuitBreaker` + `HedgedFallbackExecutor`, per-tenant `TokenBudgetRateLimiter` (with post-generation token reconciliation), `SemanticResponseCache`, and async single-flight `RequestCoalescer`.
3. **KV-aware Kubernetes autoscaling:** Export `infopt_kv_cache_utilization_ratio`, `infopt_active_batch_size`, `infopt_ttft_seconds`, and `infopt_tpot_seconds` to Prometheus/Grafana and scale GPU replicas via `HorizontalPodAutoscaler` (`deploy/k8s/hpa.yaml`) on KV saturation rather than CPU utilization.

## Consequences

- **Pros:** Enforces measurable quality floors alongside speedups, prevents noisy-neighbor starvation and cascading GPU OOMs, and ensures CI catches latency/SQNR regressions automatically via `infopt.regression_cli`.
- **Cons:** Adds gateway configuration surface area (thresholds, circuit timeouts, cache similarity cutoffs) that must be tuned per model and tenant SLA.

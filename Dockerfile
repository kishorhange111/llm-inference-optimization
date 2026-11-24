# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[serve]"

FROM python:3.12-slim AS runtime

RUN useradd --create-home --shell /bin/bash infopt
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src ./src
COPY deploy/grafana ./deploy/grafana

USER infopt
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

CMD ["python", "-c", "import uvicorn; from infopt.router import ModelEndpoint, ModelRegistry, create_inference_app; from infopt.metrics import InferenceMetrics, attach_metrics_endpoint; reg = ModelRegistry(); reg.register(ModelEndpoint(model_id='qwen2.5-7b-awq', backend='vllm'), aliases=['default']); app = create_inference_app(reg); attach_metrics_endpoint(app, InferenceMetrics()); uvicorn.run(app, host='0.0.0.0', port=8000)"]

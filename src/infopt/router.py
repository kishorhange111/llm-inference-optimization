"""OpenAI-compatible FastAPI inference router and model registry.

Provides:
- :class:`ModelRegistry` with health tracking, alias mapping, and policy-based
  routing (``lowest_latency``, ``lowest_cost``, ``highest_weight``).
- :func:`create_inference_app` exposing ``/healthz``, ``/v1/models``,
  ``/v1/chat/completions`` (JSON & SSE streaming), and ``/v1/completions``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


@dataclass
class ModelEndpoint:
    """Registered serving backend entry in :class:`ModelRegistry`."""

    model_id: str
    backend: str = "vllm"
    max_context_tokens: int = 8192
    avg_ttft_ms: float = 120.0
    cost_per_1k_input_usd: float = 0.0002
    cost_per_1k_output_usd: float = 0.0006
    weight: float = 1.0
    healthy: bool = True
    tags: tuple[str, ...] = ()
    handler: Callable[[str, int], str] | None = field(default=None, repr=False)

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        if self.handler is not None:
            return self.handler(prompt, max_tokens)
        words = prompt.split()
        preview = " ".join(words[: min(8, len(words))])
        return f"[{self.model_id}:{self.backend}] {preview}"


class ModelRegistry:
    """In-memory registry of inference endpoints with routing policies."""

    def __init__(self) -> None:
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._aliases: dict[str, str] = {}

    def register(self, endpoint: ModelEndpoint, aliases: list[str] | None = None) -> None:
        self._endpoints[endpoint.model_id] = endpoint
        for alias in aliases or []:
            self._aliases[alias] = endpoint.model_id

    def unregister(self, model_id: str) -> None:
        self._endpoints.pop(model_id, None)
        self._aliases = {k: v for k, v in self._aliases.items() if v != model_id}

    def set_health(self, model_id: str, healthy: bool) -> None:
        if model_id not in self._endpoints:
            raise KeyError(f"Unknown model {model_id!r}")
        self._endpoints[model_id].healthy = healthy

    def get(self, model_id: str) -> ModelEndpoint | None:
        resolved = self._aliases.get(model_id, model_id)
        return self._endpoints.get(resolved)

    def list_models(self, healthy_only: bool = False) -> list[ModelEndpoint]:
        models = list(self._endpoints.values())
        if healthy_only:
            models = [m for m in models if m.healthy]
        return models

    def select_model(
        self,
        requested_model: str = "auto",
        strategy: Literal["lowest_latency", "lowest_cost", "highest_weight"] = "lowest_latency",
        min_context: int = 0,
    ) -> ModelEndpoint:
        """Resolve an explicit model/alias or pick the best healthy model for ``strategy``."""
        if requested_model != "auto":
            ep = self.get(requested_model)
            if ep is None:
                raise KeyError(f"Model {requested_model!r} is not registered")
            if not ep.healthy:
                raise RuntimeError(f"Model {ep.model_id!r} is currently unhealthy")
            return ep

        candidates = [m for m in self._endpoints.values() if m.healthy and m.max_context_tokens >= min_context]
        if not candidates:
            raise RuntimeError("No healthy models satisfy the routing constraints")

        if strategy == "lowest_cost":
            return min(candidates, key=lambda m: (m.cost_per_1k_output_usd, m.avg_ttft_ms))
        if strategy == "highest_weight":
            return max(candidates, key=lambda m: (m.weight, -m.avg_ttft_ms))
        return min(candidates, key=lambda m: (m.avg_ttft_ms, m.cost_per_1k_output_usd))


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    routing_strategy: Literal["lowest_latency", "lowest_cost", "highest_weight"] = "lowest_latency"


class CompletionRequest(BaseModel):
    model: str = "auto"
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=32768)
    stream: bool = False
    routing_strategy: Literal["lowest_latency", "lowest_cost", "highest_weight"] = "lowest_latency"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def create_inference_app(registry: ModelRegistry) -> FastAPI:
    """Create an OpenAI-compatible FastAPI inference router backed by ``registry``."""
    app = FastAPI(title="infopt OpenAI-Compatible Inference Router", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        healthy = registry.list_models(healthy_only=True)
        return {
            "status": "ok" if healthy else "degraded",
            "total_models": len(registry.list_models()),
            "healthy_models": len(healthy),
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": m.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": m.backend,
                    "metadata": {
                        "healthy": m.healthy,
                        "max_context_tokens": m.max_context_tokens,
                        "avg_ttft_ms": m.avg_ttft_ms,
                    },
                }
                for m in registry.list_models()
            ],
        }

    def _resolve(model: str, strategy: Literal["lowest_latency", "lowest_cost", "highest_weight"]) -> ModelEndpoint:
        try:
            return registry.select_model(model, strategy=strategy)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Any:
        endpoint = _resolve(req.model, req.routing_strategy)
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        text = endpoint.generate(prompt, req.max_tokens)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:

            def _stream_chunks() -> Iterator[str]:
                tokens = text.split(" ")
                for idx, tok in enumerate(tokens):
                    piece = tok if idx == 0 else f" {tok}"
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": endpoint.model_id,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream_chunks(), media_type="text/event-stream")

        prompt_toks = _estimate_tokens(prompt)
        completion_toks = _estimate_tokens(text)
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": endpoint.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
            },
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> dict[str, Any]:
        endpoint = _resolve(req.model, req.routing_strategy)
        text = endpoint.generate(req.prompt, req.max_tokens)
        prompt_toks = _estimate_tokens(req.prompt)
        completion_toks = _estimate_tokens(text)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": endpoint.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
            },
        }

    return app

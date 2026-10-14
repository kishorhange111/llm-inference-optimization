"""Online serving benchmark against an OpenAI-compatible streaming endpoint (e.g. ``vllm serve``).

This measures what users experience: many concurrent clients, each streaming
its answer. A static-batch offline benchmark can't show this. With continuous
batching (vLLM), new requests join the running batch at every decode step
instead of waiting for the whole batch to finish, so throughput scales with
concurrency while TTFT stays low.

For each concurrency level C we keep exactly C requests in flight (a closed-loop
load generator) until ``n_requests`` have completed, and record per-request
TTFT/TPOT from the token stream.
"""

from __future__ import annotations

import asyncio
import json
import time

from infopt.stats import BenchmarkResult, RequestTiming, summarize


def parse_sse_line(line: str) -> str | None:
    """Return the text delta from one ``data: {...}`` SSE line of a completions stream, else None."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return None
    choice = json.loads(payload)["choices"][0]
    return choice.get("text") or (choice.get("delta") or {}).get("content")


async def _one_request(client, url: str, model: str, prompt: str, max_tokens: int) -> RequestTiming:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    first = None
    chunks = 0
    async with client.stream("POST", url, json=body, timeout=600) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if parse_sse_line(line) is not None:
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    # vLLM streams one chunk per generated token; with ignore_eos the count is exactly max_tokens.
    return RequestTiming(start, first or end, end, max(chunks, 1))


async def _run_level(
    base_url: str, model: str, prompts: list[str], concurrency: int, max_tokens: int, transport=None
) -> tuple[list[RequestTiming], float]:
    import httpx

    url = f"{base_url.rstrip('/')}/completions"
    queue = list(prompts)
    timings: list[RequestTiming] = []

    # ``transport`` lets tests plug in a fake server; in real runs it is None (normal HTTP).
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency), transport=transport) as client:

        async def worker():
            # Each worker loops: take a prompt, send it, wait for it to finish. Exactly C requests stay in flight.
            while queue:
                timings.append(await _one_request(client, url, model, queue.pop(), max_tokens))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - start
    return timings, wall


def run_concurrency_sweep(
    base_url: str, model: str, prompts: list[str], levels: list[int], max_tokens: int, prompt_tokens: int, label: str, transport=None
) -> list[BenchmarkResult]:
    """Benchmark each concurrency level with the same prompt set."""
    results = []
    asyncio.run(_run_level(base_url, model, prompts[: max(levels)], max(levels), 16, transport))  # warm-up
    for c in levels:
        n = max(c * 4, 32)  # enough requests that the steady state dominates
        timings, wall = asyncio.run(_run_level(base_url, model, (prompts * (n // len(prompts) + 1))[:n], c, max_tokens, transport))
        res = summarize(f"{label} c={c}", timings, wall, c, prompt_tokens)
        results.append(res)
        print(
            json.dumps(
                {
                    "concurrency": c,
                    "throughput": round(res.throughput_tok_s),
                    "ttft_p50_ms": round(res.ttft_ms["p50"]),
                    "tpot_p50_ms": round(res.tpot_ms["p50"], 1),
                }
            ),
            flush=True,
        )
    return results

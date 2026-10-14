"""Tests for KV-cache math, latency statistics, SSE parsing and the load generator (fake server)."""

import json

import httpx
import pytest

from infopt.kv_cache import KNOWN, kv_bytes_per_token, kv_cache_gb, max_concurrent_sequences, weights_gb
from infopt.serve_bench import parse_sse_line, run_concurrency_sweep
from infopt.stats import RequestTiming, percentiles, summarize


def test_kv_bytes_qwen7b_matches_hand_calculation():
    # 2 (K,V) x 28 layers x 4 kv heads x 128 dims x 2 bytes = 57,344 bytes per token
    assert kv_bytes_per_token(KNOWN["Qwen2.5-7B"]) == 57_344
    assert kv_bytes_per_token(KNOWN["Qwen2.5-7B"], bytes_per_value=1) == 28_672  # fp8 halves it


def test_gqa_shrinks_kv_cache():
    gqa, mha = KNOWN["Llama-3.1-8B"], KNOWN["Llama-2-7B (no GQA)"]
    assert kv_bytes_per_token(mha) / kv_bytes_per_token(gqa) == 4  # 32 kv heads vs 8


def test_capacity_planning():
    q7 = KNOWN["Qwen2.5-7B"]
    assert weights_gb(q7, 16) == pytest.approx(14.19, abs=0.01)
    assert kv_cache_gb(q7, 4096, 32) == pytest.approx(7.0, abs=0.01)
    bf16 = max_concurrent_sequences(q7, 40, 4096)
    small = max_concurrent_sequences(q7, 40, 4096, weight_bits=4, kv_bytes=1)
    assert bf16 > 0 and small > 2 * bf16  # 4-bit weights + fp8 KV more than double capacity
    assert max_concurrent_sequences(q7, 10, 4096) == 0  # weights alone don't fit


def test_request_timing_and_summary():
    t = RequestTiming(start=0.0, first_token=0.2, end=1.2, output_tokens=11)
    assert t.ttft == pytest.approx(0.2) and t.tpot == pytest.approx(0.1) and t.latency == pytest.approx(1.2)
    assert RequestTiming(0, 1, 1, 1).tpot == 0.0
    res = summarize("x", [t, t], wall_seconds=1.2, batch_size=2, prompt_tokens=10)
    # Two overlapping requests in 1.2 s: throughput counts both.
    assert res.throughput_tok_s == pytest.approx(22 / 1.2)
    assert res.ttft_ms["p50"] == pytest.approx(200)


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p99"] == pytest.approx(99.01)


def test_parse_sse_line():
    assert parse_sse_line('data: {"choices":[{"text":"hi"}]}') == "hi"
    assert parse_sse_line('data: {"choices":[{"delta":{"content":"yo"}}]}') == "yo"
    assert parse_sse_line("data: [DONE]") is None and parse_sse_line(": keep-alive") is None


def test_concurrency_sweep_against_fake_streaming_server():
    """A fake OpenAI-compatible server that streams exactly max_tokens chunks."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        chunks = "".join(f"data: {json.dumps({'choices': [{'text': 't'}]})}\n\n" for _ in range(body["max_tokens"])) + "data: [DONE]\n\n"
        return httpx.Response(200, content=chunks.encode(), headers={"content-type": "text/event-stream"})

    results = run_concurrency_sweep(
        "http://fake/v1", "m", ["p1", "p2"], [1, 2], max_tokens=8, prompt_tokens=4, label="fake", transport=httpx.MockTransport(handler)
    )
    assert [r.batch_size for r in results] == [1, 2]
    assert all(r.output_tokens == 8 and r.requests == 32 for r in results)
    assert all(b["stream"] and b["ignore_eos"] for b in seen)


def test_hf_bench_on_tiny_cpu_model(tmp_path):
    """bench_generate end to end with a tiny random GPT-2 (no GPU, no downloads)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from infopt.hf_bench import bench_generate, perplexity

    words = ["the", "history", "of", "computing", "spans", "centuries", "from", "mechanical", "calculators", "to", "modern", "processors"]
    vocab = {"<pad>": 0, "<eos>": 1, "[UNK]": 2, **{w: i + 3 for i, w in enumerate(dict.fromkeys(words))}}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="[UNK]")
    tok.padding_side = "left"
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=256, n_embd=32, n_layer=2, n_head=2)).eval()
    res = bench_generate(model, tok, "tiny", batch_size=2, prompt_tokens=16, output_tokens=5, n_batches=2)
    assert res.requests == 4 and res.output_tokens == 5 and res.throughput_tok_s > 0
    assert res.ttft_ms["p50"] <= res.latency_s["p50"] * 1000
    ppl = perplexity(model, tok, " ".join(words * 20), window=64, stride=32)
    assert 1 < ppl < 10 * len(vocab)

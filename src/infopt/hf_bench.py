"""Benchmarks with Hugging Face ``transformers``: precision, quantisation, attention kernels, speculative decoding.

Measurement rules (each one prevents a classic benchmarking mistake):

* **Warm-up first.** The first generate() call pays for CUDA context set-up,
  kernel autotuning and memory-pool growth. We run one untimed warm-up per config.
* **Synchronise before reading the clock.** CUDA kernels run asynchronously,
  so ``time.perf_counter()`` without ``torch.cuda.synchronize()`` measures how
  long it took to *queue* the work, not to do it.
* **Fixed output length.** ``min_new_tokens == max_new_tokens`` stops early EOS
  from making one config look faster just because it wrote less.
* **Same prompts everywhere**, built deterministically from a seed.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

from infopt.stats import BenchmarkResult, RequestTiming, summarize


@dataclass(frozen=True)
class HFConfig:
    """One way of loading a model.

    Attributes:
        label: Name used in result tables.
        quantization: None (bf16/fp16 weights), "int8" (LLM.int8() outlier-aware 8-bit) or "nf4" (4-bit NormalFloat, as in QLoRA).
        attn_implementation: "eager" (plain PyTorch matmuls), "sdpa" (fused FlashAttention/mem-efficient kernels) ...
        dtype: compute dtype.
        compile: torch.compile the forward with a static KV cache (removes Python overhead per decode step).
    """

    label: str
    quantization: str | None = None
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"
    compile: bool = False


def load_model(model_id: str, cfg: HFConfig):
    """Load (model, tokenizer) according to ``cfg``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "torch_dtype": getattr(torch, cfg.dtype),
        "device_map": "cuda",
        "attn_implementation": cfg.attn_implementation,
    }
    if cfg.quantization == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif cfg.quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, cfg.dtype),  # de-quantise to bf16 for the matmul
            bnb_4bit_use_double_quant=True,  # also quantise the quantisation constants (saves ~0.4 bits/param)
        )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # batched generation: every prompt must end where generation starts
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if cfg.compile:
        # A static cache gives fixed tensor shapes, which torch.compile can turn into CUDA graphs.
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    return model, tok


def make_prompts(tok, n: int, prompt_tokens: int, seed: int = 0) -> list[str]:
    """Deterministic prompts of exactly ``prompt_tokens`` tokens, built from real text so attention patterns are realistic."""
    import random

    base = (
        "The history of computing spans centuries, from mechanical calculators to modern processors. "
        "Early machines used gears and levers, while later designs relied on vacuum tubes and transistors. "
        "Software evolved from hand-wired programs to high-level languages and operating systems. "
    )
    ids = tok(base * (prompt_tokens // 40 + 2), add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        start = rng.randint(0, len(ids) - prompt_tokens - 1)  # different offsets so prompts differ
        prompts.append(tok.decode(ids[start : start + prompt_tokens]))
    return prompts


def _sync(torch_mod) -> None:
    """Wait for queued GPU work to finish (no-op on CPU, so the code is testable without a GPU)."""
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


class _StepTimer:
    """A StoppingCriteria that never stops generation; it only records when each decode step finishes.

    generate() calls stopping criteria once per new token, so the first call
    is the moment the first token exists (TTFT), and the last call is the end.
    """

    def __init__(self, torch_mod):
        self.torch = torch_mod
        self.times: list[float] = []

    def __call__(self, input_ids, scores, **kwargs):
        _sync(self.torch)
        self.times.append(time.perf_counter())
        return self.torch.zeros(input_ids.shape[0], dtype=self.torch.bool, device=input_ids.device)


def bench_generate(
    model, tok, label: str, batch_size: int, prompt_tokens: int, output_tokens: int, n_batches: int = 3, extra: dict | None = None
) -> BenchmarkResult:
    """Static-batch generation benchmark: ``n_batches`` timed batches of ``batch_size`` prompts."""
    import torch
    from transformers import StoppingCriteriaList

    prompts = make_prompts(tok, batch_size * (n_batches + 1), prompt_tokens)
    gen = {"max_new_tokens": output_tokens, "min_new_tokens": output_tokens, "do_sample": False, "pad_token_id": tok.pad_token_id}

    def run(batch_prompts):
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        timer = _StepTimer(torch)
        _sync(torch)
        start = time.perf_counter()
        with torch.inference_mode():
            model.generate(**enc, **gen, stopping_criteria=StoppingCriteriaList([timer]))
        _sync(torch)
        end = time.perf_counter()
        return start, timer.times[0], end

    run(prompts[:batch_size])  # warm-up (not timed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timings: list[RequestTiming] = []
    wall = 0.0
    for b in range(1, n_batches + 1):
        start, first, end = run(prompts[b * batch_size : (b + 1) * batch_size])
        wall += end - start
        # In a static batch every request starts, gets its first token and finishes at the same time.
        timings += [RequestTiming(start, first, end, output_tokens) for _ in range(batch_size)]
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    return summarize(label, timings, wall, batch_size, prompt_tokens, peak, extra)


def perplexity(model, tok, text: str, max_tokens: int = 32_768, window: int = 2048, stride: int = 1024) -> float:
    """Sliding-window perplexity: the standard quality check for quantisation.

    Each window is scored only on its last ``stride`` tokens, so every scored
    token has at least ``window - stride`` tokens of context. Scoring
    fixed-size chunks with no overlap would inflate perplexity at the chunk starts.
    """
    import torch

    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :max_tokens].to(model.device)
    nll_sum, count = 0.0, 0
    prev_end = 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + window, ids.size(1))
        target_len = end - prev_end  # tokens not scored by an earlier window
        input_ids = ids[:, begin:end]
        labels = input_ids.clone()
        labels[:, :-target_len] = -100  # don't score the context-only prefix
        with torch.inference_mode():
            loss = model(input_ids, labels=labels).loss
        n = (labels[:, 1:] != -100).sum().item()
        nll_sum += loss.float().item() * n
        count += n
        prev_end = end
        if end == ids.size(1):
            break
    return float(torch.exp(torch.tensor(nll_sum / count)))


def bench_speculative(target_id: str, draft_id: str, prompts: list[str], output_tokens: int = 256) -> dict:
    """Assisted (speculative) decoding: a small draft model proposes tokens, the big model verifies them in one pass.

    With greedy decoding the output is *mathematically identical* to the big
    model alone (we check this). The speed-up depends on how often the
    draft's guesses are accepted.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    gen = {"max_new_tokens": output_tokens, "do_sample": False}

    def timed(**kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = target.generate(**kw, **gen)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t

    results = []
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        if i == 0:  # warm-up both paths once
            timed(**enc)
            timed(**enc, assistant_model=draft)
        base_out, base_t = timed(**enc)
        spec_out, spec_t = timed(**enc, assistant_model=draft)
        n_new = base_out.shape[1] - enc["input_ids"].shape[1]
        results.append(
            {
                "baseline_tok_s": n_new / base_t,
                "speculative_tok_s": (spec_out.shape[1] - enc["input_ids"].shape[1]) / spec_t,
                "identical_output": bool(torch.equal(base_out, spec_out)),
            }
        )
    # Both models are freed when this function returns (they are local variables).
    base = sum(r["baseline_tok_s"] for r in results) / len(results)
    spec = sum(r["speculative_tok_s"] for r in results) / len(results)
    return {
        "target": target_id,
        "draft": draft_id,
        "baseline_tok_s": base,
        "speculative_tok_s": spec,
        "speedup": spec / base,
        "identical_outputs": sum(r["identical_output"] for r in results),
        "n_prompts": len(results),
        "per_prompt": results,
    }


def free(model) -> None:
    """Release GPU memory between configurations."""
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()

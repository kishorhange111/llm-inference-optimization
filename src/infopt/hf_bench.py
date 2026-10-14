"""Benchmarks with Hugging Face ``transformers``: precision, quantisation, attention kernels, speculative decoding.

Measurement rules (each one prevents a classic benchmarking mistake):

* **Warm-up first.** The first generate() call pays for CUDA context set-up,
  kernel autotuning and memory-pool growth. We run one untimed warm-up per config.
* **Synchronise before reading the clock.** CUDA kernels run asynchronously,
  so ``time.perf_counter()`` without ``torch.cuda.synchronize()`` measures how
  long it took to *queue* the work, not to do it. But sync only at the ends of
  a timed region: syncing every step would perturb the decode loop itself.
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


def make_prompt_ids(tok, n: int, prompt_tokens: int, seed: int = 0) -> list[list[int]]:
    """Deterministic prompts of exactly ``prompt_tokens`` token ids, built from real text so attention patterns are realistic.

    Kept as ids (not decoded text) because decode -> re-encode can change the
    length by a token or two, which would make a static-cache compiled model
    recompile in the middle of a timed run.
    """
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
        prompts.append(ids[start : start + prompt_tokens])
    return prompts


def make_prompts(tok, n: int, prompt_tokens: int, seed: int = 0) -> list[str]:
    """The same prompts as text, for HTTP servers that take strings (lengths may differ by a token after re-tokenisation)."""
    return [tok.decode(ids) for ids in make_prompt_ids(tok, n, prompt_tokens, seed)]


def _sync(torch_mod) -> None:
    """Wait for queued GPU work to finish (no-op on CPU, so the code is testable without a GPU)."""
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def bench_generate(
    model, tok, label: str, batch_size: int, prompt_tokens: int, output_tokens: int, n_batches: int = 5, extra: dict | None = None
) -> BenchmarkResult:
    """Static-batch generation benchmark: ``n_batches`` timed batches of ``batch_size`` prompts.

    TTFT is measured with a *separate* ``max_new_tokens=1`` call on the same
    batch (prefill plus one token). Hooking every decode step and calling
    ``cuda.synchronize()`` inside it would stall the GPU pipeline and inflate
    the very TPOT we want to measure, so the full-length run is timed with
    only one sync at each end. TPOT is then ``(latency - TTFT) / (tokens - 1)``.

    In a static batch every request in a batch shares the same timings, so
    the percentiles are over *batches*; with the default 5 batches, p99 is
    close to the max and should be read as "worst batch".
    """
    import torch

    prompts = make_prompt_ids(tok, batch_size * (n_batches + 1), prompt_tokens)
    gen = {"do_sample": False, "pad_token_id": tok.pad_token_id}

    def timed(batch_prompts, n_new: int) -> float:
        # Every prompt has exactly prompt_tokens ids, so no padding is needed.
        ids = torch.tensor(batch_prompts, device=model.device)
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        _sync(torch)
        start = time.perf_counter()
        with torch.inference_mode():
            model.generate(**enc, **gen, max_new_tokens=n_new, min_new_tokens=n_new)
        _sync(torch)
        return time.perf_counter() - start

    # Warm-up (not timed): both shapes, so compile / autotuning cost is paid up front.
    timed(prompts[:batch_size], 1)
    timed(prompts[:batch_size], output_tokens)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timings: list[RequestTiming] = []
    wall = 0.0
    for b in range(1, n_batches + 1):
        batch = prompts[b * batch_size : (b + 1) * batch_size]
        ttft = timed(batch, 1)
        latency = timed(batch, output_tokens)
        wall += latency
        # Express as timestamps relative to 0 so RequestTiming's properties apply unchanged.
        timings += [RequestTiming(0.0, min(ttft, latency), latency, output_tokens) for _ in range(batch_size)]
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
    # Qwen2.5 checkpoints pad their embedding matrices to different sizes (152064 for 7B, 151936
    # for 0.5B) although the tokenizer is the same. Assisted generation requires equal vocab sizes,
    # so pad the draft's (the extra rows are never produced by the tokenizer).
    if draft.config.vocab_size != target.config.vocab_size:
        draft.resize_token_embeddings(target.config.vocab_size)
    gen = {"max_new_tokens": output_tokens, "do_sample": False}

    # Count target-model forward passes: in assisted decoding each one verifies a
    # whole block of draft tokens, so "new tokens per target forward" is the
    # practical acceptance measure (1.0 = no draft token ever accepted).
    calls = {"n": 0}

    def count_forward(module, args, output):
        calls["n"] += 1

    target.register_forward_hook(count_forward)

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
        calls["n"] = 0
        spec_out, spec_t = timed(**enc, assistant_model=draft)
        n_new = base_out.shape[1] - enc["input_ids"].shape[1]
        spec_new = spec_out.shape[1] - enc["input_ids"].shape[1]
        results.append(
            {
                "baseline_tok_s": n_new / base_t,
                "speculative_tok_s": spec_new / spec_t,
                "tokens_per_target_forward": spec_new / max(calls["n"], 1),
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
        "mean_tokens_per_target_forward": sum(r["tokens_per_target_forward"] for r in results) / len(results),
        "identical_outputs": sum(r["identical_output"] for r in results),
        "n_prompts": len(results),
        "per_prompt": results,
    }


def free(model) -> None:
    """Release GPU memory between configurations.

    The caller must also drop its own reference (``model = None``): a ``del``
    inside this function only removes the local name. Moving the parameters to
    the ``meta`` device frees their GPU storage even while references remain.
    """
    import torch

    try:
        model.to("meta")
    except Exception:  # noqa: BLE001 - some quantised models refuse .to(); fall back to plain collection
        pass
    gc.collect()
    torch.cuda.empty_cache()

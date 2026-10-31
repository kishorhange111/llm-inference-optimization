"""Run the inference experiments on a GPU and write JSON results.

python scripts/run_experiments.py quant       --model Qwen/Qwen2.5-7B-Instruct
python scripts/run_experiments.py attention   --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/run_experiments.py compile     --model Qwen/Qwen2.5-1.5B-Instruct
python scripts/run_experiments.py speculative --model Qwen/Qwen2.5-7B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct
python scripts/run_experiments.py serve       --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1 --label bf16
python scripts/run_experiments.py kv          (no GPU needed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infopt.hf_bench import HFConfig, bench_generate, bench_speculative, free, load_model, make_prompt_ids, perplexity
from infopt.kv_cache import KNOWN, kv_bytes_per_token, kv_cache_gb, max_concurrent_sequences, weights_gb

SPEC_PROMPTS = [
    "Explain how a hash map works and the time complexity of its operations.",
    "Write a Python function that checks whether a string is a palindrome, with comments.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "What is the difference between TCP and UDP? Give an example use case for each.",
    "Describe the water cycle to a ten-year-old.",
    "List the steps to deploy a containerized web app to Kubernetes.",
    "Write a SQL query that returns the top 5 customers by total order value.",
    "Explain gradient descent and why the learning rate matters.",
    "Translate 'The weather is nice today, let's go for a walk' into French, Spanish and German.",
    "Write a short story about a robot learning to paint, in about 150 words.",
    "What are the pros and cons of microservices compared to a monolith?",
    "Implement binary search in Python and explain its complexity.",
    "Give a recipe for a simple vegetable curry with quantities.",
    "Explain what a transformer's attention mechanism computes, step by step.",
    "Write a bash one-liner that counts lines in all .py files in a directory tree.",
    "Compare renewable and fossil energy sources in a short table.",
    "Explain the CAP theorem with a real-world example.",
    "Draft a polite email asking a colleague to review a pull request.",
    "Describe how HTTPS keeps a connection secure.",
    "Write a Python class for a bank account with deposit and withdraw methods.",
]


def wikitext() -> str:
    """WikiText-2 test split, the usual corpus for quantisation perplexity comparisons."""
    from datasets import load_dataset

    return "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])


def exp_quant(args) -> dict:
    """bf16 vs LLM.int8() vs NF4: memory, speed at several batch sizes, and perplexity."""
    text = wikitext()
    out = {"model": args.model, "prompt_tokens": 512, "output_tokens": 128, "configs": []}
    for cfg in (HFConfig("bf16"), HFConfig("int8 (bitsandbytes)", quantization="int8"), HFConfig("nf4 (bitsandbytes)", quantization="nf4")):
        model, tok = load_model(args.model, cfg)
        entry = {
            "config": cfg.label,
            "weights_gb": model.get_memory_footprint() / 2**30,
            "perplexity_wikitext2": perplexity(model, tok, text),
            "runs": [],
        }
        for bs in (1, 8, 32):
            entry["runs"].append(bench_generate(model, tok, cfg.label, bs, 512, 128).to_dict())
            print(json.dumps({k: entry["runs"][-1][k] for k in ("config", "batch_size", "throughput_tok_s", "peak_memory_gb")}), flush=True)
        out["configs"].append(entry)
        free(model)
    return out


def exp_attention(args) -> dict:
    """Eager attention vs fused SDPA kernels at growing prompt lengths (prefill is where they differ)."""
    out = {"model": args.model, "runs": []}
    for impl in ("eager", "sdpa"):
        model, tok = load_model(args.model, HFConfig(impl, attn_implementation=impl))
        for prompt in (512, 2048, 8192):
            try:
                out["runs"].append(bench_generate(model, tok, impl, 4, prompt, 64).to_dict())
            except Exception as err:  # noqa: BLE001 - an OOM at long context is a result, not a crash
                out["runs"].append({"config": impl, "prompt_tokens": prompt, "error": type(err).__name__})
            print(json.dumps(out["runs"][-1])[:300], flush=True)
        free(model)
    return out


def exp_compile(args) -> dict:
    """Eager decode vs torch.compile + static KV cache, batch size 1 (latency-bound chat setting)."""
    out = {"model": args.model, "runs": []}
    for cfg in (HFConfig("eager decode"), HFConfig("torch.compile + static cache", compile=True)):
        model, tok = load_model(args.model, cfg)
        out["runs"].append(bench_generate(model, tok, cfg.label, 1, 256, 256, n_batches=5).to_dict())
        print(json.dumps(out["runs"][-1])[:300], flush=True)
        free(model)
    return out


def exp_speculative(args) -> dict:
    return bench_speculative(args.model, args.draft, SPEC_PROMPTS, output_tokens=256)


def exp_serve(args) -> dict:
    from transformers import AutoTokenizer

    from infopt.serve_bench import run_concurrency_sweep

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    prompts = make_prompt_ids(tok, 64, 512)  # token ids, so every request is exactly 512 tokens
    res = run_concurrency_sweep(args.base_url, args.model, prompts, [1, 4, 16, 64], max_tokens=128, prompt_tokens=512, label=args.label)
    return {"model": args.model, "label": args.label, "runs": [r.to_dict() for r in res]}


def exp_kv(args) -> dict:
    """Pure arithmetic: KV-cache size per token and concurrent 4k-token sequences on an A100 40GB."""
    rows = []
    for name, shape in KNOWN.items():
        rows.append(
            {
                "model": name,
                "kv_kib_per_token_bf16": kv_bytes_per_token(shape) / 1024,
                "kv_gib_32x4k_bf16": kv_cache_gb(shape, 4096, 32),
                "weights_gib_bf16": weights_gb(shape, 16),
                "max_4k_seqs_a100_40gb_bf16": max_concurrent_sequences(shape, 40, 4096),
                "max_4k_seqs_a100_40gb_4bit_weights_fp8_kv": max_concurrent_sequences(shape, 40, 4096, weight_bits=4, kv_bytes=1),
            }
        )
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["quant", "attention", "compile", "speculative", "serve", "kv"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--label", default="vllm")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = globals()[f"exp_{args.experiment}"](args)
    out = Path(args.out or f"results/{args.experiment}{'_' + args.label if args.experiment == 'serve' else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

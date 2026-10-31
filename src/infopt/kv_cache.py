"""KV-cache memory math: why serving is usually limited by memory, not compute.

During generation every layer stores the Key and Value vectors of every
previous token, so attention for the next token doesn't recompute them:

    bytes = 2 (K and V) x layers x kv_heads x head_dim x tokens x bytes_per_value

For Qwen2.5-7B (28 layers, 4 KV heads thanks to grouped-query attention,
head_dim 128) in bf16, that is 57 KB per token. A batch of 32 requests x 4k
tokens needs 7.5 GB of KV cache *on top of* 15 GB of weights. Two levers follow:

* **Grouped-query attention (GQA)**: with 4 KV heads instead of 28, the cache is 7x smaller.
* **KV-cache quantisation (fp8)**: 2x smaller again.

vLLM's PagedAttention stores this cache in fixed-size blocks, like OS virtual
memory, so there's no fragmentation and many more sequences fit in memory at once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    """The architecture numbers that determine KV-cache size and weight memory."""

    name: str
    n_layers: int
    n_heads: int  # query heads
    n_kv_heads: int  # key/value heads (== n_heads without GQA)
    head_dim: int
    n_params: float  # total parameters

    @classmethod
    def from_hf_config(cls, name: str, cfg) -> ModelShape:
        """Read the shape from a ``transformers`` config object."""
        heads = cfg.num_attention_heads
        return cls(
            name=name,
            n_layers=cfg.num_hidden_layers,
            n_heads=heads,
            n_kv_heads=getattr(cfg, "num_key_value_heads", heads) or heads,
            head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // heads,
            n_params=0.0,
        )


# Shapes of common open models (from their published configs).
KNOWN = {
    "Qwen2.5-0.5B": ModelShape("Qwen2.5-0.5B", 24, 14, 2, 64, 0.49e9),
    "Qwen2.5-1.5B": ModelShape("Qwen2.5-1.5B", 28, 12, 2, 128, 1.54e9),
    "Qwen2.5-7B": ModelShape("Qwen2.5-7B", 28, 28, 4, 128, 7.62e9),
    "Llama-3.1-8B": ModelShape("Llama-3.1-8B", 32, 32, 8, 128, 8.03e9),
    "Llama-2-7B (no GQA)": ModelShape("Llama-2-7B (no GQA)", 32, 32, 32, 128, 6.74e9),
}


def kv_bytes_per_token(shape: ModelShape, bytes_per_value: float = 2.0) -> float:
    """KV-cache bytes for one token across all layers (2 = bf16/fp16, 1 = fp8)."""
    return 2 * shape.n_layers * shape.n_kv_heads * shape.head_dim * bytes_per_value


def kv_cache_gb(shape: ModelShape, tokens: int, batch: int = 1, bytes_per_value: float = 2.0) -> float:
    """Total KV-cache size in GiB for ``batch`` sequences of ``tokens`` tokens each."""
    return kv_bytes_per_token(shape, bytes_per_value) * tokens * batch / 2**30


def weights_gb(shape: ModelShape, bits: float = 16) -> float:
    """Weight memory in GiB at a given precision (16 = bf16, 8 = int8, 4 = NF4/AWQ, ignoring small overheads)."""
    return shape.n_params * bits / 8 / 2**30


def max_concurrent_sequences(
    shape: ModelShape, gpu_gb: float, tokens_per_seq: int, weight_bits: float = 16, kv_bytes: float = 2.0, utilization: float = 0.9
) -> int:
    """How many sequences of ``tokens_per_seq`` fit after loading the weights.

    ``utilization`` leaves headroom for activations and the CUDA context; vLLM's
    ``gpu_memory_utilization`` defaults to the same 0.9.
    """
    free = gpu_gb * utilization - weights_gb(shape, weight_bits)
    if free <= 0:
        return 0
    return int(free // kv_cache_gb(shape, tokens_per_seq, 1, kv_bytes))

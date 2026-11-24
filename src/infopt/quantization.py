"""Quantization specifications, group-wise calibration, and accuracy-drop analyzer.

Supports BF16, FP8, INT8, NF4/INT4, AWQ, and GPTQ quantization configurations,
performs real NumPy group-wise fake-quantization calibration on CPU to measure
SQNR (dB), cosine similarity, and relative L2 error, and evaluates accuracy-drop
quality gates alongside memory/bandwidth savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from infopt.kv_cache import ModelShape, weights_gb


class QuantScheme(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4_NF4 = "int4_nf4"
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass(frozen=True)
class QuantConfig:
    """Declarative quantization specification for weight/activation compression."""

    scheme: QuantScheme
    weight_bits: float
    activation_bits: float = 16.0
    group_size: int = 128
    symmetric: bool = True
    calibration_samples: int = 128
    zero_point: bool = False
    desc_act: bool = False

    def __post_init__(self) -> None:
        if self.weight_bits <= 0 or self.weight_bits > 32:
            raise ValueError("weight_bits must be in (0, 32]")
        if self.group_size <= 0:
            raise ValueError("group_size must be > 0")
        if self.calibration_samples < 0:
            raise ValueError("calibration_samples must be >= 0")

    def effective_bits_per_weight(self) -> float:
        """Include scale/zero-point metadata overhead per group."""
        if self.weight_bits >= 16:
            return self.weight_bits
        # 16-bit scale (+ optional 4-bit zero-point) amortized over group_size
        overhead_bits = (16.0 + (4.0 if self.zero_point else 0.0)) / self.group_size
        return self.weight_bits + overhead_bits


QUANT_PRESETS: dict[QuantScheme, QuantConfig] = {
    QuantScheme.BF16: QuantConfig(QuantScheme.BF16, weight_bits=16, activation_bits=16, group_size=128, calibration_samples=0),
    QuantScheme.FP8: QuantConfig(QuantScheme.FP8, weight_bits=8, activation_bits=8, group_size=128, symmetric=True, calibration_samples=64),
    QuantScheme.INT8: QuantConfig(
        QuantScheme.INT8, weight_bits=8, activation_bits=16, group_size=128, symmetric=True, calibration_samples=64
    ),
    QuantScheme.INT4_NF4: QuantConfig(
        QuantScheme.INT4_NF4, weight_bits=4, activation_bits=16, group_size=64, symmetric=True, calibration_samples=0
    ),
    QuantScheme.AWQ: QuantConfig(
        QuantScheme.AWQ, weight_bits=4, activation_bits=16, group_size=128, symmetric=False, zero_point=True, calibration_samples=128
    ),
    QuantScheme.GPTQ: QuantConfig(
        QuantScheme.GPTQ,
        weight_bits=4,
        activation_bits=16,
        group_size=128,
        symmetric=False,
        zero_point=True,
        desc_act=True,
        calibration_samples=128,
    ),
}


def quantize_dequantize_groupwise(
    tensor: np.ndarray,
    bits: int,
    group_size: int = 128,
    symmetric: bool = True,
) -> np.ndarray:
    """Fake-quantize ``tensor`` along its last dimension in groups of ``group_size``."""
    if bits >= 16:
        return tensor.astype(np.float32, copy=True)
    if bits < 2:
        raise ValueError("bits must be >= 2")

    orig_shape = tensor.shape
    flat = tensor.astype(np.float32).reshape(-1)
    pad = (-len(flat)) % group_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    groups = flat.reshape(-1, group_size)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        max_abs = np.max(np.abs(groups), axis=1, keepdims=True)
        scale = np.where(max_abs > 1e-12, max_abs / qmax, 1.0)
        q = np.clip(np.round(groups / scale), -qmax, qmax)
        deq = q * scale
    else:
        qmax = (1 << bits) - 1
        gmin = np.min(groups, axis=1, keepdims=True)
        gmax = np.max(groups, axis=1, keepdims=True)
        span = gmax - gmin
        scale = np.where(span > 1e-12, span / qmax, 1.0)
        zp = np.round(-gmin / scale)
        q = np.clip(np.round(groups / scale + zp), 0, qmax)
        deq = (q - zp) * scale

    out = deq.reshape(-1)[: flat.size - pad]
    return out.reshape(orig_shape)


@dataclass(frozen=True)
class CalibrationReport:
    """Measured quantization fidelity and memory/bandwidth impact."""

    scheme: QuantScheme
    effective_bits: float
    sqnr_db: float
    cosine_similarity: float
    relative_l2_error: float
    max_abs_error: float
    weights_gb: float | None = None
    memory_reduction_pct: float | None = None
    estimated_decode_speedup: float | None = None


@dataclass(frozen=True)
class AccuracyGateResult:
    """Verdict comparing quantized quality metrics against acceptable drop thresholds."""

    passed: bool
    ppl_delta_pct: float
    violations: list[str]


class CalibrationPipeline:
    """Evaluates quantization schemes on calibration activations & linear weights."""

    def calibrate_linear(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        config: QuantConfig,
        model_shape: ModelShape | None = None,
    ) -> CalibrationReport:
        """Measure output reconstruction error ``X @ W^T`` vs ``X @ Q(W)^T``."""
        w = weights.astype(np.float32)
        x = activations.astype(np.float32)

        # AWQ-style activation-aware salience scaling before quantization
        if config.scheme == QuantScheme.AWQ:
            act_scale = np.mean(np.abs(x), axis=0)
            s = np.power(np.clip(act_scale, 1e-6, None), 0.5)
            w_scaled = w * s[None, :]
            w_q = quantize_dequantize_groupwise(w_scaled, int(config.weight_bits), config.group_size, config.symmetric) / s[None, :]
        else:
            w_q = quantize_dequantize_groupwise(w, int(config.weight_bits), config.group_size, config.symmetric)

        y_ref = x @ w.T
        y_q = x @ w_q.T
        err = y_ref - y_q

        sig_power = float(np.mean(y_ref**2))
        noise_power = float(np.mean(err**2))
        sqnr_db = 10.0 * float(np.log10(sig_power / noise_power)) if noise_power > 1e-18 else 120.0

        ref_norm = float(np.linalg.norm(y_ref))
        q_norm = float(np.linalg.norm(y_q))
        cos_sim = float(np.sum(y_ref * y_q) / (ref_norm * q_norm)) if ref_norm > 0 and q_norm > 0 else 1.0
        rel_l2 = float(np.linalg.norm(err) / ref_norm) if ref_norm > 0 else 0.0
        max_abs = float(np.max(np.abs(err)))

        eff_bits = config.effective_bits_per_weight()
        w_gb: float | None = None
        mem_red: float | None = None
        speedup: float | None = None
        if model_shape is not None:
            bf16_gb = weights_gb(model_shape, 16.0)
            w_gb = weights_gb(model_shape, eff_bits)
            mem_red = (1.0 - w_gb / bf16_gb) * 100.0 if bf16_gb > 0 else 0.0
            # Decode is memory-bandwidth bound; dequant overhead leaves ~85% of raw bit-ratio speedup
            raw_ratio = 16.0 / eff_bits
            speedup = 1.0 if raw_ratio <= 1.0 else 1.0 + (raw_ratio - 1.0) * 0.65

        return CalibrationReport(
            scheme=config.scheme,
            effective_bits=eff_bits,
            sqnr_db=sqnr_db,
            cosine_similarity=cos_sim,
            relative_l2_error=rel_l2,
            max_abs_error=max_abs,
            weights_gb=w_gb,
            memory_reduction_pct=mem_red,
            estimated_decode_speedup=speedup,
        )

    @staticmethod
    def evaluate_accuracy_drop(
        baseline_ppl: float,
        quantized_ppl: float,
        report: CalibrationReport,
        max_ppl_delta_pct: float = 2.0,
        min_cosine_similarity: float = 0.99,
        min_sqnr_db: float = 18.0,
    ) -> AccuracyGateResult:
        """Check whether quantization stays within perplexity and reconstruction budgets."""
        if baseline_ppl <= 0:
            raise ValueError("baseline_ppl must be > 0")
        ppl_delta_pct = ((quantized_ppl - baseline_ppl) / baseline_ppl) * 100.0
        violations: list[str] = []
        if ppl_delta_pct > max_ppl_delta_pct:
            violations.append(f"ppl_delta {ppl_delta_pct:.2f}% > {max_ppl_delta_pct:.2f}%")
        if report.cosine_similarity < min_cosine_similarity:
            violations.append(f"cosine_similarity {report.cosine_similarity:.4f} < {min_cosine_similarity:.4f}")
        if report.sqnr_db < min_sqnr_db:
            violations.append(f"sqnr_db {report.sqnr_db:.1f} < {min_sqnr_db:.1f}")
        return AccuracyGateResult(passed=len(violations) == 0, ppl_delta_pct=ppl_delta_pct, violations=violations)

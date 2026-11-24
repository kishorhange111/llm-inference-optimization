"""Tests for quantization configs, groupwise calibration, and accuracy-drop analyzer."""

import numpy as np
import pytest

from infopt.kv_cache import KNOWN
from infopt.quantization import (
    QUANT_PRESETS,
    CalibrationPipeline,
    QuantConfig,
    QuantScheme,
    quantize_dequantize_groupwise,
)


def test_presets_and_effective_bits():
    bf16 = QUANT_PRESETS[QuantScheme.BF16]
    int8 = QUANT_PRESETS[QuantScheme.INT8]
    awq = QUANT_PRESETS[QuantScheme.AWQ]
    assert bf16.effective_bits_per_weight() == 16.0
    assert 8.0 < int8.effective_bits_per_weight() < 8.2
    assert 4.0 < awq.effective_bits_per_weight() < 4.3

    with pytest.raises(ValueError):
        QuantConfig(QuantScheme.INT4_NF4, weight_bits=0)


def test_int8_higher_sqnr_than_int4_and_awq_handles_outliers():
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.05, size=(64, 128)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(32, 128)).astype(np.float32)
    # Inject salient activation channels (typical LLM outlier channels)
    x[:, :4] *= 15.0

    pipe = CalibrationPipeline()
    r_bf16 = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.BF16], KNOWN["Qwen2.5-7B"])
    r_int8 = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.INT8], KNOWN["Qwen2.5-7B"])
    r_awq = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.AWQ], KNOWN["Qwen2.5-7B"])
    r_nf4 = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.INT4_NF4], KNOWN["Qwen2.5-7B"])

    assert r_bf16.sqnr_db >= 100.0
    assert r_int8.sqnr_db > r_nf4.sqnr_db
    assert r_awq.sqnr_db > r_nf4.sqnr_db  # salience scaling protects outlier channels
    assert r_awq.memory_reduction_pct > 70.0
    assert r_awq.estimated_decode_speedup > 2.0


def test_accuracy_drop_gate_pass_and_fail():
    rng = np.random.default_rng(1)
    w = rng.normal(0.0, 0.02, size=(32, 64)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(16, 64)).astype(np.float32)
    pipe = CalibrationPipeline()
    r_int8 = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.INT8])

    ok = pipe.evaluate_accuracy_drop(baseline_ppl=7.20, quantized_ppl=7.28, report=r_int8)
    assert ok.passed
    assert ok.ppl_delta_pct == pytest.approx((0.08 / 7.20) * 100.0)

    bad = pipe.evaluate_accuracy_drop(baseline_ppl=7.20, quantized_ppl=8.10, report=r_int8, max_ppl_delta_pct=2.0)
    assert not bad.passed
    assert any("ppl_delta" in v for v in bad.violations)


def test_groupwise_quantization_padding_and_errors():
    arr = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    deq = quantize_dequantize_groupwise(arr, bits=8, group_size=16, symmetric=False)
    assert deq.shape == arr.shape
    assert np.allclose(arr, deq, atol=0.02)

    with pytest.raises(ValueError):
        quantize_dequantize_groupwise(arr, bits=1)

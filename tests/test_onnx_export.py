"""Tests for ONNX Runtime / Optimum export and graph optimization wrapper."""

import json

import numpy as np
import pytest

from infopt.onnx_export import (
    ExecutionProvider,
    GraphOptLevel,
    ONNXExportConfig,
    ONNXGraphOptimizer,
    build_optimum_export_command,
    build_ort_session_options,
)


def test_optimum_command_and_manifest_export(tmp_path):
    cfg = ONNXExportConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        opset=17,
        opt_level=GraphOptLevel.ALL,
        providers=(ExecutionProvider.CUDA, ExecutionProvider.CPU),
        quantize_dynamic_int8=True,
        fp16=True,
    )
    cmd = build_optimum_export_command(cfg, tmp_path)
    assert cmd[:3] == ["optimum-cli", "export", "onnx"]
    assert "--fp16" in cmd
    assert "O3" in cmd

    opt = ONNXGraphOptimizer(cfg)
    artifact = opt.export_and_optimize(tmp_path, dry_run=True)
    assert "attention_fusion" in artifact.graph_passes
    assert "dynamic_int8_matmul" in artifact.graph_passes

    manifest = json.loads((tmp_path / "onnx_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert manifest["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_ort_session_options_when_installed():
    pytest.importorskip("onnxruntime")
    cfg = ONNXExportConfig(model_id="tiny-gpt2", opt_level=GraphOptLevel.EXTENDED, intra_op_num_threads=2)
    opts = build_ort_session_options(cfg)
    assert opts.intra_op_num_threads == 2


def test_numerical_parity_pass_and_fail():
    rng = np.random.default_rng(9)
    ref = rng.normal(0.0, 2.0, size=(8, 16, 64)).astype(np.float32)
    close = ref + rng.normal(0.0, 1e-4, size=ref.shape).astype(np.float32)
    drifted = ref + rng.normal(0.0, 1.5, size=ref.shape).astype(np.float32)

    ok = ONNXGraphOptimizer.verify_numerical_parity(ref, close, atol=1e-2)
    assert ok.passed
    assert ok.top1_agreement >= 0.99
    assert ok.cosine_similarity > 0.9999

    bad = ONNXGraphOptimizer.verify_numerical_parity(ref, drifted, atol=1e-2)
    assert not bad.passed
    assert bad.max_abs_diff > 1e-2

    with pytest.raises(ValueError):
        ONNXGraphOptimizer.verify_numerical_parity(ref, ref[:, :2, :])


def test_config_validation():
    with pytest.raises(ValueError):
        ONNXExportConfig(model_id="   ")
    with pytest.raises(ValueError):
        ONNXExportConfig(model_id="m", opset=11)

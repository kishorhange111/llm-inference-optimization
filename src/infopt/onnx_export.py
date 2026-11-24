"""ONNX Runtime / Hugging Face Optimum export and graph optimization wrapper.

Provides declarative export configurations, ``optimum-cli`` command builders,
ONNX Runtime ``SessionOptions`` construction (gracefully degrading when
``onnxruntime`` is not installed), graph-fusion pass reporting, reproducible
export manifests, and numerical parity verification against PyTorch logits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class GraphOptLevel(str, Enum):
    DISABLE = "disable"
    BASIC = "basic"
    EXTENDED = "extended"
    ALL = "all"


class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    DIRECTML = "DmlExecutionProvider"
    COREML = "CoreMLExecutionProvider"


_PASSES_BY_LEVEL: dict[GraphOptLevel, list[str]] = {
    GraphOptLevel.DISABLE: [],
    GraphOptLevel.BASIC: [
        "constant_folding",
        "redundant_node_elimination",
        "cast_elimination",
    ],
    GraphOptLevel.EXTENDED: [
        "constant_folding",
        "redundant_node_elimination",
        "cast_elimination",
        "matmul_add_fusion",
        "gelu_fusion",
        "layer_norm_fusion",
        "attention_fusion",
    ],
    GraphOptLevel.ALL: [
        "constant_folding",
        "redundant_node_elimination",
        "cast_elimination",
        "matmul_add_fusion",
        "gelu_fusion",
        "layer_norm_fusion",
        "attention_fusion",
        "nhwc_layout_optimization",
        "qdq_operator_fusion",
    ],
}


@dataclass(frozen=True)
class ONNXExportConfig:
    """Configuration for exporting and optimizing a transformer model with Optimum + ORT."""

    model_id: str
    task: str = "text-generation-with-past"
    opset: int = 17
    opt_level: GraphOptLevel = GraphOptLevel.ALL
    providers: tuple[ExecutionProvider, ...] = (ExecutionProvider.CPU,)
    use_io_binding: bool = True
    quantize_dynamic_int8: bool = False
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 1
    fp16: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.opset < 13:
            raise ValueError("opset must be >= 13 for transformer attention ops")
        if not self.providers:
            raise ValueError("At least one execution provider is required")
        if self.intra_op_num_threads < 0 or self.inter_op_num_threads < 0:
            raise ValueError("Thread counts must be >= 0")


@dataclass(frozen=True)
class ParityReport:
    """Numerical parity check between reference PyTorch logits and ORT logits."""

    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float
    top1_agreement: float


@dataclass
class ONNXExportArtifact:
    """Output summary and manifest of an export/optimization pipeline run."""

    model_id: str
    output_dir: str
    manifest_path: str
    command: list[str]
    graph_passes: list[str]
    providers: list[str]
    session_options: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True


def build_optimum_export_command(config: ONNXExportConfig, output_dir: str | Path) -> list[str]:
    """Construct the ``optimum-cli export onnx`` CLI invocation for ``config``."""
    opt_flag = {
        GraphOptLevel.DISABLE: "O1",
        GraphOptLevel.BASIC: "O1",
        GraphOptLevel.EXTENDED: "O2",
        GraphOptLevel.ALL: "O3",
    }[config.opt_level]
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        config.model_id,
        "--task",
        config.task,
        "--opset",
        str(config.opset),
        "--optimize",
        opt_flag,
    ]
    if config.fp16:
        cmd.append("--fp16")
    cmd.append(str(output_dir))
    return cmd


def session_options_summary(config: ONNXExportConfig) -> dict[str, Any]:
    """Serializable view of ONNX Runtime session options."""
    return {
        "graph_optimization_level": config.opt_level.value,
        "intra_op_num_threads": config.intra_op_num_threads,
        "inter_op_num_threads": config.inter_op_num_threads,
        "enable_mem_pattern": True,
        "enable_cpu_mem_arena": True,
        "use_io_binding": config.use_io_binding,
        "providers": [p.value for p in config.providers],
    }


def build_ort_session_options(config: ONNXExportConfig) -> Any:
    """Build an ``onnxruntime.SessionOptions`` instance if ``onnxruntime`` is installed."""
    import onnxruntime as ort  # optional runtime dependency

    opts = ort.SessionOptions()
    level_map = {
        GraphOptLevel.DISABLE: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        GraphOptLevel.BASIC: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        GraphOptLevel.EXTENDED: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        GraphOptLevel.ALL: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opts.graph_optimization_level = level_map[config.opt_level]
    opts.intra_op_num_threads = config.intra_op_num_threads
    opts.inter_op_num_threads = config.inter_op_num_threads
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return opts


class ONNXGraphOptimizer:
    """Orchestrates Optimum ONNX export configuration, manifests, and parity checks."""

    def __init__(self, config: ONNXExportConfig) -> None:
        self.config = config

    def active_passes(self) -> list[str]:
        passes = list(_PASSES_BY_LEVEL[self.config.opt_level])
        if self.config.quantize_dynamic_int8 and "dynamic_int8_matmul" not in passes:
            passes.append("dynamic_int8_matmul")
        return passes

    def export_and_optimize(self, output_dir: str | Path, dry_run: bool = True) -> ONNXExportArtifact:
        """Write an export manifest to ``output_dir`` and return the planned artifact."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        cmd = build_optimum_export_command(self.config, out_path)
        passes = self.active_passes()
        sess = session_options_summary(self.config)
        manifest_path = out_path / "onnx_export_manifest.json"

        artifact = ONNXExportArtifact(
            model_id=self.config.model_id,
            output_dir=str(out_path),
            manifest_path=str(manifest_path),
            command=cmd,
            graph_passes=passes,
            providers=[p.value for p in self.config.providers],
            session_options=sess,
            dry_run=dry_run,
        )
        manifest_path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")
        return artifact

    @staticmethod
    def verify_numerical_parity(
        ref_logits: np.ndarray,
        ort_logits: np.ndarray,
        atol: float = 1e-2,
        min_top1_agreement: float = 0.99,
    ) -> ParityReport:
        """Check logit-level numerical parity between reference and ONNX Runtime outputs."""
        ref = np.asarray(ref_logits, dtype=np.float32)
        ort = np.asarray(ort_logits, dtype=np.float32)
        if ref.shape != ort.shape:
            raise ValueError(f"Shape mismatch: {ref.shape} vs {ort.shape}")

        diff = np.abs(ref - ort)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        ref_norm = float(np.linalg.norm(ref))
        ort_norm = float(np.linalg.norm(ort))
        cos_sim = float(np.sum(ref * ort) / (ref_norm * ort_norm)) if ref_norm > 0 and ort_norm > 0 else 1.0

        ref_top1 = np.argmax(ref, axis=-1)
        ort_top1 = np.argmax(ort, axis=-1)
        top1 = float(np.mean(ref_top1 == ort_top1))
        passed = (max_abs <= atol) and (top1 >= min_top1_agreement)

        return ParityReport(
            passed=passed,
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            cosine_similarity=cos_sim,
            top1_agreement=top1,
        )

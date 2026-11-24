"""Automated latency/memory regression benchmark CLI and report generator.

Runs a deterministic CPU-only benchmark suite across the core ``infopt``
subsystems (continuous batching, KV paging, prefix caching, quantization
calibration, and speculative decoding), generates JSON and Markdown reports,
and gates CI builds against baseline regressions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from infopt.kv_cache import KNOWN, max_concurrent_sequences
from infopt.kv_paging import compare_contiguous_vs_paged_waste
from infopt.prefix_cache import RadixPrefixCache
from infopt.quantization import QUANT_PRESETS, CalibrationPipeline, QuantScheme
from infopt.scheduler import BatchPolicy, ContinuousBatchingScheduler, InferenceRequest, SchedulerConfig
from infopt.speculative import SpeculativeConfig, SpeculativeDecodingSimulator


@dataclass(frozen=True)
class RegressionThresholds:
    """Maximum allowed regression tolerances vs a saved baseline."""

    max_ttft_increase_pct: float = 10.0
    max_tpot_increase_pct: float = 10.0
    max_throughput_drop_pct: float = 10.0
    max_sqnr_drop_db: float = 1.5
    max_speedup_drop_pct: float = 10.0


@dataclass(frozen=True)
class RegressionVerdict:
    passed: bool
    regressions: list[str] = field(default_factory=list)


def run_smoke_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic CPU benchmarks across all optimization subsystems."""

    # 1. Continuous vs Static batching
    def _build_requests() -> list[InferenceRequest]:
        return [
            InferenceRequest(
                request_id=f"req-{i}",
                prompt_tokens=128,
                max_output_tokens=96 if i % 4 == 0 else 16,
                arrival_time=0.0 if i < 6 else 0.05,
            )
            for i in range(12)
        ]

    cont = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4, policy=BatchPolicy.CONTINUOUS))
    stat = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4, policy=BatchPolicy.STATIC))
    for r in _build_requests():
        cont.submit(r)
    for r in _build_requests():
        stat.submit(r)
    res_cont = cont.run_until_complete()
    res_stat = stat.run_until_complete()

    # 2. KV-cache capacity & paging savings
    q7 = KNOWN["Qwen2.5-7B"]
    cap_bf16 = max_concurrent_sequences(q7, gpu_gb=40.0, tokens_per_seq=4096)
    cap_q4_fp8 = max_concurrent_sequences(q7, gpu_gb=40.0, tokens_per_seq=4096, weight_bits=4, kv_bytes=1)
    paging = compare_contiguous_vs_paged_waste([128, 340, 768, 256, 1024], max_model_len=4096, block_size=16)

    # 3. Quantization calibration (AWQ vs INT8)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.04, size=(64, 128)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(32, 128)).astype(np.float32)
    x[:, :4] *= 10.0
    pipe = CalibrationPipeline()
    cal_int8 = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.INT8], q7)
    cal_awq = pipe.calibrate_linear(w, x, QUANT_PRESETS[QuantScheme.AWQ], q7)

    # 4. Speculative decoding
    spec = SpeculativeDecodingSimulator(SpeculativeConfig(gamma=5), seed=seed).simulate(num_tokens=400, alpha=0.82)

    # 5. Prefix cache reuse
    pcache = RadixPrefixCache(max_cached_tokens=4096)
    sys_tokens = list(range(1, 129))
    pcache.insert(sys_tokens + [500, 501, 502])
    pmatch = pcache.match_prefix(sys_tokens + [600, 601, 602, 603])

    return {
        "scheduler": {
            "continuous_output_tok_s": round(res_cont.report.output_tok_s, 2),
            "static_output_tok_s": round(res_stat.report.output_tok_s, 2),
            "continuous_ttft_p95_ms": round(res_cont.report.ttft_ms["p95"], 2),
            "continuous_tpot_p95_ms": round(res_cont.report.tpot_ms["p95"], 2),
            "batching_speedup": round(res_cont.report.output_tok_s / max(res_stat.report.output_tok_s, 1e-9), 3),
        },
        "kv_cache": {
            "qwen7b_4k_seqs_bf16": cap_bf16,
            "qwen7b_4k_seqs_q4_fp8": cap_q4_fp8,
            "paged_memory_savings_ratio": round(paging["memory_savings_ratio"], 4),
        },
        "quantization": {
            "int8_sqnr_db": round(cal_int8.sqnr_db, 2),
            "awq_sqnr_db": round(cal_awq.sqnr_db, 2),
            "awq_memory_reduction_pct": round(cal_awq.memory_reduction_pct or 0.0, 2),
        },
        "speculative": {
            "acceptance_rate": round(spec.acceptance_rate, 3),
            "speedup": round(spec.speedup, 3),
            "mean_tokens_per_target_pass": round(spec.mean_tokens_per_target_pass, 3),
        },
        "prefix_cache": {
            "hit_ratio": round(pmatch.hit_ratio, 4),
            "matched_tokens": pmatch.matched_tokens,
        },
    }


def check_regressions(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: RegressionThresholds | None = None,
) -> RegressionVerdict:
    """Compare candidate smoke-suite metrics against baseline tolerances."""
    t = thresholds or RegressionThresholds()
    regressions: list[str] = []

    b_sched, c_sched = baseline["scheduler"], candidate["scheduler"]
    if b_sched["continuous_ttft_p95_ms"] > 0:
        ttft_inc = (c_sched["continuous_ttft_p95_ms"] - b_sched["continuous_ttft_p95_ms"]) / b_sched["continuous_ttft_p95_ms"] * 100.0
        if ttft_inc > t.max_ttft_increase_pct:
            regressions.append(f"ttft_p95 regressed by +{ttft_inc:.1f}% (limit {t.max_ttft_increase_pct:.1f}%)")

    if b_sched["continuous_tpot_p95_ms"] > 0:
        tpot_inc = (c_sched["continuous_tpot_p95_ms"] - b_sched["continuous_tpot_p95_ms"]) / b_sched["continuous_tpot_p95_ms"] * 100.0
        if tpot_inc > t.max_tpot_increase_pct:
            regressions.append(f"tpot_p95 regressed by +{tpot_inc:.1f}% (limit {t.max_tpot_increase_pct:.1f}%)")

    if b_sched["continuous_output_tok_s"] > 0:
        tput_drop = (b_sched["continuous_output_tok_s"] - c_sched["continuous_output_tok_s"]) / b_sched["continuous_output_tok_s"] * 100.0
        if tput_drop > t.max_throughput_drop_pct:
            regressions.append(f"throughput dropped by -{tput_drop:.1f}% (limit {t.max_throughput_drop_pct:.1f}%)")

    sqnr_drop = baseline["quantization"]["awq_sqnr_db"] - candidate["quantization"]["awq_sqnr_db"]
    if sqnr_drop > t.max_sqnr_drop_db:
        regressions.append(f"awq_sqnr_db dropped by -{sqnr_drop:.2f} dB (limit {t.max_sqnr_drop_db:.2f} dB)")

    b_sp = baseline["speculative"]["speedup"]
    if b_sp > 0:
        sp_drop = ((b_sp - candidate["speculative"]["speedup"]) / b_sp) * 100.0
        if sp_drop > t.max_speedup_drop_pct:
            regressions.append(f"speculative speedup dropped by -{sp_drop:.1f}% (limit {t.max_speedup_drop_pct:.1f}%)")

    return RegressionVerdict(passed=len(regressions) == 0, regressions=regressions)


def render_markdown_report(suite: dict[str, Any]) -> str:
    """Format smoke-suite metrics as a GitHub-flavored Markdown summary table."""
    s = suite["scheduler"]
    kv = suite["kv_cache"]
    q = suite["quantization"]
    sp = suite["speculative"]
    pc = suite["prefix_cache"]
    lines = [
        "# Inference Optimization Smoke Benchmark Report",
        "",
        "| Subsystem | Metric | Value |",
        "|---|---|---:|",
        f"| Scheduler (Continuous) | Output throughput | {s['continuous_output_tok_s']} tok/s |",
        f"| Scheduler (Continuous) | Batching speedup vs static | {s['batching_speedup']}x |",
        f"| Scheduler (Continuous) | TTFT p95 | {s['continuous_ttft_p95_ms']} ms |",
        f"| Scheduler (Continuous) | TPOT p95 | {s['continuous_tpot_p95_ms']} ms |",
        f"| KV Cache (Qwen2.5-7B) | 4k seqs on A100-40GB (bf16 -> Q4+FP8) | {kv['qwen7b_4k_seqs_bf16']} -> {kv['qwen7b_4k_seqs_q4_fp8']} |",
        f"| KV Paging | Memory savings vs contiguous | {kv['paged_memory_savings_ratio'] * 100:.1f}% |",
        f"| Quantization (AWQ) | SQNR / Memory reduction | {q['awq_sqnr_db']} dB / {q['awq_memory_reduction_pct']}% |",
        f"| Speculative Decoding | Speedup (alpha=0.82, gamma=5) | {sp['speedup']}x ({sp['mean_tokens_per_target_pass']} tok/pass) |",
        f"| Prefix Cache | System-prompt token reuse | {pc['hit_ratio'] * 100:.1f}% ({pc['matched_tokens']} tok) |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="infopt regression benchmark CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run deterministic CPU smoke benchmarks")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--output", type=Path, default=None, help="Write JSON report to path")
    p_run.add_argument("--markdown", type=Path, default=None, help="Write Markdown report to path")

    p_cmp = sub.add_parser("compare", help="Compare candidate report against baseline")
    p_cmp.add_argument("--baseline", type=Path, required=True)
    p_cmp.add_argument("--candidate", type=Path, required=True)
    p_cmp.add_argument("--max-regression-pct", type=float, default=10.0)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        suite = run_smoke_suite(seed=args.seed)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(suite, indent=2), encoding="utf-8")
        md = render_markdown_report(suite)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(md, encoding="utf-8")
        print(md)
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    verdict = check_regressions(
        baseline,
        candidate,
        RegressionThresholds(
            max_ttft_increase_pct=args.max_regression_pct,
            max_tpot_increase_pct=args.max_regression_pct,
            max_throughput_drop_pct=args.max_regression_pct,
            max_speedup_drop_pct=args.max_regression_pct,
        ),
    )
    print(json.dumps(asdict(verdict), indent=2))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

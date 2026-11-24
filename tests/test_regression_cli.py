"""Tests for automated latency/memory regression benchmark CLI & report generator."""

import copy
import json

from infopt.regression_cli import (
    RegressionThresholds,
    check_regressions,
    main,
    render_markdown_report,
    run_smoke_suite,
)


def test_smoke_suite_and_markdown_report():
    suite = run_smoke_suite(seed=42)
    assert suite["scheduler"]["batching_speedup"] > 1.0
    assert suite["kv_cache"]["qwen7b_4k_seqs_q4_fp8"] > suite["kv_cache"]["qwen7b_4k_seqs_bf16"]
    assert suite["quantization"]["awq_memory_reduction_pct"] > 70.0
    assert suite["speculative"]["speedup"] > 1.5
    assert suite["prefix_cache"]["hit_ratio"] > 0.9

    md = render_markdown_report(suite)
    assert "| Subsystem | Metric | Value |" in md
    assert "Speculative Decoding" in md


def test_regression_gate_pass_and_fail():
    base = run_smoke_suite(seed=42)
    assert check_regressions(base, base).passed

    regressed = copy.deepcopy(base)
    regressed["scheduler"]["continuous_ttft_p95_ms"] *= 1.5
    regressed["scheduler"]["continuous_output_tok_s"] *= 0.7
    regressed["quantization"]["awq_sqnr_db"] -= 3.0

    verdict = check_regressions(base, regressed, RegressionThresholds(max_ttft_increase_pct=10.0))
    assert not verdict.passed
    assert len(verdict.regressions) >= 3


def test_cli_run_and_compare_exit_codes(tmp_path):
    base_json = tmp_path / "base.json"
    cand_json = tmp_path / "cand.json"
    md_path = tmp_path / "report.md"

    assert main(["run", "--output", str(base_json), "--markdown", str(md_path)]) == 0
    assert base_json.exists() and md_path.exists()

    assert main(["compare", "--baseline", str(base_json), "--candidate", str(base_json)]) == 0

    bad = json.loads(base_json.read_text(encoding="utf-8"))
    bad["speculative"]["speedup"] *= 0.5
    cand_json.write_text(json.dumps(bad), encoding="utf-8")
    assert main(["compare", "--baseline", str(base_json), "--candidate", str(cand_json)]) == 1

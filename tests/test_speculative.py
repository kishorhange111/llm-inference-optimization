"""Tests for speculative decoding simulator and acceptance-rate analyzer."""

import numpy as np
import pytest

from infopt.speculative import (
    SpeculativeConfig,
    SpeculativeDecodingSimulator,
    expected_tokens_per_step,
    optimal_lookahead_gamma,
    theoretical_speedup,
    verify_draft_tokens,
)


def test_analytical_expected_tokens_and_speedup():
    assert expected_tokens_per_step(0.0, 5) == pytest.approx(1.0)
    assert expected_tokens_per_step(1.0, 5) == pytest.approx(6.0)
    assert expected_tokens_per_step(0.8, 4) == pytest.approx((1 - 0.8**5) / 0.2)

    # High acceptance + cheap draft yields >2x speedup
    sp = theoretical_speedup(alpha=0.85, gamma=5, cost_ratio=0.08, verify_overhead=1.05)
    assert sp > 2.5

    # Very low acceptance can slow down decoding (<1x)
    slow = theoretical_speedup(alpha=0.1, gamma=6, cost_ratio=0.2, verify_overhead=1.05)
    assert slow < 1.0


def test_optimal_lookahead_gamma_increases_with_alpha():
    g_low, _ = optimal_lookahead_gamma(alpha=0.4, cost_ratio=0.1)
    g_high, s_high = optimal_lookahead_gamma(alpha=0.9, cost_ratio=0.08)
    assert g_high > g_low
    assert s_high > 3.0


def test_rejection_sampling_preserves_exact_target_distribution():
    """Verify 1-step rejection sampling matches target distribution p empirically."""
    rng = np.random.default_rng(123)
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.2, 0.4, 0.2, 0.2])
    counts = np.zeros(4)
    n_trials = 8000
    for _ in range(n_trials):
        draft_tok = int(rng.choice(4, p=q))
        emitted, _ = verify_draft_tokens([draft_tok], [q], [p, p], rng=rng)
        counts[emitted[0]] += 1
    empirical = counts / n_trials
    assert np.allclose(empirical, p, atol=0.025)


def test_simulator_runs_and_reflects_alpha():
    cfg = SpeculativeConfig(gamma=5, draft_step_ms=1.5, target_step_ms=20.0)
    high = SpeculativeDecodingSimulator(cfg, seed=7).simulate(num_tokens=500, alpha=0.85)
    low = SpeculativeDecodingSimulator(cfg, seed=7).simulate(num_tokens=500, alpha=0.20)

    assert high.tokens_generated == 500
    assert high.speedup > 2.0
    assert high.mean_tokens_per_target_pass > low.mean_tokens_per_target_pass
    assert high.speedup > low.speedup

    with pytest.raises(ValueError):
        SpeculativeDecodingSimulator(cfg).simulate(num_tokens=0, alpha=0.5)
    with pytest.raises(ValueError):
        expected_tokens_per_step(1.5, 4)

"""Speculative decoding draft/verify simulator and acceptance-rate analyzer.

Implements:
1. Exact rejection-sampling verification (Leviathan et al., 2023), proving that
   draft-then-verify preserves the exact target model output distribution.
2. Analytical expected tokens per verification step, theoretical wall-clock
   speedup, and optimal lookahead length ``gamma`` search.
3. Discrete-event simulation of draft/target step latencies and acceptance rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeculativeConfig:
    """Parameters for a draft + target speculative decoding setup."""

    gamma: int = 5  # number of draft tokens proposed per target verify pass
    draft_step_ms: float = 2.0
    target_step_ms: float = 20.0
    verify_overhead_factor: float = 1.05  # verifying gamma+1 tokens in parallel vs 1 token


@dataclass(frozen=True)
class SpeculativeMetrics:
    """Measured or simulated speculative decoding run statistics."""

    tokens_generated: int
    target_passes: int
    draft_passes: int
    total_draft_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    mean_tokens_per_target_pass: float
    baseline_latency_ms: float
    speculative_latency_ms: float
    speedup: float


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    """Expected accepted tokens + 1 bonus/resampled token per target verify step.

    Formula: ``(1 - alpha^(gamma + 1)) / (1 - alpha)`` for ``alpha != 1``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if abs(1.0 - alpha) < 1e-12:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def theoretical_speedup(alpha: float, gamma: int, cost_ratio: float, verify_overhead: float = 1.0) -> float:
    """Wall-clock speedup over autoregressive target decoding.

    ``cost_ratio`` is ``T_draft / T_target`` (e.g. 0.1 for a 0.5B draft + 7B target).
    """
    if cost_ratio < 0:
        raise ValueError("cost_ratio must be >= 0")
    e_tokens = expected_tokens_per_step(alpha, gamma)
    step_cost = gamma * cost_ratio + verify_overhead
    return e_tokens / step_cost


def optimal_lookahead_gamma(
    alpha: float,
    cost_ratio: float,
    max_gamma: int = 16,
    verify_overhead: float = 1.05,
) -> tuple[int, float]:
    """Return ``(best_gamma, best_speedup)`` maximizing theoretical speedup."""
    best_g = 1
    best_s = theoretical_speedup(alpha, 1, cost_ratio, verify_overhead)
    for g in range(2, max_gamma + 1):
        s = theoretical_speedup(alpha, g, cost_ratio, verify_overhead)
        if s > best_s:
            best_g, best_s = g, s
    return best_g, best_s


def verify_draft_tokens(
    draft_tokens: list[int],
    draft_probs: list[np.ndarray],
    target_probs: list[np.ndarray],
    rng: np.random.Generator | None = None,
) -> tuple[list[int], int]:
    """Run exact speculative rejection sampling on one draft block.

    ``target_probs`` has length ``len(draft_tokens) + 1`` (one per draft position
    plus the bonus token distribution if all draft tokens are accepted).

    Returns ``(emitted_tokens, accepted_draft_count)``.
    """
    if len(draft_probs) != len(draft_tokens):
        raise ValueError("draft_probs length must match draft_tokens")
    if len(target_probs) != len(draft_tokens) + 1:
        raise ValueError("target_probs must have length len(draft_tokens) + 1")

    rng = rng or np.random.default_rng()
    emitted: list[int] = []
    accepted = 0

    for tok, q, p in zip(draft_tokens, draft_probs, target_probs[:-1]):
        q_tok = float(q[tok])
        p_tok = float(p[tok])
        accept_prob = 1.0 if q_tok <= 0 else min(1.0, p_tok / q_tok)
        if rng.random() < accept_prob:
            emitted.append(tok)
            accepted += 1
        else:
            # Sample corrected token from normalized max(0, p - q)
            residual = np.maximum(0.0, p - q)
            total = float(residual.sum())
            probs = residual / total if total > 1e-12 else p / float(p.sum())
            corrected = int(rng.choice(len(probs), p=probs))
            emitted.append(corrected)
            return emitted, accepted

    # All draft tokens accepted: sample bonus token from target_probs[-1]
    bonus_p = target_probs[-1] / float(target_probs[-1].sum())
    bonus = int(rng.choice(len(bonus_p), p=bonus_p))
    emitted.append(bonus)
    return emitted, accepted


class SpeculativeDecodingSimulator:
    """Simulates multi-step speculative decoding at a configured acceptance rate."""

    def __init__(self, config: SpeculativeConfig | None = None, seed: int = 42) -> None:
        self.config = config or SpeculativeConfig()
        self.rng = np.random.default_rng(seed)

    def simulate(self, num_tokens: int, alpha: float) -> SpeculativeMetrics:
        """Simulate generating ``num_tokens`` with per-token draft acceptance probability ``alpha``."""
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        generated = 0
        target_passes = 0
        draft_passes = 0
        total_draft = 0
        accepted_draft = 0

        while generated < num_tokens:
            k = min(self.config.gamma, num_tokens - generated)
            draft_passes += k
            total_draft += k
            target_passes += 1

            step_accepted = 0
            for _ in range(k):
                if self.rng.random() < alpha:
                    step_accepted += 1
                else:
                    break
            accepted_draft += step_accepted
            # Accepted draft tokens + 1 bonus/resampled token from target
            emitted = min(step_accepted + 1, num_tokens - generated)
            generated += emitted

        baseline_ms = num_tokens * self.config.target_step_ms
        spec_ms = draft_passes * self.config.draft_step_ms + target_passes * self.config.target_step_ms * self.config.verify_overhead_factor
        return SpeculativeMetrics(
            tokens_generated=generated,
            target_passes=target_passes,
            draft_passes=draft_passes,
            total_draft_tokens=total_draft,
            accepted_draft_tokens=accepted_draft,
            acceptance_rate=accepted_draft / total_draft if total_draft > 0 else 0.0,
            mean_tokens_per_target_pass=generated / target_passes if target_passes > 0 else 0.0,
            baseline_latency_ms=baseline_ms,
            speculative_latency_ms=spec_ms,
            speedup=baseline_ms / spec_ms if spec_ms > 0 else 0.0,
        )
